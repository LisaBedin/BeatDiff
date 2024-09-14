import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'

from glob import glob
from functools import partial

import hydra
import matplotlib.pyplot as plt
import optax
import tqdm
import wandb
import numpy as np
from flax.linen import jit as nn_jit
from flax.training.train_state import TrainState
from flax.jax_utils import replicate, unreplicate, prefetch_to_device
from jax import random, disable_jit, jit, numpy as jnp, value_and_grad, tree_map, grad, clear_caches, devices, default_device
from jax.tree_util import tree_reduce, tree_flatten
from jax.lax import pmean
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from beat_net.variance_exploding_utils import heun_sampler, make_loss_fn, skip_scaling, output_scaling, \
    input_scaling, noise_scaling
from beat_net.data_loader import numpy_collate
from beat_net.data_loader import PhysionetECG, PhysionetNPZ
from beat_net.unet_parts import DenoiserNet
import orbax.checkpoint


def array_size(x):
    return x.size * x.itemsize


def get_module_size(params):
    return tree_reduce(lambda x, y: x + y, tree_map(array_size, params)) / 1e6


def calculate_weight_once(value):
    return jnp.linalg.norm(jnp.concatenate(tree_flatten(tree_map(lambda x: x.reshape(-1), value))[0]), ord=2).item()


def calculate_weight_norm(params):
    return {f'WNorm/{key}': calculate_weight_once(value) for key, value in params['params']['DhariwalUnet_0'].items()}


def update_step(x, class_features, keys, train_state, loss_fn):
    loss, grad = loss_fn(train_state.params, x, class_features, keys)
    #grad = pmean(grad, axis_name='devices')

    train_state = train_state.apply_gradients(grads=grad)
    return train_state, loss


def cv_loss_calculation(x, class_features, keys, train_state, loss_fn):
    loss = loss_fn(train_state.params, x, class_features, keys)

    return train_state, loss


@jit
def one_step_denoising(key_noise, noises_std, tr, x, class_labels):
    noise = random.normal(key_noise, shape=x.shape)
    corrupted_data = x + noise * noises_std[:, None, None]
    pred_x = tr.apply_fn(tr.params, corrupted_data, noises_std, class_labels)
    return corrupted_data, pred_x


@partial(jit, static_argnums=(1, 2))
def generate_keys_for_epoch(epoch_key, l, n_dev):
    return random.split(epoch_key, l * n_dev).reshape(l, n_dev, -1)


def run_epoch(dataloader, epoch_key, train_devices, train_state, update_fun, train_state_ema, device_ema, decay):
    n_dev = len(train_devices)
    epoch_loss = 0
    # pbar = tqdm.tqdm(
    #     prefetch_to_device(
    #         zip(
    #             generate_keys_for_epoch(epoch_key, len(dataloader), n_dev) if n_dev > 1 else random.split(epoch_key, len(dataloader)),
    #             dataloader
    #         ),
    #         size=2,
    #         devices=train_devices),
    #     position=0, leave=True)
    pbar = tqdm.tqdm(
        zip(
            generate_keys_for_epoch(epoch_key, len(dataloader), n_dev) if n_dev > 1 else random.split(epoch_key, len(dataloader)),
            dataloader
        ),
        position=0, leave=True, total=len(dataloader))
    for batch_key, batch_ecg in pbar:

        batch_ecg, batch_class_features = batch_ecg
        #clear_caches()
        #batch_ecg.block_until_ready()
        #save_device_memory_profile("memory.prof")
        train_state, loss = update_fun(
            jnp.array(batch_ecg, dtype=jnp.float32),
            jnp.array(batch_class_features, dtype=jnp.float32),
            batch_key,
            train_state
        )
        #train_state.block_until_ready()

        #save_device_memory_profile("memory.prof")
        epoch_loss += loss.mean().item()
        train_state_ema = train_state_ema.replace(params=update_train_state_ema(params_net=train_state.params,
                                                                                params_ema=train_state_ema.params,
                                                                                decay=decay))
        pbar.set_description(f'Loss : {loss.mean().item():.2f}')
    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss, train_state, train_state_ema


@partial(jit, static_argnums=(2,))
def update_train_state_ema(params_net, params_ema, decay):
    return optax.incremental_update(new_tensors=params_net, old_tensors=params_ema, step_size=1 - decay)


# done a second one to compile faster, since changing dataset length causes recompilation
@partial(jit, static_argnums=(1, 2))
def generate_keys_for_cv(epoch_key, l, n_dev):
    return random.split(epoch_key, l * n_dev).reshape(l, n_dev, -1)


def run_cv(dataloader, epoch_key, train_state, train_state_device, update_fun):
    epoch_loss = 0
    with default_device(train_state_device):
        pbar = tqdm.tqdm(
            zip(
                random.split(epoch_key, len(dataloader)),
                dataloader
            ),
            position=0, leave=True, total=len(dataloader))
        for batch_key, batch_ecg in pbar:
            batch_ecg, batch_class_features = batch_ecg
            train_state, loss = update_fun(
                batch_ecg,
                batch_class_features,
                batch_key,
                train_state
            )
            epoch_loss += loss.mean().item()
            pbar.set_description(f'Loss : {loss.mean().item():.2f}')
        epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


def initialize_net(data,
                   class_features,
                   device_ema,
                   device_train,
                   cfg):
    net_cfg = cfg.model
    exp_cfg = cfg.train
    diffusion_cfg = cfg.diffusion
    per_gpu_batch = cfg.train.batch_size_per_gpu
    key1, key2, key3 = random.split(random.PRNGKey(net_cfg.seed), 3)
    lr_schedule = optax.linear_schedule(init_value=exp_cfg.learning_rate_start,
                                        end_value=exp_cfg.learning_rate_final,
                                        transition_steps=exp_cfg.n_transition_steps)
    tx = optax.adam(learning_rate=lr_schedule)
    model = DenoiserNet(
        skip_scaling=partial(skip_scaling, sigma_data=diffusion_cfg.sigma_data),
        output_scaling=partial(output_scaling, sigma_data=diffusion_cfg.sigma_data),
        input_scaling=partial(input_scaling, sigma_data=diffusion_cfg.sigma_data),
        noise_conditioning=noise_scaling,
        model_channels=net_cfg.model_channels,
        num_blocks=net_cfg.num_blocks,
        channel_mult=net_cfg.channel_mult,
        channel_mult_emb=net_cfg.channel_mult_emb,
        attn_resolutions=net_cfg.attn_resolutions,
        dropout_rate=net_cfg.dropout_rate,
        use_f_training=net_cfg.use_f_training,
        training=True,
        conditional=net_cfg.conditional,
        embbed_position_in_signal=net_cfg.embbed_position_in_signal,
        dtype=jnp.float32,
    )
    t_steps = jnp.ones((data.shape[0],), jnp.float32)*100.0
    init_rngs = {'params': key2, 'dropout': key3}
    if len(device_train) > 1:
        train_state = replicate(TrainState.create(apply_fn=model.apply,
                                                  params=jit(model.init)(init_rngs, data, t_steps, class_features),
                                                  tx=tx), devices=device_train)
    else:
        train_state = TrainState.create(apply_fn=model.apply,
                                        params=jit(model.init)(init_rngs, data, t_steps, class_features),
                                        tx=tx)

    with default_device(device_ema):
        model_ema = DenoiserNet(
            skip_scaling=partial(skip_scaling, sigma_data=diffusion_cfg.sigma_data),
            output_scaling=partial(output_scaling, sigma_data=diffusion_cfg.sigma_data),
            input_scaling=partial(input_scaling, sigma_data=diffusion_cfg.sigma_data),
            noise_conditioning=noise_scaling,
            model_channels=net_cfg.model_channels,
            num_blocks=net_cfg.num_blocks,
            channel_mult=net_cfg.channel_mult,
            channel_mult_emb=net_cfg.channel_mult_emb,
            attn_resolutions=net_cfg.attn_resolutions,
            dropout_rate=net_cfg.dropout_rate,
            embbed_position_in_signal=net_cfg.embbed_position_in_signal,
            use_f_training=False,
            training=False,
            conditional=net_cfg.conditional,
            dtype=jnp.float32,
        )
        t_steps = jnp.ones((data.shape[0],), dtype=jnp.float32) * 100.0
        init_rngs = {'params': key2, 'dropout': key3}

        train_state_ema = TrainState.create(apply_fn=model_ema.apply,
                                            params=jit(model_ema.init)(init_rngs, data, t_steps, class_features),
                                            tx=optax.ema(decay=exp_cfg.decay))

    loss_fn = value_and_grad(make_loss_fn(
        apply_fn=train_state.apply_fn,
        batch_size=per_gpu_batch,
        p_mean=diffusion_cfg.p_mean,
        p_std=diffusion_cfg.p_std,
        sigma_data=diffusion_cfg.sigma_data,
        sigma_max=diffusion_cfg.sigma_max,
        sigma_min=diffusion_cfg.sigma_min,
        use_f_training=net_cfg.use_f_training,
    ),
        has_aux=False)

    update_fun = jit(partial(update_step, loss_fn=loss_fn)) #pmap(partial(update_step, loss_fn=loss_fn), in_axes=(0, 0, 0, 0), axis_name='devices', devices=device_train)
    cv_loss_fun = jit(partial(cv_loss_calculation,
                              loss_fn=make_loss_fn(
                                  apply_fn=train_state_ema.apply_fn,
                                  batch_size=per_gpu_batch,
                                  p_mean=diffusion_cfg.p_mean,
                                  p_std=diffusion_cfg.p_std,
                                  sigma_data=diffusion_cfg.sigma_data,
                                  sigma_max=diffusion_cfg.sigma_max,
                                  sigma_min=diffusion_cfg.sigma_min,
                                  use_f_training=False),
                              ))
    return update_fun, cv_loss_fun, train_state, train_state_ema, lr_schedule


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    path_to_save = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'model')
    print(path_to_save)
    def best_fn(metrics) -> float:
        return metrics['loss/CV']

    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=3, best_fn=best_fn, best_mode='min')
    mngr = orbax.checkpoint.CheckpointManager(
        path_to_save, orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        options)

    os.makedirs(path_to_save, mode=0o775, exist_ok=True)
    wandb.init(
        **cfg.wandb, config=OmegaConf.to_container(cfg, resolve=True)
    )

    available_devices = devices("cuda")
    n_dev = len(available_devices)
    print(f"N devices: {n_dev}")
    ema_device = available_devices[0]
    if len(available_devices) > 1:
        training_devices = available_devices[1:]
    else:
        training_devices = available_devices

    if 'centered' in cfg.dataset.database_path:
        all_files = np.array(glob(os.path.join(cfg.dataset.database_path, '*/*.npz')))
        kept_files = []
        all_mean, all_rr, all_age, all_sex = [], [], [], []
        for file_p in tqdm.tqdm(all_files, total=len(all_files), desc='filtering empty ecg'):
            npz = np.load(file_p)
            ecg = npz['data']
            if ecg.shape[1] >= 1:
                try:
                    rr = npz['rr']
                    sex, age = npz['sex'], npz['age']
                    if (sex != 'Unknown') and (age < 300) and (age >= 0) and (not np.isnan(age)):
                        kept_files.append(file_p)
                        all_mean.append(ecg.mean())
                        all_rr.append(rr.mean())
                        all_age.append(age)
                        all_sex.append(sex)
                except:
                    ()
        kept_files = np.array(kept_files)
        all_age, all_sex, all_mean, all_rr = np.array(all_age), np.array(all_sex), np.array(all_mean), np.array(all_rr)

        ind_train, ind_val = train_test_split(np.arange(len(kept_files)), test_size=0.1, random_state=0)
        ind_val, _ = train_test_split(ind_val, test_size=0.5, random_state=0)
        ind_train, ind_val = np.array(ind_train, dtype=int), np.array(ind_val, dtype=int)
        print(f'train = {len(ind_train)} | val = {len(ind_val)}')
        train_set = PhysionetNPZ(file_paths=kept_files[ind_train], training_class='Training', **cfg.dataset)
        val_set = PhysionetNPZ(file_paths=kept_files[ind_val], training_class='CV', **cfg.dataset)
    else:
        train_set = PhysionetECG(training_class='Training', estimate_std=False, **cfg.dataset)
        val_set = PhysionetECG(training_class='CV', estimate_std=False, **cfg.dataset)
    dataloader = DataLoader(dataset=train_set,
                            batch_size=cfg.train.batch_size_per_gpu * len(training_devices),
                            shuffle=True,
                            num_workers=10,
                            drop_last=True,
                            collate_fn=numpy_collate)#partial(numpy_collate, n_devices=len(training_devices)))
    dataloader_cv = DataLoader(dataset=val_set,
                               batch_size=cfg.train.batch_size_per_gpu,
                               shuffle=True,
                               num_workers=10,
                               drop_last=True,
                               collate_fn=numpy_collate)
    data_sample, feat_sample = dataloader.dataset[0]

    update_fun, cv_loss_fun, train_state, train_state_ema, lr_schedule = initialize_net(
        data=jnp.ones((1, *data_sample.shape), dtype=jnp.float32),
        class_features=jnp.ones((1, feat_sample.shape[0]), dtype=jnp.float32),
        device_ema=ema_device,
        device_train=training_devices,
        cfg=cfg,
    )

    print(f"Model size is {get_module_size(train_state_ema.params):.2f} MB")
    if cfg.diffusion.scale_type == 'linear':
        scaling_fun = lambda t: cfg.diffusion.scaling_coeff * t
    elif cfg.diffusion.scale_type == 'one':
        scaling_fun = lambda t: 1.0
    else:
        raise NotImplemented

    noise_fun = lambda t: jnp.clip(cfg.diffusion.noise_coeff * t, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max)
    class_label = jnp.repeat(jnp.array([1] + [0]*(feat_sample.shape[0]-1), dtype=jnp.float32)[None, :], repeats=cfg.generate.n_samples, axis=0)
    generate_fun = jit(partial(heun_sampler,
                               sigma_min=cfg.diffusion.sigma_min,
                               sigma_max=cfg.diffusion.sigma_max,
                               N=25, # to reduce eval time # cfg.generate.T,
                               p=cfg.generate.rho,
                               scale_fun=scaling_fun,
                               noise_fun=noise_fun,
                               scale_fun_grad=grad(scaling_fun),
                               noise_fun_grad=grad(noise_fun),
                               class_labels=class_label))
    run_epoch_train = partial(run_epoch,
                              train_devices=training_devices,
                              dataloader=dataloader,
                              update_fun=update_fun,
                              device_ema=ema_device,
                              decay=cfg.train.decay)
    run_epoch_cv = partial(run_cv,
                           train_state_device=ema_device,
                           dataloader=dataloader_cv,
                           update_fun=cv_loss_fun)

    n_epochs = cfg.train.n_iters
    pbar = tqdm.tqdm(enumerate(random.split(random.PRNGKey(0), n_epochs)), position=0, leave=True, total=n_epochs)
    for e, epoch_key in pbar:
        with disable_jit(False):
            epoch_loss, train_state, train_state_ema = run_epoch_train(epoch_key=epoch_key,
                                                                       train_state=train_state,
                                                                       train_state_ema=train_state_ema)
        pbar.set_description(f'Epoch loss : {epoch_loss:.2f}')
        if e % cfg.train.iters_per_logging == 0:
            wandb.log({
                'loss/train': epoch_loss,
                'lr': lr_schedule(train_state.opt_state[0].count).item()
            }, step=e)
        if e % cfg.train.iters_per_ckpt == 0:
            print("Start cross valid")
            epoch_cv_loss = run_epoch_cv(epoch_key=epoch_key,
                                         train_state=train_state_ema)

            print("Calculating weight norms")
            weights_norm = calculate_weight_norm(train_state_ema.params)
            metrics = {
                'loss/CV': epoch_cv_loss,
                **weights_norm
            }
            wandb.log(metrics, step=e)
            mngr.save(step=e, items=train_state_ema, metrics=metrics)
            print("Start one step denoising")
            ## One step denoising
            for batch_num, (batch_key, batch_ecg) in enumerate(
                    zip(random.split(epoch_key, len(dataloader)), dataloader_cv)):

                batch_ecg, batch_features = batch_ecg
                features = batch_features[:4]
                x = batch_ecg[:4]
                break
            # x = batch_ecg[:4]
            # features = batch_features[:4]
            key_std, key_noise = random.split(batch_key, 2)
            noises_std = jnp.exp(
                random.normal(key=key_std, shape=(1,)) * cfg.diffusion.p_std + cfg.diffusion.p_mean).clip(cfg.diffusion.sigma_min,
                                                                                                          cfg.diffusion.sigma_max)

            corrupted_data, pred_x = one_step_denoising(key_noise, noises_std, train_state_ema, x, features)

            for j, (corrupted_track, real_track, pred_track) in enumerate(zip(corrupted_data, x, pred_x)):
                fig, ax = plt.subplots(1, 1, figsize=(5, 8))
                fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                for color, ecg in zip(('red', 'blue', 'orange'), (corrupted_track, real_track, pred_track)):
                    for i, track in enumerate(ecg.T):
                        ax.plot(track - i, color=color, alpha=.7)
                wandb.log({f"one_step_denoising/{j}": wandb.Image(fig)}, step=e)
                plt.close(fig)

            print("Start sampling")
            # sampling
            initial_samples = random.normal(key=epoch_key,
                                            shape=(cfg.generate.n_samples, *data_sample.shape)) * cfg.diffusion.sigma_max

            samples = generate_fun(train_state=train_state_ema, initial_samples=initial_samples)

            for j, gen_beat in enumerate(samples):
                fig, ax = plt.subplots(1, 1, figsize=(5, 8))
                fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                for i, track in enumerate(gen_beat.T):
                    ax.plot(track - i, color='blue', alpha=.7)
                wandb.log({f"gen/{j}": wandb.Image(fig)}, step=e)
                plt.close(fig)

    mngr.wait_until_finished()


if __name__ == "__main__":
    main()
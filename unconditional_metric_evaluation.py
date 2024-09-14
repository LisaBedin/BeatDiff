import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from beat_net.beat_net.unet_parts import load_net
from beat_net.beat_net.variance_exploding_utils import heun_sampler, heun_sampler_one_step

from jax.tree_util import Partial as partial
from jax import numpy as jnp
from jax import random, grad
from torch.utils.data import DataLoader
from beat_net.beat_net.data_loader import PhysionetECG
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import ot
import time
import json
import torch
from beat_wgan.net import Gen_ac_wgan_gp_1d
import matplotlib.pyplot as plt

def plot_ecg(ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if ecg_distributions is not None:
        for ecg in ecg_distributions:
            for i, track in enumerate(ecg):
                alpha = 0.05
                if len(ecg_distributions) == 1:
                    alpha=1
                ax.plot(track - i*1.3, c=color_posterior, alpha=alpha, linewidth=.7, rasterized=True)  # rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)
    # ax.set_ylim(-13.5, 1.5)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    # ax.set_yticks([-i*1.5 for i in range(9)])
    #ax.set_xticklabels(np.arange(0, 175, 50).astype(int), fontsize=22)
    #ax.set_yticklabels(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), fontsize=22)
    return fig
@hydra.main(version_base=None, config_path="configs/", config_name="EMD_eval")
def main(cfg: DictConfig) -> list:
    # cfg = compose(config_name="config") # for loading cfg in python console
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    train_state, net_config, ckpt_num = load_net(cfg.paths)
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho

    if net_config.diffusion.scale_type == 'linear':
        scaling_fun = lambda t: net_config.diffusion.scaling_coeff * t
    elif net_config.diffusion.scale_type == 'one':
        scaling_fun = lambda t: 1.0
    else:
        raise NotImplemented

    model_wgan = Gen_ac_wgan_gp_1d(
        noise_dim=100,
        generator_n_features=64,  # Gen data n channels
        conditional_features_dim=4,  # N classes for embedding
        sequence_length=176,  # Length for channel
        sequence_n_channels=9,  # n_channels_seq
        embedding_dim=64).cuda()
    model_wgan.load_state_dict(torch.load('/mnt/Reseau/Signal/lisa/ecg_results/models/wgan/generator_trained_cl.pt'))
    # model_wgan.load_state_dict(torch.load('/mnt/data/lisa/ecg_results/models/wgan_no_norm/generator_trained_cl.pt'))

    conditioned = ('feature' not in net_config.dataset.keys()) or net_config.dataset.feature
    print('conditioned:', conditioned)
    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=net_config.dataset.normalized,
                                                      training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 num_workers=0)


    noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)


    for batch_test in test_dataloader:
        batch_test_features = None
        if conditioned:
            batch_test, batch_test_features = batch_test
        break
    n_min = batch_test.shape[0]
    batch_test = jnp.array(batch_test.reshape(n_min, -1).numpy()/np.max(np.absolute(batch_test.numpy()), axis=(1, 2))[:, np.newaxis])
    if conditioned:
        batch_test = jnp.concatenate((jnp.array(batch_test),
                                     jnp.array(batch_test_features.numpy())), axis=-1)

    n_max = n_min  # 22589
    train_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                       categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                       normalized=net_config.dataset.normalized,
                                                       training_class='Training', all=False,
                                                       return_beat_id=False),
                                  batch_size=n_max,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0)
    MI_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                    categories_to_filter=["MI"],
                                                    normalized=net_config.dataset.normalized,
                                                    training_class='Test', all=True,
                                                    return_beat_id=False),
                               batch_size=n_max,
                               shuffle=True,
                               drop_last=False,
                               num_workers=0)
    all_train_test_ws = []
    all_train_test_slight_mod = []
    for N, batch_1 in enumerate(train_dataloader):
        batch_features_1 = None
        if conditioned:
            batch_1, batch_features_1 = batch_1
            batch_1 /= np.max(np.absolute(batch_1.numpy()),axis=(1,2))[:, np.newaxis, np.newaxis]# TODO
            batch_1 = jnp.concatenate((jnp.array(batch_1.reshape(batch_1.shape[0], -1).numpy()),
                                      jnp.array(batch_features_1.numpy())),
                                      axis=-1)
        else:
            batch_1 /= np.absolute(batch_1.numpy()).max(axis=(1,2))[:, np.newaxis, np.newaxis]# TODO
            batch_1 = jnp.array(batch_1.reshape(batch_1.shape[0], -1).numpy())
        M = ot.dist(batch_1, batch_test)
        G0 = ot.emd2(jnp.ones(n_max) / n_max, jnp.ones(n_min)/n_min, M, numItermax=1_000_000)
        all_train_test_ws.append(G0.item())
        M = ot.dist(batch_1, batch_test + np.random.randn(*batch_test.shape)*0.05)
        G0 = ot.emd2(jnp.ones(n_max) / n_max, jnp.ones(n_min) / n_min, M, numItermax=1_000_000)
        all_train_test_slight_mod.append(G0.item())
    all_MI_test_ws = []
    for N, batch_1 in enumerate(MI_dataloader):
        batch_features_1 = None
        if conditioned:
            batch_MI, batch_features_1 = batch_1
            batch_MI /= np.absolute(batch_MI.numpy()).max(axis=(1, 2))[:, np.newaxis, np.newaxis] # TODO
            batch_MI = jnp.concatenate((jnp.array(batch_MI.reshape(batch_MI.shape[0], -1).numpy()),
                                      jnp.array(batch_features_1.numpy())),
                                      axis=-1)
        else:
            batch_MI = jnp.array(batch_1.reshape(batch_1.shape[0], -1).numpy())

        M = ot.dist(batch_MI, batch_test)
        G0 = ot.emd2(jnp.ones(M.shape[0]) / M.shape[0], jnp.ones(M.shape[1])/M.shape[1], M, numItermax=1_000_000)
        all_MI_test_ws.append(G0.item())
    metrics = {
        "ref_mean": np.mean(all_train_test_ws),
        "ref_std": np.std(all_train_test_ws),
        "ref_N": N+1,
        "ref_mod_mean": np.mean(all_train_test_slight_mod),
        "ref_mod_std": np.std(all_train_test_slight_mod),
        "MI_mean": np.mean(all_MI_test_ws),
        "MI_std": np.std(all_MI_test_ws),
        "batch_size": n_min,
        "wasserstein_with_T_train": {},
        "wasserstein_with_T_test": {},
    }
    print(metrics)
    for T in tqdm([2, 10, 20, 30, 40, 50, 60, 70, 80, 100, 125, 150], total=12):
        generate_fun = partial(heun_sampler,
                               sigma_min=sigma_min,
                               sigma_max=sigma_max,
                               N=T,
                               p=p,
                               scale_fun=scaling_fun,
                               noise_fun=noise_fun,
                               scale_fun_grad=grad(scaling_fun),
                               noise_fun_grad=grad(noise_fun),
                               train_state=train_state)

        initial_samples = random.normal(key=random.PRNGKey(0),
                                        shape=(n_min, 176, 9)) * sigma_max
        t1 = time.time()
        generated_features_test = generate_fun(initial_samples,
                                               class_labels=batch_test_features).block_until_ready()
        generated_features_test = generated_features_test.reshape(n_min, -1)
        if conditioned:
            generated_features_test = jnp.concatenate((generated_features_test,
                                                      jnp.array(batch_test_features)), axis=-1)

        time_delta = time.time() - t1
        if T > 2:
            all_train_gen_ws = []
            for N, batch_1 in enumerate(train_dataloader):
                if conditioned:
                    batch_1, batch_features_1 = batch_1
                    batch_1 = jnp.concatenate((jnp.array(batch_1.reshape(batch_1.shape[0], -1).numpy()),
                                               jnp.array(batch_features_1.numpy())),
                                              axis=-1)
                else:
                    batch_1 = jnp.array(batch_1.reshape(batch_1.shape[0], -1).numpy())
                M = ot.dist(batch_1, generated_features_test)
                G0 = ot.emd2(jnp.ones(n_max) / n_max, jnp.ones(n_min) / n_min, M, numItermax=1_000_000)
                all_train_gen_ws.append(G0.item())

            metrics["wasserstein_with_T_train"][T] = {
                "mean": np.mean(all_train_gen_ws),
                "std": np.std(all_train_gen_ws),
                "time": time_delta,
                "N": N+1
            }
            M = ot.dist(batch_test, generated_features_test)
            G0 = ot.emd2(jnp.ones(n_min) / n_min, jnp.ones(n_min) / n_min, M, numItermax=1_000_000)
            metrics["wasserstein_with_T_test"][T] = {
                "mean": G0.item(),
                "std": 0,
                "time": time_delta,
                "N": 1
            }
            print(' ')
            print(metrics)
            print(' ')

    #if 'uncond' not in cfg.checkpoint:
    #    results_path = os.path.join(cfg.results_path, f'baseline/model{ckpt_num}')
    #else:
    #    results_path = os.path.join(cfg.results_path, f'baseline_uncond/model{ckpt_num}')
    if 'uncond' in cfg.paths.name:
        conditioned_suffix = '_uncond'
    else:
        conditioned_suffix = '_cond'
    results_path = os.path.join(cfg.paths.results_path, 'generation_eval/global_norm'+conditioned_suffix)
    os.makedirs(results_path, exist_ok=True)
    #with open(os.path.join(cfg.checkpoint, 'model', str(ckpt_num), 'wasserstein_generation_metrics.json'), 'w') as file:
    with open(os.path.join(results_path, 'wasserstein_generation_metrics.json'), 'w') as file:
        json.dump(metrics, file)

    # ======================== plot some images ========================= #
    n_min, T = 20, 50
    initial_samples = random.normal(key=random.PRNGKey(0),
                                    shape=(n_min, 176, 9)) * sigma_max

    generate_fun = partial(heun_sampler,
                           sigma_min=sigma_min,
                           sigma_max=sigma_max,
                           N=T,
                           p=p,
                           scale_fun=scaling_fun,
                           noise_fun=noise_fun,
                           scale_fun_grad=grad(scaling_fun),
                           noise_fun_grad=grad(noise_fun),
                           train_state=train_state)

    generated_features_test = generate_fun(initial_samples,
                                           class_labels=batch_test_features[:n_min]).block_until_ready()

    noise = torch.randn((n_min, 100, 1)).cuda()
    with torch.no_grad():
        gen_wgan_ecg = np.swapaxes(model_wgan(noise, torch.Tensor(batch_test_features[:n_min].cuda().float())).detach().cpu().numpy(), 1, 2)
    color_posterior = '#00428d'
    color_target = '#fa526c'
    target_ecg = batch_test[:n_min, :176*9].reshape(-1, 176, 9)
    for k in range(n_min):
        fig = plot_ecg(ecg_distributions=None,
                       conditioning_ecg=target_ecg[k].T,
                       color_target=color_target,
                       color_posterior=color_posterior)

        fig.savefig(f'images/real{k}.pdf')
        fig = plot_ecg(ecg_distributions=None,
                       conditioning_ecg=gen_wgan_ecg[k].T,
                       color_target=color_posterior,
                       color_posterior=color_target)

        fig.savefig(f'images/wgan{k}.pdf')
        plt.close()
        fig = plot_ecg(ecg_distributions=None,
                       conditioning_ecg=generated_features_test[k].T,
                       color_target=color_posterior,
                       color_posterior=color_target)

        fig.savefig(f'images/diff{k}.pdf')

    # ====== plot along diffusion steps ===== #
    timesteps = jnp.arange(0, T-1) / (T - 2)
    timesteps = (sigma_max ** (1/p) + timesteps * (sigma_min**(1/p) - sigma_max**(1/p)))**(p)
    timesteps = jnp.append(timesteps, values=jnp.array([0]))
    all_steps = [initial_samples]
    for i in tqdm(range(T-1)):
        single_step_diff = heun_sampler_one_step(i, all_steps[-1],
            timesteps=timesteps,
            train_state=train_state,
            scale_fun=scaling_fun,
            noise_fun=noise_fun,
            scale_fun_grad=grad(scaling_fun),
            noise_fun_grad=grad(noise_fun),
            class_labels=batch_test_features[:n_min])
        all_steps.append(single_step_diff)
    for k in range(n_min):
        for step in range(25, len(all_steps)):
            fig = plot_ecg(ecg_distributions=None,
                           conditioning_ecg=all_steps[step][k].T,
                           color_target=color_posterior,
                           color_posterior=color_target)

            fig.savefig(f'images/diff{k}_step{step}.pdf')
            plt.close()

    return metrics


if __name__ == '__main__':
    metrics = main()

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from tqdm import tqdm
from beat_net.beat_net.unet_parts import DenoiserNet
from beat_net.beat_net.variance_exploding_utils import skip_scaling, output_scaling, input_scaling, noise_scaling, heun_sampler
import optax
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from flax.linen import jit as nn_jit
from jax.tree_util import Partial as partial
from jax import numpy as jnp
from jax import random
from flax.training.train_state import TrainState
import orbax.checkpoint
from jax import jit, grad, disable_jit
import matplotlib.pyplot as plt
from typing import Callable
from torch.utils.data import DataLoader
from beat_net.beat_net.data_loader import PhysionetECG
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve #mcg_diff_single_gpu, mcg_diff_multi_gpu, prepare_mcg_diff_inputs, prepare_mcg_diff_outputs
from jax import disable_jit
import math
from jax import vmap, pmap, devices
from jax import profiler
from scipy.spatial.distance import mahalanobis
# For debugging TODO: remove for pushing


def load_net(cfg):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=3, best_fn=lambda x: x, best_mode='min')
    mngr = orbax.checkpoint.CheckpointManager(
        cfg.checkpoint, orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        options)
    net_cfg = cfg.net_config.model
    diffusion_cfg = cfg.net_config.diffusion
    model = nn_jit(DenoiserNet, backend="gpu")(
        skip_scaling=partial(skip_scaling, sigma_data=diffusion_cfg.sigma_data),
        output_scaling=partial(output_scaling, sigma_data=diffusion_cfg.sigma_data),
        input_scaling=partial(input_scaling, sigma_data=diffusion_cfg.sigma_data),
        noise_conditioning=noise_scaling,
        start_dim=net_cfg.start_dim,
        n_blocks=net_cfg.n_blocks,
        n_resnet_per_block=net_cfg.n_resnet_per_block,
        n_conv_per_resnet=net_cfg.n_conv_per_resnet,
        kernel_sizes_resnet=tuple(net_cfg.kernel_sizes_resnet),
        use_group_norm_block=net_cfg.use_group_norm_block,
        use_mid_block=net_cfg.use_group_norm_block,
        dtype=jnp.float16,
        time_embedding_type=net_cfg.time_embedding_type,
        use_time_positional=net_cfg.use_time_positional,
        dropout_probability=net_cfg.dropout_probability,
        use_f_training=net_cfg.use_f_training,
        training=False,
        use_group_norm_mid=net_cfg.use_group_norm_mid,
    )

    params = mngr.restore(mngr.latest_step())
    train_state_ema = TrainState.create(apply_fn=model.apply,
                                        params={"params": params["params"]},
                                        tx=optax.ema(decay=1))

    return train_state_ema


def plot_ecg(ecg_distributions, conditioning_ecg = None):
    fig, ax = plt.subplots(1, 1)
    for ecg in ecg_distributions:
        for i, track in enumerate(ecg):
            ax.plot(track - i*1.5, color='blue', alpha=.02)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.5, color='red', alpha=.5)
    ax.set_ylim(-13.5, 1.5)
    return fig


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    # cfg = compose(config_name="config") # for loading cfg in python console
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    train_state = load_net(cfg)
    print("Load model")

    database_path = cfg.net_config.dataset.database_path

    # results_path = cfg.analysis_path
    # os.makedirs(results_path, exist_ok=True)

    T = cfg.net_config.generate.T
    sigma_max = cfg.net_config.diffusion.sigma_max
    sigma_min = cfg.net_config.diffusion.sigma_min
    p = cfg.net_config.generate.rho

    timesteps = jnp.arange(0, T-1) / (T - 2)
    timesteps = (sigma_max ** (1/p) + timesteps * (sigma_min**(1/p) - sigma_max**(1/p)))**(p)

    conditions = jnp.array(cfg.net_config.generate_eval.conditioning_pistes)
    coordinates_mask = jnp.ones((176, 9))*False
    coordinates_mask = coordinates_mask.at[..., conditions].set(True)
    coordinates_mask = coordinates_mask.astype(jnp.bool_)


    #if lab not in ignored_diseases:
    out_dist_dataloader = DataLoader(dataset=PhysionetECG(database_path=database_path, categories_to_filter=['NSR', ],
                                                          normalized=True, training_class='Test', all=False,
                                                          return_beat_id=True),
                                     batch_size=cfg.net_config.generate_eval.n_condition_per_diseases,
                                     shuffle=True,
                                     num_workers=10)

    for i_batch, (target_ecg, beats_indices, batch_indices) in enumerate(out_dist_dataloader):
        break
    observation = jnp.array(target_ecg[0, :, :3].flatten(), dtype=jnp.float16)
    coordinates_mask = coordinates_mask.flatten().astype(jnp.bool_)
    with disable_jit(False):

        n_chains = 2
        ns_particles = []
        dists = []
        for log_4_n_particles in range(2, 7):
            n_particles = 4**log_4_n_particles
            denoiser_fn = jit(
                lambda x, sigma: train_state.apply_fn(train_state.params,
                                                      x.reshape(n_particles, 176, 9),
                                                      sigma).reshape(
                    n_particles, 176*9))
            generate_ecg_fn = jit(vmap(partial(mcg_diff_ve,
                                               timesteps=timesteps.astype(dtype=jnp.float16),
                                               observations=observation,
                                               coordinates_mask=coordinates_mask,
                                               denoiser=denoiser_fn)))

            initial_samples = random.normal(random.PRNGKey(0), shape=(n_chains, n_particles, 176*9), dtype=jnp.float16) * sigma_max
            keys = random.split(random.PRNGKey(0), n_chains)
            t1 = time.time()
            generated_ecg = generate_ecg_fn(initial_samples,
                                            keys)
            generated_ecg.block_until_ready()
            t2 = time.time()

            generated_qrs = generated_ecg[:, 0, :].reshape(-1, 176, 9)[:, 40:70]
            target_qrs = target_ecg[0, 40:70]
            # Mahanalobis
            means_qrs = generated_qrs.mean(axis=0)
            prec_qrs = np.stack(
                [np.linalg.pinv(np.corrcoef(qrs)) for qrs in np.swapaxes(generated_qrs, 0, 2)])
            dist = max(mahalanobis(*it) for it in zip(target_qrs.T, means_qrs.T, prec_qrs))
            dists.append(dist)
            ns_particles.append(n_particles)
            print(t2 - t1, n_chains, n_particles, dist)
            fig = plot_ecg(np.swapaxes(generated_ecg[:, 0, :].reshape(-1, 176, 9), 1, 2),
                           target_ecg[0:1].T)
            fig.show()

        plt.plot(ns_particles, dists)
        plt.show()


if __name__ == '__main__':
    main()

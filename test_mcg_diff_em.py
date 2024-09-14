import os

import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve
import hydra
from omegaconf import DictConfig, OmegaConf
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate
from torch.utils.data import DataLoader
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit
from jax.tree_util import Partial as partial
import ot
import json
import matplotlib.pyplot as plt
from beat_net.beat_net.variance_exploding_utils import heun_sampler

def plot_ecg(ecg_distributions, conditioning_ecg = None):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for ecg in ecg_distributions:
        for i, track in enumerate(ecg):
            ax.plot(track - i*1.3, color='blue', alpha=.05, linewidth=.7, rasterized=True)  # rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, color='red', linewidth=.7, rasterized=True)  # rasterized=True)
    # ax.set_ylim(-13.5, 1.5)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    # ax.set_yticks([-i*1.5 for i in range(9)])
    #ax.set_xticklabels(np.arange(0, 175, 50).astype(int), fontsize=22)
    #ax.set_yticklabels(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), fontsize=22)
    return fig



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    train_state, net_config, ckpt_num = load_net(cfg)
    T = net_config.generate.T
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho
    # 4.30 GiB on the first GPU here


    noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)
    if net_config.diffusion.scale_type == 'linear':
        scaling_fun = lambda t: cfg.diffusion.scaling_coeff * t
    elif net_config.diffusion.scale_type == 'one':
        scaling_fun = lambda t: 1.0
    else:
        raise NotImplemented

    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=True, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=numpy_collate)
    for (batch_test_ecg, batch_test_features) in test_dataloader:
        break

    sigma_ests = jnp.ones((3,))*0

    n_parts = 50
    timesteps = jnp.arange(0, T - 1) / (T - 2)
    timesteps = (sigma_max ** (1 / p) + timesteps * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** (p)

    observations = jnp.concatenate((batch_test_ecg[0, :, :3],
                                    jnp.nan * jnp.ones((*batch_test_ecg[0].shape[:-1], 6))), axis=-1)
    observations = observations.flatten()

    posterior_sampling_fun = jit(vmap(
        partial(mcg_diff_ve,
                timesteps=timesteps,
                observations=observations,
                class_features=batch_test_features[0],
                denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params, x.reshape(-1, 176, 9),
                                                                      sigma, feats).reshape(x.shape[0], -1))
        , in_axes=(0, 0, None)))
    for i in range(10):

        ref_samples = random.normal(random.PRNGKey(i),
                                    shape=(
                                        100, n_parts, batch_test_ecg.shape[1] * batch_test_ecg.shape[2])) * sigma_max

        coordinates_mask = jnp.where(timesteps[:, None, None] >= sigma_ests[None, None, :],
                                     True,
                                     False)  # T x 1 x nb_cond_pistes
        coordinates_mask = jnp.concatenate((jnp.repeat(coordinates_mask,
                                                       axis=1,
                                                       repeats=176),  # T x 176 x nb_cond_pistes
                                            jnp.ones((T-1, 176, 6))*False),  # T x 176 x (9-nb_cond_pistes)
                                           axis=-1)
        coordinates_mask = coordinates_mask.reshape(coordinates_mask.shape[0], -1).astype(jnp.bool_)  # T x 176*9

        keys = random.split(random.PRNGKey(i), num=100)

        ref_samples = posterior_sampling_fun(ref_samples, keys, coordinates_mask)[:, 0].reshape(-1, 176, 9)
        # with disable_jit(True):
        #     posterior_sampling_fun = partial(mcg_diff_ve,
        #                                      timesteps=timesteps,
        #                                      observations=observations,
        #                                      class_features=batch_test_features[0],
        #                                      denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params, x.reshape(-1, 176, 9),
        #                                                                                            sigma, feats).reshape(x.shape[0], -1))
        #
        #     ref_samples = posterior_sampling_fun(ref_samples[0], keys[0]).reshape(-1, 176, 9)

        new_sigma_ests = jnp.round(jnp.std(batch_test_ecg[0, :, :3][None, :] - ref_samples[:, :, :3], axis=(0, 1)),
                                   2)

        fig = plot_ecg(ecg_distributions=jnp.swapaxes(ref_samples,
                                                      1, 2),
                       conditioning_ecg=batch_test_ecg[0].T)
        fig.show()
        if (new_sigma_ests == sigma_ests).all():
            break
        sigma_ests = new_sigma_ests

        print(sigma_ests)


if __name__ == '__main__':
    metrics = main()



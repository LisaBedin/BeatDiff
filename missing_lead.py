import os

import tqdm

'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,6,7'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
'''

from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve, em_variance, generate_coordinate_mask_from_observations
import hydra
from omegaconf import DictConfig, OmegaConf
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate
from torch.utils.data import DataLoader
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from jax.tree_util import Partial as partial
import ot
import json
import matplotlib.pyplot as plt
from beat_net.beat_net.variance_exploding_utils import heun_sampler
from neurokit2.ecg import ecg_delineate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np
from matplotlib.pyplot import cm
from scipy.spatial.distance import mahalanobis


def plot_ecg(ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for ecg in ecg_distributions:
        for i, track in enumerate(ecg):
            ax.plot(track - i*1.3, c=color_posterior, alpha=.05, linewidth=.7, rasterized=True)  # rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)
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


    full_forward_dowers = np.matrix([
        [0.515, -0.768, 0.125],
        [-0.434, -0.415, 0.037],
        [-0.081, 1.184, -0.162],
        [-0.515, 0.157, -0.917],
        [0.044, 0.164, -1.387],
        [0.882, 0.098, -1.277],
        [1.213, 0.127, -0.601],
        [1.125, 0.127, -0.086],
        [0.831, 0.076, 0.230]
    ])


    color_posterior = '#00428d'
    color_target = '#fa526c'
    seed = 0
    n_patients = len(devices("cuda"))

    noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)
    if net_config.diffusion.scale_type == 'linear':
        scaling_fun = lambda t: cfg.diffusion.scaling_coeff * t
    elif net_config.diffusion.scale_type == 'one':
        scaling_fun = lambda t: 1.0
    else:
        raise NotImplemented

    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path, categories_to_filter=['NSR', 'SB', 'STach', 'SA'],
                                                      normalized=True, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=len(devices("cuda")),
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=numpy_collate)
    test_dataloader_non_normalized = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path, categories_to_filter=['NSR', 'SB', 'STach', 'SA'],
                                                      normalized=False, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=len(devices("cuda")),
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=numpy_collate)

    timesteps = jnp.arange(0, T - 1) / (T - 2)
    timesteps = (sigma_max ** (1 / p) + timesteps * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** (p)

    def posterior_sampling_per_patient(key, observation, variance, features):
        keys = random.split(key, 101)
        init_samples = random.normal(key=keys[0], shape=(100, 50, 176 * 9))
        return vmap(partial(mcg_diff_ve,
                            timesteps=timesteps,
                            denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                  x.reshape(-1, 176, 9),
                                                                                  sigma, feats).reshape(x.shape[0],
                                                                                                        -1)),
                    in_axes=(0, 0, None, None, None)
                    )(init_samples, keys[1:], features, variance, observation)[:, 0].reshape(-1, 176, 9)
    posterior_sampling_fun = pmap(posterior_sampling_per_patient)

    all_ref_samples = []
    all_dowers_reconstructions = []
    all_missing = []
    all_observations = []
    for i, ((batch_test_ecg, batch_test_features), (batch_test_ecg_non_normalized, _)) in enumerate(zip(test_dataloader, test_dataloader_non_normalized)):
        batch_missing_leads = random.randint(key=random.PRNGKey(seed + 3*(i+1)),
                                             shape=(batch_test_ecg.shape[0],),
                                             minval=0,
                                             maxval=9)
        sigma_ests = jnp.ones((n_patients, 9,))*1
        observations = jnp.array(batch_test_ecg).at[jnp.arange(batch_test_ecg.shape[0]), batch_missing_leads].set(jnp.nan)
        variances = pmap(partial(em_variance,
                                 key=random.PRNGKey(seed + 4*(i+1)),
                                 timesteps=timesteps,
                                 denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                       x.reshape(-1, 176, 9),
                                                                                       sigma, feats).reshape(x.shape[0], -1),
                                 max_iter=10,
                                 n_particles=50,
                                 n_samples=100,
                                 ),
                         in_axes=(0, 0, 0))(observations, sigma_ests, batch_test_features[:n_patients])

        ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(seed + 3 + i), n_patients),
                                             observations,
                                             variances,
                                             batch_test_features[:n_patients])

        dowers_vcg = [jnp.linalg.pinv(full_forward_dowers[[i for i in jnp.arange(9) if i != lead]]) @ np.array(obs)[
                                                                                                      :, [i for i in
                                                                                                          jnp.arange(9)
                                                                                                          if
                                                                                                          i != lead]].T
                      for obs, lead in zip(batch_test_ecg_non_normalized, batch_missing_leads)]

        dowers_reconstructions = np.array([(full_forward_dowers @ vcg) for vcg in dowers_vcg])
        dowers_reconstructions = dowers_reconstructions / jnp.abs(dowers_reconstructions).max(axis=-1)[..., None]
        all_missing.append(batch_missing_leads)
        all_observations.append(batch_test_ecg)
        all_ref_samples.append(ref_samples)
        all_dowers_reconstructions.append(dowers_reconstructions)
        # np.savez(os.path.join(cfg.checkpoint, 'missing_leads.npy'),#os.path.join(results_path, 'qt_as_rr.npz'),
        np.savez(os.path.join(cfg.results_path, 'inpainting_eval/missing_leads.npy'),
                          # os.path.join(results_path, 'qt_as_rr.npz'),
                          missing_leads=np.concatenate(all_missing),
                          posterior_samples=np.concatenate(all_ref_samples),
                          observations=np.concatenate(all_observations),
                          dowers_reconstruction=np.concatenate(all_dowers_reconstructions))
        # real_leads = batch_test_ecg[jnp.arange(batch_test_ecg.shape[0]), :, batch_missing_leads]
        # mean = jnp.mean(ref_samples, axis=1)[jnp.arange(batch_test_ecg.shape[0]), :, :, batch_missing_leads]
        # precision = jnp.stack([jnp.pinv(jnp.cov(samples[:, :, index].T)) for samples, index in zip(ref_samples, batch_missing_leads)], axis=0)# color_points = '#00428d'#'#47d1dc'
        #
        # mean_mse = 1 - (jnp.linalg.norm(real_leads - mean, axis=-1) / jnp.linalg.norm(real_leads, axis=-1))
        # rec_mahanalobis = np.array([mahalanobis(u=lead, v=rec, VI=prec) for lead, rec, prec in zip(real_leads, mean, precision)])
        #
        # dowers_mse = 1 - (jnp.nansum((real_leads - dowers_reconstructions)**2, axis=-1) / jnp.linalg.norm(real_leads, axis=-1))


if __name__ == '__main__':
    metrics = main()

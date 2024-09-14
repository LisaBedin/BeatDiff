import os

import tqdm


# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import yaml
from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve, lasso_eta, em_variance
import hydra
from omegaconf import DictConfig, OmegaConf
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate, LQT_ECG
from torch.utils.data import DataLoader
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from jax.tree_util import Partial as partial

import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np
import torch
import time
import importlib
main_model = importlib.import_module("Score-based-ECG-Denoising-main.main_model")
denoising_model_small = importlib.import_module("Score-based-ECG-Denoising-main.denoising_model_small")


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

def apply_ekgan(X, ekgan_model, kept_leads, device_baseline):
    # ==== prepare input ==== #
    n_leads = X.shape[1]
    lead_factor = int(n_leads // len(kept_leads))
    input_image = X[:, torch.tensor(kept_leads)].repeat(1, lead_factor, 1)
    top_pad = int((16 - input_image.shape[1]) // 2)
    bottom_pad = 16 - (top_pad + input_image.shape[1])
    input_image = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(input_image, 1, 2)), 1,
                                 2)
    with torch.no_grad():
        ecg_recon_ekgan = ekgan_model(input_image.unsqueeze(1).to(device_baseline))[1][:, 0,
                          top_pad:top_pad + n_leads].detach().cpu()
    ecg_recon_ekgan = torch.swapaxes(torch.nn.functional.interpolate(ecg_recon_ekgan, size=176, mode='linear'), 1,
                                     2).numpy()
    return ecg_recon_ekgan

def BW_cosine(ecg, J=1, f_s=256, f_c=0.7):
    #J = 10 # number of harmnonics
    T, L = ecg.shape
    #f_s = 256
    coeff_cos = np.random.uniform(0, 1, size=(J, 1, L)) # coeffs of the harmonics
    coeff_sin = np.random.uniform(0, 1, size=(J, 1, L)) # coeffs of the harmonics

    #phases = np.random.uniform(0, 2*np.pi, size=(J, 1, L)) # coeffs of the harmonics
    cos_fn = coeff_cos * np.cos(2*np.pi *
                    f_c*np.arange(1, J+1)[:, np.newaxis, np.newaxis]/J *
                    np.arange(T)[np.newaxis, :, np.newaxis]*4)# /f_s)
    sin_fn = coeff_sin * np.sin(2*np.pi *
                    f_c*np.arange(1, J+1)[:, np.newaxis, np.newaxis]/J *
                    np.arange(T)[np.newaxis, :, np.newaxis]*4)# /f_s)
    return ((cos_fn + sin_fn).sum(axis=0))

@hydra.main(version_base=None, config_path="configs/", config_name="EMbeat_diff_denoising")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    # cfg.checkpoint = cfg.paths.checkpoint  # unconditional model for this dataset
    train_state, net_config, ckpt_num = load_net(cfg.paths)
    T = cfg.denoising.T # net_config.generate.T
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho
    # 4.30 GiB on the first GPU here


    color_posterior = '#00428d'
    color_target = '#fa526c'
    seed = 0

    real_noise = len(cfg.paths.noise_path) > 0

    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                      categories_to_filter=['NSR'],
                                                      noise_path=os.path.join(cfg.paths.noise_path, cfg.denoising.name),
                                                      normalized=net_config.dataset.normalized,
                                                      training_class='Test',
                                                      all=False,
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
    all_observations = []
    all_feats = []
    all_variances = []
    all_real_eta, all_eta = [], []
    all_noise, all_noisy_leads = [], []
    desco_lst, all_noisy_obs = [], []

    #  === noise model === #
    J = cfg.denoising.J # number of harmonics
    # f_c = 0.7 # BW
    f_s = 25 # sampling frequency
    # 2 * jnp.pi ??
    phi = jnp.concatenate([jnp.cos(jnp.arange(176)[:, None] / f_s * (jnp.arange(J)[None] / J *
                                                                                  (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                                                                  +cfg.denoising.f_c_min)),
                           jnp.sin(jnp.arange(176)[:, None] / f_s * jnp.arange(J)[None] / J * (
                                   (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                   + cfg.denoising.f_c_min)
                                   )],
                          axis=1)
    begin_time = time.time()
    f_c = (cfg.denoising.f_c_min + cfg.denoising.f_c_max) / 2.
    suffix = f'K{cfg.denoising.T}_J{J}_{f_c:.2f}_l1reg{cfg.denoising.l1reg}'
    if real_noise:
        noise_type = cfg.denoising.name
        suffix += '_' + noise_type
    else:
        suffix += '_' + cfg.denoising.amplitude
    save_path = os.path.join(cfg.paths.results_path,
                             f'denoising_{suffix}')
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(os.path.join(cfg.paths.baseline_folder, 'DeScoD/base.yaml')), "r") as f:
        config = yaml.safe_load(f)
    base_model = denoising_model_small.ConditionalModel(config['train']['feats'], channels=9).to('cuda:0')
    model_descoD9 = main_model.DDPM(base_model, config, 'cuda:0')
    model_descoD9.load_state_dict(torch.load(os.path.join(cfg.paths.baseline_folder, 'DeScoD/model.pth')))

    for i, data_normalized in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_test_ecg, batch_test_features = data_normalized[:2]

        all_observations.append(batch_test_ecg)
        all_feats.append(batch_test_features)

        # === apply nc-mcgdiff === #
        observations = jnp.array(batch_test_ecg) #.at[:, :, np.arange(3, 9).astype(int)].set(jnp.nan)

        sigma_ests = jnp.ones((batch_test_features.shape[0], 9,))*1
        real_eta = jnp.zeros((batch_test_features.shape[0], phi.shape[1], 9))
        key_var, key_corrupt = random.split(random.PRNGKey(seed + 4 * (i + 1)), 2)

        if real_noise:
            batch_noisy_obs, batch_test_noise, batch_test_noisy_leads = data_normalized[2:]
            all_noise.append(batch_test_noise[:, 0]) # only noise type 1
            all_noisy_leads.append(batch_test_noisy_leads[:, 0]) # only noise type 1
            noisy_obs = jnp.nan * jnp.ones(batch_noisy_obs.shape)
            for k in range(batch_test_noisy_leads.shape[1]):
                noisy_obs = noisy_obs.at[np.arange(batch_test_noisy_leads.shape[0]), :, batch_test_noisy_leads[:, k]].set(
                batch_noisy_obs[np.arange(batch_test_noisy_leads.shape[0]), :, batch_test_noisy_leads[:, k]])
            #noisy_obs = noisy_obs.at[np.arange(batch_test_noisy_leads.shape[0]), :, batch_test_noisy_leads[:, 1]].set(batch_noisy_obs[np.arange(batch_test_noisy_leads.shape[0]), :, batch_test_noisy_leads[:, 1]])
        else:
            key_corrupt = random.split(key_corrupt, 3)
            amplitude = random.uniform(key_corrupt[0], shape=(batch_test_features.shape[0],), minval=0, maxval=cfg.denoising.amplitude)
            harmonics = random.randint(key_corrupt[1], shape=(batch_test_features.shape[0],), dtype=int, minval=0, maxval=J)
            phase = random.uniform(key_corrupt[2], shape=(batch_test_features.shape[0],), minval=0, maxval=2*np.pi)

            for l in range(9):
                real_eta = real_eta.at[jnp.arange(harmonics.shape[0]), harmonics, l].set(amplitude*jnp.cos(phase))
                real_eta = real_eta.at[np.arange(harmonics.shape[0]),harmonics+J, l].set(amplitude*jnp.sin(phase))
            noisy_obs = observations+phi[:, :J]@real_eta[:, :J] + phi[:, J:]@real_eta[:, J:]
        init_eta = jnp.zeros_like(real_eta)
        init_eta = pmap(partial(lasso_eta,
                               # initial_variance=sigma_ests,
                                 key=key_var,
                                 timesteps=timesteps,
                                 denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                       x.reshape(-1, 176, 9),
                                                                                       sigma, feats).reshape(x.shape[0], -1),
                                 max_iter=5,
                                 phi=phi,
                                 lreg=cfg.denoising.l1reg,
                                 n_particles=50,
                                 n_samples=100,
                                 ),
                         in_axes=(0, 0, 0, 0))(
            noisy_obs, sigma_ests, init_eta, batch_test_features)  # [:n_patients])
        harmonics_th = 1e-3
        mask_phi = np.absolute(init_eta) > harmonics_th
        #leads_list = [np.where(mask_phi[p].sum(axis=0) > 0)[0] for p in range(mask_phi.shape[0])]
        variances, final_eta = pmap(partial(em_variance,
                                 key=key_var,
                                 timesteps=timesteps,
                                 denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                       x.reshape(-1, 176, 9),
                                                                                       sigma, feats).reshape(x.shape[0], -1),
                                 max_iter=5,
                                 phi=phi,
                                 #mask_phi=mask_phi,
                                 #leads_list=leads_list,
                                 lreg=1,
                                 n_particles=50,
                                 n_samples=100,
                                 ),
                         in_axes=(0, 0, 0, 0, 0))(
            noisy_obs,
            sigma_ests, init_eta, batch_test_features, mask_phi)  # [:n_patients])

        ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(seed + 3 + i), batch_test_features.shape[0]),
                                             noisy_obs - phi[:, :J]@final_eta[:, :J] - phi[:, J:]@final_eta[:, J:],
                                             variances,
                                             batch_test_features) # [:n_patients])

        all_desco9 = []
        with torch.no_grad():
            desco_input = torch.Tensor(np.swapaxes(np.array(noisy_obs), 1, 2)).to('cuda:0')
            for _ in tqdm.tqdm(range(10)):
                all_desco9.append(np.swapaxes(model_descoD9.denoising(desco_input).detach().cpu().numpy(), 1, 2))
        all_desco9 = np.stack(all_desco9, axis=1)
        desco_lst.append(all_desco9)

        sample_id = 0
        ecg_distributions = np.swapaxes(ref_samples[sample_id], 1, 2) / batch_test_ecg[sample_id].max(axis=0)[np.newaxis, :, np.newaxis]
        conditioning_ecg = batch_test_ecg[sample_id].T / batch_test_ecg[sample_id].max(axis=0)[:, np.newaxis]
        # conditioning_ecg = noisy_obs[sample_id].T / batch_test_ecg[sample_id].max(axis=0)[:, np.newaxis]

        fig, ax = plt.subplots(1, 1, figsize=(1, 4))
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
        ax.tick_params(axis='both', which='major', labelsize=16)
        for ecg in ecg_distributions:
            for i, track in enumerate(ecg):
                ax.plot(track - i * 1.3, c=color_posterior, alpha=.05, linewidth=.7,
                        rasterized=True)  # rasterized=True)
        for i, track in enumerate(conditioning_ecg):
            ax.plot(track - i * 1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)
        ax.set_ylim(-11, 1.1)
        ax.set_xlim(0, 175)

        all_ref_samples.append(ref_samples)
        all_variances.append(variances)
        all_real_eta.append(real_eta)
        all_eta.append(final_eta)
        all_noisy_obs.append(noisy_obs)

        np.savez(os.path.join(save_path, 'NSR.npz'),
                          ground_truth=np.concatenate(all_observations),
                          features=np.concatenate(all_feats),
                          noisy_observation=np.concatenate(all_noisy_obs),
                          posterior_samples=np.concatenate(all_ref_samples),
                          variances=np.concatenate(all_variances),
                          eta=np.concatenate(all_eta),
                          gt_eta=np.concatenate(all_real_eta),
                          phi=phi,
                          DeScoD=np.concatenate(desco_lst),
                          real_noise=np.concatenate(all_noise),
                          noisy_leads=np.concatenate(all_noisy_leads)
                 )
    total_time = time.time()-begin_time
    print(total_time)

    # L2_dist = np.mean(np.sqrt(np.mean((qt_gt[:, None, 70:]- qt_pred[:, :, 70:])**2, axis=1)), axis=(1,2)) -> 56%

if __name__ == '__main__':
    main()

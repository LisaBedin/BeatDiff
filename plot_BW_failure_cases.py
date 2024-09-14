import os

import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,7'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import yaml
import wfdb
from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve, lasso_eta, em_variance
import hydra
from scipy.signal import resample
from omegaconf import DictConfig, OmegaConf
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate, LQT_ECG
from torch.utils.data import DataLoader
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from jax.tree_util import Partial as partial
from jaxopt.prox import prox_lasso, prox_ridge, prox_elastic_net
import importlib
main_model = importlib.import_module("Score-based-ECG-Denoising-main.main_model")
denoising_model_small = importlib.import_module("Score-based-ECG-Denoising-main.denoising_model_small")

import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np
import torch
import time



def plot_ecg(ecg_distributions, conditioning_ecg = None, color_posterior='#00428d', color_target='#fa526c'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i * 1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)

    if ecg_distributions is not None:
        for ecg in ecg_distributions:
            alpha = .05
            if ecg_distributions.shape[0] == 1:
                alpha = 1
            for i, track in enumerate(ecg):
                ax.plot(track - i*1.3, c=color_posterior, alpha=alpha, linewidth=.7, rasterized=True)  # rasterized=True)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    return fig
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

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint = cfg.denoising.model_path  # unconditional model for this dataset
    train_state, net_config, ckpt_num = load_net(cfg)
    T = cfg.inpainting.T # net_config.generate.T
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

    real_noise = len(cfg.denoising.noise_path) > 0

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
    all_noisy_obs = []
    all_variances = []
    all_real_eta, all_eta = [], []
    all_noise, all_noisy_leads = [], []
    all_desco1, all_desco9 = [], []
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
    suffix = f'K{cfg.inpainting.T}_J{J}_{f_c:.2f}_l1reg{cfg.denoising.l1reg}_l2reg{cfg.denoising.l2reg}'
    if real_noise:
        noise_type = cfg.denoising.noise_path.split('/')[-1]
        suffix += '_' + noise_type
    else:
        suffix += '_' + cfg.denoising.amplitude
    save_path = os.path.join(cfg.results_path,
                             f'denoising_{suffix}')

    path = "Score-based-ECG-Denoising-main/config/base.yaml"

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    base_model = denoising_model_small.ConditionalModel(config['train']['feats'], channels=1).to('cuda:0')
    model_descoD1 = main_model.DDPM(base_model, config, 'cuda:0')
    model_descoD1.load_state_dict(torch.load(os.path.join(config['results_path'], 'channels_1/model.pth')))

    base_model = denoising_model_small.ConditionalModel(config['train']['feats'], channels=9).to('cuda:0')
    model_descoD9 = main_model.DDPM(base_model, config, 'cuda:0')
    model_descoD9.load_state_dict(torch.load(os.path.join(config['results_path'], 'channels_9/model.pth')))

    # ==== Loading the computed EM and BW results ==== #
    npz_em = dict(np.load(os.path.join(cfg.results_path,
                                'denoising_K20_J100_0.50_l1reg10_l2reg5_em/NSR.npz')))
    npz_bw = dict(np.load(os.path.join(cfg.results_path,
                                'denoising_K20_J100_0.50_l1reg10_l2reg10_bw/NSR.npz')))

    # ===== loading the EM noise ==== #
    noise_main_path = '/'.join(cfg.denoising.noise_path.split('/')[:-1])
    noise_em_path = os.path.join(noise_main_path, 'em')
    em_record = wfdb.rdrecord(noise_em_path).__dict__
    noise_em = em_record['p_signal']
    f_s_prev = em_record['fs']
    new_s = int(round(noise_em.shape[0] / f_s_prev * 250))
    noise_em = resample(noise_em, num=new_s)[45139 + 361111:]

    # ==== Loading the spotted failure cases in BW ==== #
    sample_list = [0, 1, 2, 3] # [687, 876, 803, 352]
    # 637#407, 425, 27
    #637, 407
    input_BW, input_EM, input_gt, input_feats = [], [], [], []
    beatdiff_BW = []
    for sample_id in sample_list:
        clean_ecg = npz_bw['ground_truth'][sample_id]
        input_gt.append(clean_ecg)
        input_feats.append(npz_bw['features'][sample_id])
        T, L = clean_ecg.shape
        input_BW.append(npz_bw['noisy_observation'][sample_id])
        start_ind = np.random.choice(a=len(noise_em) - T,
                                     size=(L,), replace=True)
        leads = np.arange(L)
        noise = np.stack([noise_em[start_ind[k]:start_ind[k] + T, 1] for k in
                          range(L)], axis=1)
        beat_max_value = np.max(clean_ecg, axis=0) - np.min(clean_ecg, axis=0)
        noise_max_value = np.max(noise, axis=0) - np.min(noise, axis=0)
        Ase = noise_max_value / beat_max_value # [leads]
        noise = noise / Ase[np.newaxis]
        input_EM.append(clean_ecg+noise)
        #all_desco9_BW.append(npz_bw['DeScoD9'][sample_id])
        beatdiff_BW.append(npz_bw['posterior_samples'][sample_id])
    noisy_obs = jnp.array(np.stack(input_EM))
    batch_test_ecg = np.stack(input_gt)
    input_BW = np.stack(input_BW)
    #all_desco9_BW = np.stack(all_desco9_BW)
    beatdiff_BW = np.stack(beatdiff_BW)
    batch_test_features = jnp.array(np.stack(input_feats))


    sigma_ests = jnp.ones((batch_test_ecg.shape[0], 9,))*1
    real_eta = jnp.zeros((batch_test_ecg.shape[0], phi.shape[1], 9))
    key_var, key_corrupt = random.split(random.PRNGKey(seed), 2)

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
                             lreg=10,
                            solver=prox_lasso,
                             n_particles=50,
                             n_samples=100,
                             ),
                     in_axes=(0, 0, 0, 0))(
        noisy_obs, sigma_ests, init_eta, batch_test_features)  # [:n_patients])
    harmonics_th = 1e-3
    mask_phi = np.absolute(init_eta) > harmonics_th
    print('mask_phi', mask_phi[:, :, 0].sum(axis=1))
    leads_list = [np.where(mask_phi[p].sum(axis=0) > 0)[0] for p in range(mask_phi.shape[0])]
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
                             lreg=5,
                                        solver=prox_ridge,
                             n_particles=50,
                             n_samples=100,
                             ),
                     in_axes=(0, 0, 0, 0, 0))(
        noisy_obs,
        sigma_ests, init_eta, batch_test_features, mask_phi)  # [:n_patients])

    ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(seed + 3), batch_test_features.shape[0]),
                                         noisy_obs - phi[:, :J]@final_eta[:, :J] - phi[:, J:]@final_eta[:, J:],
                                         variances,
                                         batch_test_features) # [:n_patients])
    all_desco9_BW = []
    all_desco9_EM = []
    with torch.no_grad():
        desco_input = torch.Tensor(np.swapaxes(np.array(noisy_obs), 1, 2)).to('cuda:0')
        for _ in tqdm.tqdm(range(10)):
            all_desco9_EM.append(np.swapaxes(model_descoD9.denoising(desco_input).detach().cpu().numpy(), 1, 2))
        desco_input = torch.Tensor(np.swapaxes(input_BW, 1, 2)).to('cuda:0')
        for _ in tqdm.tqdm(range(10)):
            all_desco9_BW.append(np.swapaxes(model_descoD9.denoising(desco_input).detach().cpu().numpy(), 1, 2))
    all_desco9_EM = np.stack(all_desco9_EM,axis=1)
    all_desco9_BW = np.stack(all_desco9_BW, axis=1)

    for k in range(4):
        real_ecg = np.array(batch_test_ecg[k]).T
        normalizing_ecg = np.absolute(real_ecg).max(axis=1)[:, np.newaxis]
        real_ecg /= normalizing_ecg

        # =============== electrode motion ============= #
        pred_ecg = np.array(ref_samples[k].mean(axis=0)).T / normalizing_ecg
        fig = plot_ecg(pred_ecg[np.newaxis],
                       real_ecg)
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/EM_beatdiff_{k}.pdf'))
        plt.show()


        pred_ecg = np.array(all_desco9_EM[k].mean(axis=0)).T / normalizing_ecg
        fig = plot_ecg(pred_ecg[np.newaxis],
                       real_ecg)
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/EM_DeScoD_{k}.pdf'))
        plt.show()

        noisy_ecg = noisy_obs[k].T / normalizing_ecg
        fig = plot_ecg(None, noisy_ecg -np.mean(noisy_ecg, axis=1)[:, np.newaxis])
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/EM_input_{k}.pdf'))
        plt.show()


        # =============== baseline wander ============= #
        pred_ecg = np.array(beatdiff_BW[k].mean(axis=0)).T / normalizing_ecg
        fig = plot_ecg(pred_ecg[np.newaxis],
                       real_ecg)
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/BW_beatdiff_{k}.pdf'))
        plt.show()


        pred_ecg = np.array(all_desco9_BW[k].mean(axis=0)).T / normalizing_ecg
        fig = plot_ecg(pred_ecg[np.newaxis],
                       real_ecg)
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/BW_DeScoD_{k}.pdf'))
        plt.show()
        # TODO: center per lead
        noisy_ecg = input_BW[k].T / normalizing_ecg
        fig = plot_ecg(None, noisy_ecg -np.mean(noisy_ecg, axis=1)[:, np.newaxis])
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/BW_input_{k}.pdf'))
        plt.show()

if __name__ == '__main__':
    metrics = main()

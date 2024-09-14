import os

import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4,3'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve, em_variance, em_variance_only
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

def apply_ekgan(X, ekgan_model, kept_leads, device_baseline, segment_length):
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
    ecg_recon_ekgan = torch.swapaxes(torch.nn.functional.interpolate(ecg_recon_ekgan, size=segment_length, mode='linear'), 1,
                                     2).numpy()
    return ecg_recon_ekgan

@hydra.main(version_base=None, config_path="configs/", config_name="EMbeat_diff_inpainting")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    train_state, net_config, ckpt_num = load_net(cfg.paths)
    T = cfg.inpainting.T # net_config.generate.T
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho
    # 4.30 GiB on the first GPU here
    segment_length = 2500 if 'long_term' in cfg.paths.name else 176

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

    seed = 0

    if cfg.inpainting.labels[0] == 'LQTS_clean':
        # cohort of patients with genetic marker of disease,
        # much more precise than LQT annotations in physionet
        # but not publicly available
        qt_clean = True
        test_dataloader = DataLoader(dataset=LQT_ECG(database_path=cfg.paths.qt_path,
                                                     normalized='no',
                                                     return_beat_id=False),
                                     batch_size=len(devices("cuda")),
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=numpy_collate)
        test_dataloader_non_normalized = DataLoader(dataset=LQT_ECG(
            database_path=cfg.paths.qt_path,
            normalized='none',
            return_beat_id=False),
            batch_size=len(devices("cuda")),
            shuffle=False,
            num_workers=0,
            collate_fn=numpy_collate)

    else:
        qt_clean = False
        test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                          categories_to_filter=cfg.inpainting.labels,
                                                          segment_length=segment_length,
                                                          long_term='long_term' in cfg.paths.name,
                                                          normalized='no', training_class='Test', all=('NSR' not in cfg.inpainting.labels),
                                                          return_beat_id=False),
                                     batch_size=len(devices("cuda")),
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=numpy_collate)

    timesteps = jnp.arange(0, T - 1) / (T - 2)
    timesteps = (sigma_max ** (1 / p) + timesteps * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** (p)

    def posterior_sampling_per_patient(key, observation, variance, features):
        keys = random.split(key, 101)
        init_samples = random.normal(key=keys[0], shape=(100, 50, segment_length * 9))
        return vmap(partial(mcg_diff_ve,
                            timesteps=timesteps,
                            denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                  x.reshape(-1, segment_length, 9),
                                                                                  sigma, feats).reshape(x.shape[0],
                                                                                                        -1)),
                    in_axes=(0, 0, None, None, None)
                    )(init_samples, keys[1:], features, variance, observation)[:, 0].reshape(-1, segment_length, 9)
    posterior_sampling_fun = pmap(posterior_sampling_per_patient)

    # ===== for denoising the observations ==== #
    J = cfg.denoising.J # number of harmonics
    # f_c = 0.7 # BW
    f_s = 25 # sampling frequency
    # 2 * jnp.pi ??
    phi = jnp.concatenate([jnp.cos(jnp.arange(segment_length)[:, None] / f_s * (jnp.arange(J)[None] / J *
                                                                                  (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                                                                  +cfg.denoising.f_c_min)),
                           jnp.sin(jnp.arange(segment_length)[:, None] / f_s * jnp.arange(J)[None] / J * (
                                   (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                   + cfg.denoising.f_c_min)
                                   )],
                          axis=1)

    all_ref_samples = []
    all_dowers_reconstructions = []
    all_ekgan = []
    all_aae = []
    all_observations = []
    all_labels = []
    all_feats = []
    all_variances = []
    all_eta = []
    # ==== loading baselines ===) #
    ekgan_model = torch.load(os.path.join(cfg.paths.baseline_folder, f'{cfg.inpainting.ekgan_name}/best_inference_generator.pth')).cuda()
    AAE_model = torch.load(os.path.join(cfg.paths.baseline_folder, 'AAE/best_model.pth')).cuda()
    AAE_model.eval()
    AAE_model.netD.TCN_opt = False
    ekgan_model.eval()
    device_baseline = ekgan_model.all_convs[0].conv.weight.device
    save_path = os.path.join(cfg.paths.results_path, f'inpainting_samples_K{cfg.inpainting.T}')
    os.makedirs(save_path, exist_ok=True)
    begin_time = time.time()
    for i, data_raw in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_test_ecg, batch_test_features = data_raw[:2]
        if qt_clean:
            all_labels.append(data_raw[-1])
        all_observations.append(batch_test_ecg)
        all_feats.append(batch_test_features)
        # === apply ekgan === #
        X = nn.Upsample(256, mode='linear')(torch.swapaxes(torch.tensor(batch_test_ecg).to(torch.float32), 1, 2))
        try:
            if 'leadI' in cfg.inpainting.ekgan_name:
                ekgan_leads = np.array([0])
            else:
                ekgan_leads = np.arange(3)
            ecg_recon_ekgan = apply_ekgan(X, ekgan_model, ekgan_leads, device_baseline, segment_length)
            all_ekgan.append(ecg_recon_ekgan)
        except:
            breakpoint()

        # === apply AAE === #
        X = torch.Tensor(np.swapaxes(batch_test_ecg, 1, 2))
        with torch.no_grad():
            ecg_recon_AAE = np.swapaxes(AAE_model(X.to(device_baseline), None).detach().cpu().numpy(), 1, 2)
        all_aae.append(ecg_recon_AAE)

        # === apply nc-mcgdiff === #
        if cfg.inpainting.QRS_only:
            observations = jnp.array(batch_test_ecg).at[:, 70:, :].set(jnp.nan)
        elif cfg.inpainting.ST_only:
            observations = jnp.array(batch_test_ecg).at[:, :70, :].set(jnp.nan)
        elif cfg.inpainting.all_leads:
            observations = jnp.array(batch_test_ecg)
        elif cfg.inpainting.name == 'Kardia':
            observations = jnp.array(batch_test_ecg).at[:, :, np.array(cfg.inpainting.missing_leads).astype(int)].set(
                jnp.nan)
        else:
            observations = jnp.array(batch_test_ecg).at[:, :, np.arange(cfg.inpainting.start_lead, 9).astype(int)].set(jnp.nan)
        sigma_ests = jnp.ones((batch_test_features.shape[0], 9,)) * 1
        init_eta = jnp.zeros((batch_test_features.shape[0], 2 * J, 9))
        if cfg.inpainting.do_denoise:
            max_T = segment_length  # 70 if cfg.inpainting_samples_K20.QRS_only else segment_length
            variances, final_eta = pmap(partial(em_variance,
                                     key=random.PRNGKey(seed + 4*(i+1)),
                                     timesteps=timesteps,
                                     denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                           x.reshape(-1, segment_length, 9),
                                                                                           sigma, feats).reshape(x.shape[0], -1),
                                     max_iter=10,
                                                phi=phi,
                                                l1reg=cfg.denoising.l1reg,
                                                max_T=max_T,
                                     n_particles=50,
                                     n_samples=100,
                                     ),
                             in_axes=(0, 0, 0, 0))(observations, sigma_ests,
                                                init_eta, batch_test_features)  # [:n_patients])

        else:
            variances = pmap(partial(em_variance_only,
                                                key=random.PRNGKey(seed + 4 * (i + 1)),
                                                timesteps=timesteps,
                                                denoiser=lambda x, sigma, feats: train_state.apply_fn(
                                                    train_state.params,
                                                    x.reshape(-1, segment_length, 9),
                                                    sigma, feats).reshape(x.shape[0], -1),
                                                max_iter=10,
                                                n_particles=50,
                                                n_samples=100,
                                                ),
                                        in_axes=(0, 0, 0))(observations, sigma_ests, batch_test_features)

            final_eta = jnp.zeros_like(init_eta)

        ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(seed + 3 + i), batch_test_features.shape[0]),
                                             observations - phi[:, :J] @ final_eta[:, :J] - phi[:, J:] @ final_eta[:,
                                                                                                         J:],
                                             variances,
                                             batch_test_features)
        all_ref_samples.append(ref_samples)
        all_variances.append(variances)
        all_eta.append(final_eta)
        # === applying Dower reconstruction === #
        dowers_vcg = [jnp.linalg.pinv(full_forward_dowers[:6]) @ np.array(obs)[:, :6].T
                      for obs in batch_test_ecg]

        dowers_reconstructions = np.array([(full_forward_dowers @ vcg) for vcg in dowers_vcg])
        dowers_reconstructions[:, 6:] /= jnp.abs(dowers_reconstructions[:, 6:]).max(axis=(1, 2))[:, None, None]
        # dowers_reconstructions = dowers_reconstructions / jnp.abs(dowers_reconstructions).max(axis=(1, 2))[:, None, None]
        all_dowers_reconstructions.append(np.swapaxes(dowers_reconstructions, 1, 2))


        suffix = '_'.join(cfg.inpainting.labels)
        if cfg.inpainting.QRS_only:
            suffix += '_qrs'
        elif cfg.inpainting.ST_only:
            suffix += '_ST'
        elif cfg.inpainting.all_leads:
            suffix += '_allLeads'
        elif cfg.inpainting.name == 'Kardia':
            suffix += '_' + '_'.join([str(k) for k in range(9) if k not in cfg.inpainting.missing_leads])
        else:
            suffix += '_' + str(cfg.inpainting.start_lead)
        np.savez(os.path.join(save_path, f'{suffix}.npz'),
                          ground_truth=np.concatenate(all_observations),
                          genetic_annotation=np.concatenate(all_labels) if len(all_labels)> 0 else [],
                          features=np.concatenate(all_feats),
                          posterior_samples=np.concatenate(all_ref_samples),
                          variances=np.concatenate(all_variances),
                          ekgan=np.concatenate(all_ekgan),
                          AAE=np.concatenate(all_aae),
                          eta=np.concatenate(all_eta),
                          phi=phi,
                          dowers_reconstruction=np.concatenate(all_dowers_reconstructions))
        # real_leads = batch_test_ecg[jnp.arange(batch_test_ecg.shape[0]), :, batch_missing_leads]
        # mean = jnp.mean(ref_samples, axis=1)[jnp.arange(batch_test_ecg.shape[0]), :, :, batch_missing_leads]
        # precision = jnp.stack([jnp.pinv(jnp.cov(samples[:, :, index].T)) for samples, index in zip(ref_samples, batch_missing_leads)], axis=0)# color_points = '#00428d'#'#47d1dc'
        #
        # mean_mse = 1 - (jnp.linalg.norm(real_leads - mean, axis=-1) / jnp.linalg.norm(real_leads, axis=-1))
        # rec_mahanalobis = np.array([mahalanobis(u=lead, v=rec, VI=prec) for lead, rec, prec in zip(real_leads, mean, precision)])
        #
        # dowers_mse = 1 - (jnp.nansum((real_leads - dowers_reconstructions)**2, axis=-1) / jnp.linalg.norm(real_leads, axis=-1))
    total_time = time.time()-begin_time
    print(total_time)

    # L2_dist = np.mean(np.sqrt(np.mean((qt_gt[:, None, 70:]- qt_pred[:, :, 70:])**2, axis=1)), axis=(1,2)) -> 56%

if __name__ == '__main__':
    metrics = main()

import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,4,3,2,1'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.append('.')
from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import em_variance_only, lasso_eta, mcg_diff_ve, em_variance
import hydra
from omegaconf import DictConfig, OmegaConf
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from jax.tree_util import Partial as partial
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

# from mcg_diff_inpainting import apply_ekgan  # careful if CUDA_VISIBLE_DEVICES configured in this script !
from torch.utils.data import Dataset, DataLoader
from beat_net.beat_net.data_loader import numpy_collate


def get_ind_peaks(signal, first_lead=False):
    if first_lead:
        inds_peaks = np.stack([
            np.argmin(signal, axis=1),
            np.argmax(signal, axis=1),
            # np.min(signal, axis=1),
            # np.max(signal, axis=1)
        ], axis=1)
    else:
        inds_peaks = np.stack([
            np.argmax(signal[:, :80], axis=1),
            np.argmax(signal[:, 80:], axis=1),
            # np.max(signal[:, :80], axis=1),
            # np.max(signal[:, 80:], axis=1)
        ], axis=1)
    return inds_peaks


def get_target_ind(signalI, signal_target, first_lead=False):
    ind_peaksI = get_ind_peaks(signalI, first_lead=True)
    ind_peaks_target = get_ind_peaks(signal_target, first_lead=first_lead)
    dist = ((ind_peaks_target[:, np.newaxis] - ind_peaksI[np.newaxis]) ** 2).sum(axis=-1)
    return dist.argmin(axis=0)


def apply_ekgan(X, ekgan_model, kept_leads, device_baseline):
    # ==== prepare input ==== #
    n_leads = 9
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


class AppleWatchECG(Dataset):
    def __init__(self, preprocessed_path, patient_ids, n_beats, lead_type='flank', normalization='no_norm'):
        self.patient_ids = patient_ids
        self.lead_type = lead_type
        self.apple_ecg = []
        self.ind_lead = []
        self.ecg12 = []
        self.features = []
        self.n_beats = n_beats  # keep n_beats per patient
        self.normalization = normalization
        for p_id in patient_ids:
            npz = np.load(os.path.join(preprocessed_path, f'{p_id}.npz'))
            leadI = npz['leadI_wrist']
            ind_leadI = np.random.randint(low=0, high=leadI.shape[1], size=(n_beats,))
            leadI = leadI[:, ind_leadI]
            rrI = npz['rr_leadI_wrist'][ind_leadI]
            leadII, leadIII, ecg12 = npz[f'leadII_{lead_type}'], npz[f'leadIII_{lead_type}'], npz['beats12']

            ind_leadII = get_target_ind(leadI[0], leadII[0], first_lead=False)
            ind_leadIII = get_target_ind(leadI[0], leadIII[0], first_lead=False)
            ind_lead12 = get_target_ind(leadI[0], ecg12[0], first_lead=True)

            leadII, leadIII, ecg12 = leadII[:, ind_leadII], leadIII[:, ind_leadIII], ecg12[:, ind_lead12]
            self.apple_ecg.append(np.swapaxes(np.concatenate([leadI, leadII, leadIII], axis=0), 0, 1))
            self.ecg12.append(np.swapaxes(ecg12, 0, 1))
            self.ind_lead.append(np.stack([ind_leadI, ind_leadII, ind_leadIII, ind_lead12], axis=-1))
            sex = int('M' in npz['sex']) * np.ones((n_beats,))
            age = float(npz['age']) * np.ones((n_beats,))
            feats = np.stack([sex, 1-sex, (rrI - 125) / 125, (age-50)/50], axis=-1)
            self.features.append(feats)

        self.apple_ecg = np.concatenate(self.apple_ecg)
        self.ecg12 = np.concatenate(self.ecg12)
        self.ind_lead = np.concatenate(self.ind_lead)
        self.features = np.concatenate(self.features)

    def __len__(self):
        return self.apple_ecg.shape[0]

    def __getitem__(self, item):
        '''
        Args:
            item: index of the wrist beat
        Returns:
            input_ecg: (3, 176), only limb leads
            input_feats: (4,) rr of the wrist beat
            indices: (4,) index of the 3 AppleWatch ECGs,
        '''
        apple_ecg_sample = self.apple_ecg[item]
        if self.normalization == 'per_lead':
            apple_ecg_sample /= np.max(np.absolute(apple_ecg_sample), axis=-1)[:, np.newaxis]
        ecg_sample = np.concatenate([self.ecg12[item][:3], self.ecg12[item][6:]])
        if self.normalization == 'per_lead':
            ecg_sample /= np.max(np.absolute(ecg_sample), axis=-1)[:, np.newaxis]
        return apple_ecg_sample, ecg_sample, self.features[item], self.ind_lead[item]


def plot_ecg(ecg_distributions, conditioning_ecg=None, alpha=0.05, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if len(ecg_distributions.shape) == 3:
        for ecg in ecg_distributions:
            for i, track in enumerate(ecg):
                ax.plot(track - i*1.3, c=color_posterior, alpha=alpha, linewidth=.7, rasterized=True)
    else:
        for i, track in enumerate(ecg_distributions):
            ax.plot(track - i * 1.3, c=color_posterior, linewidth=.7, rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    return fig


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    print(torch.cuda._parse_visible_devices())

    cfg.checkpoint = os.path.join(cfg.checkpoint_path, f'baseline_{cfg.normalization}')
    train_state, net_config, ckpt_num = load_net(cfg)
    T = cfg.T  # net_config.generate.T
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho
    # 4.30 GiB on the first GPU here

    timesteps = jnp.arange(0, T - 1) / (T - 2)
    timesteps = (sigma_max ** (1 / p) + timesteps * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** (p)

    results_path = os.path.join(cfg.main_path, 'results')
    os.makedirs(results_path, exist_ok=True)

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

    # === create the dataset and DataLoader === #
    patient_ids = ['M21', 'F23', 'F24', 'F28']
    apple_set = AppleWatchECG(os.path.join(cfg.main_path, 'processed_data'),
                              patient_ids, cfg.n_beats, cfg.lead_type,
                              cfg.normalization)

    apple_loader = DataLoader(dataset=apple_set,
                              batch_size=len(devices("cuda")),
                              shuffle=False,
                              num_workers=0,
                              collate_fn=numpy_collate)

    # ============= EkGAN baseline ============= #
    leads_suffix = '_'.join([str(l) for l in cfg.leads])
    ekgan_model_I = torch.load(os.path.join(cfg.baseline_folder,
                                            f'Ekgan_{cfg.normalization}_{leads_suffix}/best_inference_generator.pth')).cuda()

    seed = 0

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

    # === create the saving path === #
    folder_name = f'inpainting_{cfg.normalization}_' + '_'.join([str(l) for l in cfg.leads])
    if len(cfg.leads) > 1:
        folder_name += '_' + cfg.lead_type
    if cfg.denoising.do_denoise:
        folder_name += f'_l1reg{cfg.denoising.l1reg}_l2reg{cfg.denoising.l2reg}'
    results_path = os.path.join(results_path, folder_name)
    os.makedirs(results_path, exist_ok=True)

    # === initializing for saving the data === #
    all_ref_samples = []
    all_observations = []
    all_target_ecgs = []
    all_inds = []
    all_feats = []
    all_variances = []
    all_eta = []
    all_ekgan = []
    for i, (apple_ecg, ecg, feats, inds) in tqdm.tqdm(enumerate(apple_loader), total=len(apple_loader)):

        # === baseline === #
        X = nn.Upsample(256, mode='linear')(torch.tensor(np.array(apple_ecg)).to(torch.float32))
        with torch.no_grad():
            ecg_recon_ekgan = apply_ekgan(X, ekgan_model_I, np.array(cfg.leads), 'cuda:0')
        all_ekgan.append(ecg_recon_ekgan)

        # === reconstruction === #
        input_ecg = jnp.swapaxes(apple_ecg, 1, 2)
        no_leads = [k for k in range(9) if k not in cfg.leads]
        target_ecg = jnp.swapaxes(ecg, 1, 2)
        observations = jnp.concatenate([input_ecg, jnp.zeros((*input_ecg.shape[:2], 6))], axis=-1).at[:, :, no_leads].set(jnp.nan)
        # observations = target_ecg.at[:, :, no_leads].set(jnp.nan)

        all_observations.append(input_ecg)
        all_target_ecgs.append(target_ecg)
        all_inds.append(inds)
        all_feats.append(feats)

        if not cfg.denoising.do_denoise:
            sigma_ests = jnp.ones((observations.shape[0], 9,)) * 1
            variances = pmap(partial(em_variance_only,
                                     key=random.PRNGKey(seed + i*3),
                                     timesteps=timesteps,
                                     denoiser=lambda x, sigma, feats_tmp: train_state.apply_fn(train_state.params,
                                                                                           x.reshape(-1, 176, 9),
                                                                                           sigma, feats_tmp).reshape(
                                         x.shape[0], -1),
                                     max_iter=10,
                                     n_particles=cfg.N,
                                     n_samples=100,
                                     ),
                            in_axes=(0, 0, 0))(observations, sigma_ests, feats)

            ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(seed + i*3+1), feats.shape[0]),
                                                 observations,
                                                 variances,
                                                 feats)
        else:
            sigma_ests = jnp.ones((feats.shape[0], 9,)) * 1
            init_eta = jnp.zeros((feats.shape[0], 2*J, 9))
            init_eta = pmap(partial(lasso_eta,
                                    # initial_variance=sigma_ests,
                                    key=random.PRNGKey(seed + i*3+1),
                                    timesteps=timesteps,
                                    denoiser=lambda x, sigma, feats_tmp: train_state.apply_fn(train_state.params,
                                                                                          x.reshape(-1, 176, 9),
                                                                                          sigma, feats_tmp).reshape(
                                        x.shape[0], -1),
                                    max_iter=5,
                                    phi=phi,
                                    lreg=cfg.denoising.l1reg,
                                    n_particles=50,
                                    n_samples=100,
                                    ),
                            in_axes=(0, 0, 0, 0))(
                observations, sigma_ests, init_eta, feats)  # [:n_patients])
            harmonics_th = 1e-3
            mask_phi = np.absolute(init_eta) > harmonics_th
            # leads_list = [np.where(mask_phi[p].sum(axis=0) > 0)[0] for p in range(mask_phi.shape[0])]
            variances, final_eta = pmap(partial(em_variance,
                                                key=random.PRNGKey(seed + i*3+2),
                                                timesteps=timesteps,
                                                denoiser=lambda x, sigma, feats: train_state.apply_fn(
                                                    train_state.params,
                                                    x.reshape(-1, 176, 9),
                                                    sigma, feats).reshape(x.shape[0], -1),
                                                max_iter=5,
                                                phi=phi,
                                                # mask_phi=mask_phi,
                                                # leads_list=leads_list,
                                                lreg=1,
                                                n_particles=50,
                                                n_samples=100,
                                                ),
                                        in_axes=(0, 0, 0, 0, 0))(
                observations,
                sigma_ests, init_eta, feats, mask_phi)  # [:n_patients])

            ref_samples = posterior_sampling_fun(
                random.split(random.PRNGKey(seed + 3 + i), feats.shape[0]),
                observations - phi[:, :J] @ final_eta[:, :J] - phi[:, J:] @ final_eta[:, J:],
                variances,
                feats)  # [:n_patients])

            all_eta.append(final_eta)
        all_variances.append(variances)
        all_ref_samples.append(ref_samples)
        ecg_indices = np.concatenate(all_inds)
        np.savez(os.path.join(results_path, 'inpainting_from_watch.npz'),
                 posterior_samples=np.concatenate(all_ref_samples),
                 input_ecg=np.concatenate(all_observations),
                 target_ecg=np.concatenate(all_target_ecgs),
                 feats=np.concatenate(all_feats),
                 eta=np.concatenate(all_eta) if cfg.denoising.do_denoise else [],
                 phi=phi if cfg.denoising.do_denoise else [],
                 variance=np.concatenate(all_variances),
                 ekgan=np.concatenate(all_ekgan),
                 ind_leadI=ecg_indices[:, 0],
                 ind_leadII=ecg_indices[:, 1],
                 ind_leadIII=ecg_indices[:, 2],
                 ind_lead12=ecg_indices[:, 3],
                 )

    for k in range(ref_samples.shape[0]):
        plot_ecg(np.mean(ref_samples[k], axis=0).T, target_ecg[k].T)
        plt.show()
    for k in range(ref_samples.shape[0]):
        plot_ecg(np.swapaxes(ref_samples[k], 1, 2), target_ecg[k].T)
        plt.show()
    print('ok')

if __name__ == '__main__':
    main()

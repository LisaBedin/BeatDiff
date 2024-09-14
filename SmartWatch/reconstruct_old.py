import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,4,3,2,1'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import em_variance_only, mcg_diff_ve
import hydra
from omegaconf import DictConfig, OmegaConf
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from jax.tree_util import Partial as partial
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

from EMbeat_diff_inpainting import apply_ekgan  # careful if CUDA_VISIBLE_DEVICES configured in this script !


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

    train_state, net_config, ckpt_num = load_net(cfg)
    T = cfg.T  # net_config.generate.T
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho
    # 4.30 GiB on the first GPU here

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

    # === create the dataset and DataLoader === #
    analyzed_ecgs = {'li': {'W_I': np.array([24]),
                            'W_II': np.array([0]),
                            '12L': np.array([6])},
                     'ca': {'W_I': np.array([15]),
                            'W_II': np.array([5]),
                            '12L': np.array([5])},
                     'ma': {'W_I': np.array([19]),
                            'W_II': np.array([26]),
                            '12L': np.array([4])},
                     'ch': {'W_I': np.array([39]),
                            'W_II': np.array([33]),
                            '12L': np.array([5])}
                     }
    # [(19, 4), (14, 5), (33, 2)]}
    all_beats12, all_beatsW, all_feats = [], [], []
    all_ekgan = []
    for patient_id, ecg_inds in analyzed_ecgs.items():
        npz_file = jnp.load(os.path.join(cfg.data_folder, f'patient_ID{patient_id}.npz'), allow_pickle=True)
        beats12 = npz_file['beats12'][:, analyzed_ecgs[patient_id]['12L']]
        beats12 /= jnp.absolute(beats12).max(axis=2)[..., jnp.newaxis]
        beatsW = npz_file['beatsW_I'][:, analyzed_ecgs[patient_id]['W_I']]
        beatsW /= jnp.absolute(beatsW).max(axis=2)[..., jnp.newaxis]
        if len(npz_file['beatsW_II'].shape) > 0:
            beatsW_II = npz_file['beatsW_II'][:, analyzed_ecgs[patient_id]['W_II']]
            beatsW_II /= jnp.absolute(beatsW_II).max(axis=2)[..., jnp.newaxis]
        else:
            beatsW_II = jnp.nan * jnp.ones_like(beatsW)
        beatsW = jnp.concatenate([beatsW, beatsW_II], axis=0)
        rrW = (npz_file['rrW_I'][analyzed_ecgs[patient_id]['W_I']]-125) / 125
        # rr12 = (npz_file['rr12'][analyzed_ecgs[patient_id]['12L']]-125) / 125
        sex = jnp.repeat(jnp.array([0, 1])[jnp.newaxis], rrW.shape[0], axis=0)
        age = jnp.array([(npz_file['age']-50)/50]*rrW.shape[0])[:, jnp.newaxis]
        feats = jnp.concatenate([sex, rrW[:, jnp.newaxis], age], axis=1)
        all_beatsW.append(beatsW)
        all_beats12.append(beats12)
        all_feats.append(feats)

    all_beats12 = jnp.moveaxis(jnp.concatenate(all_beats12, axis=1), [0, 1, 2], [2, 0, 1])
    all_beats9 = jnp.concatenate([all_beats12[:, :, :3], all_beats12[:, :, 6:]], axis=-1)
    all_beatsW = jnp.moveaxis(jnp.concatenate(all_beatsW, axis=1), [0, 1, 2], [2, 0, 1])
    all_feats = np.concatenate(all_feats)

    # ============= EkGAN baseline ============= #
    ekgan_model_I = torch.load(os.path.join(cfg.baseline_folder,
                                            'Ekgan_per_lead_norm_0/best_inference_generator.pth')).cuda()
    X = nn.Upsample(256, mode='linear')(torch.swapaxes(torch.tensor(np.array(all_beatsW)).to(torch.float32), 1, 2))
    with torch.no_grad():
        ecg_recon_ekgan = apply_ekgan(X, ekgan_model_I, np.array([0]), 'cuda:0')
    all_ekgan.append(ecg_recon_ekgan)

    # ============= EM-beatdiff for inpainting_samples_K20 from Apple Watch Data ============= #
    observations = jnp.concatenate([all_beatsW,
                                    jnp.zeros((*all_beatsW.shape[:2], 7))], axis=-1).at[:, :, 1:].set(jnp.nan)

    sigma_ests = jnp.ones((observations.shape[0], 9,)) * 1
    variances = pmap(partial(em_variance_only,
                             key=random.PRNGKey(0),
                             timesteps=timesteps,
                             denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                   x.reshape(-1, 176, 9),
                                                                                   sigma, feats).reshape(
                                 x.shape[0], -1),
                             max_iter=10,
                             n_particles=cfg.N,
                             n_samples=100,
                             ),
                    in_axes=(0, 0, 0))(observations, sigma_ests, all_feats)


    ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(0), all_feats.shape[0]),
                                         observations,
                                         variances,
                                         all_feats)


    # for k in range(ref_samples.shape[0]):
    #     plot_ecg(np.mean(ref_samples[k], axis=0).T, all_beats9[k].T)
    #     plt.show()
    # for k in range(ref_samples.shape[0]):
    #     plot_ecg(np.swapaxes(ref_samples[k], 1, 2), all_beats9[k].T)
    #     plt.show()
    #
    # print('ok')

    # === clustering to identify modes === #
    for k in range(ref_samples.shape[0]):
        n_clusters = 10
        spectral = cluster.SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            random_state=0,
        )
        spectral.fit(ref_samples[k].reshape(100, -1))
        cluster_labels = spectral.labels_
        clustered_samples = np.stack([ref_samples[k][cluster_labels == lab].mean(axis=0) for lab in range(n_clusters)])
        plot_ecg(np.swapaxes(clustered_samples, 1, 2), all_beats9[k].T, alpha=.5)
        plt.show()
        lab_names, lab_count = np.unique(cluster_labels, return_counts=True)
        print(lab_count)
        plot_ecg(clustered_samples[lab_names[lab_count.argmax()]].T, all_beats9[k].T, alpha=.5)
        plt.show()

    # ============= EM-beatdiff for inpainting_samples_K20 from Lead I in ECGs ============= #
    observations = all_beats9.at[:, :, 1:].set(jnp.nan)

    sigma_ests = jnp.ones((observations.shape[0], 9,)) * 1
    variances = pmap(partial(em_variance_only,
                             key=random.PRNGKey(0),
                             timesteps=timesteps,
                             denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                   x.reshape(-1, 176, 9),
                                                                                   sigma, feats).reshape(
                                 x.shape[0], -1),
                             max_iter=10,
                             n_particles=cfg.N,
                             n_samples=100,
                             ),
                    in_axes=(0, 0, 0))(observations, sigma_ests, all_feats)


    ref_samples_ECG = posterior_sampling_fun(random.split(random.PRNGKey(0), all_feats.shape[0]),
                                         observations,
                                         variances,
                                         all_feats)

    np.savez(os.path.join(cfg.results_path, 'inpainting_results.npz'),
             posterior_WATCH=ref_samples,
             posterior_ECG=ref_samples_ECG,
             feats=all_feats
             )

    print('ok')
    '''
        all_beats12 = jnp.moveaxis(jnp.concatenate(all_beats12, axis=1), [0, 1, 2], [2, 0, 1])
    all_beats9 = jnp.concatenate([all_beats12[:, :, :3], all_beats12[:, :, 6:]], axis=-1)
    all_beatsW = jnp.moveaxis(jnp.concatenate(all_beatsW, axis=1), [0, 1, 2], [2, 0, 1])
    all_feats
    '''

if __name__ == '__main__':
    main()

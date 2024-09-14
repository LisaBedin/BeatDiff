import os

import tqdm
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,'  # 4,6,7'
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


def calculate_qt(samples):

    pca = Pipeline([('Scaler', StandardScaler()),
                    ('PCA', PCA(n_components=3))])
    pca.fit(np.array(samples.reshape(-1, samples.shape[-1])))
    ref_track = pca.transform(np.array(samples.reshape(-1, samples.shape[-1])))[:, 0].reshape(samples.shape[0], samples.shape[1])
    ref_track = np.concatenate(ref_track)
    _, waves_infos = ecg_delineate(ref_track, sampling_rate=250,
                                   method='dwt')
    qts = np.array([t - q for q, t in zip(waves_infos['ECG_R_Onsets'], waves_infos['ECG_T_Offsets']) if (t - q) == (t - q)])
    if len(qts):
        return qts
    return np.array([t - q - 3 for q, t in zip(waves_infos['ECG_Q_Peaks'], waves_infos['ECG_T_Offsets']) if (t - q) == (t - q)])


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

    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path, categories_to_filter=["NSR", ],
                                                      normalized=True, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=len(devices("cuda")),
                                 shuffle=True,
                                 num_workers=4,
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
    #results_path = os.path.join(cfg.mcg_diff.results_path, cfg.mcg_diff.setting, 'qt_as_rr')
    #os.makedirs(results_path, exist_ok=True)
    rrs, qts_mean, qts_interval = [], [], []
    for (batch_test_ecg, batch_test_features) in test_dataloader:
        sigma_ests = jnp.ones((n_patients, 9,))*1
        observations = (jnp.nan * jnp.ones_like(batch_test_ecg[:n_patients])).at[:, :70].set(batch_test_ecg[:n_patients, :70])
        variances = pmap(partial(em_variance,
                                 key=random.PRNGKey(seed),
                                 timesteps=timesteps,
                                 denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params,
                                                                                       x.reshape(-1, 176, 9),
                                                                                       sigma, feats).reshape(x.shape[0], -1),
                                 max_iter=5,
                                 n_particles=50,
                                 n_samples=100,
                                 ),
                         in_axes=(0, 0, 0))(observations, sigma_ests, batch_test_features[:n_patients])

        batch_rrs, batch_qts_mean, batch_qts_interval = [], [], []
        # for i, real_rr in enumerate(jnp.linspace(600, 1200, num=20)):
        for i, real_rr in enumerate(jnp.linspace(900, 2200, num=20)):
            rr_to_put = ((real_rr / 4) - 125) / 125
            class_features = jnp.copy(batch_test_features[:n_patients])
            class_features = class_features.at[:, 2].set(rr_to_put)
            print(real_rr, class_features[:, 2])

            ref_samples = posterior_sampling_fun(random.split(random.PRNGKey(seed + 3 + i), n_patients),
                                                 observations,
                                                 variances,
                                                 class_features)
            all_qt = [calculate_qt(s) * 4 for s in ref_samples]
            batch_rrs.append(real_rr)
            batch_qts_mean.append(np.array([np.mean(q) for q in  all_qt]))
            batch_qts_interval.append(np.array([(np.std(qt) * 1.96) / (len(qt)**.5) for qt in all_qt]))
            '''
            print(batch_qts_mean[-1] / (batch_rrs[-1])**.5)
            for j, (patient_samples, ref) in enumerate(zip(ref_samples, batch_test_ecg[:n_patients])):
                fig = plot_ecg(ecg_distributions=jnp.swapaxes(patient_samples,
                                                              1, 2),
                               conditioning_ecg=ref[:70].T,
                               color_target=color_target,
                               color_posterior=color_posterior)
                fig.savefig(f'images/example_{i}_{j}_QT_as_RR_{int(real_rr)}.pdf')
                plt.close(fig)
            '''
        # break
        # color_points = '#00428d'#'#47d1dc'
        # color_curve = '#fa526c'
        batch_qts_mean = np.stack(batch_qts_mean, axis=1) / 1000
        batch_qts_interval = np.stack(batch_qts_interval, axis=1) / 1000
        batch_rrs = np.array(batch_rrs) / 1000
        rrs.append(batch_rrs)
        qts_mean.append(batch_qts_mean)
        qts_interval.append(batch_qts_interval)
        np.savez('/mnt/data/lisa/qt_as_rr.npz',#os.path.join(results_path, 'qt_as_rr.npz'),
                 qts_mean=np.concatenate(qts_mean),
                 qts_interval=np.concatenate(qts_interval),
                 rrs=np.concatenate(rrs))
        # plt.rcParams.update({'font.size': 18})
        # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # fig.subplots_adjust(right=1, left=.2, top=1, bottom=.1)
        # for qt_mean, qt_interval, color in zip(qts_mean[:, :].T,
        #                                        qts_interval[:, :-1].T,
        #                                        cm.rainbow(np.linspace(0, 1, qts_mean.shape[-1]))):
        #     lr = LinearRegression(fit_intercept=True).fit((rrs ** .5)[:, None], qt_mean)
        #     span = np.linspace(min(rrs) * .95, max(rrs) * 1.05, 100)
        #     curve = lr.predict((span ** .5)[:, None])
        #     ax.errorbar(x=rrs, y=qt_mean, yerr=qt_interval, c=color, capsize=10, fmt="none", alpha=.8)
        #     ax.plot(span, curve, c=color)
        # ax.set_xlim(min(span), max(span))
        # fig.savefig('images/QT_as_RR.pdf')
        # plt.close(fig)

if __name__ == '__main__':
    metrics = main()

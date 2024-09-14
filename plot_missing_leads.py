import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import matplotlib.pyplot
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from scipy.stats.distributions import chi2
def plot_ecg(missing_lead, ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
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
        if i == missing_lead:
            ax.plot(track - i * 1.3, c=color_target, ls='--', linewidth=.7, rasterized=True)  # rasterized=True)
        else:
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

    npz1 = np.load('/mnt/data/lisa/ecg_results/inpainting_samples_20/safe_NSR/NSR_SB_STach_SA_1.npz')
    npz3 = np.load('/mnt/data/lisa/ecg_results/inpainting_samples_20/safe_NSR/NSR_SB_STach_SA.npz')









    data = np.load(os.path.join(cfg.checkpoint, 'missing_leads.npy.npz'))

    missing_leads = data.get('missing_leads')
    posterior_samples = data.get('posterior_samples')
    observations = data.get('observations')
    dowers_reconstruction = data.get('dowers_reconstruction')

    scale_length = 25
    mahanalobis_coeff_decay = jnp.stack([jnp.exp(-(jnp.arange(176) - i)**2 / (2*scale_length**2)) for i in range(176)])
    mahanalobis_coeff_decay = mahanalobis_coeff_decay.at[jnp.abs(mahanalobis_coeff_decay) < 1e-2].set(0)
    mahanalobis_distances = []

    arange_helper = np.arange(len(missing_leads))
    posterior_mean_rec = np.mean(posterior_samples, axis=1)[arange_helper, :, missing_leads]
    dowers_rec = dowers_reconstruction[arange_helper, missing_leads, :]
    real_signal = observations[arange_helper, :, missing_leads]

    signal_energy = np.sum((real_signal-real_signal.mean(axis=1)[:, np.newaxis])**2, axis=-1)
    error_mcg_diff = np.sum((posterior_mean_rec - real_signal)**2, axis=-1)
    error_dowers = np.nansum((real_signal - dowers_rec)**2, axis=-1)

    r2_mcg_diff = 1 - (error_mcg_diff / signal_energy)
    r2_dowers = 1 - (error_dowers / signal_energy)

    posterior_lead = posterior_samples[arange_helper, :, :, missing_leads]

    # means = posterior_samples[..., missing_leads].mean(axis=1)  # on calcule mu à partir des échantillons de X_0 | y
    precs = jnp.stack([jnp.linalg.pinv(jnp.cov(track.T) * mahanalobis_coeff_decay)
        for track in posterior_lead
    ]) # on calcule Sigma à partir des échantillons de X_0 | y
    track_dists = np.stack([
        mahalanobis(t_mean, t_obs, t_prec) for t_mean, t_obs, t_prec in zip(posterior_mean_rec, real_signal, precs)])/chi2.ppf(0.999, 176)

    all_data = []
    for i, track in enumerate(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')):
        indices = np.where(missing_leads == i)[0]
        if len(indices) > 0:
            all_data.append(
                {
                    "track": track,
                    "mcg_diff": f'{track_dists[indices].mean(axis=0):.2f} ± {(1.96*r2_mcg_diff[indices].std(axis=0) / (len(indices)**.5)):.2f}',
                    "mcg_diff_num": track_dists[indices].mean(axis=0)
                }
            )
    pd.DataFrame.from_records(all_data).to_csv(os.path.join(cfg.inpainting.results_path, 'missing_leads_mahalanobis.csv'))
    indices = np.where(np.sum(np.stack([missing_leads==k for k in range(3, 9)]),axis=0)>0)[0]
    print("mcg_diff", f'{r2_mcg_diff[indices].mean(axis=0):.3f} ± {(1.96*r2_mcg_diff[indices].std(axis=0) / (len(indices)**.5)):.3f}')
    print("dowers", f'{r2_dowers[indices].mean(axis=0):.3f} ± {(1.96 * r2_dowers[indices].std(axis=0) / (len(indices) ** .5)):.3f}')
    all_data = []
    for i, track in enumerate(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')):
        indices = np.where(missing_leads == i)[0]
        if len(indices) > 0:
            all_data.append(
                {
                    "track": track,
                    "mcg_diff": f'{r2_mcg_diff[indices].mean(axis=0):.3f} ± {(1.96*r2_mcg_diff[indices].std(axis=0) / (len(indices)**.5)):.3f}',
                    "mcg_diff_num": r2_mcg_diff[indices].mean(axis=0),
                    "dowers": f'{r2_dowers[indices].mean(axis=0):.3f} ± {(1.96 * r2_dowers[indices].std(axis=0) / (len(indices) ** .5)):.3f}',
                    "dowers_num": r2_dowers[indices].mean(axis=0),
                }
            )
    pd.DataFrame.from_records(all_data).to_csv(os.path.join(cfg.inpainting.cfg.inpainting.results_path, 'missing_leads_r2.csv'))

    results_path = cfg.inpainting.results_path  # s.path.join(cfg.results_path, 'inpainting_eval')
    # =========== plot some reconstructed samples ========== #
    color_posterior = '#00428d'
    color_target = '#fa526c'
    generated_ecg_lst = [np.swapaxes(posterior_samples, 2, 3) , dowers_reconstruction[:, np.newaxis]]
    method_name_lst = ['diff', 'dower']
    n_test = posterior_samples.shape[0]
    for piste in range(9):
        # inds = np.arange(n_test)[missing_leads==piste]
        for pt in range(0, 101, 10):
            pt_value = float(np.percentile(r2_dowers[missing_leads == piste], q=pt))
            id_ = int(np.argmin(np.abs(r2_dowers[missing_leads == piste] - pt_value)))
            for method_name, generated_ecg in zip(method_name_lst, generated_ecg_lst):
                fig = plot_ecg(missing_lead=missing_leads[id_],
                               ecg_distributions=generated_ecg[id_],
                               conditioning_ecg=observations[id_].T,
                               color_target=color_target,
                               color_posterior=color_posterior)

                fig.savefig(os.path.join(results_path,
                                         f'inpainting{piste}_dower{pt}_{method_name}.pdf'))
                # fig.show()
                plt.close()
        for pt in range(0, 101, 10):
            pt_value = float(np.percentile(r2_mcg_diff[missing_leads == piste], q=pt))
            id_ = int(np.argmin(np.abs(r2_mcg_diff[missing_leads == piste] - pt_value)))
            for method_name, generated_ecg in zip(method_name_lst, generated_ecg_lst):
                fig = plot_ecg(missing_lead=missing_leads[id_],
                               ecg_distributions=generated_ecg[id_],
                               conditioning_ecg=observations[id_].T,
                               color_target=color_target,
                               color_posterior=color_posterior)

                fig.savefig(os.path.join(results_path,
                                         f'inpainting{piste}_diff{pt}_{method_name}.pdf'))
                # fig.show()
                plt.close()
if __name__ == '__main__':
    main()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import glob
import hydra
import numpy as np
from collections import Counter
from matplotlib.colors import to_rgb
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib
import ot
import torch
import torch.nn as nn
import scipy.spatial.distance
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import mahalanobis
from dtw import dtw
from tqdm import tqdm
import jax.numpy as jnp
from scipy.stats import chi2
from sklearn.metrics import roc_curve, auc, roc_auc_score

matplotlib.rc('font', **{'size'   : 22})
np.random.seed(0)


def plot_ecg(ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for ecg in ecg_distributions:
        for i, track in enumerate(ecg):
            alpha = 0.05
            if len(ecg_distributions) == 1:
                alpha=1
            ax.plot(track - i*1.3, c=color_posterior, alpha=alpha, linewidth=.7, rasterized=True)  # rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)
    # ax.set_ylim(-13.5, 1.5)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    # ax.set_yticks([-i*1.5 for i in range(9)])
    #ax.set_xticklabels(np.arange(0, 175, 50).astype(int), fontsize=22)
    #ax.set_yticklabels(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), fontsize=22)
    return fig


def get_EMD_L2(generated_ecg, target_ecg):
    N = generated_ecg.shape[0]
    source = generated_ecg.reshape((N, -1))
    target = target_ecg.reshape((1, -1))
    M = ot.dist(source, target)
    return ot.emd2(a=[], b=[], M=M)


def get_R2_score(target_ecg, generated_ecg, agg_fun):
    signal_energy = np.sum((target_ecg - target_ecg.mean(axis=1)[:, np.newaxis]) ** 2, axis=1)
    error_mcg_diff = np.sum((generated_ecg - target_ecg) ** 2, axis=1)

    r2_mcg_diff = 1 - (error_mcg_diff / signal_energy)
    return agg_fun(r2_mcg_diff, axis=-1) # aggregating over the leads


def get_mahanalobis(target_ecg, generated_ecg, agg_fun, decay):
    ecg_length = generated_ecg.shape[2]
    if decay:
        scale_length = 25
        mahanalobis_coeff_decay = np.stack([np.exp(-(jnp.arange(ecg_length) - i)**2 / (2*scale_length**2))
                                            for i in range(ecg_length)])
        mahanalobis_coeff_decay[np.abs(mahanalobis_coeff_decay) < 1e-2] = 0
        print(mahanalobis_coeff_decay[0][:10])
    else:
        mahanalobis_coeff_decay = jnp.ones((ecg_length, ecg_length))

    means_qrs = generated_ecg.mean(axis=1)

    prec_qrs = jnp.stack([
        jnp.stack([jnp.linalg.pinv(jnp.cov(qrs.T) * mahanalobis_coeff_decay)
                   for qrs in jnp.swapaxes(all_tracks_qrs, 0, 1)])
        for all_tracks_qrs in jnp.swapaxes(generated_ecg, -1, -2)
    ])

    # outlier_coeff = chi2.ppf(.999, df=176)  # TODO: 0.9, 0.95 or 0.999 ?
    dist = [agg_fun([mahalanobis(*i) for i in zip(*it)]) for it in zip(np.swapaxes(target_ecg, 1, 2),
                                                                       np.swapaxes(means_qrs, 1, 2),
                                                                       prec_qrs)]
    # return np.array(dist) / outlier_coeff
    return np.array(dist)

ALL_AGG_FUN = {'max': np.max, 'min': np.min, 'mean': np.mean, 'median': np.median, 'sum': np.sum}


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(0)
    OmegaConf.set_struct(cfg, False)

    results_path = os.path.join(cfg.results_path, f'inpainting_samples_{cfg.inpainting.T}')
    cardiac_labels = '_'.join(cfg.inpainting.labels)
    save_name = cardiac_labels + '_' + cfg.eval.metric + '_' + cfg.eval.agg_fun
    # if os.path.isfile(os.path.join(results_path, save_name + '.csv')

    suffix = ''
    if cfg.inpainting.QRS_only:
        suffix = '_qrs'
    if not cfg.inpainting.em_variance:
        suffix += '_noiseless'
    # ==== loading the inpainted data ==== #
    if cfg.inpainting.labels[0] != 'LQTS_clean':
        ctrl_npz = np.load(os.path.join(results_path, f'NSR_SB_STach_SA{suffix}.npz'))
        anomaly_npz = np.load(os.path.join(results_path, f'{cardiac_labels}{suffix}.npz'))
        mcg_ecg = np.concatenate([ctrl_npz['posterior_samples'], anomaly_npz['posterior_samples']])
        nan_inds = np.isnan(mcg_ecg[:, 0, 0, 0])
        mcg_ecg = mcg_ecg[~nan_inds]
        ctrl_ecg, anomaly_ecg = ctrl_npz['ground_truth'], anomaly_npz['ground_truth']
        labels = np.array([1]*len(ctrl_ecg) + [0]*len(anomaly_ecg))[~nan_inds] # 1 is healthy 0 is unhealthy
        target_ecg = np.concatenate([ctrl_ecg, anomaly_ecg])[~nan_inds]
        ekgan_I_ecg = np.concatenate([ctrl_npz['ekgan_I'], anomaly_npz['ekgan_I']])[~nan_inds, :, 1:]
        ekgan_limb_ecg = np.concatenate([ctrl_npz['ekgan_limb'], anomaly_npz['ekgan_limb']])[~nan_inds, :, 3:]
        AAE_ecg = np.concatenate([ctrl_npz['AAE'], anomaly_npz['AAE']])[~nan_inds]
        dower_ecg = np.concatenate([ctrl_npz['dowers_reconstruction'], anomaly_npz['dowers_reconstruction']])[~nan_inds, :, 6:]
    else:
        npz = np.load(os.path.join(results_path, f'{cardiac_labels}{suffix}.npz'))
        mcg_ecg = npz['posterior_samples']
        nan_inds = np.isnan(mcg_ecg[:, 0, 0, 0])
        mcg_ecg = mcg_ecg[~nan_inds]
        target_ecg = npz['ground_truth'][~nan_inds]
        ekgan_I_ecg = npz['ekgan_I'][~nan_inds, :, 1:]
        ekgan_limb_ecg = npz['ekgan_limb'][~nan_inds, :, 3:]
        AAE_ecg = npz['AAE'][~nan_inds]
        dower_ecg = npz['dowers_reconstruction'][~nan_inds, :, 6:]
        labels = 1-npz['genetic_annotation'][~nan_inds] # 1 is healthy (NEG), 0 is unhealthy (POS)

    # if cfg.eval.metric == 'R2_score':
    if cfg.inpainting.QRS_only:
        mcg_ecg = mcg_ecg[:, :, 70:]
    else:
        mcg_ecg = mcg_ecg[:, :, :, 3:]
    if 'mahanalobis' not in cfg.eval.metric:
        mcg_ecg = mcg_ecg.mean(axis=1)

    agg_fun = ALL_AGG_FUN[cfg.eval.agg_fun]

    ekgan_I_dist = get_R2_score(target_ecg[:, :, 1:], ekgan_I_ecg, agg_fun)
    ekgan_limb_dist = get_R2_score(target_ecg[:, :, 3:], ekgan_limb_ecg, agg_fun)
    AEE_dist = get_R2_score(target_ecg, AAE_ecg, agg_fun)
    dower_dist = get_R2_score(target_ecg[:, :, 6:], dower_ecg, agg_fun)

    target_mcg = target_ecg[:, 70:] if cfg.inpainting.QRS_only else target_ecg[:, :, 3:]
    if cfg.eval.metric == 'mahanalobis':
        mcg_dist = get_mahanalobis(target_mcg, mcg_ecg, agg_fun, 'decay' in cfg.eval.metric)
    else:
        mcg_dist = get_R2_score(target_mcg, mcg_ecg, agg_fun)

    fpr, tpr, _ = roc_curve(labels, ekgan_I_dist)
    roc_auc = auc(fpr, tpr)
    print(f'ekgan_I {roc_auc*100:.2f}')
    plt.plot(fpr, tpr, label=f"ekgan_I {roc_auc:.2f}")
    fpr, tpr, _ = roc_curve(labels, ekgan_limb_dist)
    roc_auc = auc(fpr, tpr)
    print(f'ekgan_limb {roc_auc * 100:.2f}')
    plt.plot(fpr, tpr, label=f"ekgan_limb {roc_auc:.2f}")

    fpr, tpr, _ = roc_curve(labels, AEE_dist)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AAE {roc_auc:.2f}")
    print(f'AAE {roc_auc * 100:.2f}')
    fpr, tpr, _ = roc_curve(labels, dower_dist)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"dower {roc_auc:.2f}")
    print(f'dower {roc_auc * 100:.2f}')
    if cfg.eval.metric == 'mahanalobis':
        fpr, tpr, _ = roc_curve(1-labels, mcg_dist)
    else:
        fpr, tpr, _ = roc_curve(labels, mcg_dist)
    roc_auc = auc(fpr, tpr)
    print(f'mcg {roc_auc * 100:.2f}')
    plt.plot(fpr, tpr, label=f"ours {roc_auc:.2f}")
    plt.legend()
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.025])
    plt.xticks([0.5, 1], ['0.5', '1'])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    plt.show()

    df = pd.DataFrame({'is_healthy': labels,
                       'our': mcg_dist,
                       'AAE': AEE_dist,
                       'ekgan_I': ekgan_I_dist,
                       'ekgan_limb': ekgan_limb_dist,
                       'dower': dower_dist})
    df.to_csv(os.path.join(results_path, save_name+suffix + '.csv'), index=False)

    if cfg.inpainting.QRS_only:
        ecg_length = generated_ecg.shape[-2]
        if qrs_only:
            generated_ecg = generated_ecg[:, :, 40:70]
            target_ecg = target_ecg[:, :, 40:70]
        if 'R2_score' in dist_name:
            posterior_mean_rec = np.mean(generated_ecg, axis=1)
            real_signal = target_ecg[:, 0]
            signal_energy = np.sum((real_signal - real_signal.mean(axis=1)[:, np.newaxis]) ** 2, axis=1)
            error_mcg_diff = np.sum((posterior_mean_rec - real_signal) ** 2, axis=1)

            r2_mcg_diff = 1 - (error_mcg_diff / signal_energy)
            all_dist.append(r2_mcg_diff.mean(axis=-1))  # averaging over the leads
        if 'OT_L2' in dist_name:
            print('OT')
            dist = [get_EMD_L2(generated_ecg[k], target_ecg[k])
                    for k in range(generated_ecg.shape[0])]
            all_dist.append(dist)
        if 'covariance' in dist_name:
            prec_qrs = np.stack([
                [jnp.corrcoef(qrs.T) for qrs in jnp.swapaxes(all_tracks_qrs, 0, 1)] for all_tracks_qrs in jnp.swapaxes(generated_ecg, -1, -2)
            ])

            dist = np.mean(prec_qrs, axis=(2, 3))
            all_dist.append(dist)
        if 'mahanalobis' in dist_name:
            means_qrs = generated_ecg.mean(axis=1)
            cov_kernel = np.ones((ecg_length, ecg_length))
            if 'time' in dist_name:
                cov_kernel = np.abs(np.arange(ecg_length)[None] - np.arange(ecg_length)[:, None]) + 1
            prec_qrs = jnp.stack([
                jnp.stack([jnp.linalg.pinv(jnp.cov(qrs.T)*mahanalobis_coeff_decay)#/cov_kernel)
                 for qrs in jnp.swapaxes(all_tracks_qrs, 0, 1)])
                for all_tracks_qrs in jnp.swapaxes(generated_ecg, -1, -2)
            ])
            if 'max' in dist_name:
                agg_fun = max
            elif 'min' in dist_name:
                agg_fun = min
            elif 'mean' in dist_name:
                agg_fun = np.mean
            elif 'median' in dist_name:
                agg_fun = np.median
            elif 'sum' in dist_name:
                agg_fun = np.sum
            else:
                raise NotImplementedError
            outlier_coeff = chi2.ppf(.999, df=176)  # TODO: 0.9, 0.95 or 0.999 ?
            dist = [agg_fun([mahalanobis(*i) for i in zip(*it)]) for it in zip(np.swapaxes(target_ecg[:, 0], 1, 2),
                                                                               np.swapaxes(means_qrs, 1, 2),
                                                                               prec_qrs)]
            all_dist.append(np.array(dist)/outlier_coeff)
        if dist_name[:2] == 'L2':
            L2_distance = np.sqrt(np.sum((generated_ecg-target_ecg)**2, axis=2)).min(axis=1)
            if dist_name[2:] == '_mean':
                all_dist.append(L2_distance.mean(axis=-1))
            elif dist_name[2:] == '_min':
                all_dist.append(L2_distance.min(axis=-1))
            elif dist_name[2:] == '_max':
                all_dist.append(L2_distance.max(axis=-1))
        if dist_name == 'MSE':
            L2_distance = np.sqrt(np.sum((generated_ecg-target_ecg)**2, axis=2)).mean(axis=(1, 2))
            all_dist.append(L2_distance)
        if dist_name[:3] == 'dtw':
            for patient_particles, patient_ecg in tqdm(zip(generated_ecg, target_ecg)):
                all_particles = np.concatenate((patient_particles, patient_ecg), axis=0)
                all_particles = all_particles.reshape(all_particles.shape[0], -1)
                cdists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(all_particles, lambda u, v: dtw(u, v).distance))
                cdists[np.arange(cdists.shape[0]), np.arange(cdists.shape[0])] = np.nan
                all_dist.append(np.nanmax(cdists[-1]) / np.nanmedian(cdists[:-1, :-1]))

        file_name = file_path.split('/')[-1].split('.')[0]
        n_patients = generated_ecg.shape[0]
        all_names += [file_name + f'_{i}' for i in range(n_patients)]
        all_abbrev += [file_name.split('_')[0]]*n_patients
    all_dist = np.concatenate(all_dist)
    all_sex = np.concatenate(all_sex)
    all_names = np.array(all_names)
    all_ages = np.concatenate(all_ages)
    all_abbrev = np.array(all_abbrev)
    all_gen = np.concatenate(all_gen)
    all_target = np.concatenate(all_target)
    all_baseline = np.concatenate(all_baseline)
    all_ekgan = np.concatenate(all_ekgan)
    # TODO: reweight the ages in NSR such that age distrib is the same in NSR and MI

    # ======= uniformisation of features ======== #
    kept_inds = get_equlibrated(all_ages, all_abbrev, all_sex)
    kept_inds = np.arange(all_abbrev.shape[0])
    equil_abbrev = all_abbrev[kept_inds]
    equil_dist = all_dist[kept_inds]
    equil_ekgan = all_ekgan[kept_inds]
    equil_names = all_names[kept_inds]
    equil_ages = all_ages[kept_inds]
    equil_sex = all_sex[kept_inds]
    equil_gen = all_gen[kept_inds]
    equil_target = all_target[kept_inds]
    equil_dist_baseline = all_baseline[kept_inds]
    equil_abbrev[isin_inds(equil_abbrev, ['NSR', 'SB', 'STach', 'SA'])] = 'ctrl'
    '''
    all_dist = all_dist[all_ages > 39]
    all_sex = all_sex[all_ages > 39]
    all_names = all_names[all_ages > 39]
    all_abbrev = all_abbrev[all_ages > 39]
    all_gen = all_gen[all_ages > 39]
    all_target = all_target[all_ages > 39]
    all_ages = all_ages[all_ages > 39]


    equil_abbrev, equil_dist, equil_names, equil_ages, equil_sex = [], [], [], [], []
    equil_gen, equil_target = [], []

    for sex in [0, 1]:

        ages_NSR, counts_NSR = np.unique(all_ages[isin_inds(all_abbrev, ['NSR', 'SB', 'STach', 'SA'])*(all_sex == sex)], return_counts=True)
        weight_MI = Counter(all_ages[(all_abbrev == 'MI')*(all_sex == sex)])
        weight_NSR = np.array([weight_MI[age]/count for (age, count) in zip(ages_NSR, counts_NSR)])
        weight_NSR = (weight_NSR / weight_NSR[weight_NSR>0].min()).round().astype(int)
        inds_NSR = np.concatenate([[ind]*count for ind, count in enumerate(weight_NSR) if count > 0])
        #weight_NSR = (weight_NSR / np.sum(weight_NSR))
        #inds_NSR = np.random.choice(np.arange(len(weight_NSR)), size=sum(weight_MI.values()), replace=True, p=weight_NSR)

        equil_abbrev = np.concatenate([equil_abbrev, equilibrate_ages(all_abbrev, inds_NSR, sex, all_abbrev, all_sex)])
        equil_dist = np.concatenate([equil_dist, equilibrate_ages(all_dist, inds_NSR, sex, all_abbrev, all_sex)])
        equil_names = np.concatenate([equil_names, equilibrate_ages(all_names, inds_NSR, sex, all_abbrev, all_sex)])
        equil_ages = np.concatenate([equil_ages, equilibrate_ages(all_ages, inds_NSR, sex, all_abbrev, all_sex)])
        equil_sex = np.concatenate([equil_sex, equilibrate_ages(all_sex, inds_NSR, sex, all_abbrev, all_sex)])
        if len(equil_gen) == 0:
            equil_gen = equilibrate_ages(all_gen, inds_NSR, sex, all_abbrev, all_sex)
            equil_target = equilibrate_ages(all_target, inds_NSR, sex, all_abbrev, all_sex)
        else:
            equil_gen = np.concatenate([equil_gen, equilibrate_ages(all_gen, inds_NSR, sex, all_abbrev, all_sex)])
            equil_target = np.concatenate([equil_target, equilibrate_ages(all_target, inds_NSR, sex, all_abbrev, all_sex)])

    all_dist = equil_dist
    all_sex = equil_sex
    all_names = equil_names
    all_ages = equil_ages
    all_abbrev = equil_abbrev
    equil_abbrev[isin_inds(equil_abbrev, ['NSR', 'SB', 'STach', 'SA'])] = 'control'
    all_gen = equil_gen
    all_target = equil_target
    '''
    #dic_dist = {f"L2_piste{p+4}": all_dist[:, p] for p in range(all_dist.shape[1])}
    if len(all_dist.shape) > 1:
        df_dist = pd.DataFrame({'Abbreviation': np.concatenate([all_abbrev] * 9), dist_name: np.concatenate(all_dist.T),
                                "ids": np.concatenate([all_names] * 9),
                                "piste": np.concatenate([[k] * 1395 for k in range(9)])})
        hue = 'piste'
    else:
        df_dist = pd.DataFrame({'Abbreviation': equil_abbrev,
                                dist_name: equil_dist,
                                dist_name+'_AAE': equil_dist_baseline,
                                dist_name+'_ekgan': equil_ekgan,
                                "ids": equil_names,
                                'age': equil_ages,
                                'sex': ['F'*(1-sex) + 'M'*sex for sex in equil_sex.astype(int)]})
        hue = 'sex'
        # df_dist[df_dist['age']>39]

    dist_name_save = dist_name
    if qrs_only:
        dist_name_save = dist_name + '_qrs'

    # ======= boxplot ======= #

    fig = plt.figure(figsize=(5, 5))
    PROPS = {
        'boxprops': {'edgecolor': 'k'}, # 'facecolor': 'none', alpha=0.5
        # 'medianprops': {'color': 'k'},
        'whiskerprops': {'color': 'k'},
        'capprops': {'color': 'k'}
    }
    alpha = 0.8
    color_bg = np.array(to_rgb('white'))

    color_NSR = (1-alpha)* color_bg + alpha*np.array(to_rgb('#fa526c'))
    color_MI = (1-alpha)* color_bg + alpha*np.array(to_rgb('#00428d'))
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    color_MI = cmap(0.)
    color_NSR = cmap(1.)

    sns.boxplot(data=df_dist, x='sex', y=dist_name, hue='Abbreviation', palette={"ctrl": tuple(color_NSR), "MI": tuple(color_MI)}, # fliersize=0,
                hue_order=['ctrl', 'MI'], linewidth=2,showfliers=False, order=['M', 'F'], **PROPS) #, 'IRBBB', 'RBBB', 'LBBB'])
    #fig.subplots_adjust(top=.99, left=.07, bottom=.12, right=.99)
    fig.subplots_adjust(top=1., left=0.115, bottom=0., right=1)
    fig.axes[0].set_xlabel('')
    fig.axes[0].set_ylabel('')
    if 'R2_score' in dist_name:
        plt.yticks([-1, 0, 1], ['-1', '0', '1'])
        plt.ylim(-1.2,1.1)
    else:
        plt.yticks([0, 0.1, 0.2], ['0', '0.1', '0.2'])
        plt.ylim(-0.005, 0.205)
    legend = plt.legend()
    legend.remove()

    # plt.legend(loc='upper left')
    # plt.xticks(rotation=15, fontsize=22)
    #plt.ylim(200)
    #p01 = max(0, fig.axes[0].get_ylim()[0])
    #p99 = scoreatpercentile(all_dist, 99.5)
    #plt.ylim(p01, p99)
    #ymin, ymax = int(p01+1), int(p99+1)
    #stride = int(round((ymax-ymin)/5))
    #plt.yticks(np.arange(ymin, ymax, stride).astype(int), fontsize=22)
    # df_dist = df_dist.replace(['OldMI', 'MIs'], 'MI')
    fig.savefig(os.path.join(results_path, f'{setting}_boxplot_{dist_name_save}_precordial_from_augmented.pdf'))
    fig.show()
    plt.close(fig)


    for lab in ['MI', 'LQT', 'LQRSV', 'LAD', 'LAE', 'IRBBB', 'LBBB', 'RBBB']:
        df_tmp = df_dist[df_dist['Abbreviation'].str.contains('ctrl')+df_dist['Abbreviation'].str.contains(lab)]
        fig = plt.figure(figsize=(5, 5))
        lw = 2

        if 'R2_score' in dist_name:
            y_true = [abbr == 'ctrl' for abbr in df_tmp['Abbreviation']]
            #y_true = [abbr == 'ctrl' for abbr in df_dist['Abbreviation']]
        else:
            y_true = [abbr == 'MI' for abbr in df_tmp['Abbreviation']]
            #y_true = [abbr == 'MI' for abbr in df_dist['Abbreviation']]
        #y_score = list(df_dist[dist_name])
        y_score = list(df_tmp[dist_name])
        #y_score_b = list(df_dist[dist_name+'_AAE'])
        y_score_b = list(df_tmp[dist_name+'_AAE'])
        y_score_ekgan = list(df_tmp[dist_name + '_ekgan'])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fpr_b, tpr_b, _ = roc_curve(y_true, y_score_b)
        roc_auc_b = auc(fpr_b, tpr_b)
        fpr_ekgan, tpr_ekgan, _ = roc_curve(y_true, y_score_ekgan)
        roc_auc_ekgan = auc(fpr_ekgan, tpr_ekgan)
        print('LAB', lab)
        print( roc_auc, roc_auc_b, roc_auc_ekgan, len(y_true)-np.sum(y_true))
        plt.plot(
            fpr,
            tpr,
            color='red',
            lw=lw,
            alpha=.7,
            label=f"ours {roc_auc:.2f}",
            #label=f"{roc_auc:.2f}",
        )
        plt.plot(
            fpr_b,
            tpr_b,
            color='gray',
            lw=lw,
            alpha=.7,
            ls='--',
            label=f"AAE {roc_auc_b:.2f}",
            #label=f"{roc_auc:.2f}",
        )
        plt.plot(
            fpr_ekgan,
            tpr_ekgan,
            color='darkblue',
            lw=lw,
            alpha=.7,
            ls='--',
            label=f"ekgan {roc_auc_ekgan:.2f}",
            #label=f"{roc_auc:.2f}",
        )
        '''
        plt.plot(
            fpr_b,
            tpr_b,
            # color=dic_color[sex],
            lw=lw,
            ls='--',
            alpha=0.5,
            # label=f"{sex} AAE ({roc_auc_b:.2f})",
            label=f"AAE ({roc_auc_b:.2f})",
        )
        '''
        plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.025])
        plt.xticks([0.5, 1], [ '0.5', '1'])
        plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
        plt.legend()
        #handles, labels = plt.gca().get_legend_handles_labels()
        #order = [0, 1]#[3, 2, 1, 0]
        #plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        #plt.xlabel("False Positive Rate")
        #plt.ylabel("True Positive Rate")
        #plt.title("Receiver operating characteristic example")
        #plt.legend(loc="lower right")
        fig.subplots_adjust(top=1, left=0.11, bottom=0.07, right=1)
        fig.savefig(os.path.join(results_path, f'{setting}_ROC_{dist_name_save}_{lab}_{roc_auc:.2f}_ALL.pdf'))
        plt.show()
        #plt.close(fig)

    # ======= ROC curve ======= #
    for lab in ['MI', 'LQT', 'LQRSV', 'LAD', 'LAE']:
        df_tmp = df_dist[df_dist['Abbreviation'].str.contains('ctrl')+df_dist['Abbreviation'].str.contains(lab)]
        fig = plt.figure(figsize=(5, 5))
        lw = 2
        dic_color = {'F': '#FF81C0', 'M': 'darkblue'}#'#7BC8F6'}
        for sex in ['F', 'M']:
            if 'R2_score' in dist_name:
                y_true = [abbr == 'ctrl' for abbr in df_tmp[df_tmp['sex'] == sex]['Abbreviation']]
                #y_true = [abbr == 'ctrl' for abbr in df_dist['Abbreviation']]
            else:
                y_true = [abbr == 'MI' for abbr in df_tmp[df_tmp['sex'] == sex]['Abbreviation']]
                #y_true = [abbr == 'MI' for abbr in df_dist['Abbreviation']]
            #y_score = list(df_dist[dist_name])
            y_score = list(df_tmp[df_tmp['sex'] == sex][dist_name])
            #y_score_b = list(df_dist[dist_name+'_AAE'])
            y_score_b = list(df_tmp[df_tmp['sex'] == sex][dist_name+'_AAE'])
            y_score_ekgan = list(df_tmp[df_tmp['sex'] == sex][dist_name+'_AAE'])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            fpr_b, tpr_b, _ = roc_curve(y_true, y_score_b)
            roc_auc_b = auc(fpr_b, tpr_b)
            print('LAB', lab,sex)
            print(sex, roc_auc, roc_auc_b, len(y_true)-np.sum(y_true))
            plt.plot(
                fpr,
                tpr,
                color=dic_color[sex],
                lw=lw,
                label=f"{sex} ({roc_auc:.2f})",
                #label=f"{roc_auc:.2f}",
            )
            '''
            plt.plot(
                fpr_b,
                tpr_b,
                # color=dic_color[sex],
                lw=lw,
                ls='--',
                alpha=0.5,
                # label=f"{sex} AAE ({roc_auc_b:.2f})",
                label=f"AAE ({roc_auc_b:.2f})",
            )
            '''
        plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.025])
        plt.xticks([0.5, 1], [ '0.5', '1'])
        plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
        #plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 1]#[3, 2, 1, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        #plt.xlabel("False Positive Rate")
        #plt.ylabel("True Positive Rate")
        #plt.title("Receiver operating characteristic example")
        #plt.legend(loc="lower right")
        fig.subplots_adjust(top=1, left=0.11, bottom=0.07, right=1)
        fig.savefig(os.path.join(results_path, f'{setting}_ROC_{dist_name_save}_{lab}_{roc_auc:.2f}.pdf'))
        plt.show()
        plt.close(fig)

    # ====== plot ECGs ======= #
    color_posterior = '#00428d'
    color_target = '#fa526c'
    baseline_opt = True

    for name in np.unique(equil_abbrev):
        target_ecg = equil_target[equil_abbrev == name]
        if baseline_opt:
            with torch.no_grad():
                generated_ecg = model_AAE(torch.Tensor(np.swapaxes(target_ecg, 1, 2)).cuda(), None).cpu().numpy()
        else:
            generated_ecg = equil_gen[equil_abbrev == name]

        for pt in  range(10, 100, 10):
            pt_value = float(np.percentile(equil_dist[equil_abbrev == name], q=pt))
            id_ = int(np.argmin(np.abs(equil_dist[equil_abbrev == name] - pt_value)))
            if baseline_opt:
                gen_ecg = generated_ecg[id_][np.newaxis]
            else:
                jnp.swapaxes(generated_ecg[id_],
                             1, 2)
            fig = plot_ecg(ecg_distributions=gen_ecg,
                           conditioning_ecg=target_ecg[id_].T,
                           color_target=color_target,
                           color_posterior=color_posterior)
            #fig.suptitle(f'{name} {dist_name} {dt.iloc[id_median][dist_name]:.2f}')
            if baseline_opt:
                fig.savefig(os.path.join(results_path,
                                         f'{setting}_{name}_BASELINE_{dist_name_save}_precordial_from_augmented_ptg_{pt}_dist.pdf'))
            else:
                fig.savefig(os.path.join(results_path,
                                         f'{setting}_{name}_{dist_name_save}_precordial_from_augmented_ptg_{pt}_dist.pdf'))
            fig.show()
            plt.close()

    print('ok')
    # === analyse tail of ctrl-F distribution === #
    bp = plt.boxplot(
        np.array(df_dist[(df_dist['sex'] == 'M') & (df_dist['Abbreviation'] == 'ctrl')]['mahanalobis_mean']),
        widths=0.6, patch_artist=True)
    # bp is a dict, we can extract the values of boxplot caps/whiskers
    plt.close()


if __name__ == '__main__':
    main()
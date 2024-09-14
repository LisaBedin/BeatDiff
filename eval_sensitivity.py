import os
import glob
import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import mahalanobis
from scipy import signal
from dtaidistance import dtw, preprocessing


# choose a disease and compute distances with
def plot_ecg(ecg_distributions, conditioning_ecg = None):
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(bottom=.05, top=.92, right=.99, left=.1)
    for ecg in ecg_distributions:
        for i, track in enumerate(ecg):
            ax.plot(track - i*1.5, color='blue', alpha=.1,  rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.5, color='red', alpha=.5,  rasterized=True)
    ax.set_ylim(-13.5, 1.5)
    ax.set_yticks([-i*1.5 for i in range(9)])
    ax.set_yticklabels(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'))
    return fig


def read_ecg(file_path_lst):
    target_ecg_lst, generated_ecg_lst = [], []
    for file_path in file_path_lst:
        npz = np.load(file_path)
        target_ecg_lst.append(npz['target_ecg'])
        generated_ecg_lst.append(npz['generated_ecg'])
    return target_ecg_lst[0][:10], np.concatenate(generated_ecg_lst, axis=1)[:10]


def smooth_ecg(ecgs, axis=1, smooth=0.1):
    fs = 100  # sample rate, Hz
    cutoff = fs * smooth  # cut off frequency, Hz
    nyq = 0.5 * fs  # Nyquist frequency
    b, a = signal.butter(2, cutoff / nyq, btype='low', analog=False, output='ba')
    return signal.filtfilt(b, a, ecgs, axis=axis)


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    results_path = '/mnt/data/lisa/ecg_results/evaluation/generated_ecg/nb_particles_exp1'
    part16 = [os.path.join(results_path, '2023-09-14/13-27-46/16_0_0/NSR.npz')]
    part64 = glob.glob(os.path.join(results_path, '2023-09-15/15-17-49/64_0_*/NSR.npz'))
    part256 = glob.glob(os.path.join(results_path, '2023-09-15/15-39-36/256_0_*/NSR.npz'))
    part1024 = glob.glob(os.path.join(results_path, '2023-09-15/16-26-07/1024_0_*/NSR.npz'))

    figure_path = os.path.join(results_path, 'eval_sensitivity')
    os.makedirs(figure_path, exist_ok=True)
    all_files = [part16, part64, part256, part1024]
    nb_particles = [16, 64, 256, 1024]

    qrs_only, pre_smooth = True, False

    # ============= single patient ============== #
    dist_name_lst = ['MedianAD', 'MaxAD', 'MeanAD']
    for dist_name in dist_name_lst:
        all_dist, all_abbrev, all_nb_part, all_pistes = [], [], [], []
        for nb_part, file_path in zip(nb_particles, all_files):
            target_ecg, generated_ecg = read_ecg(file_path)
            _, N_samples, T, N_pistes = generated_ecg.shape
            pistes = np.tile(np.arange(N_pistes)[None], (N_samples, 1)).flatten()
            # ids = np.tile(np.arange(N_samples)[:, None], (1, N_pistes)).flatten()
            if qrs_only:
                generated_ecg = generated_ecg[0, :, 40:70]
                target_ecg = target_ecg[0, 40:70]
            if pre_smooth:
                generated_ecg = smooth_ecg(generated_ecg, axis=2)
                target_ecg = smooth_ecg(target_ecg, axis=1)
            if 'MedianAD' in dist_name:
                center = np.median(generated_ecg, axis=0)[None]
                dist = np.median(np.abs(generated_ecg-center), axis=1).flatten()
            if 'MeanAD' in dist_name:
                center = np.mean(generated_ecg, axis=0)[None]
                dist = np.mean(np.abs(generated_ecg-center), axis=1).flatten()
            if 'MaxAD' in dist_name:
                center = np.mean(generated_ecg, axis=0)[None]
                dist = np.max(np.abs(generated_ecg-center), axis=1).flatten()
            all_dist += list(dist)
            all_pistes += list(pistes)
            file_name = file_path[0].split('/')[-1].split('.')[0]
            all_abbrev += [file_name]*len(dist)
            all_nb_part += [nb_part]*len(dist)
        suffix = 'Qrs'*qrs_only+'All'*(1-qrs_only)+'Smooth'*pre_smooth+'Raw'*(1-pre_smooth)
        save_path = os.path.join(figure_path, os.path.join(dist_name, suffix))
        os.makedirs(save_path, exist_ok=True)

        df_deviation = pd.DataFrame({'Abbreviation': all_abbrev,
                                     dist_name: all_dist,
                                     'nb_part': all_nb_part,
                                     "piste": all_pistes})
        fig = plt.figure(figsize=(7, 7))
        sns.boxplot(data=df_deviation[df_deviation['piste'] > 2], x='piste', hue='nb_part', y=dist_name)
        # sns.swarmplot(data=df_dist, x='nb_part', y=dist_name)
        # fig.subplots_adjust(top=.99, left=.04, bottom=.05, right=.99)
        fig.subplots_adjust(top=.99, left=.1, bottom=.1, right=.99)
        fig.axes[0].set_xlabel('piste')
        fig.axes[0].set_ylabel(dist_name)
        fig.savefig(os.path.join(save_path, dist_name + '.png'))
        fig.show()
        print(dist_name)

    # ============= analyzing 10 patients ============== #
    dist_name_lst = ['cov']#, 'cov', 'dtw_dist', 'L2', 'mahanalobis']

    # dist_name = 'dtw_cov'#'cov'  # 'L2'  # 'mahanalobis_min'  # 'mahanalobis_qrs'
    for dist_name in dist_name_lst:
        all_names, all_dist, all_abbrev, all_nb_part, all_pistes = [], [], [], [], []
        for nb_part, file_path in zip(nb_particles, all_files):
            #npz = np.load(file_path)
            target_ecg, generated_ecg = read_ecg(file_path)
            #generated_ecg = generated_ecg[..., 3:]#[5:6]
            #target_ecg = target_ecg[..., 3:]#[5:6]
            if qrs_only:
                generated_ecg = generated_ecg[:, :, 40:70]
                target_ecg = target_ecg[:, 40:70]
            if pre_smooth:
                generated_ecg = smooth_ecg(generated_ecg, axis=2)
                target_ecg = smooth_ecg(target_ecg, axis=1)
            n_data, T, P = target_ecg.shape
            piste_name = np.concatenate([np.arange(P)]*n_data)
            if 'mahanalobis' in dist_name:
                means_qrs = generated_ecg.mean(axis=1)
                prec_qrs = np.stack([
                    [np.linalg.pinv(np.corrcoef(qrs.T))
                     for qrs in np.swapaxes(all_tracks_qrs, 0, 1)]
                    for all_tracks_qrs in np.swapaxes(generated_ecg, -1, -2)
                ])
                dist = np.concatenate([[mahalanobis(*i) for i in zip(*it)]
                        for it in zip(np.swapaxes(target_ecg, 1, 2),
                                      np.swapaxes(means_qrs, 1, 2),
                                      prec_qrs)])
            elif 'L2' in dist_name:
                L2_distance = np.sqrt(np.sum((generated_ecg - target_ecg[:, np.newaxis]) ** 2, axis=2)).std(axis=1)
                dist = np.concatenate(L2_distance)
            elif 'dtw_pairwise' in dist_name:
                if 'diff' in dist_name:
                    smooth = None
                    if 'smooth' in dist_name:
                        smooth = 0.1
                    dist = np.concatenate([[
                        dtw.distance_matrix_fast(preprocessing.differencing(generated_ecg[i, :, :, j].astype(np.float64), smooth=smooth)).mean()
                        for j in range(P)
                    ] for i in range(n_data)])
                else:
                    dist = np.concatenate([[
                        dtw.distance_matrix_fast(generated_ecg[i, :, :, j].astype(np.float64)).mean()
                        for j in range(P)
                    ] for i in range(n_data)])
            elif 'dtw_dist' in dist_name:
                if 'diff' in dist_name:
                    smooth = None
                    if 'smooth' in dist_name:
                        smooth = 0.1
                    dist = np.concatenate([[
                        np.min(dtw.distance_matrix_fast(preprocessing.differencing(
                            np.concatenate([target_ecg[i:i + 1, :, j], generated_ecg[i, :, :, j]]).astype(np.float64),
                            smooth=smooth),
                                                        block=((1, 2), (1, generated_ecg.shape[1] + 1)), compact=True))
                        for j in range(P)
                    ] for i in range(n_data)])
                else:
                    dist = np.concatenate([[
                        np.min(dtw.distance_matrix_fast(np.concatenate([target_ecg[i:i+1, :, j],
                                                                 generated_ecg[i, :, :, j]]).astype(np.float64),
                                                 block=((1,2), (1,generated_ecg.shape[1]+1)), compact=True))
                        for j in range(P)
                    ] for i in range(n_data)])
            else:
                prec_qrs = np.stack([
                    [np.corrcoef(qrs.T)
                     for qrs in np.swapaxes(all_tracks_qrs, 0, 1)]
                    for all_tracks_qrs in np.swapaxes(generated_ecg, -1, -2)
                ]).mean(axis=(2, 3))
                dist = np.concatenate(prec_qrs)
            all_dist += list(dist)
            all_pistes += list(piste_name)
            file_name = file_path[0].split('/')[-1].split('.')[0]
            #n_patients = generated_ecg.shape[0]
            all_names += [file_name + f'_{i}' for i in range(len(dist))]
            all_abbrev += [file_name]*len(dist)
            all_nb_part += [nb_part]*len(dist)
        #all_dist = np.concatenate(all_dist)
        df_dist = pd.DataFrame({'Abbreviation': all_abbrev,
                                dist_name: all_dist,
                                "ids": all_names,
                                'nb_part': all_nb_part,
                                "piste": all_pistes})

        suffix = 'Qrs'*qrs_only+'All'*(1-qrs_only)+'Smooth'*pre_smooth+'Raw'*(1-pre_smooth)
        save_path = os.path.join(figure_path, os.path.join(dist_name, suffix))
        os.makedirs(save_path, exist_ok=True)

        fig = plt.figure(figsize=(7, 7))
        sns.boxplot(data=df_dist[df_dist['piste']>2], x='piste', hue='nb_part', y=dist_name)
        # sns.swarmplot(data=df_dist, x='nb_part', y=dist_name)
        #fig.subplots_adjust(top=.99, left=.04, bottom=.05, right=.99)
        fig.subplots_adjust(top=.99, left=.1, bottom=.1, right=.99)
        fig.axes[0].set_xlabel('piste')
        fig.axes[0].set_ylabel(dist_name)
        fig.savefig(os.path.join(save_path, dist_name + '.png'))
        fig.show()


if __name__ == '__main__':
    main()
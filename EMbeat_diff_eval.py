import os
os.environ["OMP_NUM_THREADS"] = "30"
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn import cluster
from collections.abc import Callable
from typing import Tuple, List, Union


# ===== plot function ===== #
def augmented_leads(beats: np.ndarray) -> np.ndarray:
    avr = -(beats[..., 0] + beats[..., 2]) / 2
    avl = (beats[..., 0] - beats[..., 2])/2
    avf = (beats[..., 1] + beats[..., 2]) / 2
    beats = np.concatenate([beats[..., :3], np.stack([avr, avl, avf], axis=-1), beats[..., 3:]], axis=-1)
    return beats


def plot_12leads(target_beats: np.ndarray, pred_beats: np.ndarray, target_c: str = 'r', pred_c: str = 'b') -> matplotlib.figure.Figure:
    target_beats = augmented_leads(target_beats)
    pred_beats = augmented_leads(pred_beats)

    max_H = 3.1  # cm = mV
    offset_H = 1.5
    margin_H = 0.
    margin_W = 0.2

    T = target_beats.shape[0]*4*2.5*0.001  # (*4 ms * 25 mm)
    times = np.linspace(0, T, target_beats.shape[0])

    H = max_H*3 + margin_H*4  # in cm
    W = T*4 + margin_W*5  # in cm

    fig, ax = plt.subplots(figsize=(W/2.54, H/2.54))  # 1/2.54  # centimeters in inches
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

    # grid colors
    color_major, color_minor, color_line = (1, 0, 0), (1, 0.7, 0.7), (0, 0, 0.7)

    # Ticks configuration
    ax.set_xticks(np.arange(0, W/2.54, 1/2.54))  # Major vertical ticks
    ax.set_yticks(np.arange(0, H/2.54, 1/2.54))  # Major horizontal ticks
    ax.minorticks_on()  # Activate major ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 minor ticks per major ticks
    ax.grid(which='major', linestyle='-', linewidth=0.4, color='k', alpha=0.5)
    ax.grid(which='minor', linestyle='-', linewidth=0.4, color='k', alpha=0.2)

    ax.set_xticklabels([])  # Suppress labels on X axis
    ax.set_yticklabels([])  # Suppress labels on Y axis
    ax.set_ylim(0, H/2.54)
    ax.set_xlim(0, W/2.54)
    ind_W = [1]*3 + [2]*3 + [3]*3 + [4]*3
    ind_H = np.concatenate([np.arange(3, 0, -1)]*4)

    for ecg_l, lW, lH in zip(target_beats.T, ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c=target_c, linewidth=.7,
                rasterized=True)

    if len(pred_beats.shape) == 2:
        for ecg_l, lW, lH in zip(pred_beats.T, ind_W, ind_H):
            ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                    c=pred_c, linewidth=.7,
                    rasterized=True)
    else:
        for pred_b in pred_beats:
            for ecg_l, lW, lH in zip(pred_b.T, ind_W, ind_H):
                ax.plot(((lW - 1) * T + lW * margin_W + times) / 2.54,
                        (ecg_l + offset_H + max_H * (lH - 1) + lH * margin_H) / 2.54,
                        c=pred_c, linewidth=.7, alpha=0.4, rasterized=True)
    return fig

# ===== evaluation metrics ===== #
def SSD(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # return ((x - y) ** 2).sum(axis=(1, 2))
    return ((x - y) ** 2).sum(axis=1).mean(axis=1)


def R2_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    squared_dist = ((x - y)**2).sum(axis=1)
    scaling = ((y - np.mean(y, axis=1)[:, np.newaxis])**2).sum(axis=1)
    return (1 - squared_dist / scaling).mean(axis=1)


def MASE(x, y):  # problems when y is not equal to zero...
    eps = 1e-4
    return np.mean(np.abs(x - y) / np.maximum(eps, np.mean(y[:, 1:]-y[:, :-1], axis=1)[:, np.newaxis]), axis=(1, 2))


def MAPE(x, y):
    eps = 1e-4
    return np.mean(np.abs(x - y) / np.maximum(eps, y), axis=(1, 2))


def wMAPE(x, y):
    eps = 1e-4
    return np.mean(np.abs(x - y) / np.maximum(eps, y.sum(axis=1)[:, np.newaxis]), axis=(1, 2))


def sMAPE(x, y):
    eps = 1e-4
    return np.mean(np.abs(x - y) / np.maximum(eps, (x+y)/2), axis=(1, 2))



def MSE(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # return ((x - y) ** 2).sum(axis=(1, 2))
    return np.sqrt(((x - y) ** 2).sum(axis=1)).mean(axis=1)


def MAD(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # return np.absolute(x - y).max(axis=(1, 2))
    return np.absolute(x - y).max(axis=1).mean(axis=1)


def CosDist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x_norm, y_norm = np.linalg.norm(x, axis=(1, 2)), np.linalg.norm(y, axis=(1, 2))
    # return (x * y).sum(axis=(1, 2)) / x_norm / y_norm
    x_norm, y_norm = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    return ((x * y).sum(axis=1) / x_norm / y_norm).mean(axis=1)


def get_principal_mode(posterior_samples: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    # spectral = cluster.SpectralClustering(
    #     n_clusters=n_clusters,
    #     affinity='nearest_neighbors',
    #     random_state=0,
    # )
    # # normalized_post = posterior_samples / np.sqrt((posterior_samples**2).mean(axis=1)[:, np.newaxis])
    # spectral.fit(posterior_samples.reshape(posterior_samples.shape[0], -1))
    # cluster_labels = spectral.labels_
    #
    # clustered_samples = np.stack([posterior_samples[cluster_labels == lab].mean(axis=0) for lab in range(n_clusters)])
    # lab_names, lab_count = np.unique(cluster_labels, return_counts=True)
    #
    # principal_mode = clustered_samples[lab_names[lab_count.argmax()]]
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(posterior_samples.reshape(posterior_samples.shape[0], -1))
    clustered_samples = kmeans.cluster_centers_.reshape(n_clusters, *posterior_samples.shape[1:])
    cluster_labels = kmeans.labels_
    lab_names, lab_count = np.unique(cluster_labels, return_counts=True)
    principal_mode = clustered_samples[lab_names[lab_count.argmax()]]
    return principal_mode, clustered_samples


def get_scores(pred_samples: np.ndarray,
               ground_truth: np.ndarray,
               metrics_fn: List[Callable],
               mask_times: np.ndarray,
               mask_leads: np.ndarray,
               n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    clustered_samples = None
    if len(pred_samples.shape) == 4:
        res = [get_principal_mode(pred_samples[k], n_clusters)
                                                    for k in tqdm(range(pred_samples.shape[0]),
                                                                  desc='compute principal mode')]
        pred_samples = np.stack([r[0] for r in res])
        clustered_samples = np.stack([r[1] for r in res])

    all_scores = []
    for dist_fn_ in metrics_fn:
        all_scores.append(dist_fn_(pred_samples[:, mask_times][..., mask_leads],
                                   ground_truth[:, mask_times][..., mask_leads]))
    all_scores = np.stack(all_scores)
    return pred_samples, clustered_samples, all_scores


def update_dic(dic_ci: dict, scores: np.ndarray, metric_names: List[str], method_name: str) -> dict:
    n_data = scores.shape[1]
    for k, metric_n in enumerate(metric_names):
        dic_ci[metric_n].append(f'{np.mean(scores[k]):.3f} +/- {1.96*np.std(scores[k])/n_data:.3f}')
    dic_ci['method'].append(method_name)
    return dic_ci


def eval_inpainting(cfg: DictConfig, start_lead: int = 1):
    npz = np.load(os.path.join(cfg.results_path, cfg.eval.folder_name, f"NSR_{start_lead}.npz"))
    metric_names = ['R2_score', 'MSE', 'SSD', 'MAD', 'CosDist']
    dic_ci = {k: [] for k in ['method'] + metric_names}
    principal_modes, modes, scores = get_scores(npz['posterior_samples'],
                                                npz['ground_truth'],
                                                [R2_score, MSE, SSD, MAD, CosDist],
                                                mask_times=np.arange(176),
                                                mask_leads=np.arange(start_lead, 9),
                                                n_clusters=10)
    dic_ci = update_dic(dic_ci, scores, metric_names, 'EMbeat_diff_10')


    _, _, scores = get_scores(npz['posterior_samples'].mean(axis=1),
                              npz['ground_truth'],
                              [R2_score, MSE, SSD, MAD, CosDist],
                              mask_times=np.arange(176),
                              mask_leads=np.arange(start_lead, 9),
                              n_clusters=10)
    dic_ci = update_dic(dic_ci, scores, metric_names, 'EMbeat_diff')

    _, _, scores = get_scores(npz['ekgan'],
                               npz['ground_truth'],
                               [R2_score, MSE, SSD, MAD, CosDist],
                               mask_times=np.arange(176),
                               mask_leads=np.arange(start_lead, 9),
                               n_clusters=10)

    dic_ci = update_dic(dic_ci, scores, metric_names, 'ekgan')

    # === plot figs === #


    return dic_ci


@hydra.main(version_base=None, config_path="configs/paths", config_name="beat_no_norm")
def main(cfg: DictConfig):
    # ===== Denoising ===== #

    # ===== Inpainting ===== #
    npz = np.load(os.path.join(cfg.results_path, cfg.eval.folder_name, "NSR_1.npz"))
    npz_limbs = np.load(os.path.join(cfg.results_path, cfg.eval.folder_name, "NSR_3.npz"))

    # ===== Anomaly Detection ===== #

    return


if __name__ == "__main__":
    main()


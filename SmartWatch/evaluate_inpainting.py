import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,4,3,2,1'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import em_variance_only, mcg_diff_ve
import hydra
from matplotlib.ticker import AutoMinorLocator

from omegaconf import DictConfig, OmegaConf
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from jax.tree_util import Partial as partial
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

#from mcg_diff_inpainting import apply_ekgan  # careful if CUDA_VISIBLE_DEVICES configured in this script !
from reconstruct_from_watch import apply_ekgan

def plot_ecg(ecg_distributions, conditioning_ecg=None, alpha=0.05, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if ecg_distributions is not None:
        if len(ecg_distributions.shape) == 3:
            for ecg in ecg_distributions:
                for i, track in enumerate(ecg):
                    ax.plot(track - i*1.3, c=color_posterior, alpha=alpha, linewidth=.7, rasterized=True)
        else:
            for i, track in enumerate(ecg_distributions):
                ax.plot(track - i * 1.3, c=color_posterior, linewidth=.7, rasterized=True)
    if conditioning_ecg is not None:
        for i, track in enumerate(conditioning_ecg):
            ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    return fig

def plot_12leads(ecg_9leads, color='k'):
    aVR = -(ecg_9leads[0]+ecg_9leads[2])/2
    aVL = (ecg_9leads[0]-ecg_9leads[2])/2
    aVF = (ecg_9leads[1] + ecg_9leads[2]) / 2
    ecg = np.concatenate([ecg_9leads[:3], np.stack([aVR, aVL, aVF]), ecg_9leads[3:]])

    max_H = 3.1  # cm = mV
    offset_H = 1.5
    margin_H = 0.
    margin_W = 0.2

    T = ecg.shape[1]*4*2.5*0.001  # (*4 ms * 25 mm)
    times = np.linspace(0, T, ecg.shape[1])

    H = max_H*3 + margin_H*4  # in cm
    W = T*4 + margin_W*5  # in cm

    fig, ax = plt.subplots(figsize=(W/2.54, H/2.54))  # 1/2.54  # centimeters in inches
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)


    # Couleurs des grilles
    color_major, color_minor, color_line = (1, 0, 0), (1, 0.7, 0.7), (0, 0, 0.7)

    # Configuration des ticks majeurs et mineurs
    ax.set_xticks(np.arange(0, W/2.54, 1/2.54))  # Ticks majeurs vertical
    ax.set_yticks(np.arange(0, H/2.54, 1/2.54))  # Ticks majeurs horizontal
    ax.minorticks_on()  # Activer les ticks mineurs
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 ticks mineurs par tick majeur
    ax.grid(which='major', linestyle='-', linewidth=0.4, color='k', alpha=0.5)
    ax.grid(which='minor', linestyle='-', linewidth=0.4, color='k', alpha=0.2)



    ax.set_xticklabels([])  # Supprimer les labels des ticks sur l'axe X
    ax.set_yticklabels([])  # Supprimer les labels des ticks sur l'axe Y
    ax.set_ylim(0, H/2.54)
    ax.set_xlim(0, W/2.54)
    ind_W = [1]*3 + [2]*3 + [3]*3 + [4]*3
    ind_H = np.concatenate([np.arange(3, 0, -1)]*4)

    for ecg_l, lW, lH in zip(ecg, ind_W, ind_H):
        ax.plot(((lW-1)*T+lW*margin_W + times) / 2.54, (ecg_l + offset_H + max_H*(lH-1) + lH*margin_H) / 2.54,
                c=color, linewidth=.7,
                rasterized=True)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    print(torch.cuda._parse_visible_devices())

    results_path = os.path.join(cfg.main_path, 'results', cfg.evaluate_inpainting)
    npz = np.load(os.path.join(results_path, 'inpainting_from_watch.npz'))

    ref_samples = npz['posterior_samples']
    target_ecg = npz['target_ecg']
    input_ecg = npz['input_ecg']
    ekgan_ecg = npz['ekgan']
    feats = npz['feats']

    age = (feats[:, -1] * 50 + 50).astype(int)
    # sex = ['M'*int(f[0]) + 'F'*int(f[1]) for f in feats[:, :2]]
    sex = [int(a>=22)*'F' + int(a<22)*'M' for a in age]

    count_labs = {'M21': 0, 'F23': 0, 'F24': 0, 'F28': 0}
    error_labs = {'M21': 0., 'F23': 0., 'F24': 0., 'F28': 0.}
    ekgan_error_labs = {'M21': 0., 'F23': 0., 'F24': 0., 'F28': 0.}

    '''
    ekgan_model_I = torch.load(os.path.join(cfg.baseline_folder,
                                            'Ekgan_per_lead_norm_0_1_2/best_inference_generator.pth')).cuda()

    X = nn.Upsample(256, mode='linear')(torch.tensor(np.swapaxes(npz['input_ecg'], 1, 2)).to(torch.float32))
    with torch.no_grad():
        ecg_recon_ekgan = apply_ekgan(X, ekgan_model_I, np.array([0,1, 2]), 'cuda:0')
    npz_dic = dict(npz)
    npz_dic['ekgan'] = ecg_recon_ekgan
    '''

    if '0_1_2' in cfg.evaluate_inpainting:
        leads_lst = np.array([0, 1, 2])
    elif '0_1' in cfg.evaluate_inpainting:
        leads_lst = np.array([0, 1])
    elif '0_2' in cfg.evaluate_inpainting:
        leads_lst = np.array([0, 2])
    else:
        leads_lst = np.array([0])
    # === clustering to identify modes === #
    for k in range(ref_samples.shape[0]):
        lab = sex[k] + str(age[k])
        print(lab)
        count_labs[lab] += 1
        n_clusters = 10
        spectral = cluster.SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            random_state=0,
        )
        spectral.fit(ref_samples[k].reshape(100, -1))
        cluster_labels = spectral.labels_
        clustered_samples = np.stack([ref_samples[k][cluster_labels == lab].mean(axis=0) for lab in range(n_clusters)])
        # clustered_samples /= np.absolute(clustered_samples).max(axis=1)[:, np.newaxis]
        cluster_dist = np.sqrt(((clustered_samples - target_ecg[k:k+1])**2).sum(axis=1)).mean(axis=-1)
        lab_names, lab_count = np.unique(cluster_labels, return_counts=True)
        error_labs[lab] += (cluster_dist*lab_count).sum()/lab_count.sum()

        ekgan_k = ekgan_ecg[k] #  / np.absolute(ekgan_ecg[k]).max(axis=0)[np.newaxis]
        ekgan_error_labs[lab] += ((ekgan_k-target_ecg[k])**2).sum(axis=0).mean(axis=-1)
        # TODO: save figure of input ECG
        if count_labs[lab] == 5:
            print(lab, count_labs[lab])

            fig = plot_ecg(None, input_ecg[k][:, leads_lst].T, alpha=.5)
            plt.savefig(os.path.join(results_path, lab + '_input.png'))
            plt.show()

            fig = plot_ecg(np.swapaxes(clustered_samples, 1, 2), target_ecg[k].T, alpha=.5)
            plt.savefig(os.path.join(results_path, lab + '_clustered.png'))
            plt.show()
            print(lab_count)
            # print(lab_count[cluster_dist.argmin()])
            fig = plot_ecg(clustered_samples[lab_names[lab_count.argmax()]].T, target_ecg[k].T, alpha=.5)
            # fig = plot_ecg(clustered_samples[cluster_dist.argmin()].T, target_ecg[k].T, alpha=.5)
            plt.savefig(os.path.join(results_path, lab + '_majority.png'))
            plt.show()
            fig = plot_ecg(ekgan_k.T, target_ecg[k].T, alpha=.5)
            plt.savefig(os.path.join(results_path, lab + '_ekgan.png'))
            plt.show()

    error_labs = {k: val/ref_samples.shape[0] for k, val in error_labs.items()}
    ekgan_error_labs = {k: val/ref_samples.shape[0] for k, val in ekgan_error_labs.items()}
    print(error_labs)
    print(ekgan_error_labs)
    print('ok')

if __name__ == '__main__':
    main()

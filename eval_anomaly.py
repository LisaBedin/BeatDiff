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
from sklearn.model_selection import KFold
from scipy.spatial.distance import mahalanobis
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit, pmap, devices
from scipy.stats.distributions import chi2
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
import torch
from collections import Counter
def plot_ecg_old(missing_lead, ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
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

def get_R2_score(real_signal, pred_signal, agg_fn):
    signal_energy = np.sum((real_signal - real_signal.mean(axis=1)[:, np.newaxis]) ** 2, axis=1)
    error_mcg_diff = np.sum((pred_signal - real_signal) ** 2, axis=1)
    return agg_fn(1 - (error_mcg_diff / signal_energy), axis=-1)


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

agg_functions = {'mean': np.mean, 'median': np.median}

baseline_leads = {
   #  'ekgan_I': np.arange(1, 9, dtype=int),
   # 'ekgan_limb': np.arange(3, 9, dtype=int),
    'AAE': np.arange(9, dtype=int),
    #'dowers_reconstruction': np.arange(3, 9, dtype=int)
}

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    save_path = os.path.join(cfg.results_path, 'inpainting_samples_20')  # f'inpainting_samples_{cfg.inpainting_samples_K20.T}')
    df = {'scores': [], 'labels': [], 'anomaly': []}

    AAE_model = torch.load(os.path.join(cfg.baseline_folder, 'GLOBAL_AE_disc_TCN/best_model.pth')).cuda()
    AAE_model.eval()
    AAE_model.netD.TCN_opt = True

    for anomaly_label in ['MI', 'MI_qrs', 'MI_ST', 'LAD', 'LAD_qrs', 'LAD_ST', 'LAE', 'LAE_qrs', 'LAE_ST', 'LQT', 'LQT_qrs', 'LQT_ST']: # cfg.eval.labels: # [:3]:
        print(' ')
        qrs_only = 'qrs' in anomaly_label
        st_only = '_ST' in anomaly_label
        first_lead = '_1' in anomaly_label
        if 'LQTS_clean' not in anomaly_label:
            npz0 = np.load(os.path.join(save_path, anomaly_label+'.npz'))
            qrs_suffix = qrs_only*'_qrs' + st_only*'_ST' + first_lead*'_1' + ''
            npz1 = np.load(os.path.join(save_path, f'safe_NSR/NSR_SB_STach_SA{qrs_suffix}.npz'))
            all_data = {file_n: np.concatenate([npz0[file_n], npz1[file_n]]) for file_n in npz0.files}
            labels = np.array([0]*npz0['ground_truth'].shape[0]+[1]*npz1['ground_truth'].shape[0])
            all_data['label'] = labels
        else:
            all_data = dict(np.load(os.path.join(save_path, anomaly_label+'.npz')))
            #gt_clean = np.load(os.path.join(save_path, 'LQTS_clean_allLeads.npz'))['ground_truth']
            all_data['label'] = 1-all_data['genetic_annotation']
        for method in ['posterior_samples', 'AAE']: # , 'ekgan_I', 'ekgan_limb', 'AAE', 'dowers_reconstruction']:
            gt = all_data['ground_truth']
            preds = all_data[method]
            labels = all_data['label']
            if method == 'posterior_samples':
                # TODO for confidence intervals
                kfold_splits = KFold(n_splits=10, shuffle=True, random_state=0)
                #if 'LQTS_clean' in anomaly_label:
                #    gt  = np.load(os.path.join(save_path, 'LQTS_clean_allLeads.npz'))['ground_truth']
                if qrs_only:
                    gt, preds = gt[:, 70:], preds[:, :, 70:]
                else:
                    gt, preds = gt[:, :, 3:], preds[:,:, :, 3:]
                all_scores = [get_R2_score(gt, preds[:, inds].mean(axis=1), agg_functions[cfg.eval.agg_fun])
                          for inds, _ in tqdm(kfold_splits.split(np.arange(preds.shape[1])),
                                              total=kfold_splits.n_splits)]
                all_roc_curves = [np.stack(roc_curve(labels, scores)[:2]) for scores in tqdm(all_scores, total=len(all_scores))]
                all_roc_auc = [auc(fpr, tpr) for fpr, tpr in tqdm(all_roc_curves, total=len(all_roc_curves))]
                #weights = Counter(labels)
                #sample_weight = [len(labels) / weights[lab] for lab in labels]
                #all_precision = [average_precision_score(1-labels, 1-scores, sample_weight=sample_weight) for scores in tqdm(all_scores, total=len(all_scores))]


                print(f'AUC {anomaly_label}-{method}: {np.mean(all_roc_auc)*100:.2f}+/-{100*1.96*np.std(all_roc_auc)/len(all_roc_auc)**0.5:.2f}')
                #print(f'Precision {anomaly_label}-{method}: {np.mean(all_precision)*100:.2f}+/-{100*1.96*np.std(all_precision)/len(all_roc_auc)**0.5:.2f}')

                scores = np.mean(np.stack(all_scores), axis=0)
                scores_CI = np.std(np.stack(all_scores), axis=0)
                df['labels'].append(labels)
                df['scores'].append(scores)
                # df['CI'].append(np.mean(all_roc_auc))
                df['anomaly'].append(np.array([anomaly_label]*len(scores)))

            else:
                #with torch.no_grad():
                #    preds = AAE_model.cuda()(torch.Tensor(np.swapaxes(gt, 1, 2)).cuda(), None).detach().cpu()
                #    preds = np.swapaxes(preds.numpy(), 1, 2)
                gt, preds = gt[:, :, baseline_leads[method]], preds[:, :, baseline_leads[method]]
                scores = get_R2_score(gt, preds, agg_functions[cfg.eval.agg_fun])
                fpr, tpr, _ = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                #weights = Counter(labels)
                #sample_weight = [len(labels) / weights[lab] for lab in labels]
                #precision_avg = average_precision_score(1-labels, 1-scores,  sample_weight=sample_weight)
                print(f'AUC {anomaly_label}-{method}: {100*roc_auc:.2f}')
                #print(f'Precision {anomaly_label}-{method}: {100*precision_avg:.2f}')


    for label, sample_id in zip(['MI', 'LQT_qrs'], [18, 0]): # LQT_qrs 0
        npz_file = np.load(os.path.join(save_path, label+'.npz'))
        conditioning_ecg = np.swapaxes(npz_file['ground_truth'], 1, 2)  # [:, :1]

        if 'clean' in label:
            genetic_annotation = npz_file['genetic_annotation']
            inds = np.where(genetic_annotation==1)[0]
        else:
            inds = np.arange(conditioning_ecg.shape[0])
        conditioning_ecg = conditioning_ecg[inds]
        normalization_ecg = np.absolute(conditioning_ecg).max(axis=-1)[:, np.newaxis, :, np.newaxis]
        conditioning_ecg /= normalization_ecg[:, 0]
        ecg_distributions = np.swapaxes(npz_file['posterior_samples'][inds], 2, 3) #/ normalization_ecg
        ecg_distributions /= normalization_ecg #  np.abs(ecg_distributions).max(axis=3)[:, :, :, np.newaxis]
        ecg_distributions_AAE = np.swapaxes(npz_file['AAE'][inds], 1, 2) / normalization_ecg[:, 0]

        fig = plot_ecg(ecg_distributions[sample_id].mean(axis=0)[np.newaxis], conditioning_ecg[sample_id])
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/{label}_em_beatdiff_posterior.pdf'))
        plt.show()

        fig = plot_ecg(ecg_distributions_AAE[sample_id][np.newaxis], conditioning_ecg[sample_id])
        fig.savefig(os.path.join(cfg.results_path, f'some_figures/{label}_AAE.pdf'))
        plt.show()

if __name__ == '__main__':
    main()
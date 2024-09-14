import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import hydra
import matplotlib.pyplot as plt
import neurokit2 as nk
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch

def ci_intervals(data):
    return f'{np.mean(data):.3f} +/- {((1.96*np.std(data))/(len(data)**.5)):.3f}'

def get_R2_score(real_signal, generated_ecg):
    signal_energy = np.sum((real_signal - real_signal.mean(axis=1)[:, np.newaxis]) ** 2, axis=1)
    error = np.sum((generated_ecg - real_signal) ** 2, axis=1)

    r2score = 1 - (error / signal_energy)
    return r2score #.mean(axis=-1)  # averaging over the leads

def get_absolute_error(real_stds, pred_stds):
    return np.absolute(pred_stds-real_stds)#.mean(axis=-1)
def plot_ecg(ecg_distributions, conditioning_ecg = None, color_posterior='blue', color_target='red'):
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
    # data = np.load(os.path.join(cfg.checkpoint, 'denoising.npy.npz'))
    results_path = os.path.join(cfg.results_path, 'denoising_eval')
    data = np.load(os.path.join(results_path, 'denoising_diffusion.npz'))
    std_ptg_error = data.get('std_ptg_error')
    mahanalobis_outlier_score = data.get('mahanalobis_outlier_score')
    print(ci_intervals(std_ptg_error.flatten()), ci_intervals(mahanalobis_outlier_score.flatten()))

    model_AAE = torch.load(cfg.results_path + 'baseline_AE/GOOD_AE_disc_TCN/best_model.pth').cuda()
    model_AAE.eval()
    model_AAE.netD.conditional = False

    AAE_ecg = []
    classic_ecg = {'neurokit': [], 'pantompkins1985': [],
                   'hamilton2002': [], 'elgendi2010': [], 'engzeemod2012': [], 'vg': []}
    for ecg_input in tqdm(data['noisy_ecg'], total=len(data['noisy_ecg'])):
        X_input = torch.Tensor(np.swapaxes(ecg_input, 1, 2).astype(np.float32)).cuda()
        with torch.no_grad():
            X_recon = np.swapaxes(model_AAE(X_input, None).detach().cpu().numpy(), 1, 2)
        AAE_ecg.append(X_recon)
        for method_denoise in classic_ecg.keys():
            all_pistes = []
            for piste in range(9):
                all_pistes.append(np.stack([nk.ecg_clean(ecg_input[k, :, piste], sampling_rate=250, method=method_denoise) for k in range(ecg_input.shape[0])]))
            classic_ecg[method_denoise].append(np.stack(all_pistes, axis=-1))
    AAE_ecg = np.stack(AAE_ecg)
    classic_ecg = {method_d: np.concatenate(np.stack(classic_ecg[method_d])) for method_d in classic_ecg.keys()}

    AAE_ecg = np.concatenate(AAE_ecg)
    baseline_ecg = np.concatenate(data['baseline_ecg'])
    diffusion_ecg = np.concatenate(data['diffusion_ecg'])
    target_ecg = np.concatenate(data['target_ecg'])
    noisy_ecg = np.concatenate(data['noisy_ecg'])
    real_stds = np.concatenate(data['real_stds'])
    diff_stds = np.concatenate(data['pred_stds'])

    AAE_stds = (AAE_ecg-target_ecg).std(axis=1)
    DAE_stds = (baseline_ecg-target_ecg).std(axis=1)

    # =========== R2-score by ECG ============ #
    diff_R2 = get_R2_score(target_ecg, diffusion_ecg.mean(axis=1))
    DAE_R2 = get_R2_score(target_ecg, baseline_ecg)
    AAE_R2 = get_R2_score(target_ecg, AAE_ecg)
    classic_R2 = {method_d: get_R2_score(target_ecg, classic_ecg[method_d]) for method_d in classic_ecg.keys()}
    for piste in range(9):
        print(piste)
        print('diff: '+ci_intervals(diff_R2[:, piste]))
        print('DAE: ' + ci_intervals(DAE_R2[:, piste]))
        print('AAE: ' + ci_intervals(AAE_R2[:, piste]))
        for method_d in classic_R2.keys():
            print(method_d +' ' + ci_intervals(classic_R2[method_d][:, piste]))
        print(' ')
    print('diff: '+ci_intervals(diff_R2.mean(axis=-1)))
    print('DAE: ' + ci_intervals(DAE_R2.mean(axis=-1)))
    print('AAE: ' + ci_intervals(AAE_R2.mean(axis=-1)))
    for method_d in classic_R2.keys():
        print(method_d +' ' + ci_intervals(classic_R2[method_d].mean(axis=-1)))
    # =========== absolute error on std estimation =========== #
    diff_err = get_absolute_error(real_stds, diff_stds)
    DAE_err = get_absolute_error(real_stds, DAE_stds)
    AAE_err = get_absolute_error(real_stds, AAE_stds)
    for piste in range(9):
        print(piste)
        print('diff: '+ci_intervals(diff_err[:, piste]))
        print('DAE: ' + ci_intervals(DAE_err[:, piste]))
        print('AAE: ' + ci_intervals(AAE_err[:, piste]))
        print(' ')
    print('diff: '+ci_intervals(diff_err.mean(axis=-1)))
    print('DAE: ' + ci_intervals(DAE_err.mean(axis=-1)))
    print('AAE: ' + ci_intervals(AAE_err.mean(axis=-1)))


    # =========== plot some denoised samples ========== #
    color_posterior = '#00428d'
    color_target = '#fa526c'
    generated_ecg_lst = [diffusion_ecg, baseline_ecg[:, np.newaxis], AAE_ecg[:, np.newaxis]]
    generated_ecg_lst = [np.swapaxes(arr, 2, 3) for arr in generated_ecg_lst]
    method_name_lst = ['diff', 'DAE', 'AAE']
    for k in range(20):
        fig = plot_ecg(ecg_distributions=None,
                       conditioning_ecg=noisy_ecg[k].T,
                       color_target=color_target,
                       color_posterior=color_posterior)

        fig.savefig(os.path.join(results_path,
                                 f'denoise{k}_input_noise.pdf'))
        #fig.show()
        plt.close()
        for method_name, generated_ecg in zip(method_name_lst, generated_ecg_lst):
            fig = plot_ecg(ecg_distributions=generated_ecg[k],
                           conditioning_ecg=target_ecg[k].T,
                           color_target=color_target,
                           color_posterior=color_posterior)

            fig.savefig(os.path.join(results_path,
                                     f'denoise{k}_{method_name}.pdf'))
            #fig.show()
            plt.close()

if __name__ == '__main__':
    main()
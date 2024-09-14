import os.path
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'

os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import matplotlib.pyplot as plt
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import seaborn as sns
from torch.utils.data import DataLoader
from beat_net.beat_net.data_loader import PhysionetECG
import torch
import ot
from beat_wgan.net import Gen_ac_wgan_gp_1d
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
@hydra.main(version_base=None, config_path="configs/", config_name="EMD_eval")
def main(cfg: DictConfig) -> list:
    # cfg = compose(config_name="config") # for loading cfg in python console
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    ckpts_to_load = [
        #'/mnt/data/gabriel/ecg_inpainting/models/ecg_template/template_ecg_conditional/baseline/model/6220',
        #'/mnt/data/gabriel/ecg_inpainting/models/ecg_template/template_ecg_conditional/baseline_uncond/0/model/9360',
       # '/mnt/data/gabriel/ecg_inpainting/models/ecg_template/template_ecg_conditional/deeper/0/model/4480/'
        ##'/mnt/data/lisa/ecg_results/baseline/model6220',
        ##'/mnt/data/lisa/ecg_results/baseline_uncond/model9360'
        #'/mnt/data/gabriel/ecg_inpainting/models/ecg_template/template_ecg_conditional/2024-04-23-11-47-50/model/9970'
        #os.path.join(cfg.results_path, 'generation_eval/no_norm')
        os.path.join(cfg.paths.results_path, 'generation_eval/global_norm_cond'),
        os.path.join(cfg.paths.results_path, 'generation_eval/global_norm_uncond')
    ]
    #model_wgan = torch.load('/mnt/data/gabriel/ecg_inpainting/models/wgan/generator_trained_cl.pt').cuda()
    # ========================== generate samples from the baseline ======================= #
    model_wgan = Gen_ac_wgan_gp_1d(
        noise_dim=100,
        generator_n_features=64,  # Gen data n channels
        conditional_features_dim=4,  # N classes for embedding
        sequence_length=176,  # Length for channel
        sequence_n_channels=9,  # n_channels_seq
        embedding_dim=64).cuda()
    # model_wgan.load_state_dict(torch.load('/mnt/data/lisa/ecg_results/models/wgan_no_norm/generator_trained_cl.pt'))
    model_wgan.load_state_dict(torch.load('/mnt/Reseau/Signal/lisa/ecg_results/models/wgan/generator_trained_cl.pt'))
    normalized = 'none'
    if 'global' in ckpts_to_load[0]:
        normalized = 'global'
    print(normalized)
    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=normalized, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 num_workers=0)
    train_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.paths.db_path,
                                                       categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                       normalized=normalized, training_class='Training', all=False,
                                                       return_beat_id=False),
                                  batch_size=len(test_dataloader.dataset),
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0)
    model_wgan.eval()
    print('data loaders constructed')
    for batch_idx, (batch_test, batch_test_features) in enumerate(test_dataloader):
        bs = batch_test_features.shape[0]
        #real is of shape # 2864 x 176 x 9
        batch_test = np.array(batch_test.reshape(bs, -1).numpy()) # 2864 x 176 * 9
        batch_test = np.concatenate((np.array(batch_test),
                                     np.array(batch_test_features.numpy())), axis=-1) # 2865
        feats = batch_test_features.float()  # ".cuda().float()
        noise = torch.randn((bs, 100, 1))# .cuda()
        with torch.no_grad():
            gen_wgan_ecg = np.swapaxes(model_wgan.cpu()(noise, feats).detach().cpu().numpy(), 1, 2)
        gen_wgan = np.concatenate((gen_wgan_ecg.reshape(bs, -1), np.array(batch_test_features.numpy())), axis=-1)
        M = ot.dist(gen_wgan, batch_test)
        G0_wgan_test = ot.emd2(np.ones(bs) / bs, np.ones(bs)/bs, M, numItermax=1_000_000)

    all_wgan_train = []
    for batch_idx, (batch_train, batch_train_features) in enumerate(train_dataloader):
        bs = batch_train_features.shape[0]
        #real is of shape # 2864 x 176 x 9
        batch_train = np.array(batch_train.reshape(bs, -1).numpy()) # 2864 x 176 * 9
        batch_train = np.concatenate((batch_train,
                                     np.array(batch_train_features.numpy())), axis=-1) # 2865
        # feats = batch_train_features.cuda().float()
        # noise = torch.randn((bs, 100, 1)).cuda()
        # with torch.no_grad():
        #     gen_wgan = np.swapaxes(model_wgan(noise, feats).detach().cpu().numpy(), 1, 2).reshape(bs, -1)
        #  = np.concatenate((gen_wgan, np.array(batch_train_features.numpy())), axis=-1)
        M = ot.dist(gen_wgan, batch_train)
        all_wgan_train.append(ot.emd2(np.ones(bs) / bs, np.ones(bs)/bs, M, numItermax=1_000_000))
    print('wgan vs. test', G0_wgan_test)
    wgan_mean, wgan_std = np.mean(all_wgan_train), np.std(all_wgan_train)
    N = len(all_wgan_train)
    wgan_sup, wgan_inf = wgan_mean + wgan_std*1.96 /(N**.5),  wgan_mean - wgan_std*1.96 /(N**.5)

    print('wgan_sup, wgan_inf', wgan_sup, wgan_inf)

    # ========================= plot some samples from the wgan but also the model ============================ #
    '''
    # better in the unconditional_metric_evaluation.py
    color_posterior = '#00428d'
    color_target = '#fa526c'
    for k in range(10):
        fig = plot_ecg(ecg_distributions=None,
                       conditioning_ecg=gen_wgan_ecg[k].T,
                       color_target=color_posterior,
                       color_posterior=color_target)

        fig.savefig(f'images/wgan{k}.pdf')
        plt.show()    
    '''

    # ========================= plot the metric =============================== #
    data = {}
    min_time, max_time = 10_000, 0
    min_T, max_T = 10_000, 0
    for fname in ckpts_to_load:
        with open(os.path.join(fname, 'wasserstein_generation_metrics.json'), 'rb') as f:
            data[fname] = json.load(f)
        max_time = max(max_time, max([i["time"] for i in data[fname]["wasserstein_with_T_train"].values()]))
        max_T = max(max_T, max([int(i) for i in data[fname]["wasserstein_with_T_train"].keys()]))
        min_time = min(min_time, min([i["time"] for i in data[fname]["wasserstein_with_T_train"].values()]))
        min_T = min(min_T, min([int(i) for i in data[fname]["wasserstein_with_T_train"].keys()]))

    colors = {fname: np.random.rand(3,) for fname in ckpts_to_load}
    print(colors)
    ref_mean, ref_std = data[fname]['ref_mean'], data[fname]['ref_std']
    ref_mod_mean, ref_mod_std = data[fname]['ref_mod_mean'], data[fname]['ref_mod_std']
    N = data[fname]['ref_N']
    ref_sup, ref_inf = ref_mean + ref_std*1.96 /(N**.5),  ref_mean - ref_std*1.96 /(N**.5)
    ref_mod_sup, ref_mod_inf = ref_mod_mean + ref_mod_std*1.96 /(N**.5),  ref_mod_mean - ref_mod_std*1.96 /(N**.5)

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    all_colors = [cmap(k) for k in [0, 0.5, 0.7, 1.]]

    color_ref = 'red'
    color_mod = 'green'

    color_mod = all_colors[0]  #"'cyan'  #00428d
    color_ref = all_colors[-1] # '#fa526c'
    colors = [all_colors[2], 'gray', 'pink'] #, all_colors[1]]  #['#00428d', 'gray']
    colors = {fname: col for col, fname in zip(colors, ckpts_to_load)}

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    fig.subplots_adjust(top=1, bottom=0.1, left=0.1, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(.9*min_time, 1.1*max_time)
    #ax.set_ylim(ref_inf, ref_mod_sup + 1)
    ax.axhline(ref_mean, color=color_ref)#, linestyle='dashed')
    ax.fill_between(x=[.9*min_time, 1*max_time],
                    y1=ref_sup,
                    y2=ref_inf,
                    color=color_ref,
                    alpha=.2,
                    interpolate=True)
    ax.axhline(ref_mod_mean, color=color_mod)#, linestyle='dashed')
    ax.fill_between(x=[.9*min_time, 1*max_time],
                    y1=ref_mod_sup,
                    y2=ref_mod_inf,
                    color=color_mod,
                    alpha=.2,
                    interpolate=True)
    for name, dt in data.items():
        times = [m['time'] for m in dt["wasserstein_with_T_train"].values()]
        mean_train = [m['mean'] for m in dt["wasserstein_with_T_train"].values()]
        mean_test = [m['mean'] for m in dt["wasserstein_with_T_test"].values()]
        int_length_train = [m['std']*1.96 / (m['N']**.5) for m in dt["wasserstein_with_T_train"].values()]
        c = colors[name]
        ax.errorbar(x=times,
                    y=mean_train,
                    yerr=int_length_train,
                    linestyle='solid',
                    capsize=5,
                    linewidth=2,
                    c=c)
        ax.plot(times,
                mean_test,
                linestyle='dashed',
                linewidth=2,
                c=c)
    fig.show()
    fig.savefig(os.path.join(cfg.paths.results_path, f'gen_emd_as_time_{normalized}.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    fig.subplots_adjust(top=1, bottom=0.1, left=0.1, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(.9*min_T, max_T*1.1)
    #ax.set_ylim(ref_inf, ref_mod_sup + 1)
    ax.axhline(ref_mean,xmax=0.9, color=color_ref)#, linestyle='dashed')
    ax.fill_between(x=[.9*min_T, max_T],
                    y1=ref_sup,
                    y2=ref_inf,
                    color=color_ref,
                    alpha=.2,
                    interpolate=True)
    ax.axhline(ref_mod_mean,xmax=0.9,  color=color_mod)#, linestyle='dashed')
    ax.fill_between(x=[.9*min_T, max_T],
                    y1=ref_mod_sup,
                    y2=ref_mod_inf,
                    color=color_mod,
                    alpha=.2,
                    interpolate=True)
    #ax.axhline(wgan_mean,xmax=0.9,  color='gray')#, linestyle='dashed')  # REMOVE THAT
    #ax.axhline(G0_wgan_test,xmax=0.9,  color='gray', linestyle='dashed')  # REMOVE THAT
    # ax.fill_between(x=[.9*min_T, max_T], # REMOVE THAT
    #                 y1=wgan_sup,
    #                 y2=wgan_inf,
    #                 color='gray',
    #                 alpha=.2,
    #                 interpolate=True)
    for name, dt in data.items():
        times = [int(m) for m in dt["wasserstein_with_T_train"].keys()]
        mean_train = [m['mean'] for m in dt["wasserstein_with_T_train"].values()]
        int_length_train = [m['std']*1.96 / (m['N']**.5) for m in dt["wasserstein_with_T_train"].values()]
        mean_test = [m['mean'] for m in dt["wasserstein_with_T_test"].values()]
        c = colors[name]

        ax.errorbar(x=times,
                    y=mean_train,
                    yerr=int_length_train,
                    linestyle='solid',
                    capsize=5,
                    linewidth=2,
                    c=c)
        ax.plot(times,
                mean_test,
                linestyle='dashed',
                    linewidth=2,
                c=c)
        # break  # REMOVE THAT
    #plt.xlim(0, 150)
    # fig.savefig('images/gen_emd_as_T_all'+(len(colors)==3)*'_ALL'+'.pdf')
    fig.savefig(os.path.join(cfg.paths.results_path, f'gen_emd_as_K_{normalized}.pdf'))

    fig.show()
    plt.close(fig)
    # ref_mean = metrics[0]['mean_wasserstein_train_test']
    # ref_sup = ref_mean + metrics[0]['std_wasserstein_train_test']*1.96 /(metrics[0]['N']**.5)
    # ref_inf = ref_mean - metrics[0]['std_wasserstein_train_test']*1.96 /(metrics[0]['N']**.5)
    #
    # fig.show()

if __name__ == '__main__':
    main()
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from jax import random

from beat_net.beat_net.data_loader import PhysionetECG
from baselines import AE, VAE, discriminator
from baselines.utils import train_epoch, val_epoch, test_epoch
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

@hydra.main(version_base=None, config_path="configs/", config_name="baseline_AE")
def main(cfg: DictConfig) -> list:
    yaml_conf = str(OmegaConf.to_yaml(cfg))

    OmegaConf.set_struct(cfg, False)
    print(yaml_conf)
    # net_config = OmegaConf.load(os.path.join(cfg.paths.checkpoint, '.hydra/config.yaml'))
    # print("Load model")

    # ============ prepare checkpoint ============= #
    denoising = bool(cfg.baseline.denoising)
    adversarial = bool(cfg.baseline.adversarial)
    conditional = bool(cfg.baseline.conditional)
    TCN_opt = bool(cfg.baseline.TCN_opt)
    new_model_name = cfg.baseline.model_type + '_disc'*adversarial + '_cond'*conditional + '_TCN'*(TCN_opt)
    if 'VAE' in new_model_name:
        new_model_name += '_kl{cfg.baseline.kl}'
    noise_type = cfg.baseline.noise_type
    path_to_save = os.path.join(cfg.paths.baseline_folder, new_model_name+f'_denoised{noise_type}'*denoising)
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(path_to_save, mode=0o775, exist_ok=True)
    wandb.init(
        **cfg.wandb, config=OmegaConf.to_container(cfg, resolve=True)
    )


    # ============== train data ================ #
    database_path = cfg.paths.db_path
    noise_path = '' + cfg.paths.noise_path * cfg.baseline.denoising
    batch_size = 128

    train_set = PhysionetECG(training_class='Training', all=False, estimate_std=False,
                             database_path=database_path,
                             noise_path=noise_path,
                             categories_to_filter=['NSR', 'SB', 'STach', 'SA'],
                             normalized='no')
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=10,
                              drop_last=False)
    val_set = PhysionetECG(training_class='CV',  all=False, estimate_std=False,
                           database_path=database_path,
                           noise_path=noise_path,
                           categories_to_filter=['NSR', 'MI', 'SB', 'STach', 'SA'],
                           normalized='no')

    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=10,
                            drop_last=False)
    # ============== test data =============== #
    test_set = PhysionetECG(database_path=database_path,
                            categories_to_filter=['NSR', 'SB', 'STach', 'SA'],
                            normalized='no',
                            training_class='Test',
                            all=False,
                            return_beat_id=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
    MI_set = PhysionetECG(database_path=database_path,
                          categories_to_filter=["MI"],
                          normalized='no',
                          training_class='Test',
                          all=True,
                          return_beat_id=False)
    MI_loader = DataLoader(dataset=MI_set,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=8)

    # ============ create model and optimizer =============== #
    N, L, nz, nef, ngpu, ndf = 176, 9, 50, 32, 1, 32
    if cfg.baseline.model_type == 'AE':
        model = AE.AE(N=N, L=L, nz=nz, nef=nef, ngpu=ngpu, denoising=denoising, TCN_opt=TCN_opt, conditional=conditional).cuda()
        loss_fn = nn.MSELoss().cuda()
    else:
        loss_fn = VAE.VariationalLoss(kld_weight=cfg.baseline.kl).cuda()
        model = VAE.VAE(N=N, L=L, nz=nz, nef=nef, ngpu=ngpu, denoising=denoising,
                  conditional=conditional).cuda()
    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    model_dis, optimizer_dis = None, None
    n_critic = 1
    if adversarial:
        model_dis = discriminator.Discriminator(L=L, nz=nz, nef=nef, TCN_opt=TCN_opt).cuda()
        optimizer_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1, beta2))
        # n_critic = 1
    # data_inp = next(iter(train_loader))
    # X_input, feats = get_input(random.PRNGKey(0), X, feats, denoising)

    # ============ train ================ #
    num_epochs = 50
    model.initialize()

    key = random.PRNGKey(0)
    val_loss_best = 100000
    epoch = 0
    for epoch in range(num_epochs):
        train_loss, key = train_epoch(key, epoch, num_epochs, train_loader, model, loss_fn, optimizer, model_dis, optimizer_dis, n_critic)
        val_loss, key, X_input, X, X_rec = val_epoch(key, epoch, num_epochs, val_loader, model, loss_fn, model_dis)
        # evaluate anomaly detection on test set just to verify the model is intersting
        roc_auc = test_epoch(epoch, num_epochs, test_loader, MI_loader, model)
        if type(train_loss) == float:
            metrics = {'loss/train': train_loss, 'loss/val': val_loss, 'anomaly': roc_auc}
        else:
            train_loss, train_loss_d = train_loss
            val_loss, val_loss_d = val_loss
            metrics = {'loss/train': train_loss, 'loss/train_disc': train_loss_d, 'loss/val': val_loss, 'loss/val_disc': val_loss_d, 'anomaly': roc_auc}
        #val_metrics['anomaly'] = roc_auc
        wandb.log(metrics, step=epoch)
        # wandb.log(val_metrics, step=epoch)
        if len(X_rec.shape) == 3:
            for j, (corrupted_track, real_track, pred_track) in enumerate(zip(X_input, X, X_rec)):
                if j== 5:
                    break
                fig, ax = plt.subplots(1, 1, figsize=(5, 8))
                fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                for color, ecg in zip(('red', 'blue', 'orange'), (corrupted_track, real_track, pred_track)):
                    for i, track in enumerate(ecg):
                        ax.plot(track - i, color=color, alpha=.7)
                wandb.log({f"reconstruction/{j}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
        else:
            for j, (corrupted_track, real_track, pred_track) in enumerate(zip(X_input, X, X_rec)):
                fig = plot_ecg(pred_track, real_track)
                wandb.log({f"reconstruction/{j}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
        if adversarial:
            #if val_loss_best >= val_loss+val_loss_d:
            #    torch.save(model, os.path.join(path_to_save, "best_model.pth"))
            torch.save(model_dis, os.path.join(path_to_save, "best_model_dis.pth"))
            torch.save(model, os.path.join(path_to_save, "best_model.pth"))

            #    val_loss_best = val_loss+val_loss_d
        else:
            #if val_loss_best >= val_loss:
            torch.save(model, os.path.join(path_to_save, "best_model.pth"))

            #val_loss_best = val_loss

    # ============ evaluation ============ #
    # /mnt/data/lisa/ecg_results/baseline_AE/GOOD_AE_disc_TCN/best_model


if __name__ == '__main__':
    metrics = main()
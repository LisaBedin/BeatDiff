import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from tqdm import tqdm
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
from UncertaintyEnsemble.models import Network
from Ekgan.models import InferenceGenerator, LabelGenerator, Discriminator
from Ekgan.loss  import inference_generator_loss, label_generator_loss, discriminator_loss
from torch.optim.lr_scheduler import StepLR
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

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.2)

def train_step(input_image, target, inference_generator, discriminator, ig_optimizer, disc_optimizer, label_generator, lg_optimizer, lambda_, alpha):

    ig_lv, ig_output = inference_generator(input_image)
    lg_lv, lg_output = label_generator(input_image)

    disc_real_output = discriminator(torch.cat((input_image, target), dim=1))
    disc_generated_output = discriminator(torch.cat((input_image, ig_output), dim=1))

    total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, input_image)

    total_ig_loss, ig_adversarial_loss, ig_l1_loss, vector_loss  = inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv.detach(), alpha)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    discriminator.zero_grad()
    disc_loss.backward(retain_graph=True)

    label_generator.zero_grad()
    total_lg_loss.backward()

    inference_generator.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        total_ig_loss.backward()
    disc_optimizer.step()
    lg_optimizer.step()
    ig_optimizer.step()

    # print('epoch {} gen_total_loss {} ig_adversarial_loss {} ig_l1_loss {} lg_l2_loss {} vector_loss {}'.format(epoch, total_ig_loss, ig_adversarial_loss, ig_l1_loss, lg_l1_loss, vector_loss))
    metrics = {
        'train/ig_loss': total_ig_loss,
        'train/ig_adv_loss': ig_adversarial_loss,
        'train/ig_l1_loss': ig_l1_loss,
        'train/lg_l1_loss': lg_l1_loss,
        'train/vector_loss': vector_loss
    }

    return metrics


def eval_step(input_image, target, inference_generator, discriminator, label_generator, lambda_, alpha):

    ig_lv, ig_output = inference_generator(input_image)
    lg_lv, lg_output = label_generator(input_image)

    disc_real_output = discriminator(torch.cat((input_image, target), dim=1))
    disc_generated_output = discriminator(torch.cat((input_image, ig_output), dim=1))

    total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, input_image)

    total_ig_loss, ig_adversarial_loss, ig_l1_loss, vector_loss  = inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv, alpha)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # print('epoch {} gen_total_loss {} ig_adversarial_loss {} ig_l1_loss {} lg_l2_loss {} vector_loss {}'.format(epoch, total_ig_loss, ig_adversarial_loss, ig_l1_loss, lg_l1_loss, vector_loss))
    metrics = {
        'val/ig_loss': total_ig_loss,
        'val/ig_adv_loss': ig_adversarial_loss,
        'val/ig_l1_loss': ig_l1_loss,
        'val/lg_l1_loss': lg_l1_loss,
        'val/vector_loss': vector_loss
    }

    return metrics, ig_output[:, 0]

@hydra.main(version_base=None, config_path="configs/", config_name="baseline_ekgan")
def main(cfg: DictConfig) -> list:
    yaml_conf = str(OmegaConf.to_yaml(cfg))

    OmegaConf.set_struct(cfg, False)
    print(yaml_conf)
    net_config = OmegaConf.load(os.path.join(cfg.paths.checkpoint, '.hydra/config.yaml'))
    print("Load model")

    # ============ prepare checkpoint ============= #
    suffix = bool(cfg.ekgan.all_limb)*'12limb_' + '_'.join([str(inp) for inp in cfg.ekgan.input_leads])
    if 'none' in net_config.dataset.normalized or 'no_norm' in net_config.dataset.normalized:
        normalized_prefix = 'no'
    else:
        normalized_prefix = net_config.dataset.normalized
    path_to_save = os.path.join(cfg.paths.baseline_folder, f'Ekgan_{normalized_prefix}_norm_'+suffix+'_epochs200_tmp')
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(path_to_save, mode=0o775, exist_ok=True)
    print(path_to_save)
    wandb.init(
        **cfg.wandb, config=OmegaConf.to_container(cfg, resolve=True)
    )


    # ============== train data ================ #
    database_path = net_config.dataset.database_path
    batch_size = 128

    train_set = PhysionetECG(training_class='Training', all=False, estimate_std=False,
                             database_path=database_path,
                             categories_to_filter=['NSR', 'SB', 'STach', 'SA'],
                             normalized=net_config.dataset.normalized, all_limb=bool(cfg.ekgan.all_limb))
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=10,
                              drop_last=False)
    val_set = PhysionetECG(training_class='CV',  all=False, estimate_std=False,
                           database_path=database_path,
                           categories_to_filter=['NSR', 'MI', 'SB', 'STach', 'SA'],
                           normalized=net_config.dataset.normalized, all_limb=bool(cfg.ekgan.all_limb))

    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=10,
                            drop_last=False)


    n_epochs = 200
    inference_generator = InferenceGenerator()
    discriminator = Discriminator()
    label_generator = LabelGenerator()

    inference_generator.apply(weights_init)
    discriminator.apply(weights_init)
    label_generator.apply(weights_init)

    inference_generator = inference_generator.cuda()
    discriminator = discriminator.cuda()
    label_generator = label_generator.cuda()

    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    ig_optimizer = optim.Adam(inference_generator.parameters(), lr=lr, betas=(beta1, beta2))
    ig_scheduler = StepLR(ig_optimizer, step_size=1, gamma=0.95)
    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    disc_scheduler = StepLR(disc_optimizer, step_size=1, gamma=0.95)
    lr, beta1, beta2 = 0.0001, 0.5, 0.999
    lg_optimizer = optim.Adam(label_generator.parameters(), lr=lr, betas=(beta1, beta2))
    lg_scheduler = StepLR(lg_optimizer, step_size=1, gamma=0.95)

    lambda_, alpha = 50, 1
    best_epoch, best_loss = 0, 100000000
    for epoch in range(n_epochs):
        train_metrics = {'train/ig_loss': 0,
                       'train/ig_adv_loss': 0,
                       'train/ig_l1_loss': 0,
                       'train/lg_l1_loss': 0,
                       'train/vector_loss': 0}
        inference_generator.train()
        label_generator.train()
        discriminator.train()
        for i, (X, feats) in tqdm(enumerate(train_loader), total=len(train_loader)):
            X = nn.Upsample(256, mode='linear')(torch.swapaxes(X.to(torch.float32), 1, 2))

            # ==== prepare input ==== #
            n_leads = X.shape[1]
            lead_factor = int(n_leads // len(cfg.ekgan.input_leads))
            input_image = X[:, torch.tensor(cfg.ekgan.input_leads)].repeat(1, lead_factor, 1)
            top_pad = int((16 -input_image.shape[1]) //2)
            bottom_pad = 16 - (top_pad + input_image.shape[1])
            input_image = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(input_image, 1, 2)), 1, 2)

            # ==== prepare target ==== #
            top_pad = int((16 -X.shape[1]) //2)
            bottom_pad = 16 - (top_pad + X.shape[1])
            target = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(X, 1, 2)), 1, 2)

            metrics = train_step(input_image.unsqueeze(1).cuda(), target.unsqueeze(1).cuda(), inference_generator, discriminator, ig_optimizer, disc_optimizer, label_generator, lg_optimizer, lambda_, alpha)
            train_metrics = {k: train_metrics[k] + metrics[k] for k in train_metrics.keys()}
        train_metrics = {k: train_metrics[k]/len(train_loader) for k in train_metrics.keys()}
        learning_rate = disc_optimizer.param_groups[0]['lr']
        train_metrics['train/lr'] = learning_rate
        wandb.log(train_metrics, step=epoch)

        if epoch >= 150:
            ig_scheduler.step()
            disc_scheduler.step()
            lg_scheduler.step()
        with torch.no_grad():
            inference_generator.eval()
            label_generator.eval()
            discriminator.eval()
            val_metrics = {'val/ig_loss': 0,
                            'val/ig_adv_loss': 0,
                            'val/ig_l1_loss': 0,
                            'val/lg_l1_loss': 0,
                            'val/vector_loss': 0}
            for i, (X, feats) in tqdm(enumerate(val_loader), total=len(val_loader)):
                X = nn.Upsample(256, mode='linear')(torch.swapaxes(X.to(torch.float32), 1, 2))

                # ==== prepare input ==== #
                n_leads = X.shape[1]
                lead_factor = int(n_leads // len(cfg.ekgan.input_leads))
                input_image = X[:, torch.tensor(cfg.ekgan.input_leads)].repeat(1, lead_factor, 1)
                top_pad = int((16 - input_image.shape[1]) // 2)
                bottom_pad = 16 - (top_pad + input_image.shape[1])
                input_image = torch.swapaxes(
                    nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(input_image, 1, 2)), 1, 2)

                # ==== prepare target ==== #
                top_pad = int((16 - X.shape[1]) // 2)
                bottom_pad = 16 - (top_pad + X.shape[1])
                target = torch.swapaxes(nn.ConstantPad1d((top_pad, bottom_pad), 0)(torch.swapaxes(X, 1, 2)), 1, 2)

                metrics, ig_output = eval_step(input_image.cuda().unsqueeze(1), target.cuda().unsqueeze(1), inference_generator, discriminator, label_generator, lambda_, alpha)
                val_metrics = {k: val_metrics[k]+metrics[k] for k in val_metrics.keys()}
            val_metrics = {k: val_metrics[k]/len(val_loader) for k in val_metrics.keys()}
            wandb.log(val_metrics, step=epoch)

            # === visualize reconstruction === #
            for j, (corrupted_track, real_track, pred_track) in enumerate(zip(input_image.numpy(), target.numpy(), ig_output.detach().cpu().numpy())):
                if j== 5:
                    break
                fig, ax = plt.subplots(1, 1, figsize=(5, 8))
                fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                for color, ecg in zip(('red', 'blue', 'orange'), (corrupted_track, real_track, pred_track)):
                    for i, track in enumerate(ecg):
                        ax.plot(track - i, color=color, alpha=.7)
                wandb.log({f"reconstruction/{j}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)

            # === saving model === #
            if val_metrics['val/ig_loss'] < best_loss:
                best_loss = val_metrics['val/ig_loss']
                best_epoch = epoch
                torch.save(inference_generator, os.path.join(path_to_save, "best_inference_generator.pth"))
                torch.save(label_generator, os.path.join(path_to_save, "best_label_generator.pth"))
                torch.save(discriminator, os.path.join(path_to_save, "best_discriminator.pth"))

    '''
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
    '''


if __name__ == '__main__':
    metrics = main()
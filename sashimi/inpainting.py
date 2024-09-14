import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'

import time
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mcg_diff.particle_filter_old import mcg_diff
from mcg_diff.sgm_old import ScoreModel

# from dataset_sc import load_Speech_commands
# from dataset_ljspeech import load_LJSpeech
from dataloaders import dataloader
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from generate import generate

from models import construct_model


class ModelWrapper(nn.Module):
    def __init__(self, model, in_channels=9):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.in_channels = in_channels

    def forward(self, x, time_step):
        '''

        Args:
            x: flatten batch tensor of shape (B, C*seq_length)
            time_step: time_step for the backward diffusion, tensor of shape T

        Returns:

        '''
        x = x.view(x.size(0), self.in_channels, -1)
        return self.model((x, time_step), mel_spec=None).view(x.size(0), -1)


def run_inpainting(
    diffusion_cfg, model_cfg, dataset_cfg, generate_cfg, # dist_cfg, wandb_cfg, # train_cfg,
    ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
    learning_rate, batch_size_per_gpu,
    # n_samples,
    name=None,
    results_path='exp'
    # mel_path=None,
):
    """
    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    n_samples (int):                audio samples to generate and log per checkpoint
    name (str):                     prefix in front of experiment name
    mel_path (str):                 for vocoding, path to mel spectrograms (TODO generate these on the fly)
    """

    local_path, checkpoint_directory = local_directory(name, results_path, model_cfg, diffusion_cfg, dataset_cfg, 'checkpoint')

    # map diffusion hyperparameters to gpu
    # diffusion_cfg.T = 10000
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    dataset_cfg.training_class = 'Test' # TODO uncoment that, when no debugging !
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=1, unconditional=True, shuffle=False)
    print('Data loaded')

    # predefine model
    net = construct_model(model_cfg).cuda()
    print_size(net, verbose=False)

    model_path = os.path.join(checkpoint_directory, 'checkpoint.pkl') # '{}.pkl'.format(ckpt_iter))
    checkpoint = torch.load(model_path, map_location='cpu')

    # feed model dict and optimizer state
    net.load_state_dict(checkpoint['model_state_dict'])
    net = ModelWrapper(net, model_cfg.in_channels)
    for i, data in tqdm(enumerate(trainloader)):
        if i == 0:
            audio, _ = data
            audio = torch.swapaxes(audio, 1, 2).cuda()
            break
    size = (generate_cfg.n_samples, model_cfg.in_channels*dataset_cfg.segment_length)
    # ===== try wrapped model for generation ===== #
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    # Alpha = Alpha[::10]
    # Alpha_bar = Alpha_bar[::10]
    # Sigma = Sigma[::10]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 2


    print('begin sampling, total number of reverse steps = %s' % T)

    x = torch.normal(0, 1, size=size).cuda()
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net(x, diffusion_steps)  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=size).cuda()  # add the variance term to x_{t-1}
    samples = x.view(generate_cfg.n_samples, model_cfg.in_channels, -1).detach().cpu()

    for j, gen_beat in enumerate(samples[:, :, :176 * 5].detach().cpu()):  # [:, :176]):
        fig, ax = plt.subplots(1, 1, figsize=(5, 8))
        fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
        for i, track in enumerate(gen_beat):
            ax.plot(track - i, color='blue', alpha=.7)
        plt.show()
    audio = x.view(generate_cfg.n_samples, model_cfg.in_channels, -1).detach().cuda()

    #t=torch.linspace(0, 1, steps=diffusion_cfg['T']).to(audio.device)
    # alphas_cumprod = torch.exp(-.5*(diffusion_cfg['beta_T']-diffusion_cfg['beta_0'])*(t**2) - diffusion_cfg['beta_0']*t)
    timesteps = torch.arange(0, diffusion_cfg['T'], 10)
    # timesteps[-1] -= 1
    N_particles = 100
    # n_lead = 4
    observed_leads = np.array([0, 1, 2, 4, 6]) # Kardia12L
    # observed_leads = np.array([0, 1, 2, 3, 5]) # 8 Watch
    obs_ind = 1
    observation = audio[obs_ind, observed_leads].flatten()
    init_particles = torch.normal(0, 1, size=(N_particles,  model_cfg.in_channels*dataset_cfg.segment_length))
    # coordinate_mask = torch.cat([torch.ones_like(audio[0, :n_lead]), torch.zeros_like(audio[0, n_lead:])]).to(torch.bool).flatten()
    coordinate_mask = (torch.ones_like(audio[obs_ind])*False).to(bool)
    coordinate_mask[observed_leads] = True
    coordinate_mask = coordinate_mask.flatten().cuda()
    all_samples = []
    with torch.no_grad():
        for sample_k in tqdm(range(10)):
            samples, lw = mcg_diff(
                initial_particles=init_particles.to(audio.device),
                observation=observation,
                var_observation=0, # TODO: maybe 0.5 ?
                score_model=ScoreModel(
                    net=net,
                    alphas_cumprod=diffusion_hyperparams["Alpha_bar"],
                    device='cuda:0'
                ),
                likelihood_diagonal=torch.ones_like(observation),
                coordinates_mask=coordinate_mask,
                timesteps=timesteps.to(audio.device),
                gaussian_var=1e-4,
            )
            samples = samples.cpu().view(-1, model_cfg.in_channels, dataset_cfg.segment_length)
            all_samples.append(samples)
            target_beat = audio[obs_ind].cpu()  # , :, :176 * 5].detach().cpu()
            # for j, gen_beat in enumerate(samples.detach().cpu()):  # [:, :176]):
            #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            #     fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
            #     for i, (track, target_track) in enumerate(zip(gen_beat, target_beat)):
            #         ax.plot(track - i, color='blue', alpha=.7)
            #         ax.plot(target_track - i, color='red', alpha=.7)
            #     plt.show()
            #     break
            gen_beat = np.array(samples[0].detach().cpu())
            real_beat = np.array(audio[obs_ind].cpu())
            gen_beat /= np.max(np.absolute(gen_beat), axis=1)[:, np.newaxis]
            real_beat /= np.max(np.absolute(real_beat), axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(1, 1, figsize=(25, 8))
            file_path = '/mnt/data/lisa/ecg_results/inpainting_10s'
            fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
            for i, (track, track_real) in enumerate(zip(gen_beat, real_beat)):
                ax.plot(track - i, color='blue', alpha=.7, lw=3)
                ax.plot(track_real - i, color='red', alpha=.7, lw=3)
            fig.savefig(os.path.join(file_path, f'limbs_V2_V4_AF1_{sample_k}.pdf'))
            plt.show()

        np.savez('/mnt/data/lisa/ecg_results/inpainting_10s/limbs_V2_V4_AF1.npz', posterior=np.stack(all_samples)[:, 0],
                 real=audio[0].cpu())
            # np.savez('/mnt/data/lisa/ecg_results/inpainting_10s.npz', target_ecg=audio[0], posterior_samples=np.stack(all_samples))


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    os.makedirs(cfg.train.results_path, mode=0o775, exist_ok=True)  # TODO folder to save the experiments

    num_gpus = torch.cuda.device_count()
    run_inpainting(
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )



if __name__ == "__main__":
    main()

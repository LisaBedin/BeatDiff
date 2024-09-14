import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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

from jaxopt import ProximalGradient
from jaxopt.prox import prox_lasso, prox_ridge, prox_elastic_net
from jax import numpy as jnp

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


def cos_sin_proj(X, w):
    J = int(X.shape[0] // 2)
    return w[:, :J]@X[:J] + w[:, J:]@X[J:]

def least_squares(w, data):
    X, y = data
    residuals = jnp.nansum((cos_sin_proj(X, w) - y)**2, axis=-1)#  / jnp.nansum(y**2, axis=-1)
    return jnp.nanmean(residuals)


def run_inpainting(
    cfg, diffusion_cfg, model_cfg, dataset_cfg, generate_cfg, # dist_cfg, wandb_cfg, # train_cfg,
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
    # dataset_cfg.training_class = 'Test' # TODO uncoment that, when no debugging !
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
            denoised_ecg, features, audio, noise, leads = data
            audio = torch.swapaxes(audio, 1, 2).cuda()
            denoised_ecg = np.swapaxes(denoised_ecg, 1, 2)
            break

    #t=torch.linspace(0, 1, steps=diffusion_cfg['T']).to(audio.device)
    # alphas_cumprod = torch.exp(-.5*(diffusion_cfg['beta_T']-diffusion_cfg['beta_0'])*(t**2) - diffusion_cfg['beta_0']*t)

    #  === noise model === #
    J = cfg.denoising.J # number of harmonics
    # f_c = 0.7 # BW
    f_s = 25 # sampling frequency
    # 2 * jnp.pi ??
    phi = np.concatenate([jnp.cos(jnp.arange(dataset_cfg.segment_length)[:, None] / f_s * (jnp.arange(J)[None] / J *
                                                                                  (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                                                                  +cfg.denoising.f_c_min)),
                           jnp.sin(jnp.arange(dataset_cfg.segment_length)[:, None] / f_s * np.arange(J)[None] / J * (
                                   (cfg.denoising.f_c_max - cfg.denoising.f_c_min)
                                   + cfg.denoising.f_c_min)
                                   )],
                          axis=1).T
    f_c = (cfg.denoising.f_c_min + cfg.denoising.f_c_max) / 2.
    eta_param = jnp.zeros((model_cfg.in_channels, phi.shape[0]))
    var_obs = 1

    # ==== inputs ==== #
    timesteps = torch.arange(0, diffusion_cfg['T'], 20)
    # timesteps[-1] -= 1
    N_particles = 50
    n_lead = 9
    observation = audio[0, :n_lead].flatten()
    # init_particles = torch.randn(size=(N_particles,  model_cfg.in_channels*dataset_cfg.segment_length)).to(audio.device)
    # init_particles = torch.normal(0, 1, size=(N_particles,  model_cfg.in_channels*dataset_cfg.segment_length))
    coordinates_mask = torch.cat([torch.ones_like(audio[0, :n_lead]), torch.zeros_like(audio[0, n_lead:])]).to(torch.bool).flatten()

    # ==== algo ==== #
    lreg = 10.
    all_gen = []
    with torch.no_grad():
        for k in tqdm(range(7)):  # Lasso iters
            init_particles = torch.normal(0, 1, size=(N_particles, model_cfg.in_channels * dataset_cfg.segment_length))
            samples, lw = mcg_diff(
                initial_particles=init_particles.to(audio.device),
                observation=observation-torch.tensor(np.array(cos_sin_proj(phi, eta_param))[:n_lead]).flatten().cuda(),
                var_observation=var_obs,  # TODO: maybe 0.5 ?
                score_model=ScoreModel(
                    net=net,
                    alphas_cumprod=diffusion_hyperparams["Alpha_bar"],
                    device='cuda:0'
                ),
                likelihood_diagonal=torch.ones_like(observation),
                coordinates_mask=coordinates_mask,
                timesteps=timesteps.to(audio.device),
                gaussian_var=1e-4,
            )
            samples = samples.cpu().view(-1, model_cfg.in_channels, dataset_cfg.segment_length)
            all_gen.append(samples)
            new_eta_param = []
            for l in range(n_lead):
                y = jnp.array(observation[None, l].cpu() - samples[:, l])
                pg = ProximalGradient(fun=least_squares, prox=prox_lasso, tol=0.0001, maxiter=5, maxls=100, decrease_factor=0.5)
                pg_sol = pg.run(eta_param[l:l+1], hyperparams_prox=lreg, data=(phi, y)).params
                new_eta_param.append(pg_sol)
            eta_param = np.concatenate(new_eta_param)

            target_beat = denoised_ecg[0]  # , :, :176 * 5].detach().cpu()
            for j, gen_beat in enumerate(samples.detach().cpu()):  # [:, :176]):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                for i, (track, target_track) in enumerate(zip(gen_beat, target_beat)):
                    ax.plot(track - i, color='blue', alpha=.7)
                    ax.plot(target_track - i, color='red', alpha=.7)
                plt.show()
                break


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    os.makedirs(cfg.train.results_path, mode=0o775, exist_ok=True)  # TODO folder to save the experiments

    num_gpus = torch.cuda.device_count()
    run_inpainting(
        cfg,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )



if __name__ == "__main__":
    main()

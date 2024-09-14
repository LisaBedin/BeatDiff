import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ['PATH'] = '/usr/local/cuda-12.1/bin'

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

from mcg_diff.dps import Solver
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
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    dataset_cfg.training_class = 'Test'
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=1, unconditional=True)
    print('Data loaded')

    # predefine model
    net = construct_model(model_cfg).cuda()
    print_size(net, verbose=False)

    model_path = os.path.join(checkpoint_directory, 'checkpoint.pkl') # '{}.pkl'.format(ckpt_iter))
    # model_path = '/mnt/data/lisa/ecg_results/sashimi/unet_d64_n4_pool_3_expand2_ff2_T1000_betaT0.02_uncond/checkpoint/checkpoint.pkl'
    checkpoint = torch.load(model_path, map_location='cpu')

    # feed model dict and optimizer state
    net.load_state_dict(checkpoint['model_state_dict'])
    #net = ModelWrapper(net)
    for data in tqdm(trainloader):
        audio, _ = data
        audio = torch.swapaxes(audio, 1, 2).cuda()
        break
    size = (generate_cfg.n_samples, model_cfg.in_channels*dataset_cfg.segment_length)
    # ===== try wrapped model for generation ===== #
    # _dh = diffusion_hyperparams
    # T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    # # Alpha = Alpha[::10]
    # # Alpha_bar = Alpha_bar[::10]
    # # Sigma = Sigma[::10]
    # assert len(Alpha) == T
    # assert len(Alpha_bar) == T
    # assert len(Sigma) == T
    # assert len(size) == 2
    # 
    # 
    # print('begin sampling, total number of reverse steps = %s' % T)
    # 
    # x = torch.normal(0, 1, size=size).cuda()
    # with torch.no_grad():
    #     for t in tqdm(range(T-1, -1, -1)):
    #         diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
    #         epsilon_theta = net(x, diffusion_steps)  # predict \epsilon according to \epsilon_\theta
    #         x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
    #         if t > 0:
    #             x = x + Sigma[t] * torch.normal(0, 1, size=size).cuda()  # add the variance term to x_{t-1}
    # samples = x.view(generate_cfg.n_samples, model_cfg.in_channels, -1).detach().cpu()
    #
    # for j, gen_beat in enumerate(samples[:, :, :176 * 5].detach().cpu()):  # [:, :176]):
    #     fig, ax = plt.subplots(1, 1, figsize=(5, 8))
    #     fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
    #     for i, track in enumerate(gen_beat):
    #         ax.plot(track - i, color='blue', alpha=.7)
    #     plt.show()
    #
    with torch.no_grad():
        n_leads = 6
        dps_solver = Solver()
        dps_solver.set_objective(audio[0, :n_leads].cuda(),
                                 # bservation_noise,
                                 forward_operator_l=torch.eye(model_cfg.in_channels)[:n_leads].cuda(),
                                 forward_operator_t=torch.eye(dataset_cfg.segment_length).cuda(),
                                 score_network=net,
                                 alphas_cumprod=diffusion_hyperparams["Alpha_bar"].cuda(),
                                 n_samples=10)
        dps_solver.run(10)
        samples = dps_solver.get_result()['samples'].cpu()

    target_beat = audio[0].cpu()  # , :, :176 * 5].detach().cpu()
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
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )



if __name__ == "__main__":
    main()

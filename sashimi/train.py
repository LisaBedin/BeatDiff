import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
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
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# from dataset_sc import load_Speech_commands
# from dataset_ljspeech import load_LJSpeech
from dataloaders import dataloader
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from generate import generate

from models import construct_model

def distributed_train(rank, num_gpus, group_name, cfg):
    # Initialize logger
    if rank == 0 and cfg.wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(
            **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )

def train(
    rank, num_gpus,
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
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, unconditional=model_cfg.unconditional, shuffle=True)
    print('Data loaded')

    # predefine model
    net = construct_model(model_cfg).cuda()
    print_size(net, verbose=False)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if model_cfg.scheduler:
        sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(n_iters/10))
    else:
        sched = None

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, 'checkpoint.pkl') # '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch.")
            ckpt_iter = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_iter = -1

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            if 'physionet' in dataset_cfg._name_:
                audio, _ = data
                audio = torch.swapaxes(audio, 1, 2).cuda()
                mel_spectrogram = None
            elif model_cfg["unconditional"]:
                audio, _, _ = data
                # load audio
                audio = audio.cuda()
                mel_spectrogram = None
            else:
                mel_spectrogram, audio = data
                mel_spectrogram = mel_spectrogram.cuda()
                audio = audio.cuda()

            # back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), audio, diffusion_hyperparams, mel_spec=mel_spectrogram)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()
            if sched is not None:
                sched.step()
            epoch_loss += reduced_loss

            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = 'checkpoint.pkl'  # '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            # output to log
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                # print("iteration: {} \treduced loss: {} \tloss: {}".format(n_iter, reduced_loss, loss.item()))
                # tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                # tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
                wandb.log({
                    'train/loss': reduced_loss,
                    'train/log_loss': np.log(reduced_loss),
                    'train/lr': float(optimizer.param_groups[0]['lr']) if sched is None else float(sched.get_last_lr()[0]),
                }, step=n_iter)

                # Generate samples
                # if model_cfg["unconditional"]:
                #     mel_path = None
                #     mel_name = None
                # else:
                #     assert mel_path is not None
                #     mel_name=generate_cfg.mel_name # "LJ001-0001"
                if not model_cfg["unconditional"]: assert generate_cfg.mel_name is not None
                generate_cfg["ckpt_iter"] = n_iter
                samples = generate(
                    rank, # n_iter,
                    results_path,
                    diffusion_cfg, model_cfg, dataset_cfg, net,
                    name=name,
                    **generate_cfg,
                    # n_samples, n_iter, name,
                    # mel_path=mel_path,
                    # mel_name=mel_name,
                )
                if 'physionet' not in dataset_cfg._name_:
                    samples = [wandb.Audio(sample.squeeze().cpu(), sample_rate=dataset_cfg['sampling_rate']) for sample in samples]
                    # TODO: ploting ECG, multiple beats...
                    wandb.log(
                        {'inference/audio': samples},
                        step=n_iter,
                        # commit=False,
                    )
                else:
                    for j, gen_beat in enumerate(samples[:, :, :min(176*5, dataset_cfg.segment_length)].detach().cpu()):  # [:, :176]):
                        fig, ax = plt.subplots(1, 1, figsize=(5, 8))
                        fig.subplots_adjust(left=0.01, right=.99, top=.99, bottom=.01)
                        for i, track in enumerate(gen_beat):
                            ax.plot(track - i, color='blue', alpha=.7)
                        wandb.log({f"gen/{j}": wandb.Image(fig)}, step=n_iter)
                        plt.close(fig)

            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            wandb.log({'train/loss_epoch': epoch_loss, 'train/log_loss_epoch': np.log(epoch_loss)}, step=n_iter)

    # Close logger
    if rank == 0:
        # tb.close()
        wandb.finish()

def training_loss(net, loss_fn, audio, diffusion_hyperparams, mel_spec=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    # audio = X
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = torch.normal(0, 1, size=audio.shape).cuda()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net((transformed_X, diffusion_steps.view(B,1),), mel_spec=mel_spec)  # predict \epsilon according to \epsilon_\theta
    return loss_fn(epsilon_theta, z)



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    os.makedirs(cfg.train.results_path, mode=0o775, exist_ok=True)  # TODO folder to save the experiments

    num_gpus = torch.cuda.device_count()
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()

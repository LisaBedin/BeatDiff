import os
import json
import jax
from tqdm import tqdm
from beat_net.beat_net.unet_parts import load_net
from beat_net.beat_net.variance_exploding_utils import heun_sampler
from beat_db.physionet_tools import filter_ecg
from beat_db.generate_db import get_beats_from_ecg
import numpy as np
from jax.tree_util import Partial as partial
from jax import numpy as jnp
from jax import random, grad
from torch.utils.data import DataLoader
from beat_net.beat_net.data_loader import PhysionetECG
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import ot
import time
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import sys
import pandas as pd
from flax import linen as nn  # Linen API
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax
import matplotlib.pyplot as plt
from flax.training import orbax_utils
import orbax.checkpoint
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from beat_wgan.gen import Gen_ac_wgan_gp_1d, Z_DIM, N_CLASS, DATA_TIME, DATA_N_CHANNELS
from SSSD_ECG.src.sssd.models.SSSD_ECG import SSSD_ECG
from SSSD_ECG.src.sssd.utils.util import sampling_label, find_max_epoch, print_size, calc_diffusion_hyperparams
import torch
import warnings
warnings.filterwarnings("ignore")


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def generate_sssd(features, net, diffusion_hyperparams, label_dim=134, nsr_label_id=75, normalization='global'):
    num_samples = features.shape[0]

    conditioning = torch.zeros((num_samples, label_dim), dtype=torch.float32)
    conditioning[:, nsr_label_id] = 1.0
    conditioning[:, 0] = features[:, 0]

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_ecg = sampling_label(net, (num_samples, 9, 2500),
                                   diffusion_hyperparams,
                                   cond=conditioning.cuda())

    end.record()
    torch.cuda.synchronize()
    #print(int(start.elapsed_time(end) / 1000))
    beats = []
    for g_ecg in generated_ecg:
        filtered_ecg = filter_ecg(final_freq=250, freq=250, original_recording=g_ecg.cpu().numpy())
        try:
            _beats, rrs = get_beats_from_ecg(filtered_ecg)
            beats.append(_beats)
        except:
            beats.append(filtered_ecg[:, :176][:, None])
    beats = np.swapaxes(np.swapaxes(np.concatenate(beats, axis=1), 0, 1), 1, 2)
    if normalization == 'global':
        max_val = np.abs(beats).max(axis=1).clip(1e-4, 10)
        beats = beats / max_val[..., None, :]
    return beats

def load_sssd(config_path='SSSD_ECG/src/sssd/config/config_SSSD_ECG.json',
              ckpt_path = '/mnt/data/gabriel/ecg_inpainting/models/alcaraz_250/sssd_sex_label_cond/ch256_T200_betaT0.02/'):
    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]  # training parameters
    trainset_config = config["trainset_config"]  # to load trainset
    diffusion_config = config["diffusion_config"]  # basic hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters
    model_config = config['wavenet_config']
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config).cuda()
    print_size(net)
    ckpt_iter = 'max'

    # This is needed unfortunately, to initialize omega and Z params of the SSSD with the same batch size as training.
    net((torch.ones(4, 9, 2500).cuda(), torch.ones(4, 134).cuda(), torch.ones(4, 1).cuda())).cpu()

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')
    return lambda cond: sampling_label(net, (cond.shape[0], 9, 2500),
                                   diffusion_hyperparams,
                                   cond=cond)

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    # cfg = compose(config_name="config") # for loading cfg in python console
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    models = load_models(cfg)
    stats = {}
    for model_name, model_info in models.items():
        if model_info["library"] == "torch":
            stats[model_name] = get_stats_torch_model(load_fun=model_info["load_fun"],
                                          variable_creation_fun=model_info["variable_creator_fun"],
                                                      start_batch_size_log2=model_info["start_batch_size_log2"])
        elif model_info["library"] == "jax":
            stats[model_name] = get_stats_jax_model(load_fun=model_info["load_fun"],
                                          variable_creation_fun=model_info["variable_creator_fun"],
                                                    start_batch_size_log2=model_info["start_batch_size_log2"])
        print(stats)
        json.dump(stats, open("data/perf_samples.json", 'w'))

def get_stats_torch_model(load_fun, variable_creation_fun, start_batch_size_log2=1):
    stats = {}
    torch.cuda.reset_peak_memory_stats()
    #First, lets get memory footprint
    zero_mem = torch.cuda.max_memory_allocated(device=None)
    model = load_fun()
    model_mem = torch.cuda.max_memory_allocated(device=None)
    stats["stand_alone_memory_usage"] = sizeof_fmt(model_mem - zero_mem)

    for log2 in range(start_batch_size_log2, 20):
        batch_size = 2**log2
        try:
            features = variable_creation_fun(batch_size)
            features = tuple(f.cuda() for f in features)
            all_elapsed_times = []
            for n in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(*features)
                end.record()
                torch.cuda.synchronize()
                all_elapsed_times.append(start.elapsed_time(end))
            mean_elapsed_time = f'{np.mean(all_elapsed_times):.2f} +/- {np.std(all_elapsed_times) * (1.96 / len(all_elapsed_times)**.5):.2f}'
            #print(mean_elapsed_time)
        except RuntimeError as e:
            stats["max_batch_size"] = batch_size
            stats["gen_time_max_batch_size"] = mean_elapsed_time
            break
    #print(stats)
    return stats


def get_stats_jax_model(load_fun, variable_creation_fun, start_batch_size_log2=1):
    import nvidia_smi

    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info_start = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    stats = {}
    #First, lets get memory footprint
    model = load_fun()

    info_end = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    stats["stand_alone_memory_usage"] = sizeof_fmt(info_end.used - info_start.used)

    for log2 in range(start_batch_size_log2, 30):
        batch_size = 2**log2
        try:
            features = variable_creation_fun(batch_size)
            all_elapsed_times = []
            for n in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(*features).block_until_ready()
                end.record()
                torch.cuda.synchronize()
                all_elapsed_times.append(start.elapsed_time(end))
            mean_elapsed_time = f'{np.mean(all_elapsed_times):.2f} +/- {np.std(all_elapsed_times) * (1.96 / len(all_elapsed_times)**.5):.2f}'
            #print(mean_elapsed_time, batch_size)
        except Exception as e:
            stats["max_batch_size"] = batch_size
            stats["gen_time_max_batch_size"] = mean_elapsed_time
            break
    return stats

def load_models(cfg):
    models = {}

    #SSSD loading
    def dummy_variable_creator_sssd(batch_size):
        conditioning = torch.zeros((batch_size, 134), dtype=torch.float32)
        return (conditioning,)

    models["SSSD"] = {"library": "torch", "load_fun": load_sssd, "variable_creator_fun": dummy_variable_creator_sssd, "start_batch_size_log2": 2}


    #WGAN Load
    load_fun = load_wgan_function_generator(cfg)
    def dummy_variable_creator_wgan(batch_size):
        features = torch.tensor([[0, 1, 0., 0.],]*batch_size).float()
        noise = torch.randn((batch_size, Z_DIM, 1)).float()
        return (noise, features)

    models["WGAN"] = {"library": "torch", "load_fun": load_fun, "variable_creator_fun": dummy_variable_creator_wgan, "start_batch_size_log2":18}
    # Ours loading
    def dummy_variable_creator_mcg_diff(batch_size):

        features = np.stack((np.zeros((batch_size,)),
                             np.ones((batch_size,)),
                             np.zeros((batch_size,)),
                             np.zeros((batch_size,))),
                            dtype=np.float16,
                            axis=-1)
        key = random.PRNGKey(0)
        initial_samples = random.normal(key=key,
                                        shape=(batch_size, 176, 9))*80
        return (initial_samples, features)

    models["MCG-Diff"] = {"library": "jax", "load_fun": load_ours(cfg), "variable_creator_fun": dummy_variable_creator_mcg_diff, "start_batch_size_log2": 10}


    return models


def load_ours(cfg):
    def load_fn():
        train_state, net_config, ckpt_num = load_net(cfg)
        sigma_max = net_config.diffusion.sigma_max
        sigma_min = net_config.diffusion.sigma_min
        p = net_config.generate.rho
        if net_config.diffusion.scale_type == 'linear':
            scaling_fun = lambda t: cfg.diffusion.scaling_coeff * t
        elif net_config.diffusion.scale_type == 'one':
            scaling_fun = lambda t: 1.0
        else:
            raise NotImplemented
        noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)
        generate_fun = jax.jit(lambda initial_samples, cond: partial(heun_sampler,
                                       sigma_min=sigma_min,
                                       sigma_max=sigma_max,
                                       N=100,
                                       p=p,
                                       scale_fun=scaling_fun,
                                       noise_fun=noise_fun,
                                       scale_fun_grad=grad(scaling_fun),
                                       noise_fun_grad=grad(noise_fun),
                                       train_state=train_state)(initial_samples,class_labels=cond))
        return generate_fun
    return load_fn


def load_wgan_function_generator(cfg):
    def load():
        gen = Gen_ac_wgan_gp_1d(
            noise_dim=Z_DIM,
            generator_n_features=64,  # Gen data n channels
            conditional_features_dim=N_CLASS,  # N classes for embedding
            sequence_length=DATA_TIME,  # Length for channel
            sequence_n_channels=DATA_N_CHANNELS,  # n_channels_seq
            embedding_dim=64).cuda()
        gen.load_state_dict(torch.load(os.path.join(cfg.wgan.path, "generator_trained_cl.pt")))
        gen.eval()
        return gen.cuda()
    return load


if __name__ == '__main__':
    metrics = main()

import os

import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from beat_net.beat_net.unet_parts import load_net
from ecg_inpainting.variance_exploding_kernels import mcg_diff_ve
import hydra
from omegaconf import DictConfig, OmegaConf
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate
from torch.utils.data import DataLoader
from jax import vmap, jit, random, numpy as jnp, grad, disable_jit
from jax.tree_util import Partial as partial
import ot
import json
import matplotlib.pyplot as plt
from beat_net.beat_net.variance_exploding_utils import heun_sampler

def plot_ecg(ecg_1, ecg_2 = None):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for color, ecg in zip(('r', 'b'), (ecg_1, ecg_2)):
        for i, track in enumerate(ecg):
            ax.plot(track - i*1.3, color=color, alpha=.7, linewidth=.7, rasterized=True)  # rasterized=True)
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

    train_state, net_config, ckpt_num = load_net(cfg)
    T = net_config.generate.T
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho
    # 4.30 GiB on the first GPU here


    noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)
    if net_config.diffusion.scale_type == 'linear':
        scaling_fun = lambda t: cfg.diffusion.scaling_coeff * t
    elif net_config.diffusion.scale_type == 'one':
        scaling_fun = lambda t: 1.0
    else:
        raise NotImplemented

    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path, categories_to_filter=["NSR", ],
                                                      normalized=True, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=numpy_collate)
    for (batch_test_ecg, batch_test_features) in test_dataloader:
        break

    sigma_ests = jnp.ones((9,))*1

    n_parts = 50
    timesteps = jnp.arange(0, T - 1) / (T - 2)
    timesteps = (sigma_max ** (1 / p) + timesteps * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** (p)


    # Male vs Female
    male_samples = heun_sampler(
        initial_samples=random.normal(key=random.PRNGKey(0),
                                      shape=(1, 176, 9)),
        train_state=train_state,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=25,
        p=7,
        scale_fun=scaling_fun,
        noise_fun=noise_fun,
        scale_fun_grad=grad(scaling_fun),
        noise_fun_grad=grad(noise_fun),
        class_labels=jnp.array([[1, 0, 0, 0]], dtype=jnp.float32)
    )

    female_samples = heun_sampler(
        initial_samples=random.normal(key=random.PRNGKey(0),
                                      shape=(1, 176, 9)),
        train_state=train_state,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=25,
        p=7,
        scale_fun=scaling_fun,
        noise_fun=noise_fun,
        scale_fun_grad=grad(scaling_fun),
        noise_fun_grad=grad(noise_fun),
        class_labels=jnp.array([[0, 1, 0, 0]], dtype=jnp.float32)
    )
    fig = plot_ecg(ecg_1=female_samples[0].T,
                   ecg_2=male_samples[0].T)
    fig.show()

    young_samples = heun_sampler(
        initial_samples=random.normal(key=random.PRNGKey(0),
                                      shape=(1, 176, 9)),
        train_state=train_state,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=25,
        p=7,
        scale_fun=scaling_fun,
        noise_fun=noise_fun,
        scale_fun_grad=grad(scaling_fun),
        noise_fun_grad=grad(noise_fun),
        class_labels=jnp.array([[0, 1, 0, -.5]], dtype=jnp.float32)
    )

    old_samples = heun_sampler(
        initial_samples=random.normal(key=random.PRNGKey(0),
                                      shape=(1, 176, 9)),
        train_state=train_state,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=25,
        p=7,
        scale_fun=scaling_fun,
        noise_fun=noise_fun,
        scale_fun_grad=grad(scaling_fun),
        noise_fun_grad=grad(noise_fun),
        class_labels=jnp.array([[0, 1, 0, .5]], dtype=jnp.float32)
    )
    fig = plot_ecg(ecg_1=young_samples[0].T,
                   ecg_2=old_samples[0].T)
    fig.show()

    short_rr_samples = heun_sampler(
        initial_samples=random.normal(key=random.PRNGKey(0),
                                      shape=(1, 176, 9)),
        train_state=train_state,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=25,
        p=7,
        scale_fun=scaling_fun,
        noise_fun=noise_fun,
        scale_fun_grad=grad(scaling_fun),
        noise_fun_grad=grad(noise_fun),
        class_labels=jnp.array([[0, 1, .4, 0]], dtype=jnp.float32)
    )

    long_rr_samples = heun_sampler(
        initial_samples=random.normal(key=random.PRNGKey(0),
                                      shape=(1, 176, 9)),
        train_state=train_state,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=25,
        p=7,
        scale_fun=scaling_fun,
        noise_fun=noise_fun,
        scale_fun_grad=grad(scaling_fun),
        noise_fun_grad=grad(noise_fun),
        class_labels=jnp.array([[0, 1, 1, 0]], dtype=jnp.float32)
    )
    fig = plot_ecg(ecg_1=short_rr_samples[0].T,
                   ecg_2=long_rr_samples[0].T)
    fig.show()

    print("damn")

if __name__ == '__main__':
    metrics = main()

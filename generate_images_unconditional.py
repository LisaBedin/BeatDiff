import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from beat_net.beat_net.unet_parts import load_net
from beat_net.beat_net.variance_exploding_utils import heun_sampler
import matplotlib.pyplot as plt
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

def plot_ecg(conditioning_ecg, color_target):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    #fig.subplots_adjust(bottom=.05, top=.99, right=.99, left=.07)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)  # rasterized=True)
    # ax.set_ylim(-13.5, 1.5)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    # ax.set_yticks([-i*1.5 for i in range(9)])
    #ax.set_xticklabels(np.arange(0, 175, 50).astype(int), fontsize=22)
    #ax.set_yticklabels(('aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'), fontsize=22)
    return fig


def generate_ECG(dataloader, generate_fun, sigma_max, key):
    all_targets, all_gen, all_feats = [], [], []
    for (batch_ecg, batch_features) in dataloader:
        n_min = batch_ecg.shape[0]
        initial_samples = random.normal(key=key,
                                        shape=(n_min, 176, 9)) * sigma_max
        generated_samples_test = generate_fun(initial_samples,
                                              class_labels=batch_features).block_until_ready()
        all_gen.append(generated_samples_test)
        all_feats.append(batch_features.numpy())
        all_targets.append(batch_ecg.numpy())
        key = random.split(key)[0]
    return jnp.concatenate(all_targets), jnp.concatenate(all_gen), jnp.concatenate(all_feats), key

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    # cfg = compose(config_name="config") # for loading cfg in python console
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

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


    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=True, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 num_workers=0)




    train_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                       categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                       normalized=True, training_class='Training',
                                                       all=False,
                                                       return_beat_id=False),
                                 batch_size=1000,
                                 shuffle=True,
                                 num_workers=0)


    noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)

    generate_fun = partial(heun_sampler,
                           sigma_min=sigma_min,
                           sigma_max=sigma_max,
                           N=100,
                           p=p,
                           scale_fun=scaling_fun,
                           noise_fun=noise_fun,
                           scale_fun_grad=grad(scaling_fun),
                           noise_fun_grad=grad(noise_fun),
                           train_state=train_state)

    target_samples_test, generated_samples_test, test_features, key = generate_ECG(test_dataloader, generate_fun, sigma_max, random.PRNGKey(0))
    results_path = os.path.join(cfg.results_path, 'generated_samples/'+ 'baseline'*('baseline' in cfg.checkpoint)+'deeper'*('baseline' not in cfg.checkpoint)+'_uncond'*('uncond' in cfg.checkpoint))
    os.makedirs(results_path, exist_ok=True)
    np.savez(os.path.join(results_path, 'test_gen.npz'),
             target_samples=target_samples_test,
             generated_samples=generated_samples_test,
             class_features=test_features)

    for j, sample in []: # enumerate(generated_samples_test[:20]):
        color_target = '#00428d' # '#fa526c'
        fig = plot_ecg(sample.T,color_target)
        fig.show()
        fig.savefig(f'images/uncoditional_generation_{j}.pdf')
        plt.close(fig)

    target_samples_train, generated_samples_train, train_features, key = generate_ECG(train_dataloader, generate_fun, sigma_max, key)
    np.savez(os.path.join(results_path, 'train_gen.npz'),
             target_samples=target_samples_train,
             generated_samples=generated_samples_train,
             class_features=train_features)
    print('ok')


if __name__ == '__main__':
    metrics = main()

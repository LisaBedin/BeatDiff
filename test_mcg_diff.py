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
from jax import vmap, jit, random, numpy as jnp
from jax.tree_util import Partial as partial
import ot
import json


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
    timesteps = jnp.arange(0, T-1) / (T - 2)
    timesteps = (sigma_max ** (1/p) + timesteps * (sigma_min**(1/p) - sigma_max**(1/p)))**(p)


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
                                 num_workers=0,
                                 collate_fn=numpy_collate)
    for (batch_test_ecg, batch_test_features) in test_dataloader:
        break

    coordinates_mask = jnp.ones(batch_test_ecg.shape[1:])
    coordinates_mask = coordinates_mask.at[:, 3:].set(0)
    posterior_sampling_fun = jit(vmap(
        partial(mcg_diff_ve,
                observations=batch_test_ecg[0, :, :3].flatten(),
                class_features=batch_test_features[0],
                timesteps=timesteps,
                coordinates_mask=coordinates_mask.flatten()==1,
                denoiser=lambda x, sigma, feats: train_state.apply_fn(train_state.params, x.reshape(-1, 176, 9), sigma, feats).reshape(x.shape[0], -1))
    ))


    n_parts = 50# 10_000
    total_samples = 150 #1_000
    ref_samples = []
    for i in tqdm.tqdm(range(total_samples)):
        initial_samples = random.normal(random.PRNGKey(i),
                                        shape=(
                                            1, n_parts, batch_test_ecg.shape[1] * batch_test_ecg.shape[2])) * sigma_max
        keys = random.split(random.PRNGKey(i), num=1)
        ref_samples.append(posterior_sampling_fun(initial_samples, keys)[:, 0])
    ref_samples = jnp.concatenate(ref_samples)
    dist_per_part = {}
    for n_parts, batch_size in [(2, 1000), (10, 1000), (25, 100), (50, 100), (100, 100), (200, 50), (1000, 10)]:
        all_samples = []
        for batch_index in tqdm.tqdm(range(total_samples // batch_size)):
            initial_samples = random.normal(random.PRNGKey(batch_index),
                                            shape=(
                                                batch_size, n_parts,
                                                batch_test_ecg.shape[1] * batch_test_ecg.shape[2])) * sigma_max
            keys = random.split(random.PRNGKey(batch_index), num=batch_size)
            all_samples.append(posterior_sampling_fun(initial_samples, keys)[:, 0])
        all_samples = jnp.concatenate(all_samples, axis=0)
        M = ot.dist(ref_samples,
                    all_samples)
        dist = ot.emd2(jnp.ones(total_samples)/total_samples,
                       jnp.ones(total_samples)/total_samples,
                       M=M,
                       numItermax=1_000_000)
        dist_per_part[n_parts] = dist.item()
        print(dist_per_part)
    with open(os.path.join(cfg.checkpoint, 'model', str(ckpt_num), 'wasserstein_n_particles.json'), 'w') as file:
        json.dump(dist_per_part, file)


if __name__ == '__main__':
    metrics = main()



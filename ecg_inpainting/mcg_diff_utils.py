import numpy as np
import jax.numpy as jnp
from jax import random


def prepare_mcg_diff_inputs(target_ecg: jnp.ndarray,
                            class_features: jnp.ndarray,
                            setting_mask: jnp.ndarray,
                            key: random.PRNGKeyArray,
                            n_gpu: int,
                            D_batch: int,
                            N_batch:int,
                            N: int):
    observations = (target_ecg*setting_mask + np.nan*(~setting_mask)).reshape(D_batch, -1)
    if n_gpu == 1:
        init_particles = random.normal(key=key, shape=(D_batch, N_batch, N, 176*9))
        gen_keys = random.split(key, D_batch*N_batch).reshape(D_batch, N_batch, -1)
        # observations = observations.reshape(D_batch, -1)
        return init_particles, gen_keys, observations, class_features
    init_particles = random.normal(key=key, shape=((n_gpu, D_batch, N_batch, N, 176*9)))
    observations = jnp.tile(observations, (n_gpu, 1, 1))   # .reshape(n_gpu, D_batch, -1)
    class_features = jnp.tile(class_features, (n_gpu, 1, 1))
    key_shape = (n_gpu, D_batch, N_batch, 2)
    key_num = n_gpu * D_batch * N_batch
    gen_keys = random.split(key, key_num).reshape(*key_shape)
    return init_particles, gen_keys, observations, class_features


def get_coordinate_mask(sigma_ests: jnp.ndarray,  # n_patients x 9
                        timesteps: jnp.ndarray,  # usually 24 timesteps
                        setting_mask: jnp.ndarray,  # 176 x 9
                        n_gpu: int):
    coordinates_mask = jnp.where(timesteps[None, :, None, None] >= sigma_ests[:, None, None, :],
                                 True,
                                 False)  # n_patients x T x 1 x 9
    coordinates_mask = jnp.repeat(coordinates_mask, axis=2, repeats=176)  # n_patients x T x 176 x 9

    coordinates_mask = coordinates_mask * setting_mask[None, None]
    coordinates_mask = coordinates_mask.reshape(coordinates_mask.shape[0], -1, 176*9).astype(jnp.bool_)  # n_patients x T x 176*9
    if n_gpu > 1:
        return jnp.repeat(coordinates_mask[None], axis=0, repeats=n_gpu)
    return coordinates_mask


def prepare_mcg_diff_outputs(generated_ecg: jnp.ndarray,
                             init_particles: jnp.ndarray,
                             keys_gen: jnp.ndarray,
                             n_gpu: int,
                             D_batch: int,
                             N: int):
    if n_gpu > 1:
        generated_ecg = generated_ecg.reshape(n_gpu, D_batch, -1, 176, 9)
        generated_ecg = np.swapaxes(generated_ecg, 0, 1)
        init_particles = np.swapaxes(init_particles,0, 1).reshape(D_batch,-1, N, 176*9)
        keys_gen = np.swapaxes(keys_gen, 0, 1).reshape(D_batch, -1, 2)
    generated_ecg = generated_ecg.reshape(D_batch, -1, 176, 9)
    # keys_gen = generated_ecg.reshape()
    generated_ecg = generated_ecg[:, ::N]  # comment this if you want to keep all particules
    return generated_ecg, init_particles, keys_gen


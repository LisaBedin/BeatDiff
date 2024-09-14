from functools import partial

from jax import numpy as jnp, random
from jax._src.lax.control_flow import cond, fori_loop

def heun_sampler_one_step(i, samples, timesteps, train_state, scale_fun, noise_fun, scale_fun_grad, noise_fun_grad, class_labels=None):
    t = timesteps[i]
    t_next = timesteps[i+1]

    delta_t = t_next - t
    coeff_sample_fun = lambda x: ((noise_fun_grad(x) / noise_fun(x)) + (scale_fun_grad(x) / scale_fun(x)))
    coeff_net_fun = lambda x: (noise_fun_grad(x) * scale_fun(x)) / noise_fun(x)

    d_i = coeff_sample_fun(t) * samples - coeff_net_fun(t) * train_state.apply_fn(train_state.params,
                                                                                  samples,
                                                                                  jnp.array([noise_fun(t),]*samples.shape[0]),
                                                                                  class_labels)
    samples_next = samples + delta_t*d_i
    d_i_prime = cond(
        noise_fun(t_next) > 0,
        lambda x: coeff_sample_fun(x) * samples_next - coeff_net_fun(x) * train_state.apply_fn(train_state.params,
                                                                                               samples_next,
                                                                                               jnp.array([noise_fun(x),]*samples.shape[0]),
                                                                                               class_labels),
        lambda x: d_i,
        t_next
    )
    return samples + delta_t * (d_i + d_i_prime) * .5


def heun_sampler(initial_samples, train_state, sigma_min, sigma_max, N, p, scale_fun, noise_fun, scale_fun_grad, noise_fun_grad, class_labels=None):
    timesteps = jnp.arange(0, N-1) / (N - 2)
    timesteps = (sigma_max ** (1/p) + timesteps * (sigma_min**(1/p) - sigma_max**(1/p)))**(p)
    timesteps = jnp.append(timesteps, values=jnp.array([0]))
    # print('class_labels', class_labels.shape)
    samples = fori_loop(
        lower=0,
        upper=N-1,
        body_fun=partial(heun_sampler_one_step,
                         timesteps=timesteps,
                         train_state=train_state,
                         scale_fun=scale_fun,
                         noise_fun=noise_fun,
                         scale_fun_grad=scale_fun_grad,
                         noise_fun_grad=noise_fun_grad,
                         class_labels=class_labels),
        init_val=initial_samples
    )
    return samples


def make_loss_fn(apply_fn, batch_size, p_mean, p_std, sigma_data, sigma_min, sigma_max, use_f_training):
    def corrupt_data(x, key):
        key_std, key_noise, dropout_key = random.split(key, 3)
        noises_std = jnp.exp(random.normal(key=key_std, shape=(batch_size,), dtype=x.dtype) * p_std + p_mean).clip(sigma_min,
                                                                                                    sigma_max)

        noise = random.normal(key_noise, shape=x.shape, dtype=x.dtype)
        corrupted_data = x + noise * noises_std[:, None, None]
        return corrupted_data, noises_std

    if use_f_training:
        def loss(params, x, class_features, key):
            dropout_key, key_corrupt = random.split(key, 2)
            corrupted_data, noises_std = corrupt_data(x, key_corrupt)
            pred_x = apply_fn(params,
                              corrupted_data,
                              noises_std,
                              class_features,
                              rngs={'dropout': dropout_key})
            l2_norms = jnp.linalg.norm(pred_x -
                                       (x - skip_scaling(noises_std, sigma_data)[:, None, None]*corrupted_data) /
                                       output_scaling(noises_std, sigma_data)[:, None, None], axis=(1, 2))**2
            return l2_norms.mean()
    else:
        def loss(params, x, class_features, key):
            dropout_key, key_corrupt = random.split(key, 2)
            corrupted_data, noises_std = corrupt_data(x, key_corrupt)
            pred_x = apply_fn(params, corrupted_data, noises_std,
                              class_features,
                              rngs={'dropout': dropout_key})
            l2_norms = ((noises_std**2 + sigma_data**2) / (noises_std*sigma_data)**2) * (jnp.linalg.norm(pred_x - x, ord=2, axis=(1, 2))**2)
            return l2_norms.mean()

    return loss


def skip_scaling(sigma, sigma_data):
    return (sigma_data ** 2) / (sigma_data**2 + sigma**2)


def output_scaling(sigma, sigma_data):
    return sigma * sigma_data / ((sigma_data**2 + sigma**2)**.5)


def input_scaling(sigma, sigma_data):
    return 1 / ((sigma_data**2 + sigma**2)**.5)


def noise_scaling(sigma):
    return jnp.log(sigma) / 4

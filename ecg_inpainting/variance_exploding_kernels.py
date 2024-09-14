from jax import numpy as jnp
import numpy as np
from jax import random
from jax.lax import fori_loop, while_loop
from jax.tree_util import Partial as partial
from typing import Callable, Tuple, Union
from jax import vmap, pmap
import jax.debug as debug
from jaxopt import ProximalGradient
from jaxopt.prox import prox_lasso, prox_ridge, prox_elastic_net


def bridge_mean(x_0: jnp.ndarray,
                x_t: jnp.ndarray,
                sigma_t: jnp.ndarray,
                sigma_t_1: jnp.ndarray):
    return (x_t*(sigma_t_1**2) + (sigma_t**2 - sigma_t_1**2) * x_0) / (sigma_t**2)


def bridge_kernel(x_0: jnp.ndarray,
                  x_t: jnp.ndarray,
                  sigma_t: jnp.ndarray,
                  sigma_t_1: jnp.ndarray,
                  key: random.PRNGKeyArray) -> jnp.ndarray:
    mean = bridge_mean(x_0=x_0,
                       x_t=x_t,
                       sigma_t=sigma_t,
                       sigma_t_1=sigma_t_1)
    std = (sigma_t_1/sigma_t) * ((sigma_t**2 - sigma_t_1**2)**.5)
    return mean + std*random.normal(key=key, shape=(x_t.shape[0],))


def backward_kernel(x_t: jnp.ndarray,
                    sigma_t: jnp.ndarray,
                    sigma_t_1: jnp.ndarray,
                    denoiser_fun: Callable[[jnp.ndarray], jnp.ndarray],
                    key: random.PRNGKeyArray) -> jnp.ndarray:
    return bridge_kernel(x_0=denoiser_fun(x_t),
                         x_t=x_t,
                         sigma_t=sigma_t,
                         sigma_t_1=sigma_t_1,
                         key=key)


def _mcg_diff_ve_one_step(i: int,
                          particles: jnp.ndarray,
                          class_features: jnp.ndarray,
                          coordinates_mask: jnp.ndarray,
                          timesteps: jnp.ndarray,
                          observation: jnp.ndarray,
                          keys: jnp.ndarray,
                          start_times: jnp.ndarray,
                          denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    sigma_t_pred, sigma_t_particles, current_coordinates_mask = timesteps[i+1], timesteps[i], coordinates_mask[i+1]
    key_ancestor, key_kernel = random.split(keys[i], 2)
    x_0_pred = denoiser(particles,
                        sigma_t_particles.repeat(particles.shape[0]),
                        jnp.tile(class_features[None], (particles.shape[0], 1)))
    pred_particle_mean_unconditional = bridge_mean(x_0=x_0_pred,
                                                   x_t=particles,
                                                   sigma_t=sigma_t_particles,
                                                   sigma_t_1=sigma_t_pred)
    sigma_bridge = (sigma_t_pred / sigma_t_particles) * ((sigma_t_particles ** 2 - sigma_t_pred ** 2) ** .5)

    likelihood_std = jnp.where(current_coordinates_mask,
                               jnp.where((i+1)*jnp.ones_like(start_times) != start_times,
                                         (sigma_t_pred**2 - timesteps[start_times]**2)**.5,
                                         1e-8),
                               jnp.ones_like(start_times) * sigma_t_pred)

    aux_kernel_std = (sigma_bridge ** 2 + likelihood_std ** 2)**.5
    #weight calculation
    lw = jnp.nansum((((observation - particles) / sigma_t_particles)**2)*current_coordinates_mask[None, ...], axis=-1) - \
         jnp.nansum((((observation - pred_particle_mean_unconditional) / aux_kernel_std[None, :]) ** 2)*current_coordinates_mask[None, ...],
                    axis=(-1,))
    #ancestor sampling
    ancestors = random.categorical(key=key_ancestor, logits=lw, shape=(particles.shape[0],))
    #print('i', i)
    #print('ancestors', ancestors)
    #Propagating through auxiliary kernel
    next_particles_mean = pred_particle_mean_unconditional[ancestors]
    ks = sigma_bridge ** 2 / (sigma_bridge ** 2 + likelihood_std ** 2)
    next_particles_mean = jnp.where(current_coordinates_mask,
                                    next_particles_mean*(1 - ks) + ks*observation,
                                    next_particles_mean)
    noise = random.normal(key=key_kernel, shape=particles.shape, dtype=particles.dtype) * (
            (current_coordinates_mask) * (ks ** .5) * likelihood_std + (~current_coordinates_mask) * sigma_bridge)
    return next_particles_mean + noise


def mcg_diff_ve(
        initial_particles: jnp.ndarray,
        key: random.PRNGKey,
        class_features: jnp.ndarray,
        variances: jnp.ndarray,
        observation: jnp.ndarray,
        timesteps: jnp.ndarray,
        denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    coordinates_mask = generate_coordinate_mask_from_observations(observation,
                                                                  timesteps,
                                                                  variances)
    start_times = jnp.argmin(coordinates_mask, axis=0)
    particles = fori_loop(lower=0,
                          upper=len(timesteps)-1,
                          body_fun=partial(_mcg_diff_ve_one_step,
                                           observation=observation.flatten(),
                                           coordinates_mask=coordinates_mask,
                                           class_features=class_features,
                                           timesteps=timesteps,
                                           keys=random.split(key, len(timesteps) - 1),
                                           denoiser=denoiser,
                                           start_times=start_times),
                          init_val=initial_particles)
    return particles


def mcg_diff_single_gpu(initial_samples: jnp.ndarray,  # N_batch x N x -1,
                        keys: random.PRNGKeyArray,
                        coordinates_mask: jnp.ndarray,
                        observations: jnp.ndarray,
                        class_features: jnp.ndarray,
                        timesteps: jnp.ndarray,
                        denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    return vmap(partial(mcg_diff_ve,
                        timesteps=timesteps,
                        observation=observations,
                        class_features=class_features,
                        coordinates_mask=coordinates_mask,
                        denoiser=denoiser))(initial_samples, keys)


def mcg_diff_multi_gpu(initial_samples: jnp.ndarray,  # D_batch x N_batch x N x -1,
                       keys: random.PRNGKeyArray,
                       coordinates_mask: jnp.ndarray,
                       observations: jnp.ndarray,
                       class_features: jnp.ndarray,
                       timesteps: jnp.ndarray,
                       denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    return vmap(partial(mcg_diff_single_gpu,
                        coordinates_mask=coordinates_mask,
                        timesteps=timesteps,
                        denoiser=denoiser))(initial_samples, keys, observations, class_features)

def _lasso_body(x: Tuple[int, jnp.ndarray, jnp.ndarray],
             keys: random.PRNGKeyArray,
             variances: jnp.ndarray,
             lreg:int, # Tuple[int, int],
             observation: jnp.ndarray,
             timesteps: jnp.ndarray,
             phi: jnp.ndarray, # jnp.cos(4 * 2 * jnp.pi * jnp.arange(T)[:, None] * jnp.arange(J)[None] / J * 0.7)  # baseline wander
             class_features: Union[jnp.ndarray, None],
             denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                solver=prox_lasso,
             max_T: int= 176,
             n_particles: int = 50,
             n_samples: int = 100) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    i, eta_param, eta_param_prev = x
    key_part, key_mcg_diff = random.split(keys[i], 2)
    initial_particles = random.normal(key=key_part,
                                      shape=(n_samples, n_particles, *observation.shape)).reshape(n_samples, n_particles, -1)

    posterior_samples = vmap(partial(mcg_diff_ve,
                                     variances=variances,
                                     observation=observation-cos_sin_proj(phi,eta_param),
                                     class_features=class_features,
                                     timesteps=timesteps,
                                     denoiser=denoiser),
                             in_axes=(0, 0))(initial_particles, random.split(key_mcg_diff, n_samples))
    posterior_samples = posterior_samples[:, 0].reshape(n_samples, *observation.shape)

    #'''
    new_eta_param = []
    # TODO ATTENTION UNCOMMENT THAT
    phi_estim = phi[:max_T]
    for l in range(posterior_samples.shape[-1]):
        y = observation[None, :max_T, l] - posterior_samples[:, :max_T, l]
        pg = ProximalGradient(fun=least_squares, prox=solver, tol=0.0001, maxiter=5, maxls=100, decrease_factor=0.5)
        pg_sol = pg.run(eta_param[:, l], hyperparams_prox=lreg, data=(phi_estim, y)).params
        new_eta_param.append(pg_sol)
    new_eta_param = jnp.stack(new_eta_param, axis=-1)
    #'''
    #new_eta_param = jnp.ones_like(eta_param)*jnp.nan
    #new_sigma_ests = jnp.round(jnp.nanstd(observation[None, :] - posterior_samples - cos_sin_proj(phi, new_eta_param)[None], axis=(0, 1)),
    #                           2) # EM
    #new_var = new_sigma_ests ** 2 # EM
    return i+1, new_eta_param, eta_param


def _em_body(x: Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
             keys: random.PRNGKeyArray,
             lreg:int, # Tuple[int, int],
             observation: jnp.ndarray,
             timesteps: jnp.ndarray,
             phi: jnp.ndarray, # jnp.cos(4 * 2 * jnp.pi * jnp.arange(T)[:, None] * jnp.arange(J)[None] / J * 0.7)  # baseline wander
             mask_phi: jnp.ndarray,
             leads_list: np.ndarray,
             class_features: Union[jnp.ndarray, None],
             denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
             solver=prox_lasso,
             max_T: int= 176,
             n_particles: int = 50,
             n_samples: int = 100) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    i, var, var_prev, eta_param, eta_param_prev = x
    key_part, key_mcg_diff = random.split(keys[i], 2)
    initial_particles = random.normal(key=key_part,
                                      shape=(n_samples, n_particles, *observation.shape)).reshape(n_samples, n_particles, -1)

    posterior_samples = vmap(partial(mcg_diff_ve,
                                     variances=var,
                                     observation=observation-cos_sin_proj(phi,eta_param),
                                     class_features=class_features,
                                     timesteps=timesteps,
                                     denoiser=denoiser),
                             in_axes=(0, 0))(initial_particles, random.split(key_mcg_diff, n_samples))
    posterior_samples = posterior_samples[:, 0].reshape(n_samples, *observation.shape)

    #'''
    # TODO ATTENTION UNCOMMENT THAT
    new_eta_param = [] # jnp.copy(eta_param)
    phi_estim = phi[:max_T]
    for l in leads_list: # range(posterior_samples.shape[-1]):
        y = observation[None, :max_T, l] - posterior_samples[:, :max_T, l]
        pg = ProximalGradient(fun=least_squares,
                              prox=solver,
                              tol=0.0001, maxiter=5,
                              maxls=100,
                              decrease_factor=0.5)
        pg_sol = pg.run(mask_phi[:, l]*eta_param[:, l],
                        hyperparams_prox=lreg,
                        data=(mask_phi[jnp.newaxis,:, l]*phi_estim, y)).params
        new_eta_param.append(pg_sol)
    new_eta_param = jnp.stack(new_eta_param, axis=-1)
    #'''
    #new_eta_param = jnp.ones_like(eta_param)*jnp.nan
    new_sigma_ests = jnp.round(jnp.nanstd(observation[None, :] - posterior_samples - cos_sin_proj(phi, new_eta_param)[None], axis=(0, 1)),
                               2) # EM
    new_var = new_sigma_ests ** 2 # EM
    return i+1, new_var, var, new_eta_param, eta_param



def _em_body_sigma(x: Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
             keys: random.PRNGKeyArray,
             observation: jnp.ndarray,
             timesteps: jnp.ndarray,
             class_features: Union[jnp.ndarray, None],
             denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
             n_particles: int = 50,
             n_samples: int = 100) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    i, var, var_prev = x
    key_part, key_mcg_diff = random.split(keys[i], 2)
    initial_particles = random.normal(key=key_part,
                                      shape=(n_samples, n_particles, *observation.shape)).reshape(n_samples, n_particles, -1)

    posterior_samples = vmap(partial(mcg_diff_ve,
                                     variances=var,
                                     observation=observation,
                                     class_features=class_features,
                                     timesteps=timesteps,
                                     denoiser=denoiser),
                             in_axes=(0, 0))(initial_particles, random.split(key_mcg_diff, n_samples))
    posterior_samples = posterior_samples[:, 0].reshape(n_samples, *observation.shape)

    new_sigma_ests = jnp.round(jnp.nanstd(observation[None, :] - posterior_samples, axis=(0, 1)),
                               2) # EM
    new_var = new_sigma_ests ** 2 # EM
    return i+1, new_var, var



def cos_sin_proj(X, w):
    J = int(X.shape[1] // 2)
    return X[:, :J]@w[:J] + X[:, J:]@w[J:]

def least_squares(w, data):
    X, y = data
    residuals = jnp.nansum((cos_sin_proj(X, w)[None] - y)**2, axis=-1)#  / jnp.nansum(y**2, axis=-1)
    return jnp.nanmean(residuals)


def generate_coordinate_mask_from_observations(observation, timesteps, var):
    base_coordinate_mask = ~jnp.isnan(observation)
    coordinates_mask = jnp.where(
        jnp.logical_and(timesteps[:, None, None] >= (var ** .5)[None, None, :], base_coordinate_mask[None, ...]),
        True,
        False)
    coordinates_mask = coordinates_mask.reshape(coordinates_mask.shape[0], -1).astype(jnp.bool_)
    return coordinates_mask

def lasso_eta(observation: jnp.ndarray,
                initial_variance: jnp.ndarray,
                eta_init: jnp.ndarray,
                class_features: Union[jnp.ndarray, None],
                key: random.PRNGKeyArray,
                lreg: int,
                timesteps: jnp.ndarray,
                phi: jnp.ndarray,
                denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
              solver=prox_lasso,
                max_iter: int = 7,
                max_T: int = 176,
                n_particles: int = 50,
                n_samples: int = 100) -> jnp.ndarray:
    def cond_fun(x):
        i, eta, eta_prev = x
        iter_valid = (i < max_iter)
        #var_valid = jnp.allclose(var, var_prev, equal_nan=True)
        return iter_valid # & (~ var_valid)

    x = while_loop(
        cond_fun=cond_fun,
        body_fun=partial(_lasso_body,
                         keys=random.split(key, max_iter),
                         variances=initial_variance,
                         lreg=lreg,
                         observation=observation,
                         timesteps=timesteps,
                         phi=phi,
                         class_features=class_features,
                         denoiser=denoiser,
                         solver=solver,
                         max_T=max_T,
                         n_particles=n_particles,
                         n_samples=n_samples
                         ),
        init_val=(0, eta_init, eta_init+1)
    )
    i, eta, _= x
    return eta

def em_variance(observation: jnp.ndarray,
                initial_variance: jnp.ndarray,
                eta_init: jnp.ndarray,
                class_features: Union[jnp.ndarray, None],
                mask_phi: np.ndarray,
                key: random.PRNGKeyArray,
                lreg: int,
                timesteps: jnp.ndarray,
                phi: jnp.ndarray,
                denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                leads_list: np.ndarray = np.arange(9),
                solver=prox_ridge,
                max_iter: int = 5,
                max_T=176,
                n_particles: int = 50,
                n_samples: int = 100) -> jnp.ndarray:

    #mask_phi = np.absolute(eta) > harmonics_th
    #leads_list = np.where(mask_phi.sum(axis=0) > 0)[0]
    # TODO filtering phi with the threshold (keep the indices)
    def cond_fun(x):
        i, var, var_prev, eta, eta_prev = x
        iter_valid = (i < max_iter)
        #var_valid = jnp.allclose(var, var_prev, equal_nan=True)
        return iter_valid # & (~ var_valid)

    x = while_loop(
        cond_fun=cond_fun,
        body_fun=partial(_em_body,
                         keys=random.split(key, max_iter),
                         lreg=lreg,
                         observation=observation,
                         timesteps=timesteps,
                         phi=phi,
                         mask_phi=mask_phi,
                         leads_list=leads_list,
                         class_features=class_features,
                         denoiser=denoiser,
                         solver=solver,
                         max_T=max_T,
                         n_particles=n_particles,
                         n_samples=n_samples
                         ),
        init_val=(0, initial_variance, initial_variance + .1, eta_init, eta_init+1)
    )
    i, var, _, eta_new, _= x
    return var, eta_new

def em_variance_only(observation: jnp.ndarray,
                initial_variance: jnp.ndarray,
                class_features: Union[jnp.ndarray, None],
                key: random.PRNGKeyArray,
                timesteps: jnp.ndarray,
                denoiser: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                max_iter: int = 5,
                n_particles: int = 50,
                n_samples: int = 100) -> jnp.ndarray:
    def cond_fun(x):
        i, var, var_prev = x
        iter_valid = (i < max_iter)
        var_valid = jnp.allclose(var, var_prev, equal_nan=True)
        return iter_valid &(~ var_valid)

    x = while_loop(
        cond_fun=cond_fun,
        body_fun=partial(_em_body_sigma,
                         keys=random.split(key, max_iter),
                         observation=observation,
                         timesteps=timesteps,
                         class_features=class_features,
                         denoiser=denoiser,
                         n_particles=n_particles,
                         n_samples=n_samples
                         ),
        init_val=(0, initial_variance, initial_variance + .1)
    )
    i, var, _ = x
    return var


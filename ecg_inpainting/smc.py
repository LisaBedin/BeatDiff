import jax.numpy as jnp
from jax import random, vmap
from jax.lax import fori_loop
from jax.tree_util import Partial as partial
from typing import Callable, Tuple


def _one_step_smc(i: int,
                  particles: jnp.ndarray,
                  proposal: Callable[[int, jnp.ndarray, random.PRNGKeyArray], jnp.ndarray],
                  weight_function: Callable[[int, jnp.ndarray], jnp.ndarray],
                  keys: random.PRNGKeyArray) -> jnp.ndarray:
    key_cat, key_proposal = random.split(keys[i], 2)
    ancestors = random.categorical(key=key_cat, logits=weight_function(i, particles), shape=(particles.shape[0],))
    next_particles = proposal(i, particles[ancestors], key_proposal)
    return next_particles


def smc(initial_particles: jnp.ndarray,
        key: random.PRNGKeyArray,
        proposal: Callable[[int, int, jnp.ndarray, random.PRNGKeyArray], jnp.ndarray],
        weight_function: Callable[[int, jnp.ndarray], jnp.ndarray],
        n_steps: int) -> jnp.ndarray:
    particles = fori_loop(lower=0,
                          upper=n_steps,
                          body_fun=partial(_one_step_smc,
                                           proposal=proposal,
                                           weight_function=weight_function,
                                           keys=random.split(key, n_steps)),
                          init_val=initial_particles)
    return particles
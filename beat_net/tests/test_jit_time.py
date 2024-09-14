import pytest
from beat_net.beat_net.unet_parts import ResnetBlock
from jax import jit, numpy as jnp, random
from timeit import timeit


def test_resnet_compiling_time():
    model = ResnetBlock(n_conv=1,
                        training=False,
                        dtype=jnp.float16)
    _, params = model.init(random.PRNGKey(0),
                        jnp.ones((1, 100, 9)))

    apply_fn = jit(model.apply)

    time = timeit(stmt='apply_fn(params, jnp.ones((1, 100, 9)).block_until_ready()',
                  globals=globals(),
                  number=1)
    print(time)
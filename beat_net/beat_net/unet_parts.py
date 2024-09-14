import os

import flax.linen as nn
from typing import Any, Callable, Sequence, Tuple, List, Optional, Union

import optax
import orbax.checkpoint
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax._src.tree_util import Partial as partial
from jax.image import resize
from jax import random
from jax.lax import cond
from dataclasses import field

from omegaconf import OmegaConf

try:
    print('beat_net/beat_net/unet_parts.py')
    from beat_net.beat_net.variance_exploding_utils import skip_scaling, output_scaling, input_scaling, noise_scaling
except ModuleNotFoundError:
    try:
        print('beat_net/unet_parts.py')
        from beat_net.variance_exploding_utils import skip_scaling, output_scaling, input_scaling, noise_scaling
    except ModuleNotFoundError:
        print('unet_parts.py')
        from variance_exploding_utils import skip_scaling, output_scaling, input_scaling, noise_scaling



class UNetBlock(nn.Module):
    '''
    Inspired by https://github.com/NVlabs/edm/blob/main/training/networks.py
    out_channels: int
    num_heads: int
    dropout_rate: float
    skip_scale: float
    dtype = jnp.float16
    training: bool
    down: bool = False
    up: bool = False
    '''
    out_channels: int
    num_heads: int
    dropout_rate: float
    skip_scale: float = 1
    dtype: Any = jnp.float32
    training: bool = False
    down: bool = False
    up: bool = False
    @nn.compact
    def __call__(self, x, emb):
        orig = x
        if not self.up:
            x = nn.Conv(self.out_channels,
                        kernel_size=(3,),
                        strides=(1,) if not self.down else (2,),
                        padding=1,
                        dtype=self.dtype)(x)
            skip = nn.Conv(self.out_channels,
                           kernel_size=(1,),
                           strides=(1,) if not self.down else (2,),
                           padding=0,
                           dtype=self.dtype)(orig)
        else:
            x = nn.ConvTranspose(self.out_channels,
                                 kernel_size=3,
                                 padding='SAME',
                                 strides=(2,),
                                 dtype=self.dtype)(x)
            skip = nn.ConvTranspose(self.out_channels,
                                    kernel_size=3,
                                    padding='SAME',
                                    strides=(2,),
                                    dtype=self.dtype)(orig)
        params = nn.Dense(features=self.out_channels,
                          dtype=self.dtype)(emb)
        x = nn.GroupNorm(dtype=self.dtype)(x + params)
        x = nn.activation.silu(x)

        x = nn.Conv(self.out_channels,
                    kernel_size=(3,),
                    strides=(1,),
                    padding=1,
                    dtype=self.dtype)(nn.Dropout(self.dropout_rate,
                                                 deterministic=not self.training)(x))

        x = x + skip
        x = x * self.skip_scale

        if self.num_heads:
            x = nn.SelfAttention(num_heads=self.num_heads,
                                 dtype=self.dtype,
                                 out_features=self.out_channels)(nn.GroupNorm(dtype=self.dtype)(x)) + x
            x = x * self.skip_scale
        return x


class DhariwalUnet(nn.Module):
    model_channels: int = 192
    channel_mult: list = field(default_factory=lambda: [1, 2, 3, 4])
    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_resolutions: list = field(default_factory=lambda:  [32, 16, 8])
    dropout_rate: float = .10
    dtype: Any = jnp.float32
    training: bool = False
    embbed_position_in_signal: bool = True
    conditional: bool = True

    @nn.compact
    def __call__(self, x, t_step, class_features=None):
        #Mapping
        B, L, C = x.shape
        emb = PositionalEmbedding(num_channels=self.model_channels,
                                  dtype=self.dtype)(t_step)
        emb_channels = self.model_channels * self.channel_mult_emb
        emb = nn.activation.silu(nn.Dense(emb_channels,
                                          dtype=self.dtype)(emb))
        emb = nn.Dense(emb_channels, dtype=self.dtype)(emb)
        if self.conditional:
            class_emb = nn.Dense(
                emb_channels,
                dtype=self.dtype
            )(
                nn.activation.silu(
                    nn.Dense(
                        2*emb_channels,
                        dtype=self.dtype
                    )(class_features)
                )
            )

            emb = emb + class_emb
        if self.embbed_position_in_signal:
            length_embedding = PositionalEmbedding(num_channels=emb_channels,
                                                   dtype=self.dtype)(jnp.linspace(0, 1, L, dtype=emb.dtype))
            emb = emb[:, None, :] + length_embedding[None, :]
        else:
            emb = emb[:, None, :]
        emb = nn.activation.silu(emb)

        downsampled_embeddings = {int(2**i): nn.avg_pool(emb, window_shape=(2**i,), strides=(2**i,)) if i > 0 else emb for i in range(len(self.channel_mult))}
        #Encoder
        skips_values = []
        skips_channels_infos = []
        cout = C
        for level, mult in enumerate(self.channel_mult):
            res = L >> level
            if level == 0:
                cout = self.model_channels * mult
                encoder = nn.Conv(cout,
                                  kernel_size=(3,),
                                  padding=1,
                                  strides=(1,),
                                  dtype=self.dtype)
                cin = x.shape[-1]
                x = encoder(x)
            else:
                encoder = UNetBlock(out_channels=cout,
                                    dtype=self.dtype,
                                    num_heads=cout // 64,
                                    down=True,
                                    dropout_rate=self.dropout_rate,
                                    training=self.training)
                x = encoder(x, downsampled_embeddings[emb.shape[1] // (x.shape[1]//2)])
                cin = x.shape[-1]
            skips_values.append(x)
            skips_channels_infos.append((cin, cout))
            for idx in range(self.num_blocks):
                cout = self.model_channels * mult
                cin = x.shape[-1]
                x = UNetBlock(
                    out_channels=cout,
                    dtype=self.dtype,
                    num_heads=cout // 64 if res in self.attn_resolutions else 0,
                    down=False,
                    dropout_rate=self.dropout_rate,
                    training=self.training
                )(x, downsampled_embeddings[emb.shape[1] // x.shape[1]])
                skips_values.append(x)
                skips_channels_infos.append((cin, cout))

        #Decoder
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            res = L >> level
            if level == len(self.channel_mult) - 1:
                for i in range(2):
                    x = UNetBlock(
                        out_channels=cout,
                        dtype=self.dtype,
                        num_heads=cout // 64,
                        down=False,
                        dropout_rate=self.dropout_rate,
                        training=self.training
                    )(x, downsampled_embeddings[emb.shape[1] // x.shape[1]])
            else:
                x = UNetBlock(
                    out_channels=cout,
                    dtype=self.dtype,
                    num_heads=cout // 64,
                    up=True,
                    dropout_rate=self.dropout_rate,
                    training=self.training
                )(x, downsampled_embeddings[emb.shape[1] // (2*x.shape[1])])
            for idx in range(self.num_blocks + 1):
                cin_skip, cout_skip = skips_channels_infos.pop()
                cin = cout_skip + cout
                cout = self.model_channels * mult
                if x.shape[-1] != cin:
                    x = jnp.concatenate((x, skips_values.pop()), axis=-1)
                x = UNetBlock(
                    out_channels=cout,
                    dtype=self.dtype,
                    num_heads=cout // 64 if res in self.attn_resolutions else 0,
                    down=False,
                    dropout_rate=self.dropout_rate,
                    training=self.training
                )(x, downsampled_embeddings[emb.shape[1] // x.shape[1]])
        x = nn.Conv(C,
                    kernel_size=(3,),
                    padding=1,
                    strides=(1,),
                    dtype=self.dtype)(nn.activation.silu(nn.GroupNorm(dtype=self.dtype)(x)))
        return x


# Timestep embedding used in DDPM
class PositionalEmbedding(nn.Module):

    num_channels: int
    max_positions: int = 10_000
    endpoint: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        freqs = jnp.arange(start=0, stop=self.num_channels//2, dtype=self.dtype)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = jnp.outer(x, freqs)
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)
        return x


class DenoiserNet(nn.Module):
    skip_scaling: Callable[[jnp.ndarray], jnp.ndarray]
    output_scaling: Callable[[jnp.ndarray], jnp.ndarray]
    input_scaling: Callable[[jnp.ndarray], jnp.ndarray]
    noise_conditioning: Callable[[jnp.ndarray], jnp.ndarray]
    model_channels: int = 192
    embbed_position_in_signal: bool = True
    channel_mult: list = field(default_factory=lambda: [1, 2, 3, 4])
    channel_mult_emb: int = 4
    num_blocks: int = 3
    attn_resolutions: list = field(default_factory=lambda:  [32, 16, 8])
    dropout_rate: float = .10
    dtype: Any = jnp.float32
    training: bool = False
    use_f_training: bool = True
    conditional: bool = True

    @nn.compact
    def __call__(self, x, noise_std, class_features=None):
        skip_scaling = self.skip_scaling(noise_std)
        output_scaling = self.output_scaling(noise_std)
        input_scaling = self.input_scaling(noise_std)
        noise_scaling = self.noise_conditioning(noise_std)

        F_net = DhariwalUnet(
            model_channels=self.model_channels,
            channel_mult=self.channel_mult,
            embbed_position_in_signal=self.embbed_position_in_signal,
            channel_mult_emb=self.channel_mult_emb,
            num_blocks=self.num_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            training=self.training,
            conditional=self.conditional,
        )(input_scaling[:, None, None]*x,
          noise_scaling,
          class_features)
        return cond(self.training and self.use_f_training,
                    lambda _: F_net,
                    lambda _: skip_scaling[:, None, None]*x + output_scaling[:, None, None]*F_net,
                    None)


def load_net(cfg):
    print('loading')
    def best_fn(metrics) -> float:
        return metrics['loss/CV']

    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=3, best_fn=best_fn, best_mode='min')
    mngr = orbax.checkpoint.CheckpointManager(
        os.path.join(cfg.checkpoint, 'model'),
        orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        options)
    net_config = OmegaConf.load(os.path.join(cfg.checkpoint, '.hydra/config.yaml'))
    print(net_config)
    ckpt_num = mngr.best_step()
    net_cfg = net_config.model
    diffusion_cfg =net_config.diffusion
    model = DenoiserNet(
        skip_scaling=partial(skip_scaling, sigma_data=diffusion_cfg.sigma_data),
        output_scaling=partial(output_scaling, sigma_data=diffusion_cfg.sigma_data),
        input_scaling=partial(input_scaling, sigma_data=diffusion_cfg.sigma_data),
        noise_conditioning=noise_scaling,
        model_channels=net_cfg.model_channels,
        num_blocks=net_cfg.num_blocks,
        channel_mult=net_cfg.channel_mult,
        channel_mult_emb=net_cfg.channel_mult_emb,
        attn_resolutions=net_cfg.attn_resolutions,
        dropout_rate=net_cfg.dropout_rate,
        use_f_training=False,
        training=False,
        conditional=net_cfg.conditional,
        embbed_position_in_signal=net_cfg.embbed_position_in_signal,
        dtype=jnp.float32,
    )

    params = mngr.restore(ckpt_num)
    train_state_ema = TrainState.create(apply_fn=model.apply,
                                        params=params["params"],
                                        tx=optax.ema(decay=1))

    return train_state_ema, net_config, ckpt_num

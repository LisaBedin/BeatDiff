import sys
import numpy as np

from beat_net.unet_parts import Unet
from beat_net.data_loader import PhysionetECG
from torch.utils.data import DataLoader
from jax import default_device, devices
from jax import random, numpy as jnp, value_and_grad, jit
from jax.tree_util import Partial as partial
import optax
import matplotlib.pyplot as plt
import pandas as pd


def update_step(apply_fn, x, opt_state, params, tx, y):
    def loss(params):
        t_steps = jnp.ones((x.shape[0], 1))*100
        pred_y = apply_fn(params,
                          x,
                          t_steps)
        l = ((pred_y - y) ** 2).mean() # Replace with your loss here.
        return l, pred_y

    (l, pred_y), grads = value_and_grad(
        loss, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)  # Defined below.
    params = optax.apply_updates(params, updates)
    return opt_state, params, l, pred_y


def initialize_net(key_initialization, kernel_size, n_convs,
                   lr,
                   n_layers=1):
    key1, key2 = random.split(key_initialization, 2)
    tx = optax.adam(learning_rate=lr)
    model = Unet(start_dim=128,
                 n_blocks=n_layers,
                 n_conv_per_block=n_convs,
                 kernel_size_block=kernel_size,
                 use_group_norm_block=True,
                 use_mid_block=False,
                 dtype=jnp.float16)
    t_steps = jnp.ones((data.shape[0], 1))*100
    params = model.init(key2, data, t_steps)
    opt_state = tx.init(params)
    update_fun = jit(partial(update_step,
                             apply_fn=model.apply,
                             tx=tx,
                             y=data))
    return update_fun, opt_state, params


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22})
    database_path = sys.argv[1]
    parameter_to_validate = sys.argv[2]


    dist_dataloader = DataLoader(dataset=PhysionetECG(database_path=database_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=False),
                                 batch_size=512,
                                 shuffle=True,
                                 num_workers=10)
    for data in dist_dataloader:
        break
    np.save('batch_example.npy', data[0].permute(0, 2, 1).numpy())
    with default_device(devices("cuda")[0]):
        data = jnp.array(data[0].permute(0, 2, 1).numpy())
        # getting a sample dataset
        if parameter_to_validate == 'kernel_size':
            config_infos = []
            for j in range(10):
                key_initialization, key_epochs = random.split(random.PRNGKey(j), 2)
                for kernel_size in [3, 5, 7, 9]:
                    update_fun, opt_state, params = initialize_net(key_initialization,
                                                                   kernel_size=kernel_size,
                                                                   n_convs=1,
                                                                   lr=1e-4)
                    print("Start training")
                    losses = []
                    for i, key in enumerate(random.split(key_epochs, 1000)):
                        opt_state, params, l, rec = update_fun(x=data + random.normal(key=key, shape=data.shape),
                                                               opt_state=opt_state,
                                                               params=params,
                                                               )
                        print(i, l)
                        losses.append(l.item())
                        if l < 1e-2:
                            break
                    config_infos.append({"kernel_size": kernel_size, "loss": losses, "index": i, "key_starter": j})

            kernel_size_infos = pd.DataFrame.from_records(config_infos)
            agg_to_show = kernel_size_infos.groupby(['kernel_size'])[['loss']].agg(
                lambda
                    x: f'{np.mean([a[-1] for a in x]):.4f} ± {np.std([a[-1] for a in x], axis=0) * 1.96 / (len(x) ** .5):.4f}')

            agg_to_show.reset_index()
            agg_to_show.to_csv('data/kernel_size_validation.csv')
            agg_loss = kernel_size_infos.groupby(['kernel_size'])[['loss']].agg(
                [('mean_', lambda x: np.mean(np.stack(x, axis=0), axis=0)),
                 ('width_', lambda x: np.std(np.stack(x, axis=0), axis=0) * 1.96 / (len(x) ** .5))])

            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(top=.99, bottom=.1, left=.15, right=.99)
            for k_size, infos in agg_loss.iterrows():
                mu, wdt = infos.loss.mean_, infos.loss.width_
                ax.errorbar(x=np.arange(len(mu)) + 1, y=mu, yerr=wdt, label=k_size)
            fig.legend(fontsize=20)
            ax.set_yscale('log')
            fig.savefig('images/validation_losses_kernel_size.pdf')

        if parameter_to_validate == 'n_convs':

            config_infos = []

            for j in range(10):

                key_initialization, key_epochs = random.split(random.PRNGKey(j), 2)
                for n_convs in range(1, 6):
                    update_fun, opt_state, params = initialize_net(key_initialization, kernel_size=5,
                                                                   n_convs=n_convs,
                                                                   lr=1e-4)
                    print("Start training")
                    losses = []
                    for i, key in enumerate(random.split(key_epochs, 1000)):
                        opt_state, params, l, rec = update_fun(x=data + random.normal(key=key, shape=data.shape),
                                                               opt_state=opt_state,
                                                               params=params,
                                                               )
                        print(i, l)
                        losses.append(l.item())
                        if l < 1e-2:
                            break
                    config_infos.append({"n_convs": n_convs, "loss": losses, "index": i, "key_starter": j})

            n_convs_infos = pd.DataFrame.from_records(config_infos)
            agg_to_show = n_convs_infos.groupby(['n_convs'])[['loss']].agg(
                lambda
                    x: f'{np.mean([a[-1] for a in x]):.4f} ± {np.std([a[-1] for a in x], axis=0) * 1.96 / (len(x) ** .5):.4f}')

            agg_to_show.reset_index()
            agg_to_show.to_csv('data/n_convs_validation.csv')
            agg_loss = n_convs_infos.groupby(['n_convs'])[['loss']].agg(
                [('mean_', lambda x: np.mean(np.stack(x, axis=0), axis=0)),
                 ('width_', lambda x: np.std(np.stack(x, axis=0), axis=0) * 1.96 / (len(x) ** .5))])

            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(top=.99, bottom=.1, left=.15, right=.99)
            for k_size, infos in agg_loss.iterrows():
                mu, wdt = infos.loss.mean_, infos.loss.width_
                ax.errorbar(x=np.arange(len(mu)) + 1, y=mu, yerr=wdt, label=k_size)
            fig.legend(fontsize=20)
            ax.set_yscale('log')
            fig.savefig('images/validation_losses_n_convs.pdf')

        if parameter_to_validate == 'n_layers':

            config_infos = []

            for j in range(10):
                key_initialization, key_epochs = random.split(random.PRNGKey(j), 2)
                for n_layers in range(1, 5):
                    update_fun, opt_state, params = initialize_net(key_initialization, kernel_size=5,
                                                                   n_convs=3,
                                                                   n_layers=n_layers,
                                                                   lr=1e-4)
                    print("Start training")
                    losses = []
                    for i, key in enumerate(random.split(key_epochs, 5000)):
                        opt_state, params, l, rec = update_fun(x=data + random.normal(key=key, shape=data.shape),
                                                               opt_state=opt_state,
                                                               params=params,
                                                               )
                        #print(i, l)
                        losses.append(l.item())
                    config_infos.append({"n_layers": n_layers, "loss": losses, "index": i, "key_starter": j})
                    print('=============')
                    print(n_layers, l)
                    print('=============')
            n_layers_infos = pd.DataFrame.from_records(config_infos)
            agg_to_show = n_layers_infos.groupby(['n_layers'])[['loss']].agg(
                lambda
                    x: f'{np.mean([a[-1] for a in x]):.4f} ± {np.std([a[-1] for a in x], axis=0) * 1.96 / (len(x) ** .5):.4f}')

            agg_to_show.reset_index()
            agg_to_show.to_csv('data/n_layers_validation.csv')
            agg_loss = n_layers_infos.groupby(['n_layers'])[['loss']].agg(
                [('mean_', lambda x: np.mean(np.stack(x, axis=0), axis=0)),
                 ('width_', lambda x: np.std(np.stack(x, axis=0), axis=0) * 1.96 / (len(x) ** .5))])

            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(top=.99, bottom=.1, left=.15, right=.99)
            for k_size, infos in agg_loss.iterrows():
                mu, wdt = infos.loss.mean_, infos.loss.width_
                ax.errorbar(x=np.arange(len(mu)) + 1, y=mu, yerr=wdt, label=k_size)
            fig.legend(fontsize=20)
            ax.set_yscale('log')
            fig.savefig('images/validation_losses_n_layers.pdf')
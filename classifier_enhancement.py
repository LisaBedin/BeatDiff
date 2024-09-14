import os

import jax
from tqdm import tqdm
from beat_net.beat_net.unet_parts import load_net
from beat_net.beat_net.variance_exploding_utils import heun_sampler
from beat_db.physionet_tools import filter_ecg
from beat_db.generate_db import get_beats_from_ecg
import numpy as np
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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import sys
import pandas as pd
from flax import linen as nn  # Linen API
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax
import matplotlib.pyplot as plt
from flax.training import orbax_utils
import orbax.checkpoint
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from beat_wgan.gen import Gen_ac_wgan_gp_1d, Z_DIM, N_CLASS, DATA_TIME, DATA_N_CHANNELS
from SSSD_ECG.src.sssd.models.SSSD_ECG import SSSD_ECG
from SSSD_ECG.src.sssd.utils.util import sampling_label, find_max_epoch, print_size, calc_diffusion_hyperparams
import torch
from SSSD_ECG.src.sssd.train import get_trainloader
from pathlib import Path
import math


def generate_sssd(features, net, diffusion_hyperparams, n_beats=1, label_dim=134, nsr_label_id=76, normalization='global'):
    num_samples = features.shape[0]

    conditioning = torch.zeros((num_samples, label_dim), dtype=torch.float32)
    conditioning[:, nsr_label_id] = 1.0
    conditioning[:, 0] = features[:, 0]

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_ecg = sampling_label(net,
                                   (num_samples, 9, 2500),
                                   diffusion_hyperparams,
                                   cond=conditioning.cuda(),
                                   skip_steps=10)

    end.record()
    torch.cuda.synchronize()
    #print(int(start.elapsed_time(end) / 1000))
    beats = []
    feats = []
    for g_ecg, g_feat in zip(generated_ecg, conditioning):
        filtered_ecg = filter_ecg(final_freq=250, freq=250, original_recording=g_ecg.cpu().numpy())
        try:
            _beats, rrs = get_beats_from_ecg(filtered_ecg)
            beats.append(_beats[:, 1])
            feats.append(g_feat)
        except:
            beats.append(filtered_ecg[:, :176][:, None])
    beats = np.swapaxes(np.stack(beats, axis=0), 1, 2)
    feats = np.stack(feats, axis=0)
    if normalization == 'global':
        max_val = np.abs(beats).max(axis=(1, 2)).clip(1e-4, 10)
        beats = beats / max_val[..., None, None]
    return beats, feats


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, 176, 9]))['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())


class Classifier(nn.Module):
    """A simple CNN model."""
    n_class: int = 2
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3,))(x)
        x = nn.relu(nn.LayerNorm()(x))
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Conv(features=128, kernel_size=(3,))(x)
        x = nn.relu(nn.LayerNorm()(x))
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = nn.Conv(features=256, kernel_size=(3,))(x)
        x = nn.relu(nn.LayerNorm()(x))
        x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
        x = x.mean(axis=-2)  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(nn.LayerNorm()(x))
        x = nn.Dense(features=self.n_class)(x)
        return x


@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['ecg'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['ecg'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def train(cfg,
          dataset,
          n_steps,
          batch_size,
          learning_rate,
          key,
          model_name,
          num_steps_per_epoch=10):

    dataset_length = dataset['ecgs'].shape[0]
    metrics_history = {'train_loss': [],
                       'train_accuracy': [],
                       'test_loss': [],
                       'test_accuracy': []}
    cv_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                    categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                    normalized="global", training_class='CV', all=False,
                                                    return_beat_id=False),
                               batch_size=10_000,
                               shuffle=True,
                               drop_last=False,
                               num_workers=0)
    #model and optimizer initialization
    model = Classifier()
    state = create_train_state(model, key, learning_rate)
    pbar = tqdm(range(n_steps), desc=f'{model_name} training', dynamic_ncols=True)
    for step in pbar:
        batch_indexes = np.random.randint(low=0, high=dataset_length, size=(batch_size,))
        ecg, label = jnp.array(dataset['ecgs'][batch_indexes]), jnp.array(dataset['label'][batch_indexes])
        batch = {"ecg": ecg, "label": label}
        state = train_step(state, batch) # get updated train state (which contains the updated parameters)
        state = compute_metrics(state=state, batch=batch) # aggregate batch metrics

        if (step+1) % num_steps_per_epoch == 0: # one training epoch has passed
            for metric,value in state.metrics.compute().items(): # compute metrics
                metrics_history[f'train_{metric}'].append(value) # record metrics
            state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

            # Compute metrics on the test set after each training epoch
            test_state = state
            for test_batch in cv_dataloader:
                batch_ecg, batch_features = test_batch
                ecg, label = jnp.array(batch_ecg), jnp.array(batch_features[:, 0]).astype(jnp.integer)
                batch = {"ecg": ecg, "label": label}
                test_state = compute_metrics(state=test_state, batch=batch)

            for metric,value in test_state.metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            pbar.set_postfix({'train_accuracy': metrics_history['train_accuracy'][-1] * 100,
                              'cv_accuracy': metrics_history['test_accuracy'][-1] * 100})

    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized="global", training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 drop_last=False,
                                 num_workers=0)
    test_logits = []
    test_labels = []
    for test_batch in test_dataloader:
        batch_ecg, batch_features = test_batch
        ecg, label = jnp.array(batch_ecg), jnp.array(batch_features[:, 0]).astype(jnp.integer)
        batch = {"ecg": ecg, "label": label}
        logits = state.apply_fn({'params': state.params}, batch['ecg'])
        test_labels.append(np.array(label))
        test_logits.append(np.array(logits))
    test_labels = np.concatenate(test_labels)
    test_logits = np.concatenate(test_logits)
    test_metrics = {
        "test_accuracy": accuracy_score(y_true=test_labels, y_pred=test_logits.argmax(axis=-1)),
        "test_f1": f1_score(y_true=test_labels, y_pred=test_logits.argmax(axis=-1)),
        "test_auc": roc_auc_score(y_true=get_one_hot(test_labels, test_logits.shape[-1]),
                                  y_score=nn.activation.softmax(test_logits, axis=-1)),
    }
    return state, metrics_history, test_metrics



def add_metrics_to_dict(train_metrics, test_metrics, exp_data, key):
    if key in exp_data:
        exp_data[key]["train"] = {k: v + [float(train_metrics[k][-1]), ] for k, v in exp_data[key][
            "train"].items()}
        exp_data[key]["test"] = {k: v + [test_metrics[k],] for k, v in exp_data[key][
            "test"].items()}
    else:
        exp_data[key] = {
            "train": {k: [float(v[-1])] for k, v in train_metrics.items()},
            "test": {k: [v] for k, v in test_metrics.items()},
    }
    return exp_data


def clt_conf_int(samples):
    return f'{np.mean(samples):.2f} +/- {np.std(samples) * 1.96 / (len(samples)**.5):.2f}'


def exp_data_clt(exp_data):
    clt_exp_data = {}
    for k in exp_data:
        clt_exp_data[k] = {"train": {k1: clt_conf_int(v1) for k1, v1 in exp_data[k]["train"].items()},
                           "test": {k1.replace("test_", ""): clt_conf_int(v1) for k1, v1 in exp_data[k]["test"].items()}
                           }
    return clt_exp_data





def load_sssd(config_path='SSSD_ECG/src/sssd/config/config_SSSD_ECG.json',
              ckpt_path = '/mnt/data/gabriel/ecg_inpainting/models/alcaraz_250/sssd_sex_label_cond/ch256_T1000_betaT0.02/',
              test_ecg=None):
    with open(config_path) as f:
        data = f.read()
    config = json.loads(data)
    print(config)
    train_config = config["train_config"]  # training parameters
    trainset_config = config["trainset_config"]  # to load trainset
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config, eta=.01)  # dictionary of all diffusion hyperparameters
    model_config = config['wavenet_config']
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config).cuda().eval()
    print_size(net)
    ckpt_iter = 'max'

    # This is needed unfortunately, to initialize omega and Z params of the SSSD with the same batch size as training.
    net((torch.ones(4, 9, 2500).cuda(), torch.ones(4, 134).cuda(), torch.ones(4, 1).cuda())).cpu()

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except Exception as e:
        raise e
        print('No valid model found, continuing anyways seems to work')

    if test_ecg is not None:
        print("Get trainloader")
        index, train_loader = get_trainloader(4, Path("/mnt/data/gabriel/physionet.org/alcaraz_250"), False)

        for audio, label in train_loader:
            audio = torch.index_select(audio, 2, index).float().cuda().permute(0, 2, 1)
            label = label.float().cuda()
            break

        print("Start denoising evaluationg")
        with torch.no_grad():
            diffusion_steps = (torch.tensor([100, 250, 500, 750])).cuda().reshape(-1, 1,
                                                                                  1).long()  # randomly sample diffusion steps from 1~T
            alpha = diffusion_hyperparams['Alpha_bar'][diffusion_steps]
            noised_audio = torch.sqrt(alpha) * audio[0:1].repeat(4, 1, 1) + torch.sqrt(
                1 - alpha) * torch.randn_like(audio)
            epsilon_theta = net((noised_audio, label[0:1].repeat(4, 1), diffusion_steps.view(4, 1),))
            pred_x0 = (noised_audio - epsilon_theta * torch.sqrt(1 - alpha)) / torch.sqrt(alpha)
            res = torch.linalg.norm((pred_x0 - audio[0:1]).reshape(4, -1), dim=-1)
            print(list(zip(res.cpu().tolist(), diffusion_steps.flatten().tolist())))
            for p, p_noise in zip(pred_x0, noised_audio):
                plt.plot(audio[0, 1, :250].cpu(), color='red')
                plt.plot(p[1, :250].cpu(), alpha=.3, color='blue')
                plt.plot(p_noise[1, :250].cpu(), alpha=.2, color='green')
                plt.show()
    return net, diffusion_hyperparams



def build_sex_balanced_dataset(cfg):
    N_beats = 1
    engine = create_engine(f'sqlite:///{cfg.db_path}/database.db')
    # First get stats
    categories_to_filter = ["NSR", "SB", "STach", "SA"]
    query_string = '(' + ' or '.join([f"target_classes = '{i}'" for i in categories_to_filter]) + ')'
    training_class = 'Training%'
    query = text(
        "select sex, count(distinct dataset_id) as N from records where partition_attribution like '" + training_class + "' and " + query_string + " and age IS NOT NULL and sex IS NOT NULL group by sex")
    count_table_sex = pd.read_sql(query, engine)
    count_table_sex = count_table_sex.loc[count_table_sex.sex.isin(['Male', 'Female'])]
    n_male = count_table_sex.loc[count_table_sex.sex=='Male', "N"].iloc[0]
    count_table_sex = count_table_sex.assign(n_to_add=(count_table_sex.sex == 'Female')*(4 * (n_male // 5)))

    print(count_table_sex)
    # Generate unbalanced_train_data
    if "normal_dataset.npz" not in os.listdir('data/classifier_enheancement/sex'):
        print("Normal dataset not found")
        dataset = PhysionetECG(database_path=cfg.db_path,
                               categories_to_filter=["NSR", "SB", "STach", "SA"],
                               normalized='global', training_class='Training', all=False,
                               return_beat_id=False)


        counter_female = 0
        counter_male = 0
        ids_to_keep = []
        for id_ in dataset._ids:
            if id_[3] == 'Female':
                counter_female +=1
                ids_to_keep.append(id_)
            elif id_[3] == 'Male':
                counter_male += 1
                ids_to_keep.append(id_)
        dataset._ids = ids_to_keep
        dataset_stats = {"n_male": counter_male, "n_female": counter_female}

        train_dataloader = DataLoader(dataset=dataset,
                                      batch_size=2048,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0)

        normal_dataset_labels, normal_dataset_ecgs, normal_dataset_features = [], [], []
        for i in range(N_beats):
            for N, batch_1 in enumerate(train_dataloader):
                batch_ecg, batch_features = batch_1
                normal_dataset_ecgs.append(batch_ecg)
                normal_dataset_features.append(batch_features[:, 2:])
                normal_dataset_labels.append(batch_features[:, 0])

        normal_dataset_features = np.concatenate(normal_dataset_features)
        normal_dataset_labels = np.concatenate(normal_dataset_labels)
        normal_dataset_ecgs = np.concatenate(normal_dataset_ecgs)

        np.savez_compressed(os.path.abspath(
            'data/classifier_enheancement/sex/normal_dataset.npz'
        ),
            features=normal_dataset_features,
            ecgs=normal_dataset_ecgs,
            label=normal_dataset_labels,
            n_female=counter_female,
            n_male=counter_male)
        del normal_dataset_ecgs
        del normal_dataset_labels
        del normal_dataset_features
    normal_data = np.load('data/classifier_enheancement/sex/normal_dataset.npz', allow_pickle=True)

    stats = {"n_male": int(normal_data["n_male"]), "n_female": int(normal_data["n_female"])}
    print(f"Dataset statistics {stats}")
    n_female_to_generate = stats['n_female']
    features_table = pd.read_sql(text("select age from records where partition_attribution like 'Training%'"),
                                 engine)
    ages_to_gen = features_table['age'].sample(n=n_female_to_generate)
    rrs_to_gen = np.random.uniform(low=400, high=1000, size=(n_female_to_generate,))

    features = np.stack((np.zeros((n_female_to_generate,)),
                         np.ones((n_female_to_generate,)),
                         (rrs_to_gen - 125) / 125,
                         (ages_to_gen - 50) / 50),
                        dtype=np.float16,
                        axis=-1)

    if ('female_data_wgan.npz' not in os.listdir('data/classifier_enheancement/sex')):
        print("Female data not found for WGAN")
        gen = Gen_ac_wgan_gp_1d(
            noise_dim=Z_DIM,
            generator_n_features=64,  # Gen data n channels
            conditional_features_dim=N_CLASS,  # N classes for embedding
            sequence_length=DATA_TIME,  # Length for channel
            sequence_n_channels=DATA_N_CHANNELS,  # n_channels_seq
            embedding_dim=64).cuda()
        gen.load_state_dict(torch.load(os.path.join(cfg.wgan.path, "generator_trained_cl.pt")))
        gen.eval()
        print("WGAN model loaded")
        print(f"Start generation of {n_female_to_generate}")

        generated_ecgs_to_rebalance_wgan = []
        print("Start generating female data")
        for _ in range(N_beats):
            feats = torch.from_numpy(features).cuda().float()
            noise = torch.randn((feats.shape[0], Z_DIM, 1)).cuda()
            with torch.no_grad():
                ecg_wgan = gen(noise, feats).cpu().permute(0, 2, 1)

            generated_ecgs_to_rebalance_wgan.append(ecg_wgan.numpy())

        generated_ecgs_to_rebalance_wgan = np.concatenate(generated_ecgs_to_rebalance_wgan)

        np.savez_compressed("data/classifier_enheancement/sex/female_data_wgan.npz",
                            features=np.concatenate([features[:, 2:], ] * N_beats),
                            ecgs=generated_ecgs_to_rebalance_wgan,
                            label=np.concatenate([features[:, 0], ] * N_beats))
    else:
        print("female data found for WGAN")
    if ('female_data_diff.npz' not in os.listdir('data/classifier_enheancement/sex')):
        #or ('balanced_data_wgan.npz' not in os.listdir('data/classifier_enheancement/sex'))):
        print("female data not found for beat DIFF")

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
        noise_fun = lambda t: jnp.clip(net_config.diffusion.noise_coeff * t, sigma_min, sigma_max)
        generate_fun = jax.jit(partial(heun_sampler,
                                       sigma_min=sigma_min,
                                       sigma_max=sigma_max,
                                       N=100,
                                       p=p,
                                       scale_fun=scaling_fun,
                                       noise_fun=noise_fun,
                                       scale_fun_grad=grad(scaling_fun),
                                       noise_fun_grad=grad(noise_fun),
                                       train_state=train_state))


        print('Networks Loaded')
        generated_ecgs_to_rebalance_diff = []
        print("Start generating balacing data")

        for key in random.split(random.PRNGKey(0), N_beats):
            initial_samples = random.normal(key=key,
                                            shape=(features.shape[0], 176, 9)) * sigma_max
            generated_ecgs_to_rebalance_diff.append(jax.device_put(generate_fun(initial_samples,
                                                                                class_labels=features).block_until_ready(),
                                                                   jax.devices("cpu")[0]))
            print('Beat Done')

        generated_ecgs_to_rebalance_diff = np.concatenate(generated_ecgs_to_rebalance_diff)
        print(generated_ecgs_to_rebalance_diff.shape)
        np.savez_compressed("data/classifier_enheancement/sex/female_data_diff.npz",
                            features=np.concatenate([features[:, 2:], ] * N_beats),
                            ecgs=generated_ecgs_to_rebalance_diff,
                            label=np.concatenate([features[:, 0], ] * N_beats))
    else:
        print("Female data found for Beat DIFF")

    if ('female_data_sssd.npz' not in os.listdir('data/classifier_enheancement/sex')):
        #or ('balanced_data_wgan.npz' not in os.listdir('data/classifier_enheancement/sex'))):
        print("Female data not found for SSSD")

        #Loading SSSD
        net, diffusion_hyperparams = load_sssd(test_ecg=None)

        print('Network loaded')

        max_batch_size_sssd = 32
        generated_ecgs_to_rebalance_sssd = []
        generated_ecgs_features = []
        print("Start generating balacing data")
        print("Start generating balacing data")


        feats = torch.from_numpy(features).float()
        with torch.no_grad():
            for batch_feats in tqdm(feats[:(feats.shape[0]//max_batch_size_sssd)*max_batch_size_sssd].reshape(-1, max_batch_size_sssd, *feats.shape[1:]), desc='SSSD gen'):
                ecgs_sssd, feats_sssd = generate_sssd(batch_feats, net, diffusion_hyperparams)
                generated_ecgs_to_rebalance_sssd.append(ecgs_sssd)
                generated_ecgs_features.append(feats_sssd)

        generated_ecgs_to_rebalance_sssd = np.concatenate(generated_ecgs_to_rebalance_sssd)
        features_ecgs_to_rebalance_sssd = np.concatenate(generated_ecgs_features)
        np.savez_compressed("data/classifier_enheancement/sex/female_data_sssd.npz",
                            features=features_ecgs_to_rebalance_sssd,
                            ecgs=generated_ecgs_to_rebalance_sssd,
                            label=features_ecgs_to_rebalance_sssd[:, 0])
        del generated_ecgs_to_rebalance_sssd
        del features_ecgs_to_rebalance_sssd
    else:
        print("female data found for SSSD")

    balanced_data_diff = np.load('data/classifier_enheancement/sex/female_data_diff.npz')
    balanced_data_wgan = np.load('data/classifier_enheancement/sex/female_data_wgan.npz')
    balanced_data_sssd = np.load('data/classifier_enheancement/sex/female_data_sssd.npz')

    return balanced_data_diff, balanced_data_wgan, balanced_data_sssd, normal_data

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    # cfg = compose(config_name="config") # for loading cfg in python console
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    females_diff, females_wgan, females_sssd, all_dataset = build_sex_balanced_dataset(cfg)
    LR = 5e-3
    N_STEPS = 20_000
    BATCH_SIZE = 4096
    TEST_PERIOD = 500

    N_REP = 5
    exp_data = {}
    for i in range(N_REP):
        all_data_state, all_data_metrics, all_data_test_metrics = train(
            cfg,
            {"ecgs": all_dataset["ecgs"],
             "label": all_dataset["label"].astype(int)},
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            key=random.PRNGKey(10),
            num_steps_per_epoch=TEST_PERIOD,
            model_name='All data')

        female_indices = np.where(all_dataset["label"] == 0)[0]
        male_indices = np.where(all_dataset["label"] == 1)[0]
        males_from_dataset_ecgs = all_dataset["ecgs"][male_indices]
        exp_data = add_metrics_to_dict(train_metrics=all_data_metrics, test_metrics=all_data_test_metrics, exp_data=exp_data, key="All data")
        for ptg in (0.1, 0.05):
            indices_to_keep = female_indices[np.random.randint(0, len(female_indices), math.floor(len(female_indices)*ptg))]
            females_from_dataset_to_keep = all_dataset["ecgs"][indices_to_keep]
            n_from_gen_model = len(female_indices) - len(indices_to_keep)

            # Generate unbalaced classiifer
            unbalanced_state, unbalanced_metrics, unbalanced_test_metrics = train(
                cfg,
                {"ecgs": np.concatenate((males_from_dataset_ecgs, females_from_dataset_to_keep), axis=0),
                 "label": np.concatenate((np.ones(males_from_dataset_ecgs.shape[0]), np.zeros(females_from_dataset_to_keep.shape[0])), axis=0).astype(int)},
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                learning_rate=LR,
                key=random.PRNGKey(10),
                num_steps_per_epoch=TEST_PERIOD,
                model_name=f'Unbalanced data {ptg}')

            exp_data = add_metrics_to_dict(train_metrics=unbalanced_metrics, test_metrics=unbalanced_test_metrics,
                                           exp_data=exp_data, key=f"Unbalanced {ptg}")
            print(json.dumps({k: v["test"] for k, v in exp_data_clt(exp_data).items()},
            sort_keys = True, indent=4))

            for name, females_gen in [("WGAN", females_wgan), ("Beat DIFF", females_diff), ("SSSD", females_sssd)]:
                indices_from_gen = np.random.randint(0, len(females_gen["ecgs"]), n_from_gen_model)
                balanced_dataset = {
                    "ecgs": np.concatenate((males_from_dataset_ecgs,
                                            females_from_dataset_to_keep,
                                            females_gen["ecgs"][indices_from_gen]), axis=0),
                    "label": np.concatenate((np.ones(males_from_dataset_ecgs.shape[0]), np.zeros(females_from_dataset_to_keep.shape[0] + n_from_gen_model)), axis=0).astype(int)
                }
                # removing nans that can occur in the generation
                index_to_keep = ~jnp.isnan(balanced_dataset["ecgs"]).any(axis=(1, 2))
                balanced_dataset = {k: v[index_to_keep] for k, v in balanced_dataset.items()}

                balanced_gen_state, balanced_gen_metrics, balanced_gen_test_metrics = train(cfg,
                                                                                            balanced_dataset,
                                                                                            n_steps=N_STEPS,
                                                                                            batch_size=BATCH_SIZE,
                                                                                            learning_rate=LR,
                                                                                            key=random.PRNGKey(10),
                                                                                            num_steps_per_epoch=TEST_PERIOD,
                                                                                            model_name=name + f' {ptg}')
                exp_data = add_metrics_to_dict(train_metrics=balanced_gen_metrics, test_metrics=balanced_gen_test_metrics,
                                               exp_data=exp_data, key=f"{name} {ptg}")

                print(json.dumps({k: v["test"] for k, v in exp_data_clt(exp_data).items()},
                                 sort_keys=True, indent=4))
        with open('data/classifier_enheancement/sex/test_metrics_bkp.json', 'w') as file:
            json.dump(exp_data,
                      file)




if __name__ == '__main__':
    metrics = main()

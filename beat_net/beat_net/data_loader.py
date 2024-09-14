## Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py

import os
import time
import timeit
import glob
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from typing import Tuple

import wfdb
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import List
from jax import random, jit
from sqlalchemy import create_engine, text
from scipy.signal import resample

def read_memmap_from_position(filename,
                              shape,
                              beat_index):
    fp = np.memmap(filename=filename,
                   mode='r',
                   dtype='float16',
                   shape=shape,
                   offset=0)
    data = np.asarray(fp[:, beat_index])
    return data


class PhysionetNPZ(Dataset):
    def __init__(self, file_paths: List[str], training_class='Train', normalized: str = 'no',
                 return_beat_id: bool = False, all_limb=False,
                 noise_path='', segment_length=750,
                 **kwargs):
        self.file_paths = file_paths
        # self.all_npz = [np.load(file_p) for file_p in tqdm(self.file_paths, total=len(self.file_paths), desc='data loading')]
        self.normalized = normalized
        self.return_beat_id = return_beat_id
        self.all_limb = all_limb
        if segment_length < 750:
            self.start, self.end = 200, 712
        else:
            self.start, self.end = 0, segment_length
        self.noise_path = noise_path
        if len(self.noise_path) > 0:
            record = wfdb.rdrecord(self.noise_path).__dict__
            noise = record['p_signal']
            f_s_prev = record['fs']
            new_s = int(round(noise.shape[0] / f_s_prev * 250))
            self.noise_stress = resample(noise, num=new_s)
            if 'Train' in training_class or 'CV' in training_class:
                self.alphas = np.random.randint(low=20, high=200, size=len(self.file_paths)) / 100
            else:
                self.alphas = np.ones(len(self.file_paths))

    def _get_ecg(self, n: int) -> Tuple:
        npz = np.load(self.file_paths[n])
        ecg, rr = npz['data'], npz['rr']
        n_beats = ecg.shape[1]
        beat_index = np.random.randint(low=0,
                                       high=n_beats,
                                       size=(1,))[0]
        ecg, rr = ecg[:, beat_index, self.start:self.end], rr[:, beat_index]
        sex, age = npz['sex'], npz['age']
        start = 0
        signal_length = ecg.shape[1]

        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)

        if len(self.noise_path) > 0:

            start_ind = np.random.choice(a=len(self.noise_stress)-signal_length,
                                         size=(ecg.shape[0],),replace=True)
            leads = np.arange(ecg.shape[0])
            np.random.shuffle(leads)
            # corruption -> 2 leads une seule lead ?
            noise = np.stack([self.noise_stress[start_ind[k]:start_ind[k] + signal_length,0]
                              if k < ecg.shape[0] /2
                              else self.noise_stress[start_ind[k]:start_ind[k] + signal_length,1]
                              for k in range(ecg.shape[0])])
            beat_max_value = np.max(ecg, axis=1) - np.min(ecg, axis=1)
            noise_max_value = np.max(noise, axis=1) - np.min(noise, axis=1)
            Ase = noise_max_value / beat_max_value[leads]
            denoised_ecg = ecg.copy().T
            noise = noise/Ase[:, np.newaxis]*self.alphas[n]
            ecg[leads] += noise

        if self.normalized == 'per_lead':
            max_val = np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        elif self.normalized == 'global':
            max_val = np.abs(ecg).max().clip(1e-4, 10)
        else:
            max_val = 1.
        ecg = ecg / max_val
        ecg = ecg.T

        features = np.array([sex == 'Male',
                             sex=='Female',
                             (rr[0] - 125) / 125,
                             # (rr[1] - 125) / 125,
                             (age - 50) / 50], dtype=np.float16)

        if self.return_beat_id:
            return ecg, features, start, n
        if len(self.noise_path) > 0:
            return denoised_ecg, features, ecg, noise, leads
        return ecg, features

    def __getitem__(self, n: int) -> Tuple:
        return self._get_ecg(n)

    def __len__(self) -> int:
        return len(self.file_paths)


class PhysionetECG(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, database_path: str, categories_to_filter: List[str], segment_length: int = 176,
                 estimate_noise_std: bool = False, normalized: str = 'no', training_class: str = 'Training',
                 all: bool = False, return_beat_id: bool = False, all_limb=False, noise_path='', long_term=False, **kwargs):
        engine = create_engine(f'sqlite:///{database_path}/database.db')
        with engine.connect() as conn:
            query_string = '(' + ' or '.join([f"target_classes = '{i}'" for i in categories_to_filter]) + ')'
            if all:
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, n_beats, sex, age from records where " + query_string + " and age IS NOT NULL and sex IS NOT NULL group by dataset_name, dataset_id, n_beats")).fetchall()
            else:
                training_class = 'Training%' if training_class == 'Training' else training_class
                ids = conn.execute(text(
                    "select dataset_name, dataset_id, n_beats, sex, age from records where partition_attribution like '" + training_class + "' and " + query_string + " and age IS NOT NULL and sex IS NOT NULL group by dataset_name, dataset_id, n_beats")).fetchall()
        self.long_term = long_term
        self.centered = 'center' in database_path
        if self.long_term and not self.centered:
            ignored_ids = ['E00841', 'E00906', 'E00915', 'E00947', 'E08668', 'E08686',
                           'E08744', 'E08767', 'E08822', 'E08876', 'E09904', 'E09923',
                           'E09965', 'E09980', 'E09983', 'E09988', 'E10002',  # training
                           'E08366', 'E08717']  # cv
            ids = [id for id in ids if not (id.dataset_id in ignored_ids and id.dataset_name == 'georgia')]

        self.database_path = database_path
        self._ids = ids
        self.signal_length = segment_length
        self.estimate_noise_std = estimate_noise_std
        self.normalized = normalized
        self.return_beat_id = return_beat_id
        self.all_limb = all_limb
        self.noise_path = noise_path
        if len(self.noise_path) > 0:
            record = wfdb.rdrecord(self.noise_path).__dict__
            noise = record['p_signal']
            f_s_prev = record['fs']
            new_s = int(round(noise.shape[0]/f_s_prev*250))
            self.noise_stress = resample(noise, num=new_s)
            if ('Train' in training_class or 'CV' in training_class)and 'NSR' in categories_to_filter:
                self.alphas = np.random.randint(low=20, high=200, size=len(self._ids)) / 100
            else:
                self.alphas = np.ones(len(self._ids))


    def _get_beat(self, n:int) -> Tuple:
        row = self._ids[n]
        beat_index = np.random.randint(low=0,
                                       high=row.n_beats,
                                       size=(1,))[0]
        path_to_raw_data = os.path.join(self.database_path, row.dataset_name, f'{row.dataset_id}_beats.npy')
        ecg = read_memmap_from_position(filename=path_to_raw_data,
                                        shape=(12, row.n_beats, self.signal_length),
                                        beat_index=beat_index)
        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)
        if len(self.noise_path) > 0:
            # windowing
            #start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
            #                             size=(2,),replace=True)
            #leads = np.random.choice(a=ecg.shape[0], size=(2,), replace=False)
            start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
                                         size=(ecg.shape[0],),replace=True)
            leads = np.arange(ecg.shape[0])
            np.random.shuffle(leads)
            # corruption -> 2 leads une seule lead ?
            noise = np.stack([self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,0]
                              if k < ecg.shape[0] /2
                              else self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,1]
                              for k in range(ecg.shape[0])])
            beat_max_value = np.max(ecg, axis=1) - np.min(ecg, axis=1)
            noise_max_value = np.max(noise, axis=1) - np.min(noise, axis=1)
            Ase = noise_max_value / beat_max_value[leads]
            denoised_ecg = ecg.copy().T
            noise = noise/Ase[:, np.newaxis]*self.alphas[n]
            ecg[leads] += noise
        if self.normalized == 'per_lead':
            max_val = np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        elif self.normalized == 'global':
            max_val = np.abs(ecg).max().clip(1e-4, 10)
        else:
            max_val = 1.
        ecg = ecg / max_val
        ecg = ecg.T
        fp = np.memmap(filename=os.path.join(self.database_path, row.dataset_name, f'{row.dataset_id}_rr.npy'),
                       mode='r',
                       dtype='int',
                       shape=(row.n_beats,),
                       offset=0)
        rr = np.asarray(fp[beat_index])

        features = np.array([row.sex == 'Male',
                             row.sex=='Female',
                             (rr - 125) / 125,
                             (row.age - 50) / 50], dtype=np.float16)
        if self.estimate_noise_std:
            smooth_ecg = torch.nn.functional.avg_pool1d(input=ecg[None, :, :],
                                                        kernel_size=7, stride=1, padding=3)[0]
            err = smooth_ecg - ecg
            err[:, 75:125] = 0
            noise_level = np.std(err)
        else:
            smooth_ecg = ecg
            noise_level = 0
        if self.return_beat_id:
            return ecg, features, beat_index, n
        if len(self.noise_path) > 0:
            return denoised_ecg, features, ecg, noise, leads
        return ecg, features


    def _get_ecg(self, n:int) -> Tuple:
        row = self._ids[n]
        path_to_raw_data = os.path.join(self.database_path, row.dataset_name, f'{row.dataset_id}.npz')

        npz = np.load(path_to_raw_data)
        ecg = npz['data']
        T = ecg.shape[1]
        start = 0
        if T > self.signal_length:
            start = torch.randint(0, T-self.signal_length, size=(1,))[0]
            end = start + self.signal_length
            ecg = ecg[:, start:end]

        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)
        if len(self.noise_path) > 0:
            # windowing
            #start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
            #                             size=(2,),replace=True)
            #leads = np.random.choice(a=ecg.shape[0], size=(2,), replace=False)
            start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
                                         size=(ecg.shape[0],),replace=True)
            leads = np.arange(ecg.shape[0])
            np.random.shuffle(leads)
            # corruption -> 2 leads une seule lead ?
            noise = np.stack([self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,0]
                              if k < ecg.shape[0] /2
                              else self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,1]
                              for k in range(ecg.shape[0])])
            beat_max_value = np.max(ecg, axis=1) - np.min(ecg, axis=1)
            noise_max_value = np.max(noise, axis=1) - np.min(noise, axis=1)
            Ase = noise_max_value / beat_max_value[leads]
            denoised_ecg = ecg.copy().T
            noise = noise/Ase[:, np.newaxis]*self.alphas[n]
            ecg[leads] += noise
        if self.normalized == 'per_lead':
            max_val = np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        elif self.normalized == 'global':
            max_val = np.abs(ecg).max().clip(1e-4, 10)
        else:
            max_val = 1.
        ecg = ecg / max_val
        ecg = ecg.T

        features = np.array([row.sex == 'Male',
                             row.sex=='Female',
                             (row.age - 50) / 50], dtype=np.float16)
        if self.return_beat_id:
            return ecg, features, start, n
        if len(self.noise_path) > 0:
            return denoised_ecg, features, ecg, noise, leads
        return ecg, features


    def _get_centered_ecg(self, n:int) -> Tuple:
        row = self._ids[n]
        beat_index = np.random.randint(low=0,
                                       high=row.n_beats,
                                       size=(1,))[0]
        path_to_raw_data = os.path.join(self.database_path, row.dataset_name, f'{row.dataset_id}.npz')

        npz = np.load(path_to_raw_data)
        ecg, rr = npz['data'][:, beat_index], npz['rr'][:, beat_index]
        T = ecg.shape[1]
        start = 0
        if T > self.signal_length:
            start = torch.randint(0, T-self.signal_length, size=(1,))[0]
            end = start + self.signal_length
            ecg = ecg[:, start:end]

        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)
        if len(self.noise_path) > 0:
            # windowing
            #start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
            #                             size=(2,),replace=True)
            #leads = np.random.choice(a=ecg.shape[0], size=(2,), replace=False)
            start_ind = np.random.choice(a=len(self.noise_stress)-self.signal_length,
                                         size=(ecg.shape[0],),replace=True)
            leads = np.arange(ecg.shape[0])
            np.random.shuffle(leads)
            # corruption -> 2 leads une seule lead ?
            noise = np.stack([self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,0]
                              if k < ecg.shape[0] /2
                              else self.noise_stress[start_ind[k]:start_ind[k] + self.signal_length,1]
                              for k in range(ecg.shape[0])])
            beat_max_value = np.max(ecg, axis=1) - np.min(ecg, axis=1)
            noise_max_value = np.max(noise, axis=1) - np.min(noise, axis=1)
            Ase = noise_max_value / beat_max_value[leads]
            denoised_ecg = ecg.copy().T
            noise = noise/Ase[:, np.newaxis]*self.alphas[n]
            ecg[leads] += noise
        if self.normalized == 'per_lead':
            max_val = np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        elif self.normalized == 'global':
            max_val = np.abs(ecg).max().clip(1e-4, 10)
        else:
            max_val = 1.
        ecg = ecg / max_val
        ecg = ecg.T

        features = np.array([row.sex == 'Male',
                             row.sex=='Female',
                             (rr[0] - 125) / 125,
                             (rr[1] - 125) / 125,
                             (row.age - 50) / 50], dtype=np.float16)

        if self.return_beat_id:
            return ecg, features, start, n
        if len(self.noise_path) > 0:
            return denoised_ecg, features, ecg, noise, leads
        return ecg, features

    def __getitem__(self, n: int) -> Tuple:
        if self.long_term:
            if self.centered:
                return self._get_centered_ecg(n)
            return self._get_ecg(n)
        return self._get_beat(n)

    def __len__(self) -> int:
        return len(self._ids)

class LQT_ECG(Dataset):
    """s
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, database_path: str, segment_length: int = 176,
                 normalized: str = 'global',
                 return_beat_id: bool = False, all_limb=False):
        self.database_path = database_path
        self.annotations = pd.read_csv(os.path.join(database_path, 'annotations_clean.csv'))
        self.annotations = self.annotations[self.annotations['status'].isin(['POS', 'NEG'])]
        self.signal_length = segment_length
        self.normalized = normalized
        self.return_beat_id = return_beat_id
        self.all_limb = all_limb

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        row = self.annotations.iloc[n]
        file_name = row.id
        path_to_raw_data = os.path.join(self.database_path, file_name + '_beats.npy')
        n_beats = row.n_beats
        beat_index = np.random.randint(low=0,
                                       high=n_beats,
                                       size=(1,))[0]

        ecg = read_memmap_from_position(filename=path_to_raw_data,
                                        shape=(12, n_beats, self.signal_length),
                                        beat_index=beat_index)
        if not self.all_limb:
            ecg = np.concatenate([ecg[:3], ecg[6:]])
        ecg = ecg.astype(np.float16)
        if self.normalized == 'per_lead':
            ecg = ecg / np.abs(ecg).max(axis=-1).clip(1e-4, 10)[..., None]
        if self.normalized == 'global':
            ecg = ecg / np.abs(ecg).max().clip(1e-4, 10)
        ecg = ecg.T
        fp = np.memmap(filename=os.path.join(self.database_path, f'{file_name}_rr.npy'),
                       mode='r',
                       dtype='int',
                       shape=(n_beats,),
                       offset=0)
        rr = np.asarray(fp[beat_index])

        features = np.array([row.sex == 'M',
                             row.sex =='F',
                             (rr - 125) / 125,
                             (row.age - 50) / 50], dtype=np.float16)
        label = int(row.status == 'POS')
        if self.return_beat_id:
            return ecg, features, beat_index, n, label
        return ecg, features, label

    def __len__(self) -> int:
        return self.annotations.shape[0]




class PhysionetRerun(PhysionetECG):

    def __init__(self, npz_path: str):
        npz = np.load(npz_path)
        self._ids = np.stack([npz['dataset'], npz['patient_id'], npz['n_beat']], axis=1)
        self._beat_id = npz['beat_id']
        self._target_ecg = npz['target_ecg']

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        ecg = self._target_ecg[n]
        beat_index = self._beat_id[n]
        return ecg, beat_index, n


def numpy_collate(batch, n_devices=0):
    if isinstance(batch[0], np.ndarray):
        batch = np.stack(batch)
        if n_devices == 0:
            return batch
        else:
            return batch.reshape(n_devices, -1, *batch.shape[1:])
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples, n_devices=n_devices) for samples in transposed]

    else:
        if n_devices > 0:
            return np.array(batch).reshape(n_devices, -1)
        else:
            return np.array(batch)


if __name__ == '__main__':
    dataset = PhysionetECG(database_path='/mnt/data/gabriel/physionet.org/beats_db_more_meta',
                           categories_to_filter=["NSR", "SB", "STach", "SA"], normalized=True, training_class=True,
                           estimate_std=False)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=512,
                            shuffle=True,
                            num_workers=10,
                            drop_last=True,
                            collate_fn=numpy_collate)
    fun = jit(lambda x, y: random.split(x, y), static_argnums=1)
    res = fun(random.PRNGKey(0), len(dataloader))
    t1 = time.time()
    for i, _ in enumerate(zip(dataloader, res)):
        pass
    t2 = time.time()
    print(i, t2 - t1)
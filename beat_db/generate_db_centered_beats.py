import sys
import json
sys.path.append('.')
sys.path.append('..')
from beat_db.physionet_tools import find_challenge_files, get_filtered_ecg_from_header, get_labels, get_recording_id, get_age, get_sex
import numpy as np
from beat_db.db_orm import Record

import os
from functools import partial
from time import time
from multiprocessing.pool import Pool
from random import random
import pandas as pd
from scipy.signal import savgol_filter, iirnotch, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from neurokit2 import ecg_findpeaks, ecg_clean
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session


def extract_qrs_offsets(signals, freq, method='pca', n_components=3):
    if method == 'pca':
        pca = Pipeline([('Scaler', StandardScaler()),
                        ('PCA', PCA(n_components=n_components))])

        pca_signals = pca.fit_transform(signals)
        reference_track = savgol_filter(pca_signals[:, 0], window_length=15, polyorder=3)
    else:
        reference_track = ecg_clean(signals[:, 0], sampling_rate=freq)
    r_peaks = ecg_findpeaks(reference_track, sampling_rate=freq)['ECG_R_Peaks']

    return r_peaks

def get_beat_ecg_from_headers(x, Dx_map):
    leads_in_order = ('I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    header, recording_file = x
    lab = get_labels_from_header(header, Dx_map)
    # if lab in ['NSR', "SB", "STach", "SA", 'MI', 'LQT', 'LAD', 'LAE']:
    if 'NSR' in lab or 'nsr' in lab:
        ecg = get_filtered_ecg_from_header(header, leads_in_order, recording_file, final_freq=250)
        beats, rr = get_beats_from_ecg(ecg)
        id_in_dataset = get_recording_id(header)
        age = get_age(header)
        sex = get_sex(header)
        return beats, id_in_dataset, lab, sex, age, rr
    return None


def get_beats_from_ecg(ecg, method='pca', n_components=3):
    r_peaks = extract_qrs_offsets(ecg.T, freq=250, method=method, n_components=n_components)
    is_r_peak_valid = np.array(
        [(r - 48 > 0 and r + 128 < ecg.shape[-1]) and (np.std(ecg[:, np.arange(r - 20, r + 20)], axis=1).min() > .05)
         for r in r_peaks])
    kept_r_peak = np.array(
        [(r - 375 > 0 and r + 375 < ecg.shape[-1]) and (np.std(ecg[:, np.arange(r - 20, r + 20)], axis=1).min() > .05)
         for r in r_peaks])
    if is_r_peak_valid.mean() < 0.01:
        raise ValueError(f"Not enough valid beats {is_r_peak_valid.mean()}")
    kept_r_peak_inds = np.where(kept_r_peak)[0]
    beats = ecg[:, [np.arange(peak - 375, peak + 375) for peak in r_peaks[kept_r_peak]]]
    prev_rr = np.array([r1 - r2 if r1 - r2 < 500 else None for r1, r2 in zip(r_peaks[kept_r_peak_inds], r_peaks[kept_r_peak_inds-1])])
    next_rr = np.array([r1 - r2 if r1 - r2 < 500 else None for r1, r2 in zip(r_peaks[1+kept_r_peak_inds], r_peaks[kept_r_peak_inds])])
    # beats = beats[:, 1:]
    return beats, np.stack([prev_rr, next_rr])


def get_labels_from_header(header, Dx_map):
    labels = get_labels(header)
    class_labels = []
    for label in labels:
        try:
            current_class_label = Dx_map.loc[Dx_map['SNOMED CT Code'] == int(label), 'Abbreviation'].iloc[0]
        except:
            current_class_label = np.nan
            pass
        if current_class_label == current_class_label:
            class_labels.append(current_class_label)
    class_labels = sorted(list(set(class_labels)))
    return class_labels


def attribute_dataset_to_id():
    is_training = random() < .8
    new_random = random()
    if is_training:
        if new_random < .1:
            if new_random < .01:
                label = 'Training_1%'
            else:
                label = 'Training_10%'
        else:
            label = 'Training_100%'
    else:
        if new_random < .5:
            label = 'Test'
        else:
            label = 'CV'
    return label


def save_numpy(data, name, n_beats, length, n_channels, dataset_id, data_destination):
    fp = np.memmap(os.path.join(data_destination, f'{dataset_id}_{name}.npy'),
                   dtype='float16',
                   mode='w+',
                   shape=(n_channels, n_beats, length))
    fp[:] = data[:length]
    fp.flush()


def save_numpy_compact(x, name, n_channels, data_destination):
    data, length, dataset_id = x
    return save_numpy(data, name, length, n_channels, dataset_id, data_destination)


def save_lead_fields(x, data_path):
    lf, dataset_id = x
    np.save(os.path.join(data_path, f'{dataset_id}_lead_fields.npy'),
            lf)


def generate_data_for_db(inc, path):
    header, recording_file = inc
    try:
        beats, id_in_dataset, labels, sex, age, rr = get_beat_ecg_from_headers((header, recording_file), Dx_map=Dx_map)

        metadata = {
            "sex": sex,
            "age": age,
            'dataset_id': id_in_dataset,
            'dataset_name': database,
            'n_beats': beats.shape[1],
            'partition_attribution': attribute_dataset_to_id(),
            'target_classes': '-'.join(labels)
        }
        np.savez(os.path.join(path, metadata['dataset_id']+'.npz'),
                 data=beats,
                 rr=rr,
                 # name='ecg',
                 # n_channels=12,
                 # data_destination=path,
                 # n_beats=metadata['n_beats'],
                 # length=beats.shape[-1],
                 #dataset_id=metadata['dataset_id'])
                 **metadata)
        # beats_normalized = beats - np.median(np.concatenate((beats[:, :, :10], beats[:, :, -10:]), axis=-1), axis=-1)[:, :,
        #                            None]
        # beats_normalized = beats_normalized / np.clip(np.abs(beats_normalized[:, :, 75:125]).max(axis=-1)[:, :, None], 1e-2, 10)
        # metadata = {
        #     "sex": sex,
        #     "age": age,
        #     'dataset_id': id_in_dataset,
        #     'dataset_name': database,
        #     'n_beats': beats_normalized.shape[1],
        #     'partition_attribution': attribute_dataset_to_id(),
        #     'target_classes': '-'.join(labels)
        # }
        # save_numpy(data=beats,
        #            name='beats',
        #            n_channels=12,
        #            data_destination=path,
        #            n_beats=metadata['n_beats'],
        #            length=beats.shape[-1],
        #            dataset_id=metadata['dataset_id'])
        # save_numpy(data=beats_normalized,
        #            name='beats_normalized',
        #            n_channels=12,
        #            data_destination=path,
        #            n_beats=metadata['n_beats'],
        #            length=beats.shape[-1],
        #            dataset_id=metadata['dataset_id'])
        # fp = np.memmap(os.path.join(path, f'{id_in_dataset}_rr.npy'),
        #                dtype='int',
        #                mode='w+',
        #                shape=(len(rr),))
        # fp[:] = np.array([r if r else -1 for r in rr])
        # fp.flush()
        return metadata
    except Exception as e:
        # print(e)
        return None


if __name__ == '__main__':
    data_source = '/mnt/data/gabriel/physionet.org/files/challenge-2021/1.0.3/training'  # sys.argv[1]
    data_destination = '/mnt/data/lisa/physionet.org/centered_beats'  # sys.argv[2]
    n_procs = 50  # int(sys.argv[3])
    available_databases = [f for f in os.listdir(data_source) if os.path.isdir(os.path.join(data_source, f))]
    print(available_databases)
    n_batches = 1000
    try:
        Dx_map = pd.read_csv('Dx_map_physionet.csv')
    except:
        Dx_map = pd.read_csv('../Dx_map_physionet.csv')
    engine = create_engine(f'sqlite:///{data_destination}/database.db')
    with Session(engine) as session:
        for database in available_databases:
            print(f"Doing {database}")
            database_path = os.path.join(data_source, database)
            if not os.path.exists(os.path.join(data_destination, database)):
                os.makedirs(os.path.join(data_destination, database))

            subfolders = [f for f in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, f))]
            path = os.path.join(data_destination, database)
            generate_data_fun = partial(generate_data_for_db, path=path)
            for folder in subfolders:
                if n_procs > 0:
                    with Pool(n_procs) as p:
                        batch_meta = p.map(func=generate_data_fun, iterable=find_challenge_files(os.path.join(database_path, folder)))
                else:
                    batch_meta = [generate_data_fun(f) for f in find_challenge_files(os.path.join(database_path, folder))]
                batch_meta_clean = [b for b in batch_meta if b is not None]
                print(f"Inserting {len(batch_meta_clean)} / {len(batch_meta)} in {database}")
                new_records = session.execute(
                    insert(Record),
                    batch_meta_clean
                )
                session.commit()

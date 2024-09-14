from dataclasses import dataclass, field
import numpy as np
import h5py
from typing import Union

import torch
from pyedflib import highlevel
import pickle
import os

from scipy.signal import butter, sosfiltfilt, resample


@dataclass
class BioSemiData:
    """Create a patient with original signals from BDF file and markers """
    original_signal: np.ndarray = field(repr=False)
    respiratory_signal: np.ndarray = field(repr=False)
    processed_signal: np.ndarray = field(repr=False)
    processed_list: list() = field(repr=False)
    bad_leads: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    markers: list() = field(repr=False)
    bdf_file: str
    ID: str
    WCT: np.ndarray = field(repr=False)
    index_WCT: np.ndarray = field(repr=False)

    def reload_bdf(self, path='.'):
        """ Function to relaod the orignal signals from the BDF file"""
        patient_path = self.ID
        BDF_file = self.bdf_file
        try:
            signals, signal_headers, header = highlevel.read_edf(os.path.join(path,patient_path,'Raw Data',BDF_file))
            self.original_signal = signals
        except:
            print(f'{path}/{patient_path}/Raw Data/{BDF_file} not found')
        return


def save_to_disk(biosemi, filename):
    """Save Biosemi patient from disk (pickle file) """
    original_signal = biosemi.original_signal
    biosemi.original_signal = []
    error_flg = 0;
    if len(filename) == 0:
        filename = biosemi.ID + '.pickle'

    try:
        with open(filename, 'wb') as f1:
            pickle.dump([biosemi.original_signal,
            biosemi.respiratory_signal,
            biosemi.processed_signal,
            biosemi.processed_list,
            biosemi.bad_leads,
            biosemi.labels,
            biosemi.markers,
            biosemi.bdf_file,
            biosemi.ID,
            biosemi.WCT,
            biosemi.index_WCT], f1, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
        error_flg = 1;
    biosemi.original_signal = original_signal
    return error_flg


def load_from_disk(filename) -> BioSemiData:
    """ Load Biosemi patient from disk (pickle file)"""
    try:
        with open(filename, 'rb') as f1:
            original_signal,\
            respiratory_signal,\
            processed_signal,\
            processed_list,\
            bad_leads,\
            labels,\
            markers,\
            bdf_file,\
            ID,\
            WCT,\
            index_WCT = pickle.load(f1)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)
        return
    return BioSemiData(original_signal=original_signal,\
            respiratory_signal=respiratory_signal,\
            processed_signal=processed_signal,\
            processed_list=processed_list,\
            bad_leads=bad_leads,\
            labels=labels,\
            markers=markers,\
            bdf_file=bdf_file,\
            ID=ID,\
            WCT=WCT,\
            index_WCT=index_WCT)


def calculate_12Leads_ECG(struct,bad_leads=[],ecg_length='all'):

    if isinstance(struct,BioSemiData):
        signal = struct.processed_signal
        bad_leads = struct.bad_leads
        flg_struct = True
    else:
        signal = struct
        bad_leads = np.zeros(signal.shape[0])
        flg_struct = False
    if signal.shape[0]>256:
        index_electrodes =  [(257, 10,11,21,22,),(256, 117,118,106,107),(258, 115,116,126,127, 105, 94),         # WCT_R, WCT_L, WCT_F
                             (43,54,44,55),(65,75,66,76),(77,88,89,78,87),              # V1, V2, V3
                             (99,100),(110,111),(121,122)] # V4, V5, V6
    else:
        index_electrodes = [(129, 10,11,21,22,),(128, 117,118,106,107),(130, 115,116,126,127,94),               # WCT_R, WCT_L, WCT_F
                            (43, 54, 44, 55), (65, 75, 66, 76), (77, 88, 89, 78, 87),  # V1, V2, V3
                            (99, 100), (110, 111), (121, 122)]  # V4, V5, V6
    index_12_label = ['WCT_R', 'WCT_L','WCT_F','V1','V2','V3','V4','V5','V6']
    lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    sample_rate = 2048.
    electrode_12D = []
    sig_12D = []
    bad_leads_12D = []
    lead_index = 0
    for n_electrodes in index_electrodes:
        for i_electrode in n_electrodes:
            if bad_leads[i_electrode] == 0:
                electrode_12D.append(i_electrode)
                # sig_12D.append(signal[i_electrode])
                break
            if i_electrode == n_electrodes[-1]:
                bad_leads_12D.append(index_12_label[lead_index])
        lead_index += 1
    if len(electrode_12D) !=9:
        return None, bad_leads_12D
    if ecg_length == 'all':
        start, stop = 0, signal.shape[1]
    else:
        start, stop = int((len(signal[0]) / 2) - ecg_length * .5 * sample_rate), int(
            (len(signal[0]) / 2) + ecg_length * .5 * sample_rate)
    ecg = np.zeros((12, stop - start))
    ecg[0, :] = signal[electrode_12D[1], start:stop] - signal[electrode_12D[0], start:stop]  # I
    ecg[1, :] = signal[electrode_12D[2], start:stop] - signal[electrode_12D[0], start:stop]  # II
    ecg[2, :] = signal[electrode_12D[2], start:stop] - signal[electrode_12D[1], start:stop]  # III
    ecg[3, :] = -0.5 * (ecg[0, :] + ecg[1, :])  # aVR
    ecg[4, :] = -0.5 * (ecg[2, :] - ecg[0, :])  # aVL
    ecg[5, :] = 0.5 * (ecg[1, :] + ecg[2, :])  # aVF
    ecg[6, :] = signal[electrode_12D[3], start:stop]  # V1
    ecg[7, :] = signal[electrode_12D[4], start:stop]  # V2
    ecg[8, :] = signal[electrode_12D[5], start:stop]  # V3
    ecg[9, :] = signal[electrode_12D[6], start:stop]  # V4
    ecg[10, :] = signal[electrode_12D[7], start:stop]  # V5
    ecg[11, :] = signal[electrode_12D[8], start:stop]  # V6
    ecg = ecg / 1000.
    return ecg,lead_name


def load_ecg(data, freq=2000):
    ecgs, leads = calculate_12Leads_ECG(struct=data,
                                        bad_leads=data.bad_leads)
    ecgs = np.stack([torch.from_numpy(ecgs[leads.index(l), :2048 * 4]) for l in
                     ['aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']],
                    axis=0)
    high_pass_filter = butter(N=5, Wn=.6, btype='highpass', fs=freq, output='sos')
    left_pad = np.stack([np.linspace(0, o, 1000) for o in ecgs[:, 0]], axis=0)
    right_pad = np.stack([np.linspace(o, 0, 1000) for o in ecgs[:, -1]], axis=0)
    padded_original_recording = np.concatenate((left_pad, ecgs, right_pad), axis=1)
    filtered_recording = sosfiltfilt(high_pass_filter,
                                     padded_original_recording,
                                     axis=-1)[:, 1000:-1000]
    if freq != 500:
        recording = resample(filtered_recording,
                             500,
                             int(freq),
                             axis=1)
    return torch.from_numpy(recording).type(torch.FloatTensor)

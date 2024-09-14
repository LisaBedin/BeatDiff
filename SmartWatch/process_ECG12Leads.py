import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import pydicom
from pydicom.waveforms import multiplex_array
from beat_db.physionet_tools import filter_ecg
from beat_db.generate_db import get_beats_from_ecg
from dtaidistance import dtw
from SmartWatch.utils import pdf_to_image, extract_values, get_signals


def extract_12leads(dicom_p):
    ds = pydicom.dcmread(dicom_p)

    # Récupérer les séquences de formes d'onde
    waveforms = ds.WaveformSequence
    multiplex = waveforms[0]

    freq = float(multiplex.SamplingFrequency)
    age = 2024 - int(ds.data_element('PatientBirthDate')[:4])
    sex = ds.data_element('PatientSex')
    print('Sampling frequency', freq)
    print('Age', age)
    print('Name', ds.data_element('PatientName'))
    print('Sex', ds.data_element('PatientSex'))

    # Récupérer les données brutes de forme d'onde
    original_recording = get_signals(multiplex)

    channel_names = [channel.ChannelSourceSequence[0].CodeMeaning
                     for channel in multiplex.ChannelDefinitionSequence]

    final_freq = 250
    # === change the frequency of the signal === #
    recording = filter_ecg(final_freq, freq, original_recording)

    beats, rr = get_beats_from_ecg(recording, method='pca', n_components=1)

    return beats, rr, age, sex, channel_names


def extract_Watch(watch_p):
    op_list = pdf_to_image(watch_p)
    ecg_data = extract_values(op_list)  # / 1000
    final_freq, freq = 250, 512
    recording = filter_ecg(final_freq, freq, ecg_data[np.newaxis])

    return get_beats_from_ecg(recording, method='pca', n_components=1)


def get_ind_peaks(signal, first_lead=False):
    if first_lead:
        inds_peaks = np.stack([
            np.argmin(signal, axis=1),
            np.argmax(signal, axis=1),
            # np.min(signal, axis=1),
            # np.max(signal, axis=1)
        ], axis=1)
    else:
        inds_peaks = np.stack([
            np.argmax(signal[:, :80], axis=1),
            np.argmax(signal[:, 80:], axis=1),
            # np.max(signal[:, :80], axis=1),
            # np.max(signal[:, 80:], axis=1)
        ], axis=1)
    return inds_peaks


def plot_ecg(ecg_distributions, conditioning_ecg=None, alpha=0.05, color_posterior='blue', color_target='red'):
    fig, ax = plt.subplots(1, 1, figsize=(1, 4))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if len(ecg_distributions.shape) == 3:
        for ecg in ecg_distributions:
            for i, track in enumerate(ecg):
                ax.plot(track - i*1.3, c=color_posterior, alpha=alpha, linewidth=.7, rasterized=True)
    else:
        for i, track in enumerate(ecg_distributions):
            ax.plot(track - i * 1.3, c=color_posterior, linewidth=.7, rasterized=True)
    for i, track in enumerate(conditioning_ecg):
        ax.plot(track - i*1.3, c=color_target, linewidth=.7, rasterized=True)
    ax.set_ylim(-11, 1.1)
    ax.set_xlim(0, 175)
    return fig


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    patient_ids = ['M21', 'F23', 'F24', 'F28']
    dicom_path = os.path.join(cfg.main_path, 'raw_data/ECG12D')
    watch_path = os.path.join(cfg.main_path, 'raw_data/Watch')
    results_path = os.path.join(cfg.main_path, 'processed_data')
    os.makedirs(results_path, exist_ok=True)

    for patient_n in patient_ids:
        # ===== extract 12-leads ECGs ===== #
        dicom_lst = glob.glob(os.path.join(dicom_path, f'ECG12D_{patient_n}_test*.dcm'))
        all_beats12, all_rr12 = [], []
        for dc_p in dicom_lst:
            beats12, rr12, age, sex, channel_names = extract_12leads(dc_p)
            print(channel_names)
            all_beats12.append(beats12[:, 1:-1])
            all_rr12.append(rr12[1:-1])
        all_beats12 = np.concatenate(all_beats12, axis=1)
        #ind_peaks12 = np.concatenate([np.argmin(all_beats12, axis=)])
        all_rr12 = np.concatenate(all_rr12, axis=0)

        # ===== extract Watch ECGs ===== #
        watch_dic = {'leadI_wrist': [],
                     'leadII_flank': [],
                     'leadII_knee': [],
                     'leadII_ankle': [],
                     'leadIII_flank': [],
                     'leadIII_knee': [],
                     'leadIII_ankle': [],
                     'rr_leadI_wrist': [],
                     'rr_leadII_flank': [],
                     'rr_leadII_knee': [],
                     'rr_leadII_ankle': [],
                     'rr_leadIII_flank': [],
                     'rr_leadIII_knee': [],
                     'rr_leadIII_ankle': []}
        watch_lst = glob.glob(os.path.join(watch_path, f'Watch_{patient_n}_*.pdf'))
        for w_p in watch_lst:
            if os.path.exists(w_p):
                suffix = '_'.join(w_p.split('/')[-1].split('_')[2:4])
                beats, rr = extract_Watch(w_p)
                watch_dic[suffix].append(beats[:, 2:-2])
                watch_dic['rr_'+suffix].append(rr[2:-2])
        for watch_k, watch_v in watch_dic.items():
            if len(watch_v) > 0:
                watch_dic[watch_k] = np.concatenate(watch_v, axis=1-int('rr' in watch_k))

        # ==== visualize closests leads ==== #
        nW = watch_dic['leadI_wrist'].shape[1]
        nE = all_beats12.shape[1]
        timeseries = np.concatenate([watch_dic['leadI_wrist'][0], all_beats12[0]])

        #dist = dtw.distance_matrix(timeseries, block=((0, nW), (nW, nW+nE)))[:nW, nW:nW+nE]
        #inds = dist.argmin(axis=1)
        #dist = ((all_beats12[0, :, np.newaxis]-watch_dic['leadI_wrist'])**2).sum(axis=-1)
        #inds = dist.argmin(axis=0)
        dist = (all_rr12[:, np.newaxis]-watch_dic['rr_leadI_wrist'][np.newaxis])**2
        inds = dist.argmin(axis=0)

        inds_peaks12 = get_ind_peaks(all_beats12[0], first_lead=True)
        inds_peaksWI = get_ind_peaks(watch_dic['leadI_wrist'][0], first_lead=True)
        inds_peaksWII = get_ind_peaks(watch_dic['leadII_knee'][0], first_lead=False)
        inds_peaksWIII = get_ind_peaks(watch_dic['leadIII_knee'][0], first_lead=False)

        dist = ((inds_peaks12[:, np.newaxis]-inds_peaksWI[np.newaxis])**2).sum(axis=-1)
        inds12 = dist.argmin(axis=0)
        dist = ((inds_peaksWII[:, np.newaxis]-inds_peaksWI[np.newaxis])**2).sum(axis=-1)
        indsII = dist.argmin(axis=0)
        dist = ((inds_peaksWIII[:, np.newaxis]-inds_peaksWI[np.newaxis])**2).sum(axis=-1)
        indsIII = dist.argmin(axis=0)

        ecgW = np.concatenate([
            watch_dic['leadI_wrist'],
            watch_dic['leadII_flank'][:, indsII],
            watch_dic['leadIII_flank'][:, indsIII]])
        # ecgW /= np.absolute(ecgW).max(axis=-1)[:, :, np.newaxis]
        ecg12 = all_beats12[:, inds12]
        # ecg12 /= np.absolute(ecg12).max(axis=-1)[:, :, np.newaxis]

        plot_i = int(np.random.randint(low=0, high=ecgW.shape[1], size=(1,))[0])
        # shift = min(ecg12[0, plot_i].argmin() - ecgW[0, plot_i].argmin(), ecg12[0, plot_i].argmax() - ecgW[0, plot_i].argmax())
        plot_ecg(ecgW[:, plot_i], np.concatenate([ecg12[:3, plot_i], ecg12[6:, plot_i]]))  #, shift:])
        plt.show()

        # for iW, i12 in enumerate(inds):
        #     beatW = watch_dic['leadI_wrist'][0, iW] / np.absolute(watch_dic['leadI_wrist'][0, iW]).max()
        #     beatE = all_beats12[0, i12] / np.absolute(all_beats12[0, i12]).max()
        #     # warping_path = np.array(dtw.warping_path_prob(beatW, beatE, 1./len(beatE)))
        #     # new_beatW = np.zeros_like(beatW)
        #     # counts = np.zeros_like(beatW)
        #     # for s, t in warping_path:
        #     #     new_beatW[t] += beatW[s]
        #     #     counts[t] += 1
        #     # new_beatW = new_beatW / counts
        #     plt.title(patient_n)
        #     plt.plot(beatW, label=f'Watch I {iW}')
        #     # plt.plot(new_beatW, label='wrapped watch')
        #     plt.plot(beatE, label=f'12leads {i12}')
        #     plt.legend()
        #     plt.show()

        np.savez(os.path.join(results_path, f'{patient_n}.npz'),
                 beats12=all_beats12,
                 rr12=all_rr12,
                 **watch_dic,
                 sex=patient_n[:1],
                 age=int(patient_n[1:]))
    print('ok')


if __name__ == '__main__':
    main()

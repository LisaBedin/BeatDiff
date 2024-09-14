import os
from SmartWatch.utils import pdf_to_image, extract_values, get_signals
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import numpy as np
import pydicom
import pandas as pd

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Chemin vers vos fichiers
    patient_id = 'F28'
    # PDF = '/mnt/Reseau/Signal/lisa/Remi_Montre/PDF_AppleWatch/Lisa_poignet.pdf'  # Camille_poignet.pdf'
    # CSV = '/mnt/Reseau/Signal/lisa/Remi_Montre/CSV_AppleWatch_new/Lisa_poignet.csv'  # Camille_poignet.csv'
    PDF = os.path.join(cfg.main_path, f'raw_data/Watch/Watch_{patient_id}_leadII_knee_sample1.pdf')


    # Extract data from pdf
    op_list = pdf_to_image(PDF)
    ecg_data = extract_values(op_list)
    plt.plot(ecg_data[1200:3000])
    plt.show()

    # === 12 Lead ECG to verify that we
    dicom_path = os.path.join(cfg.main_path,
                              f'raw_data/ECG12D/ECG12D_{patient_id}_test1.dcm')
    ds = pydicom.dcmread(dicom_path)
    waveforms = ds.WaveformSequence
    multiplex = waveforms[0]
    signals = get_signals(multiplex)
    plt.plot(signals[1][1200:3000])
    plt.show()

    # raw = process_data(CSV)
    raw = ecg_data   # [186:] / 1000


    # raw = pd.read_csv(CSV)

    """
    # Save the extracted data to a CSV file
    save_to_csv(ecg_data, CSV)
    print(f"ECG data saved to {CSV}")
    """


    plt.figure(figsize=(30, 5))

    sampling_frequency = 511.844
    time = np.arange(len(raw)) / sampling_frequency

    plt.plot(time, raw, label='Raw')
    #plt.plot(time, ecg_data, label = 'PDF')

    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)

    plt.legend(fontsize=20)

    plt.xticks(fontsize=20)  # Adjust font size for x-axis tick labels
    plt.yticks(fontsize=20)  # Adjust font size for y-axis tick labels

    # Add x-label
    plt.xlabel('Seconds')
    plt.show()



if __name__ == '__main__':
    main()
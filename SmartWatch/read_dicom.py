import os
import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import pydicom
from pydicom.waveforms import multiplex_array
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from utils import get_signals

# Parcourir la séquence de définition des canaux pour obtenir les noms des canaux et les unités
def names_units(multiplex, channel_names):
    for ii, channel in enumerate(multiplex.ChannelDefinitionSequence):
        source = channel.ChannelSourceSequence[0].CodeMeaning
        # units = 'unitless'
        # if 'ChannelSensitivity' in channel:  # Type 1C, peut être absent
        # units = channel.ChannelSensitivityUnitsSequence[0].CodeMeaning
        channel_names.append(source)
    return (channel_names)


# Diviser le temps de chaque signal
def shorting(signals, factor):
    len_col_raw = int(signals.shape[0] / factor)
    raw2 = signals[:len_col_raw, :]
    return (raw2)


# Afficher les signaux de tous les canaux
def display(signals, channel_names):
    fig, axs = plt.subplots(len(channel_names), figsize=(15, 15), sharex=True)
    fig.suptitle("ECG Channels")

    for ii, ax in enumerate(axs):
        ax.plot(signals[:, ii])
        ax.set_ylabel(channel_names[ii])
    plt.xlabel("Time")
    plt.show()


# Fonction pour tracer les signaux ECG
def plot_12Leads_ECG(twelveD_ecg, names, results_path, title='ECG', fig=None, ax=None):
    # Configuration initiale
    sample_rate = 500  # Taux d'échantillonnage en Hz
    columns, rows, row_height = 4, 3, 500  # Nombre de colonnes, de lignes et hauteur des lignes
    lead_order = list(range(0, len(twelveD_ecg)))  # Ordre des pistes
    secs = len(twelveD_ecg[0]) / sample_rate  # Durée en secondes de chaque signal ECG
    display_factor, line_width = 1, 1  # Facteurs de mise à l'échelle et épaisseur des lignes

    # Création de la figure et de l'axe
    fig, ax = plt.subplots(figsize=(14, 40))

    # Définition des limites des axes
    x_min, x_max = 0, np.ceil(
        columns * secs)  # Limites de l'axe X : début à 0, fin à columns * secs (arrondi vers le haut)
    y_min, y_max = 0, np.ceil(row_height * 4 + row_height)  # Limites de l'axe Y pour ajuster l'affichage des lignes ECG

    # Couleurs des grilles
    color_major, color_minor, color_line = (1, 0, 0), (1, 0.7, 0.7), (0, 0, 0.7)

    # Configuration des ticks majeurs et mineurs
    ax.set_xticks(np.arange(x_min, x_max, 0.2))  # Ticks majeurs vertical
    ax.set_yticks(np.arange(y_min, y_max, 100))  # Ticks majeurs horizontal
    ax.minorticks_on()  # Activer les ticks mineurs
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # 5 ticks mineurs par tick majeur
    ax.grid(which='major', linestyle='-', linewidth=0.4 * display_factor, color=color_major)
    ax.grid(which='minor', linestyle='-', linewidth=0.4 * display_factor, color=color_minor)

    # Définir les limites des axes
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.format_coord = lambda x, y: ''  # Supprimer les coordonnées formatées
    fig.suptitle(title)  # Titre de la figure
    ax.set_xticklabels([])  # Supprimer les labels des ticks sur l'axe X
    ax.set_yticklabels([])  # Supprimer les labels des ticks sur l'axe Y

    # Configuration de la taille de la figure
    largeur, hauteur = 1404, 756

    # Boucle pour tracer chaque signal ECG
    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < len(lead_order)):
                y_offset = y_max - row_height * (
                            1.5 * i + 1)  # Décalage vertical pour chaque piste
                # Complètement du bidouillage parce que c'est plus jolie)

                if i == 3:
                    x_offset = c * secs  # Décalage horizontal pour la colonne
                else:
                    x_offset = (0.05 + secs) * c

                t_lead = lead_order[c * rows + i]  # Sélectionner la piste actuelle

                if (c > 0):
                    if i != 3:
                        # Tracer une ligne de séparation entre les signaux
                        line_height = 1  # Correspond à 1 mV (1 cm)
                        ax.plot([x_offset - 0.025, x_offset - 0.025],
                                [y_offset - line_height / 2, y_offset + line_height / 2],
                                linewidth=line_width * display_factor, color='black')

                # if all(twelveD_ecg[t_lead]):   # Vérifier si tous les points de la piste sont non nuls
                # color_line = (0, 0, 0.7)  # Couleur de la ligne pour les pistes valides

                # else:
                # color_line = 'grey'     # Couleur de la ligne pour les pistes non valides

                step = 1.0 / sample_rate  # Intervalle de temps entre les échantillons

                ax.text(x_offset + 0.07, y_offset + row_height * 0.4, names[t_lead],  # Ajouter le nom de la piste
                        fontsize=9 * display_factor,
                        color='black')

                if i == 3:
                    y_offset = y_offset  # Maintenir le décalage vertical

                # Tracer le signal ECG

                ax.plot(np.arange(0, len(twelveD_ecg[t_lead]) * step, step) + x_offset,
                        twelveD_ecg[t_lead] + y_offset,
                        linewidth=0.4,
                        color='black')

    # Ajouter les annotations de calibration et filtre
    calibration = "25mm/s  10mm/mV"
    # filtre = f"{len(twelveD_ecg[0])} ~ 60Hz"
    plt.text(0.35, 20, f'{calibration}       ')

    # Définir la limite de l'axe X finale
    ax.set_xlim([0, len(twelveD_ecg[t_lead]) * step + x_offset])

    plt.gcf().set_size_inches(largeur / 100, hauteur / 100)

    # Sauvegarder la figure en PNG avec haute résolution
    plt.savefig(results_path, dpi=1000, bbox_inches='tight')


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    dicom_path = os.path.join(cfg.main_path, 'raw_data/ECG12D')
    results_path = os.path.join(dicom_path, 'Visualization')
    os.makedirs(results_path, exist_ok=True)
    all_dicom_files = glob.glob(os.path.join(dicom_path, '*.dcm'))
    for dicom_file in all_dicom_files:
        file_name = dicom_file.split('/')[-1].split('.')[0]
        # ds = pydicom.dcmread(dicom_file)
        ds = pydicom.read_file(dicom_file)

        # Récupérer les séquences de formes d'onde
        waveforms = ds.WaveformSequence
        sequence_item = waveforms[0]

        print('Sampling frequency', float(sequence_item.SamplingFrequency))
        print('Age', 2024 - int(ds.data_element('PatientBirthDate')[:4]))
        print('Name', ds.data_element('PatientName'))
        print('Sex', ds.data_element('PatientSex'))

        channel_definitions = sequence_item.ChannelDefinitionSequence
        wavewform_data = sequence_item.WaveformData
        channels_no = sequence_item.NumberOfWaveformChannels
        samples = sequence_item.NumberOfWaveformSamples
        sampling_frequency = sequence_item.SamplingFrequency
        duration = samples / sampling_frequency
        #  mm_s = width / duration

        # Récupérer les données brutes de forme d'onde
        #raw = multiplex_array(ds, 0, as_raw=True)
        raw = get_signals(sequence_item)

        # Initialiser une liste pour stocker les noms des canaux
        channel_names = []
        names = names_units(sequence_item, channel_names)

        # Diviser la longueur du signal
        raw = shorting(raw.T, 4.5)

        # Afficher les 12 dérivations de l'ECG
        plot_12Leads_ECG(raw.T, names, os.path.join(results_path, file_name + '_new.pdf'))
        plt.show()


if __name__ == '__main__':
    main()

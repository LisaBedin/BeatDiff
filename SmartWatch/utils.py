# IMPORT
import pandas as pd
import PyPDF2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy.signal import butter, lfilter
import struct


# CONVERT PDF INTO A LIST OF GRAPHIC OPERATIONS
def pdf_to_image(pdf_path):
    op_list = []
    pdf = PyPDF2.PdfReader(open(pdf_path, 'rb'))
    page = pdf.pages[0]  # Assuming you want to process the first page

    # Extract page dimensions
    page_width = int(page.mediabox.width)
    page_height = int(page.mediabox.height)
    print('page_width', page_width)
    print('page_height', page_height)

    # Create a blank image
    image = Image.new('RGB', (page_width, page_height), 'white')
    draw = ImageDraw.Draw(image)

    def visitor_svg(op, args, cm, tm):
        op_list.append((op, args, cm, tm))

    page.extract_text(visitor_operand_before=visitor_svg)

    return op_list


# EXTRACT VALUES FROM PDF
def extract_values(op_list):
    cs_correct, sc_correct = False, False
    # Flag to indicate if 'cs' (color space) and 'sc' (color encoding) are correct

    last_move = 0
    y_list, y_lists = [], []
    skiped_ops = []
    # Iterate over the list of operations
    for idx, (op, args, cm, tm) in tqdm(enumerate(op_list)):

        if op == b'm':  # Move to
            if cs_correct and sc_correct:
                if idx - last_move != 132 and len(
                        y_list):  # Skip lines that are not part of the three long lines of red points
                    y_lists.append(y_list)
                    y_list = []
                last_move = idx

        elif op == b'l':  # Line to
            end_x, end_y = args
            # print(end_x, end_y)
            if cs_correct and sc_correct:
                y_list.append(float(end_y))

        elif op == b'sc' or op == b'cs' or op == b'SC' or op == b'CS':
            if op == b'cs' or op == b'CS':
                cs_correct = '/Cs3' in args
            if op == b'sc' or op == b'SC':
                sc_correct = all([f'{float(x):.2}' in ['0.8', '0.039', '0.13'] for x in args if
                                  isinstance(x, PyPDF2.generic._base.FloatObject)])

        else:
            skiped_ops.append(op)

    y_lists.append(y_list)

    inter_val_1 = (y_lists[0][-1] + y_lists[1][0]) / 2
    inter_val_2 = (y_lists[1][-1] + y_lists[2][0]) / 2

    y_combined = np.concatenate(
        [y_lists[0], [inter_val_1], y_lists[1], [inter_val_2], y_lists[2]])  # Combined list of y values
    # y = y_combined / -28.3465 + 2  # Convert to mV and apply a 1.5 mV offset
    # y = y - np.mean(y)
    y = -2.54 * y_combined / 72  # 72 dpi and 2.54 to convert inches to cm
    return y - np.mean(y)


# SAVE DATA AS CSV
def save_to_csv(data, output_path):
    df = pd.DataFrame(data, columns=['ECG'])
    df.to_csv(output_path, index=False)


def process_data(fname):
    df_raw_original = pd.read_csv(fname, delimiter='.', decimal=',', skiprows=13, header=None)
    df_raw_original = df_raw_original.iloc[:, 0].values
    # print("Length of 'Raw:", len(df_raw_original))
    df_raw_new = df_raw_original[186:]
    df_raw_new = ((df_raw_new / 1000))
    return df_raw_new



def butter_lowpass(highcut, sampfreq, order):
    """Supporting function.

    Prepare some data and actually call the scipy butter function.
    """

    nyquist_freq = .5 * sampfreq
    high = highcut / nyquist_freq
    num, denom = butter(order, high, btype='lowpass')
    return num, denom


def butter_lowpass_filter(data, highcut, sampfreq, order):
    """Apply the Butterworth lowpass filter to the DICOM waveform.

    @param data: the waveform data.
    @param highcut: the frequencies from which apply the cut.
    @param sampfreq: the sampling frequency.
    @param order: the filter order.
    """

    num, denom = butter_lowpass(highcut, sampfreq, order=order)
    return lfilter(num, denom, data)

def get_signals(sequence_item):
    """
    Retrieve the signals from the DICOM WaveformData object.

    sequence_item := dicom.dataset.FileDataset.WaveformData[n]

    @return: a list of signals.
    @rtype: C{list}
    """
    channel_definitions = sequence_item.ChannelDefinitionSequence
    wavewform_data = sequence_item.WaveformData
    channels_no = sequence_item.NumberOfWaveformChannels
    samples = sequence_item.NumberOfWaveformSamples
    sampling_frequency = sequence_item.SamplingFrequency
    duration = samples / sampling_frequency

    factor = np.zeros(channels_no) + 1
    baseln = np.zeros(channels_no)
    units = []
    for idx in range(channels_no):
        definition = channel_definitions[idx]

        assert (definition.WaveformBitsStored == 16)

        if definition.get('ChannelSensitivity'):
            factor[idx] = (
                float(definition.ChannelSensitivity) *
                float(definition.ChannelSensitivityCorrectionFactor)
            )

        if definition.get('ChannelBaseline'):
            baseln[idx] = float(definition.get('ChannelBaseline'))

        units.append(
            definition.ChannelSensitivityUnitsSequence[0].CodeValue
        )

    unpack_fmt = '<%dh' % (len(wavewform_data) / 2)
    unpacked_waveform_data = struct.unpack(unpack_fmt, wavewform_data)
    signals = np.asarray(
        unpacked_waveform_data,
        dtype=np.float32).reshape(
        samples,
        channels_no).transpose()

    for channel in range(channels_no):
        signals[channel] = (
            (signals[channel] + baseln[channel]) * factor[channel]
        )

    high = 40.0

    # conversion factor to obtain millivolts values
    millivolts = {'uV': 1000.0, 'mV': 1.0}

    for i, signal in enumerate(signals):
        signals[i] = (butter_lowpass_filter(
            np.asarray(signal),
            high,
            sampling_frequency,
            order=2
        ) / millivolts[units[i]])
        # signals[i] /= millivolts[units[i]]

    return signals - np.mean(signals, axis=1)[:, np.newaxis]
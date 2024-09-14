import glob
import os
import csv
import array
import xmltodict

import numpy as np


def main():
    data_path = '/mnt/data/lisa/LQTS_data'
    all_path = glob.glob(os.path.join(data_path, '*.xml'))
    with open(all_path[0], 'rb') as xml:
        ECG = xmltodict.parse(xml.read().decode('utf8'))

    for k, val in ECG['AnnotatedECG'].items():
        if type(val) == dict:
            print(k, val.keys())
        else:
            print(k, val)

    ecg_cpts = ECG['AnnotatedECG']['component']['series']['component']['sequenceSet']['component']
    ecg, lead_inds = [], []
    lead_inds_dic = {l: i for i, l in enumerate(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])}
    for lead_dic in ecg_cpts:
        lead_name = lead_dic['sequence']['code']['@code']
        if 'LEAD' in lead_name:
            seq = lead_dic['sequence']['value']
            scale = float(seq['scale']['@value'])
            lead = np.array([float(s) for s in seq['digits'].split(' ')])/scale / 1000
            ecg.append(lead)
            lead_inds.append(lead_inds_dic[lead_name.split('_')[-1]])
    ecg = np.stack(ecg)[np.array(lead_inds)]

    # TODO: sanity check with aVL aVR
    # TODO: what is the sampling frequency ?
    print(all_path)
    print('ok')


if __name__ == '__main__':
    main()
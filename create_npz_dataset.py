import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra
from omegaconf import DictConfig, OmegaConf
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate, LQT_ECG
from torch.utils.data import DataLoader
import numpy as np
import wfdb
import _pickle as pickle


def prepare(NSTDBPath='/mnt/data/lisa/physionet.org/files/nstdb/1.0.0/'):
    bw_signals, bw_fields = wfdb.rdsamp(NSTDBPath + 'bw')
    em_signals, em_fields = wfdb.rdsamp(NSTDBPath + 'em')
    ma_signals, ma_fields = wfdb.rdsamp(NSTDBPath + 'ma')

    for key in bw_fields:
        print(key, bw_fields[key])

    for key in em_fields:
        print(key, em_fields[key])

    for key in ma_fields:
        print(key, ma_fields[key])

        # Save Data
    with open('/mnt/data/lisa/Score-based-ECG-Denoising-main/data/NoiseBWL.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump([bw_signals, em_signals, ma_signals], output)
    print('=========================================================')
    print('MIT BIH data noise stress test database (NSTDB) saved as pickle')


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    all_labs_lst = [
        ['NSR', 'SB', 'STach', 'SA'],
        ['MI'],
        ['LQT'],
        ['LQRSV'],
        ['LAD'],
        ['LAE'],
        ['LBBB'],
        ['RBBB'],
    ]

    for lab_lst in all_labs_lst:

        test_dataloader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                          categories_to_filter=lab_lst,
                                                          noise_path=cfg.denoising.noise_path,
                                                          normalized='global', training_class='Test',
                                                          all=('NSR' not in cfg.inpainting.labels),
                                                          return_beat_id=False),
                                     batch_size=10,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=numpy_collate)

if __name__ == '__main__':
    metrics = main()

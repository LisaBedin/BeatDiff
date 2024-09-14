"""
from GitHub: https://github.com/mah533/Augmentation-of-ECG-Training-Dataset-with-CGAN/blob/main/main_wgan_gp_ecg.py
"""
from beat_net.beat_net.data_loader import PhysionetECG, numpy_collate
import datetime
import os
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from beat_wgan.net import Disc_ac_wgan_gp_1d, Gen_ac_wgan_gp_1d, initialize_weights
from beat_wgan.utils import gradient_penalty, grid_plot_save, normalize
from tqdm import tqdm
import ot
import numpy as np

# start_time = datetime.datetime.now()
# print(("\n" + "*" * 50 + "\n\t\tstart time:      {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 50).format(
#     start_time.hour, start_time.minute, start_time.second))
#
# drive = "E:\\"
# myPath_base = os.path.join(drive, "UTSA")
# myPath_dataset = os.path.join(drive, "UTSA\\PycharmProjects_F\\DM_post_processing")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2000
CRITIC_ITERATIONS = 5
DATA_N_CHANNELS = 9
DATA_TIME = 176
IMAGE_SIZE = 64
FEATURES_DISC = 64
FEATURES_GEN = 64
LEARNING_RATE = 1e-4
LAMBDA_GP = 10
Z_DIM = 100
N_CLASS = 4# to study the effect of support set number of samples (shorter train sets)
if __name__ == '__main__':
    myPath_save = '/mnt/data/gabriel/ecg_inpainting/models/wgan/'
    db_path = '/mnt/data/gabriel/physionet.org/beats_db_more_meta_no_zeros'
    test_dataloader = DataLoader(dataset=PhysionetECG(database_path=db_path,
                                                      categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                      normalized=True, training_class='Test', all=False,
                                                      return_beat_id=False),
                                 batch_size=10_000,
                                 shuffle=True,
                                 num_workers=0)

    for batch_test in test_dataloader:
        ecgs_test, batch_test_features = batch_test
        break
    ecgs_test = ecgs_test.permute(0, 2, 1)
    n_min = ecgs_test.shape[0]
    batch_test = ecgs_test.reshape(n_min, -1)

    batch_test = torch.concatenate((batch_test,
                                    batch_test_features),
                                   axis=-1)

    n_max = n_min  # 22589
    train_dataloader = DataLoader(dataset=PhysionetECG(database_path=db_path,
                                                       categories_to_filter=["NSR", "SB", "STach", "SA"],
                                                       normalized=True, training_class='Training', all=False,
                                                       return_beat_id=False),
                                  batch_size=n_max,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0)
    MI_dataloader = DataLoader(dataset=PhysionetECG(database_path=db_path,
                                                    categories_to_filter=["MI"],
                                                    normalized=True, training_class='Test', all=True,
                                                    return_beat_id=False),
                               batch_size=n_max,
                               shuffle=True,
                               drop_last=False,
                               num_workers=0)
    all_train_test_ws = []
    gen = Gen_ac_wgan_gp_1d(
        noise_dim=Z_DIM,
        generator_n_features=64,  # Gen data n channels
        conditional_features_dim=N_CLASS,  # N classes for embedding
        sequence_length=DATA_TIME,  # Length for channel
        sequence_n_channels=DATA_N_CHANNELS,  # n_channels_seq
        embedding_dim=64).to(device)
    gen.load_state_dict(torch.load(os.path.join(myPath_save, "generator_trained_cl.pt")))
    critic = Disc_ac_wgan_gp_1d(
        features_d=FEATURES_DISC,
        num_classes=N_CLASS,
        sequence_n_channels=DATA_N_CHANNELS,
        sequence_length=DATA_TIME
    ).to(device)
    critic.load_state_dict(torch.load(os.path.join(myPath_save, "discriminator_trained_cl.pt")))

    fixed_noise = torch.randn(32, Z_DIM, 1).to(device)

    gen.eval()
    critic.eval()

    ecgs_test.permute(0, 2, 1)
    all_train_gen_ws = []
    for batch_idx, (ecgs, feats) in enumerate(train_dataloader):
        feats = feats.to(device).float()
        ecgs = ecgs.permute(0, 2, 1).to(device).float()
        noise = torch.randn((ecgs.shape[0], Z_DIM, 1)).to(device)
        batch_1 = torch.concatenate((ecgs.reshape(ecgs.shape[0], -1),
                                     feats),
                                    axis=-1)
        with torch.no_grad():
            fake = gen(noise, feats)
            generated_features_test = torch.concatenate((fake.reshape(ecgs.shape[0], -1),
                                                         feats),
                                                        axis=-1)
            M = ot.dist(batch_1, generated_features_test)
            G0 = ot.emd2(torch.ones(n_max) /n_max,
                         torch.ones(n_min) / n_min, M, numItermax=1_000_000)
            all_train_gen_ws.append(G0.item())
        print(np.mean(all_train_gen_ws))
    with torch.no_grad():
        noise = torch.randn((ecgs.shape[0], Z_DIM, 1)).to(device)
        fake = gen(noise, batch_test_features.float().to(device))
        generated_features_test = torch.concatenate((fake.reshape(ecgs.shape[0], -1),
                                                     feats),
                                                    axis=-1)

        M = ot.dist(batch_test.float().to(device), generated_features_test)
        G0 = ot.emd2(torch.ones(n_max) /n_max,
                         torch.ones(n_min) / n_min, M, numItermax=1_000_000)
        test_gen_ws = G0.item()
    print(test_gen_ws)
    for f, r in zip(fake[:10], ecgs_test[:10]):
        fig, ax = plt.subplots(1, 1, figsize=(4, 8))

        for i, ecg in enumerate(f.cpu()):
            ax.plot(ecg - i, color='blue')
        for i, ecg in enumerate(r.cpu()):
            ax.plot(ecg - i, color='red')
        fig.show()
# print("\ntotal elapsed time: {}".format(now - start_time))

"""
from GitHub: https://github.com/mah533/Augmentation-of-ECG-Training-Dataset-with-CGAN/blob/main/main_wgan_gp_ecg.py
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
import sys
sys.path.append('.')
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

# start_time = datetime.datetime.now()
# print(("\n" + "*" * 50 + "\n\t\tstart time:      {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 50).format(
#     start_time.hour, start_time.minute, start_time.second))
#
# drive = "E:\\"
# myPath_base = os.path.join(drive, "UTSA")
# myPath_dataset = os.path.join(drive, "UTSA\\PycharmProjects_F\\DM_post_processing")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters etc.
dry_run = False
if dry_run:
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 60

BATCH_SIZE = 128
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
myPath_save = '/mnt/data/lisa/ecg_results/models/debug'
dataset = PhysionetECG(database_path='/mnt/data/gabriel/physionet.org/beats_db_more_meta_no_zeros',
                       categories_to_filter=["NSR", "SB", "STach", "SA"],
                       normalized='none',
                       training_class='Training',
                       estimate_std=False)
dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=10,
                        drop_last=True)

gen = Gen_ac_wgan_gp_1d(
    noise_dim=Z_DIM,
    generator_n_features=64,  # Gen data n channels
    conditional_features_dim=N_CLASS,  # N classes for embedding
    sequence_length=DATA_TIME,  # Length for channel
    sequence_n_channels=DATA_N_CHANNELS,  # n_channels_seq
    embedding_dim=64).to(device)
critic = Disc_ac_wgan_gp_1d(
    features_d=FEATURES_DISC,
    num_classes=N_CLASS,
    sequence_n_channels=DATA_N_CHANNELS,
    sequence_length=DATA_TIME
).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1).to(device)

gen.train()
critic.train()
pbar = tqdm(range(NUM_EPOCHS))
for epoch in pbar:
    # Target labels not needed! <3 unsupervised
    epoch_loss_D = 0
    epoch_loss_G = 0
    for batch_idx, (real, feats) in enumerate(dataloader):
        real = real.permute(0, 2, 1).to(device).float()
        feats = feats.to(device).float()
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1)).to(device)
            fake = gen(noise, feats)
            critic_real = critic(real, feats).reshape(-1)
            critic_fake = critic(fake, feats).reshape(-1)
            gp = gradient_penalty(critic, real, fake, labels=feats, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # Train Generator: min -E[critic(gen_fake)]
            output = critic(fake, feats).reshape(-1)
            loss_gen = -torch.mean(output)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            epoch_loss_G += loss_gen.item()
            epoch_loss_D += loss_critic.item()

        # Print losses occasionally and print to tensorboard
    pbar.set_description(f"Loss D/G {epoch_loss_D / (CRITIC_ITERATIONS*(batch_idx + 1)):.4} {epoch_loss_G / (CRITIC_ITERATIONS*(batch_idx + 1)):.4}")

#             with torch.no_grad():
#                 fake = gen(noise)
#                 path = os.path.join(myPath_save, 'images')
#                 plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.99, wspace=0.99)
#                 grid_plot_save(n_row=4, n_col=4, signal=fake.squeeze().to("cpu"), path=path,
#                                f_name='wgangp_gb_ep{}_{}.png'.format(epoch, batch_idx))
#
# # save model
    torch.save(gen.state_dict(), os.path.join(myPath_save, "generator_trained_cl.pt"))
    torch.save(critic.state_dict(), os.path.join(myPath_save, "discriminator_trained_cl.pt"))
    #
# now = datetime.datetime.now()
# print("\ntotal elapsed time: {}".format(now - start_time))

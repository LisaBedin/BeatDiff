import math

import torch
import torch.nn as nn


def disc_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False,  # since BN is used, no need to bias
        ),
        nn.InstanceNorm1d(out_channels, affine=True),    # LayerNorm ←→ InstanceNorm
        # nn.LayerNorm(out_channels),
        nn.LeakyReLU(0.2)
    )


class Disc_ac_wgan_gp_1d(nn.Module):
    def __init__(self,
                 features_d,
                 num_classes,
                 sequence_n_channels,
                 sequence_length
                 ):
        super().__init__()
        self.img_size = sequence_length
        self.disc = nn.Sequential(
            # Input: batch_size X n_channels X length
            nn.Conv1d(sequence_n_channels + 1, features_d, kernel_size=4, stride=2, padding=1),  # batch_size x features_d x length/2
            nn.LeakyReLU(0.2),
            # disc_block(in_channels, out_channels, kernel_size, stride, padding)
            disc_block(features_d, features_d * 2, 4, 2, 1),       # batch_size x 2features_d x length/4
            disc_block(features_d * 2, features_d * 4, 4, 2, 1),   # batch_size x 4features_d x length/8
            disc_block(features_d * 4, features_d * 8, 4, 2, 1),   # batch_size X 8features_d X length/16

            # After all _block img output is *x* (Conv1d below makes into *x*)
            nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0, dilation=1),
            # batch_size X 1 X (l / 32  - 1)
            nn.Linear(math.floor(sequence_length //32 - 1), 1)         # 3 for ECG500 and 7 for MIT_BIH
        )
        self.embed = nn.Sequential(
            nn.Linear(num_classes, sequence_length*8),
            nn.InstanceNorm1d(sequence_length*8),
            #nn.LayerNorm(sequence_length * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(sequence_length*8, sequence_length)
        )#nn.Embedding(num_classes, sequence_length)

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        out = self.disc(x)
        return out


def gen_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )


class Gen_ac_wgan_gp_1d(nn.Module):
    def __init__(self,
                 noise_dim,  # Channel dim for noise
                 generator_n_features,  # Gen data n channels
                 conditional_features_dim,  # N classes for embedding
                 sequence_length,  # Length for channel
                 sequence_n_channels,  # n_channels_seq
                 embedding_dim,  #dimension for embedding
                 ):
        super().__init__()
        self.gen = nn.Sequential(
            # input: batch_size, T_gen, n_gen
            gen_block(noise_dim + embedding_dim, generator_n_features * 16, 4, 1, 0),  # batch_size, features_g*16, 4
            gen_block(generator_n_features * 16, generator_n_features * 8, 4, 2, 1),  # 64, features_g * 8, 8
            gen_block(generator_n_features * 8, generator_n_features * 4, 4, 2, 1),  # 64, features_g * 4, 16
            gen_block(generator_n_features * 4, generator_n_features * 2, 4, 2, 1),  # 64, features_g * 2, 32
            nn.ConvTranspose1d(
                generator_n_features * 2, sequence_n_channels, kernel_size=4, stride=2, padding=1
            ),  # 64, features_g, 64
            nn.Linear(64, sequence_length, bias=False),     # 64x1x256 for MIT-BIH and 64x1x140 for ECG5000
            nn.Hardtanh(),  # [-1, 1]
        )
        self.embed = nn.Sequential(
            nn.Linear(conditional_features_dim, embedding_dim * 8),
            nn.InstanceNorm1d(embedding_dim * 8),
            # nn.LayerNorm(embedding_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim * 8, embedding_dim)
        )
        #self.embed = nn.Linear(conditional_features_dim, embedding_dim)#nn.Embedding(conditional_features_dim, embedding_dim)

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1
        temp = self.embed(labels)
        embedding = temp.unsqueeze(2)
        x = torch.cat([x, embedding], dim=1)
        out = self.gen(x)
        return out


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

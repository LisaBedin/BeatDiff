import torch.nn as nn
from pytorch_tcn import TCN

class Discriminator(nn.Module):
    def __init__(self, L=9, ngpu=1, nz=32, nef=32, TCN_opt=False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nef = nef
        self.L = L
        if not TCN_opt:
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x N
                nn.Conv1d(L, nef, 4, 2, 1, bias=False), # bs x nef x 88
                nn.BatchNorm1d(nef),
                nn.LeakyReLU(0.2,True),
                # state size (nef) x 128
                nn.Conv1d(nef, nef * 2, 4, 2, 1, bias=False), # bs x nef * 2 x 44
                nn.BatchNorm1d(nef * 2),
                nn.LeakyReLU(0.2,True),
                # state size. (nef*2) x 64
                nn.Conv1d(nef * 2, nef * 4, 4, 2, 1, bias=False), # bs x nef * 4 x 22
                nn.BatchNorm1d(nef * 4),
                nn.LeakyReLU(0.2,True),
                # state size. (nef*4) x 32
                nn.Conv1d(nef * 4, nef * 8, 4, 2, 0, bias=False),  # bs x nef * 8 * 10
                nn.BatchNorm1d(nef * 8),
                nn.LeakyReLU(0.2,True),
                # state size. (nef*8) x 16
                nn.Conv1d(nef * 8, nef * 16, 4, 2, 0, bias=False), # bs x nef * 16 x 4
                nn.BatchNorm1d(nef * 16),
                nn.LeakyReLU(0.2,True),
                # state size. (nef * 16) x 8
                nn.Conv1d(nef * 16, nz, 4, 1, 0),  # bs x nz x 1
                nn.BatchNorm1d(nz),
                nn.LeakyReLU(0.2, True),
                # state size. (nef * 16) x 8
                nn.Conv1d(nz, 1, 1, 1, 0),  # bs x nz x 1
                nn.Flatten(),
                nn.Sigmoid())
        else:
            self.conv = nn.Sequential(
                # input Beat dimensions L x N => L x N
                TCN(9, [32, 32], kernel_size=9, use_skip_connections=True), # bs x nef x 88
                # state size 32 x N
                nn.MaxPool1d(5),
                # state size 32 x 35
                TCN(32, [16, 16], kernel_size=9, use_skip_connections=True), # bs x nef x 88
                # state size 16 x 35
                nn.MaxPool1d(5),
                # state size 16 x 7
                TCN(16, [8, 8], kernel_size=9, use_skip_connections=True), # bs x nef x 88
                # state size 8 x 7
                nn.MaxPool1d(5),
                # state size 8 x 1
                nn.Conv1d(8, 8, 1, 1, 0),  # bs x nz x 1
                nn.Conv1d(8, 1, 1, 1, 0),  # bs x nz x 1
                # state size 8 x 1
                nn.Flatten(),
                nn.Sigmoid()
            )
        self.conv.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        return self.conv(input)
import torch
import torch.nn as nn


class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding, bias=False)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x

class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding, bias=False)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class InferenceGenerator(nn.Module):
    def __init__(self):
        super(InferenceGenerator, self).__init__()

        '''
        self.all_convs = [
            nn.Sequential(nn.Conv2d(1, 64, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=(2, 4), stride=2, padding=1)),
            nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=2, padding=1)),
        ]
        '''
        self.all_convs = nn.ModuleList([
            DownSampleConv(1, 64, kernel=(4, 4), strides=2, padding=1, batchnorm=False, activation=True),
            DownSampleConv(64, 128, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            DownSampleConv(128, 256, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            DownSampleConv(256, 512, kernel=(2, 4), strides=2, padding=1, batchnorm=True, activation=True),
            DownSampleConv(512, 1024, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),

        ])

        '''
        self.all_deconv = [
            nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=(2, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=2, padding=1)),
        ]
        '''

        self.all_deconv = nn.ModuleList([
            UpSampleConv(1024, 512, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(1024, 256, kernel=(2, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(512, 128, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(256, 64, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(128, 1, kernel=(4, 4), strides=2, padding=1, batchnorm=False, activation=False),

        ])

    def forward(self, x):
        skip_conns = []
        for conv_down in self.all_convs:
            x = conv_down(x)
            skip_conns.append(x)

        out = self.all_deconv[0](skip_conns[-1])
        rev_skip_conns = skip_conns[:-1]
        rev_skip_conns = rev_skip_conns[::-1]
        for skip, deconv in zip(rev_skip_conns, self.all_deconv[1:]):

            out = deconv(torch.cat([out, skip], dim=1))
        return skip_conns[-1], out

class LabelGenerator(nn.Module):
    def __init__(self):
        super(LabelGenerator, self).__init__()
        '''
        self.all_convs = [
            nn.Sequential(nn.Conv2d(1, 64, kernel_size=(4, 4), stride=2, padding=1), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=(2, 4), stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU()),
            nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=2, padding=1), nn.BatchNorm2d(1024), nn.LeakyReLU()),
        ]
        '''
        self.all_convs = nn.ModuleList([
            DownSampleConv(1, 64, kernel=(4, 4), strides=2, padding=1, batchnorm=False, activation=True),
            DownSampleConv(64, 128, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            DownSampleConv(128, 256, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            DownSampleConv(256, 512, kernel=(2, 4), strides=2, padding=1, batchnorm=True, activation=True),
            DownSampleConv(512, 1024, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),

        ])
        '''
        self.all_deconv = [
            nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(2, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=2, padding=1)),
        ]
        '''
        self.all_deconv = nn.ModuleList([
            UpSampleConv(1024, 512, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(512, 256, kernel=(2, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(256, 128, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(128, 64, kernel=(4, 4), strides=2, padding=1, batchnorm=True, activation=True),
            UpSampleConv(64, 1, kernel=(4, 4), strides=2, padding=1, batchnorm=False, activation=False),

        ])

    def forward(self, x):
        for conv_down in self.all_convs:
            x = conv_down(x)
        z = x
        for deconv in self.all_deconv:
            x = deconv(x)
        return z, x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        '''
        self.all_convs = [
            nn.Sequential(nn.Conv2d(2, 32, kernel_size=(1, 64), stride=(1, 2), padding=(0, 31))),
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 32), stride=(1, 2), padding=(0, 15))),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7))),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1, 8), stride=(1, 2), padding=(0, 3))),
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))),
        ]
        '''
        self.all_convs = nn.ModuleList([
            DownSampleConv(2, 32, kernel=(1, 64), strides=(1, 2), padding=(0, 31), batchnorm=False, activation=True),
            DownSampleConv(32, 64, kernel=(1, 32), strides=(1, 2), padding=(0, 15), batchnorm=True, activation=True),
            DownSampleConv(64, 128, kernel=(1, 16), strides=(1, 2), padding=(0, 7), batchnorm=True, activation=True),
            DownSampleConv(128, 256, kernel=(1, 8), strides=(1, 2), padding=(0, 3), batchnorm=True, activation=True),
            DownSampleConv(256, 512, kernel=(1, 4), strides=(1, 2), padding=(0, 1), batchnorm=True, activation=True)
        ])
        '''
        self.all_deconv = [
            nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))),
            nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=(1, 8), stride=(1, 2), padding=(0, 3))),
            nn.Sequential(nn.ConvTranspose2d(128, 64,kernel_size=(1, 16), stride=(1, 2), padding=(0, 7))),
            nn.Sequential(nn.ConvTranspose2d(64, 32,  kernel_size=(1,32), stride=(1, 2), padding=(0, 15))),
            nn.Sequential(nn.ConvTranspose2d(32, 1,  kernel_size=(1, 64), stride=(1, 2), padding=(0, 31))),  #  sigmoid
        ]
        '''
        self.all_deconv = nn.ModuleList([
            UpSampleConv(512, 256, kernel=(1, 4), strides=(1, 2), padding=(0, 1), batchnorm=True, activation=True),
            UpSampleConv(256, 128, kernel=(1, 8), strides=(1, 2), padding=(0, 3), batchnorm=True, activation=True),
            UpSampleConv(128, 64, kernel=(1, 16), strides=(1, 2), padding=(0, 7), batchnorm=True, activation=True),
            UpSampleConv(64, 32, kernel=(1, 32), strides=(1, 2), padding=(0, 15), batchnorm=True, activation=True),
            UpSampleConv(32, 1, kernel=(1, 64), strides=(1, 2), padding=(0, 31), batchnorm=False, activation=False),

        ])

    def forward(self, x):
        for conv_down in self.all_convs:
            x = conv_down(x)
        z = x
        for deconv in self.all_deconv:
            x = deconv(x)
        return x

'''


all_convs = [
    nn.Sequential(nn.Conv2d(1, 64, kernel_size=(2, 4), stride=2, padding=1)),
    nn.Sequential(nn.Conv2d(64, 128, kernel_size=(2, 4), stride=2, padding=1)),
    nn.Sequential(nn.Conv2d(128, 256, kernel_size=(2, 4), stride=2, padding=1)),
    nn.Sequential(nn.Conv2d(256, 512, kernel_size=(2, 4), stride=2, padding=1)),
    nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(2, 4), stride=(1, 2), padding=1)),
    ]

all_deconv = [
    nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=(2, 4), stride=(1, 2), padding=1)),
    nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=(2, 4), stride=(3, 2), padding=1)),
    nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=(3, 4), stride=(2, 2), padding=1)),
    nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=(3, 4), stride=(2, 2), padding=1)),
    nn.Sequential(nn.ConvTranspose2d(128, 1, kernel_size=(2, 4), stride=(2, 2), padding=1)),
]

x = torch.randn(2, 1, 16, 256)
skip_conns = []
out = x
for conv_down in all_convs:
    print(conv_down)
    print(out.shape)
    out = conv_down(out)
    print(out.shape)
    skip_conns.append(out)

out = all_deconv[0](skip_conns[-1])
rev_skip_conns = skip_conns[:-1]
rev_skip_conns = rev_skip_conns[::-1]
for skip, deconv in zip(rev_skip_conns, all_deconv[1:]):
    print(deconv)
    print(out.shape, skip.shape)
    out = deconv(torch.cat([out, skip], dim=1))
    
'''
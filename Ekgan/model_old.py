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

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

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

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

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

class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512, kernel=(2, 4)),  # bs x 512 x 16 x 16
            # DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            # DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            # DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False, kernel=(2, 4)),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True, kernel=(2, 4)),  # bs x 512 x 2 x 2
            #UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            #UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            # UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256, kernel=(2, 4)),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        #self.tanh = nn.Tanh()

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return x # self.tanh(x)

class Discriminator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False, kernel=(1, 32), strides=2),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128, kernel=(1, 16), strides=2),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256, kernel=(1, 8), strides=2),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512, kernel=(1, 4), strides=2),  # bs x 512 x 16 x 16
            # DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            # DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            # DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            # DownSampleConv(512, 512, batchnorm=False, kernel=4),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            # UpSampleConv(512, 512, kernel=(1, 4)),  # bs x 512 x 2 x 2
            #UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            #UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            # UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(512, 256, kernel=(1, 8), strides=2),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128, kernel=(1, 16), strides=2),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64, kernel=(1, 32), strides=2),  # bs x 64 x 128 x 128
        ]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=(1, 64), stride=2, padding=1)
        #self.tanh = nn.Tanh()

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return x # self.tanh(x)


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
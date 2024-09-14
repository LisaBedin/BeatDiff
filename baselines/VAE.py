import torch
import torch.nn as nn
import numpy as np

class VariationalLoss(nn.Module):

    def __init__(self, kld_weight=1, recon_loss_type='mse', reduction='mean'):
        super(VariationalLoss, self).__init__()
        self.kld_weight = kld_weight
        self.recon_loss_type = recon_loss_type
        self.reduction = reduction
        if recon_loss_type == 'bce':
            self.recon_loss_fn = nn.BCELoss(reduction=reduction)
        elif recon_loss_type == 'l1':
            self.recon_loss_fn = nn.L1Loss(reduction=reduction)
        elif recon_loss_type == 'mse':
            self.recon_loss_fn = nn.MSELoss(reduction=reduction)  # nn.L1Loss()

    def kld_loss_fn(self, mu, logvar):
        kld = torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.reduction == 'mean':
            kld /= mu.shape[0]
        return -0.5 * kld

    def forward(self, x, x_hat, mu, logvar):
        recon_loss = self.recon_loss_fn(x_hat, x)
        kld_loss = self.kld_loss_fn(mu, logvar)

        vae_loss = recon_loss + self.kld_weight * kld_loss

        return vae_loss  #, recon_loss.item(), kld_loss.item()

class Encoder(nn.Module):
    def __init__(self, L, nef, nz, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nef = nef
        self.L = L
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
            nn.Flatten())

        self.mu = nn.Sequential(
            nn.Conv1d( nef, nz, 1, 1, 0 ),
            # state size. (nef * 32) x 1
            nn.Flatten()
            )
        self.logvar = nn.Sequential(
            nn.Conv1d( nef, nz, 1,1, 0),
            # state size. (nef * 32) x 1
            nn.Flatten()
            )

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def features(self, x):
        return self.conv(x)

    def forward(self, x):
        f = self.conv(x)

        f = f.unsqueeze(-1)
        mu_f, logvar_f = self.mu(f), self.logvar(f)
        return self.reparametrization(mu_f, logvar_f), mu_f, logvar_f
class Decoder(nn.Module):
    def __init__(self, nef, nz, L, ngpu, conditional, nc=4):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz = nz
        n_input = nz
        if conditional:
            n_input += nc
        upconv_lst = [
            # input nz x 1
            nn.ConvTranspose1d(n_input, nef * 16, 4, 1, 0, bias=False), # bs x nef *16 x 4
            nn.BatchNorm1d(nef * 16),
            nn.LeakyReLU(0.2,True),
            # state size (nef*16) x 8
            nn.ConvTranspose1d(nef * 16, nef * 8, 4, 2, 0, bias=False),  # bs x nef * 8 x 10
            nn.BatchNorm1d(nef * 8),
            nn.LeakyReLU(0.2,True),
            # state size. (nef*8) x 16
            nn.ConvTranspose1d(nef * 8, nef * 4, 4, 2, 0, bias=False),  # bs x nef * 4 x 22
            nn.BatchNorm1d(nef * 4),
            nn.LeakyReLU(0.2,True),
            # state size. (nef*4) x 32
            nn.ConvTranspose1d(nef * 4, nef * 2, 4, 2, 1, bias=False),  # bs x nef * 2 x 44
            nn.BatchNorm1d(nef * 2),
            nn.LeakyReLU(0.2,True),
            # state size. (nef*2) x 64
            nn.ConvTranspose1d(nef * 2, nef, 4, 2, 1, bias=False), # bs x nef x 32
            nn.BatchNorm1d(nef),
            nn.LeakyReLU(0.2,True),
            # state size. (nef) x 128
            nn.ConvTranspose1d(nef, L, 4, 2, 1, bias=False),  # bs x L x N
            # state size. (L) x 256
        ]
        upconv_lst.append(nn.Tanh())
        self.upconv = nn.Sequential(*upconv_lst)
        self.conditional = conditional

    def forward(self, z, labels=None):
        if self.conditional:
            z = torch.cat((z,labels), dim = 1)
        return self.upconv(z.unsqueeze(2))

class VAE(nn.Module):
    def __init__(self, device=torch.device('cuda'), N=176,
                 L=9, nz=32, nef=32, ngpu=1,
                 lambda_tv=0.001,
                 denoising=False, conditional=False):
        super(VAE, self).__init__()
        # Lenght of a beat.
        self.N = N  # never used ??
        # TODO inclure une option de denoising !
        self.denoising = denoising

        # Number of leads
        self.L = L

        # Size of z latent vector
        self.nz = nz

        # Size of feature maps in encoder
        self.nef = nef

        self.ngpu = ngpu
        self.lambda_tv = lambda_tv

        self.netE = Encoder(L, nef, nz, ngpu)
        self.netD = Decoder(nef, nz, L, ngpu, conditional=conditional)
        self.netE.to(device)
        self.netD.to(device)

        self.initialize()

        self.trained = False

        # Initialize the threeshold
        self.threeshold = 0

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

    def initialize(self):
        self.netE.apply(self.weights_init)
        self.netD.apply(self.weights_init)

    def get_anomaly_score(self, X):
        b_s = X.size(0)  # Batch size
        X_rec = self.netD(self.netE(X)).view(b_s, -1).detach().cpu().numpy()
        X = X.view(b_s, -1).detach().cpu().numpy()
        AS = np.mean(np.square(X - X_rec), axis=1)

    def forward(self, X, feats):
        code, mu_f, logvar_f = self.netE(X)
        return self.netD(code, feats), mu_f, logvar_f

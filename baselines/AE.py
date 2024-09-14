import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, fbeta_score
from pytorch_tcn import TCN
from baselines.utils import get_threeshold, plot_error_mixing, plot_some_beat

# TCN(9, [32, 32], kernel_size=4, use_skip_connections=True).cuda()(X_input)
# nn.MaxPool1d(5)


class Encoder(nn.Module):
    def __init__(self, L, nef, nz, ngpu, TCN_opt=False):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nef = nef
        self.L = L
        if TCN_opt: # like Shan et al. for anomaly detection
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
                # state size 8 x 1
                nn.Flatten())
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2))


    def forward(self, x):
        f = self.conv(x)
        return f


class Decoder(nn.Module):
    def __init__(self, nef, nz, L, ngpu, conditional, nc=4, TCN_opt=False):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz = nz
        n_input = nz
        if conditional:
            n_input += nc
        if not TCN_opt:
            upconv_lst = [
                nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=32, out_channels=9, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(9),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=3, stride=2, padding=1, output_padding=1),
                #nn.BatchNorm1d(9),
                nn.Tanh()
            ]
        else:
            upconv_lst = [
                # input nz x 1
                nn.ConvTranspose1d(8, 8, 1, 1, 0, bias=False), # bs x nef *16 x 4
                nn.Upsample(scale_factor=11),
                TCN(8, [8, 8], kernel_size=9, use_skip_connections=True), # bs x nef x 88
                # state size (nef*16) x 8
                nn.Upsample(scale_factor=4),
                TCN(8, [16, 16], kernel_size=9, use_skip_connections=True), # bs x nef x 88
                # state size (nef*16) x 8
                nn.Upsample(scale_factor=4),
                TCN(16, [32, 32], kernel_size=9, use_skip_connections=True), # bs x nef x 88
                # state size. (nef) x 128
                nn.Conv1d(32, L, 1, 1, 0, bias=False),  # bs x L x N
                # state size. (L) x 256
            ]
        self.upconv = nn.Sequential(*upconv_lst)
        self.conditional = conditional
        self.TCN_opt = TCN_opt

    def forward(self, z, labels=None):
        if self.conditional:
            z = torch.cat((z,labels), dim = 1)
        if not self.TCN_opt:
            return self.upconv(z)
        return self.upconv(z.unsqueeze(2))

class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nz=32, ndf=32, lambda_gp=10, lambda_adv=0.001):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.lambda_adv = lambda_adv
        self.lambda_gp = lambda_gp
        self.main = nn.Sequential(
            nn.Linear(in_features=(nz), out_features=ndf),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=ndf, out_features=ndf // 2),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=ndf // 2, out_features=ndf // 4),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=ndf // 4, out_features=1)
        )
        self.main.apply(self.weights_init)

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

    def sample_noise(self, b_s, uniform_noise = False):
        if uniform_noise:
            # Uniform Distribution [-1, 1]
            a = -1
            b = 1
            noise = (a - b) * torch.rand((b_s, self.nz), dtype=torch.float).cuda() + b
        else:
            noise = torch.randn((b_s, self.nz), dtype = torch.float).cuda()
        return noise

    def compute_gradient_penalty(self, real_samples, fake_samples, labels = None):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        if labels is None:
            alpha = torch.rand((real_samples.size(0), 1)).cuda()
        else:
            alpha = torch.rand((real_samples.size(0), 1, 1)).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        if labels is None:
            d_interpolates = self.forward(interpolates)
        fake = torch.ones((real_samples.size(0), 1)).cuda()
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, input):
        return self.main(input)

class AE(nn.Module):
    def __init__(self, device=torch.device('cuda'), N=176,
                 L=9, nz=32, nef=32, ngpu=1,
                 lambda_tv=0.001,
                 TCN_opt=False,
                 denoising=False, conditional=False):
        super(AE, self).__init__()
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

        self.netE = Encoder(L, nef, nz, ngpu, TCN_opt=TCN_opt)
        self.netD = Decoder(nef, nz, L, ngpu, conditional=conditional, TCN_opt=TCN_opt)
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
        code = self.netE(X)
        return self.netD(code, feats)

    def train_(self, train_dl, valid_dl, num_epochs, lr=0.0001, early_stop=False,
              beta1=0.5, models_folder="models/AE/"):
        rec_losses = []
        example = next(iter(train_dl))
        examples_data = example[1][:4].to(self.device)
        beat_list = []

        MSE = nn.MSELoss().to(self.device)
        optimizerE = optim.Adam(self.netE.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            for i, data in enumerate(train_dl, 0):
                self.netE.zero_grad()
                self.netD.zero_grad()

                X = data[0].to(self.device)
                X_rec = self.netD(self.netE(X))

                ED_rec_loss = MSE(X, X_rec)
                ED_rec_loss.backward()
                optimizerD.step()
                optimizerE.step()

                ## Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\n\t Rec Loss: %.4f'
                          % (epoch, num_epochs, i, len(train_dl),
                             ED_rec_loss.item()))

                rec_losses.append(ED_rec_loss.item())

        torch.save(self.netE.state_dict(), models_folder + "encoder.mod")
        torch.save(self.netD.state_dict(), models_folder + "decoder.mod")
        pred = None
        labels = None
        print("\t Computing the optimal threeshold")
        pbar = tqdm.tqdm(total=len(valid_dl))
        print("\t Compute errors")
        for i, data in enumerate(valid_dl, 0):
            beats = data[0].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal'  # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            pbar.update(n=1)
        pbar.close()
        # Compute threeshold using both errors
        self.threeshold = get_threeshold(pred, labels=labels, verbose=True)

        # Save the best threeshold
        np.save(models_folder + "thr.npy", np.array([self.threeshold]))
        f, ax = plt.subplots()
        ax.set_title("Reconstruction error ")
        ax.plot(rec_losses)
        plt.xlabel("iterations")
        plt.show()
        for i in range(len(beat_list)):
            plot_some_beat(beat_list[i], suptitle='epoch: ' + str(i),
                           titles_list=[('M' if examples_data[j, 0] == 0 else 'F') for j in range(len(beat_list[i]))])
        self.trained = True

    def test_(self, test_dl, result_folder="models/AE/"):
        if not self.trained:
            print("train or load a model")
            return None
        print("Starting Testing Loop...")
        pred = None
        labels = None
        pbar = tqdm.tqdm(total=len(test_dl))
        for i, data in enumerate(test_dl, 0):
            beats = data[0].to(self.device)
            label = np.array(data[2]['label']) == 'abnormal'  # 0 normale 1 anormale
            ano_sc = self.get_anomaly_score(beats)
            if pred is None:
                pred = ano_sc
                labels = label
            else:
                pred = np.concatenate((pred, ano_sc), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            pbar.update(n=1)

        clrp = classification_report(labels, pred >= self.threeshold,
                                     target_names=['normal', 'abnormal'])
        print("\n")
        print(clrp)
        plot_error_mixing(pred, labels, self.threeshold)
        rep = classification_report(labels, pred >= self.threeshold,
                                    target_names=['normal', 'abnormal'],
                                    output_dict=True)
        auc = roc_auc_score(labels, pred)
        pr_auc = average_precision_score(labels, pred)
        result = {'rep': rep['abnormal'], 'roc_auc': auc, 'pr_auc': pr_auc,
                  'pr_curve': precision_recall_curve(labels, pred),
                  'f2score': fbeta_score(labels, pred >= self.threeshold, beta=2)}

        with open(result_folder + 'result.pickle', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return classification_report(labels, pred >= self.threeshold,
                                     target_names=['normal', 'abnormal'],
                                     output_dict=True)

    def load_model(self, model_folder='models/AE/'):
        self.netE.load_state_dict(torch.load(model_folder + "encoder.mod", map_location=self.device))
        self.netD.load_state_dict(torch.load(model_folder + "decoder.mod", map_location=self.device))
        # Load the threeshold
        self.threeshold = np.load(model_folder + "thr.npy")
        self.trained = True
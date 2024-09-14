import matplotlib.pyplot as plt
import numpy as np
import torch
from jax import random
from statsmodels.distributions.empirical_distribution import ECDF
from joblib import Parallel, delayed
from sklearn.metrics import fbeta_score, roc_curve, auc
num_cores = 10
import torch.nn as nn
def corrupt_batch(x, key):
    key_std, key_epsilon = random.split(key, num=2)
    stds = torch.Tensor(np.array(random.exponential(key=key_std,
                              shape=(x.shape[0], x.shape[-1],)) / 4)).to(x.device)
    epsilon = torch.Tensor(np.array(random.normal(key=key_epsilon, shape=x.shape))).to(x.device)
    return x + stds[..., None, :] * epsilon, stds

def get_input(key, X, feats, denoising=False):
    feats[:, -1] /= 100
    if denoising:
        X_input, stds = corrupt_batch(X, key)
        key = random.split(key)[0]
        X_input = torch.swapaxes(X_input, 1, 2).to(torch.float32).cuda()
    else:
        X_input = torch.swapaxes(X, 1, 2).to(torch.float32).cuda()
    return X_input, feats.cuda()

def train_epoch(key, epoch, num_epochs, train_loader, model, loss_fn, optimizer, model_dis, optimizer_dis, n_critic=5):
    for i, data_inp in enumerate(train_loader):
        X_input, feats = data_inp[:2]
        X_input = torch.swapaxes(X_input, 1, 2).to(torch.float32).cuda()
        if len(train_loader.dataset.noise_path) > 0:
            X_corrupt = data_inp[2]
            X_corrupt = torch.swapaxes(X_corrupt, 1, 2).to(torch.float32).cuda()
        for _ in range(n_critic):
            #X_input, feats = get_input(key, X, feats, model.denoising)
            if len(train_loader.dataset.noise_path) > 0:
                X_rec = model(X_corrupt, feats)
            else:
                X_rec = model(X_input, feats)
            if len(X_rec) == 3:
                X_rec, mu, logvar = X_rec
                loss = loss_fn(X_input, X_rec, mu, logvar)
            else:
                loss = loss_fn(X_input, X_rec)

            if model_dis is not None:
                critic_real = model_dis(X_input)
                critic_fake = model_dis(X_rec)

                #loss_critic = torch.mean(torch.log(critic_real) + torch.log(1-critic_fake))
                targets = torch.cat([torch.ones_like(critic_real), torch.zeros_like(critic_fake)])#.flatten()
                loss_critic = nn.BCELoss()(torch.cat([critic_real, critic_fake]), targets)
                model_dis.zero_grad()
                loss_critic.backward(retain_graph=True)
                optimizer_dis.step()
            model.zero_grad()
            loss.backward()
            optimizer.step()

        ## Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\n\t Train Loss: %.4f'
              % (epoch, num_epochs, i, len(train_loader),
                 loss.item()))
    loss = loss.item()
    if model_dis is not None:
        loss = loss, loss_critic.item()
    return loss, key

def val_epoch(key, epoch, num_epochs, val_loader, model, loss_fn, model_dis):
    model.eval()
    val_loss = 0
    val_critic = 0
    with torch.no_grad():
        for i, data_inp in enumerate(val_loader):
            # X_input, feats = get_input(key, X, feats, model.denoising)
            X_input, feats = data_inp[:2]
            X_input = torch.swapaxes(X_input, 1, 2).to(torch.float32).cuda()
            if len(val_loader.dataset.noise_path) > 0:
                X_corrupt = data_inp[2]
                X_corrupt = torch.swapaxes(X_corrupt, 1, 2).to(torch.float32).cuda()
                X_rec = model(X_corrupt, feats)
            else:
                X_rec = model(X_input, feats)
            if len(X_rec) == 3:
                VAE = True
                X_rec, mu, logvar = X_rec
                loss = loss_fn(X_input, X_rec, mu, logvar)
            else:
                VAE = False
                loss = loss_fn(X_input, X_rec)

            if model_dis is not None:
                model_dis.eval()
                critic_real = model_dis(X_input)
                critic_fake = model_dis(X_rec)

                targets = torch.cat([torch.ones_like(critic_real), torch.zeros_like(critic_fake)])#.flatten()
                loss_critic = nn.BCELoss()(torch.cat([critic_real, critic_fake]), targets)
                val_critic += loss_critic.item()
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_critic /= len(val_loader)
    print('[%d/%d] Val Loss: %.4f'
          % (epoch, num_epochs, val_loss))
    X_rec = X_rec.detach().cpu().numpy()[:5]
    if VAE:
        all_X_rec = []
        for _ in range(10):
            X_rec, mu, logvar = model(X_input[:5], feats[:5])
            all_X_rec.append(X_rec.detach().cpu().numpy())
        X_rec = np.stack(all_X_rec, axis=1)
    if len(val_loader.dataset.noise_path) > 0:
        X = X_corrupt.detach().cpu().numpy()[:5]
    else:
        X = X_input.detach().cpu().numpy()[:5]
    X_input = X_input.detach().cpu().numpy()[:5]
    if model_dis is not None:
        val_loss = (val_loss, val_critic)
    return val_loss, key, X_input, X, X_rec

def test_epoch(epoch, num_epochs, test_loader, MI_loader, model):
    model.eval()
    with torch.no_grad():
        test_scores, MI_scores = [], []
        all_test, all_MI = 0, 0
        for i, (X, feats) in enumerate(test_loader):
            #X_input, feats = get_input(random.PRNGKey(0), X, feats, False)
            X_input = torch.swapaxes(X, 1, 2).to(torch.float32).cuda()
            X_rec = model(X_input, feats)
            if len(X_rec) == 3:
                X_rec, mu, logvar = X_rec
            test_scores.append(np.array(((X_input-X_rec)**2).mean(dim=(1,2)).detach().cpu()))
            all_test += X.shape[0]

        for i, (X, feats) in enumerate(MI_loader):
            #X_input, feats = get_input(random.PRNGKey(0), X, feats, False)
            X_input = torch.swapaxes(X, 1, 2).to(torch.float32).cuda()
            X_rec = model(X_input, feats)
            if len(X_rec) == 3:
                X_rec, mu, logvar = X_rec
            MI_scores.append(np.array(((X_input - X_rec) ** 2).mean(dim=(1, 2)).detach().cpu()))
            all_MI += X.shape[0]
    y_true = np.array([0]*all_test+[1]*all_MI)
    y_score = np.concatenate(test_scores+MI_scores)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print('[%d/%d] Anomaly AUC: %.4f'
          % (epoch, num_epochs, roc_auc))
    return roc_auc

def plot_some_beat(beat_list, titles_list = None, suptitle = None, savefig = False, file= None):
    n = len(beat_list)
    nl = len(beat_list[0])
    f, ax = plt.subplots(nl, n, sharey=True)
    if not(suptitle is None):
        f.suptitle(suptitle)
    for i in range(n):
        if not (titles_list is None):
            if nl > 1:
                ax[0,i].set_title(titles_list[i])
            else:
                ax[i].set_title(titles_list[i])
        if nl == 1:
            ax[i].plot(beat_list[i][0])
            ax[i].set_ylim(bottom = -1, top = 1)
        else:
            for j in range(nl):
                ax[j,i].plot(beat_list[i][j])
                ax[j,i].set_ylim(bottom = -1, top = 1)
    if savefig:
        plt.savefig(file, dpi = 300)
    return ax

def plot_error_mixing(errors, labels, best_thr, est_thr=None):
    plt.figure()
    h = np.histogram(errors[np.where(labels == 0)])
    plt.bar(h[1][:-1], h[0] / np.sum(h[0]), label='normal', width=0.001)
    h = np.histogram(errors[np.where(labels == 1)])
    plt.bar(h[1][:-1], h[0] / np.sum(h[0]), label='abnormal', width=0.001)
    plt.ylim([0, 1.001])
    plt.xlim([0, np.max(errors)])
    if est_thr is None:
        plt.vlines(x=best_thr, ymin=0, ymax=1, colors='b', label='threeshold')
    else:
        plt.vlines(x=est_thr, ymin=0, ymax=1, colors='b', label='alpha-thr')
        plt.vlines(x=best_thr, ymin=0, ymax=1, colors='r', label='best f1-thr')
    plt.legend()
    plt.show()


def get_threeshold(errors, labels=None, alpha=0.05, verbose=False):
    # Estimate threeshold only with normal data
    # fixing false positive rate to alpha.
    # Then, if provided, use the labels to compute the threeshold
    # which gives the best recall ( = accuracy)
    # if the threeshold is greater than the previously computed
    # that means that we have decreased also the false positive rate
    # Hence we select the new threeshold

    if verbose:
        print("estimate threeshold with only normal data")
    if labels is None:
        ecdf = ECDF(errors)
    else:
        ecdf = ECDF(errors[np.where(labels == 0)])
    thr = ecdf.x[np.where(ecdf.y >= (1 - alpha))[0][0]]
    quantile_thr = thr
    if labels is None:
        return thr
    best_thr = 0
    # Estimate threeshold with both
    thrs = np.linspace(np.min(errors), np.max(errors), num=errors.shape[0] * 2)
    if verbose:
        print("estimate threeshold with all data")
    e = Parallel(n_jobs=num_cores)(delayed(fbeta_score)(errors >= t, labels, beta=2)
                                   for t in thrs)
    ind = np.argmax(np.array(e))
    best_thr = thrs[ind]
    if verbose:
        plot_error_mixing(errors, labels, best_thr, est_thr=thr)

    return quantile_thr, best_thr

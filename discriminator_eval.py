import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from beat_net.beat_net.unet_parts import load_net
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from collections import Counter
import torch
import glob
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib
from matplotlib.colors import to_rgb
import seaborn as sns
from UncertaintyEnsemble.models import TorchUncertaintyEnsemble
from UncertaintyEnsemble.data_loader import UncertaintyDataset
from beat_net.beat_net.data_loader import PhysionetECG
from UncertaintyEnsemble.training import train_uncertainty
from sklearn.metrics import roc_curve, auc

matplotlib.rc('font', **{'size'   : 22})


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=9, inplanes=64, num_classes=2):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        return x.view(x.size(0), -1)  # return self.fc(x)

def resnetSmall(**kwargs):
    model = ResNet1d(BasicBlock1d, [1], **kwargs)
    return model
def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model

class DiscriminatorDataset(Dataset):
    def __init__(self, npz_data, piste):
        self.piste = piste
        self.ecgs = np.concatenate([npz_data['target_samples'], npz_data['generated_samples']])
        n_train = self.ecgs.shape[0]
        labels = np.zeros(n_train)
        labels[int(n_train // 2):] = 1
        self.labels = labels
        self.feats = np.concatenate([npz_data['class_features']]*2)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, item):
        ECG = torch.Tensor(self.ecgs[item].T)
        if self.piste >= 0 and self.piste<9:
            ECG = ECG[self.piste].unsqueeze(0)
        if bool(self.labels[item]):
            lab = torch.Tensor([0, 1])
        else:
            lab = torch.Tensor([1, 0])
        return ECG, lab


def train(dataloader, net, criterion, epoch, optimizer, device, model_path):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    #aucs = roc_auc_score(y_trues, y_scores, average=None)
    #avg_auc = np.mean(aucs)
    #print('AUCs:', aucs)
    #print('Avg AUC: %.4f' % avg_auc)
    acc = (y_trues[:, 1] == y_scores.argmax(axis=1)).sum() / len(y_trues)
    print('acc: %.4f' % acc)
    torch.save(net.state_dict(), model_path)
    return running_loss, acc#avg_auc


def evaluate(dataloader, net, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    #aucs = roc_auc_score(y_trues, y_scores, average=None)
    acc = (y_trues[:, 1] == y_scores.argmax(axis=1)).sum() / len(y_trues)
    #print('AUCs:', aucs)
    #avg_auc = np.mean(aucs)
    print('acc: %.4f' % acc)
    return running_loss, acc# avg_auc


def isin_inds(arr, lst):
    inds = np.sum(np.stack([(arr==lst[k]) for k in range(len(lst))]), axis=0)
    return inds > 0


def get_equlibrated(all_ages, all_abbrev, all_sex):
    control_labs = ['train', 'test', 'gen']
    all_kept_inds = []
    for sex in [0, 1]:
        ages_MI = all_ages[(all_abbrev == 'MI')*(all_sex == sex)]
        ages_control = all_ages[isin_inds(all_abbrev, control_labs)*(all_sex == sex)]
        ages_control_counts = Counter(ages_control)
        # get the density of ages in MI
        # bandwidth: 2 for Male, .. for Female ?
        kde = KernelDensity(kernel='gaussian', bandwidth=2.).fit(ages_MI[:, None])
        ages_prob = np.exp(kde.score_samples(ages_control[:, None]))
        #  ages_prob = scipy.stats.norm.pdf(ages_control, ages_MI.mean(), ages_MI.std())
        ages_prob /= ages_prob.sum()
        ages_prob = np.array([prob/ages_control_counts[age] for age, prob in zip(ages_control, ages_prob)])
        ages_prob /= ages_prob.max()
        ages_prob_dic = {age: prob for age, prob in zip(ages_control, ages_prob)}
        binom_control = np.random.binomial(n=1, p=np.array([ages_prob_dic[age] for age in ages_control])).astype(bool)
        inds_control = np.where(isin_inds(all_abbrev, control_labs)*(all_sex == sex))[0][binom_control]
        inds_MI = np.where((all_abbrev == 'MI')*(all_sex == sex))[0]
        all_kept_inds.append(np.concatenate([inds_control, inds_MI]))
        all_kept_inds.append(np.arange(all_ages.shape[0])[~isin_inds(all_abbrev, control_labs)*(all_sex == sex)])
    all_kept_inds = np.concatenate(all_kept_inds)
    return all_kept_inds

def equilibrate_ages(lst, inds, sex, all_abbrev, all_sex):
    new_NSR = lst[isin_inds(all_abbrev, ['train', 'test', 'gen'])*(all_sex == sex)]
    # if sex == 0:
    new_NSR = new_NSR[inds]
    new_MI = lst[(all_abbrev=='MI')*(all_sex == sex)]
    return np.concatenate([new_NSR, new_MI])

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    train_state, net_config, ckpt_num = load_net(cfg)
    sigma_max = net_config.diffusion.sigma_max
    sigma_min = net_config.diffusion.sigma_min
    p = net_config.generate.rho

    if net_config.diffusion.scale_type == 'linear':
        scaling_fun = lambda t: cfg.diffusion.scaling_coeff * t
    elif net_config.diffusion.scale_type == 'one':
        scaling_fun = lambda t: 1.0
    else:
        raise NotImplemented

    npz_path = os.path.join(cfg.results_path,
                            'generated_samples/' + 'baseline' * ('baseline' in cfg.checkpoint) + 'deeper' * (
                                        'baseline' not in cfg.checkpoint) + '_uncond' * ('uncond' in cfg.checkpoint))
    npz_train = np.load(os.path.join(npz_path, 'train_gen.npz'))
    npz_test = np.load(os.path.join(npz_path, 'test_gen.npz'))

    model_AAE = torch.load('/mnt/data/lisa/ecg_results/baseline_AE/GOOD_AE_disc_TCN/best_model.pth').cuda()
    model_AAE.eval()
    model_AAE.netD.conditional = False


    X_train = np.concatenate([npz_train['target_samples'], npz_train['generated_samples']])
    n_train = X_train.shape[0]
    y_train = np.zeros(n_train)
    y_train[int(n_train//2):] = 1
    X_test = np.concatenate([npz_test['target_samples'], npz_test['generated_samples']])
    n_test = X_test.shape[0]
    y_test = np.zeros(n_test)
    y_test[int(n_test//2):] = 1

    for piste in range(9):
        clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train[:, piste], y_train)
        y_pred = clf.predict(X_train[:, piste])
        acc = (y_pred==y_train).sum() / len(y_pred)
        print(f'train: piste={piste} | acc={acc:.2f}')
        y_pred = clf.predict(X_test[:, piste])
        y_prob = clf.predict_proba(X_test[:, piste])
        acc = (y_pred==y_test).sum() / len(y_pred)
        print(f'val: piste={piste} | acc={acc:.2f}')
        print(' ')

    device =  'cuda:0'
    # ====================== uncertainty measurement ================== #
    batch_size = 64
    piste = 9
    conds = np.array(cfg.eval_mcg_diff.ood_condition)
    conds_name = '_'.join(conds.astype(str))
    baseline_path = os.path.join(cfg.results_path, 'generated_samples/baseline')

    '''
    generated_ecg_lst, target_ecg_lst, class_features_lst = [], [], []
    MI_cond_path = glob.glob('/mnt/data/lisa/ecg_results/S1/EMinit1_EMsteps10_N50_0_0/MI_*.npz')
    for MI_p in MI_cond_path:
        npz = np.load(MI_p)
        generated_ecg_lst.append(npz['generated_ecg'][:, 0]) # on calcule seulement pour un sample ?
        target_ecg_lst.append(npz['target_ecg'])
        class_features_lst.append(npz['class_features'])
    np.savez(os.path.join(baseline_path, 'MI_cond.npz'),
             generated_samples=np.concatenate(generated_ecg_lst),
             target_samples=np.concatenate(target_ecg_lst),
             class_features=np.concatenate(class_features_lst))
    '''

    # model_path = os.path.join(cfg.results_path, 'generated_samples/baseline/piste9/uncertainties_final/epochs100_out1024')
    test_loader = DataLoader(UncertaintyDataset(np.load(os.path.join(baseline_path, 'test_gen.npz')), piste=conds, prefix='target'), batch_size=batch_size, shuffle=False, num_workers=8)
    train_loader = DataLoader(UncertaintyDataset(np.load(os.path.join(baseline_path, 'train_gen.npz')), piste=conds, prefix='target'), batch_size=batch_size, shuffle=True, num_workers=8)
    gen_loader = DataLoader(UncertaintyDataset(npz_test, piste=conds, prefix='generated'), batch_size=batch_size, shuffle=False, num_workers=8)
    MI_cond_loader = DataLoader(UncertaintyDataset(np.load(os.path.join(baseline_path, 'MI_cond.npz')), piste=conds, prefix='generated'), batch_size=batch_size, shuffle=False, num_workers=8)
    results_path = os.path.join(npz_path, f'piste{piste}', 'uncertainties_final', 'epochs100_out1024', conds_name)
    os.makedirs(results_path, exist_ok=True)
    print(results_path)


    model = TorchUncertaintyEnsemble(c_in=len(conds),
                                     ensemble_size=10,
                                     output_size=1024,
                                     init_scaling=2.0,
                                     output_weight=1.0,
                                     gp_weight=0.,
                                     beta=1.,
                                     conditioned=False).to(device).half()
    trainable_params = lambda model: filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.Adam(trainable_params(model), lr=0.0001, weight_decay=5e-4 * batch_size, eps=1e-4)

    # TODO create the train/test/gen loaders
    '''
    epochs = 100
    train_uncertainty(model, opt, train_loader, test_loader, gen_loader, epochs, results_path)
    '''

    model = torch.load(f"{results_path}/uncertainty_model.pth").to(device)
    stats = pd.read_csv(os.path.join(results_path, 'uncertainty_stats.csv'))
    epoch_infos = pd.read_csv(os.path.join(results_path, 'uncertainty_gen_test.csv'))

    model.train(False)
    model.eval()
    train_metrics = []
    train_gender = []
    train_age = []
    for (ecg_batch, feats_batch, labels_batch) in tqdm(train_loader, total=len(train_loader)):
        batch = {'input': ecg_batch.cuda().half(), 'feats': feats_batch.cuda().half()}
        output = model(batch)
        train_metrics.append(output['uncertainties'].detach().cpu().numpy())
        train_gender.append(feats_batch.numpy()[:, 0])
        train_age.append(feats_batch.numpy()[:, 3]*50+50)
    train_metrics = np.concatenate(train_metrics)
    train_gender = np.concatenate(train_gender)
    train_age = np.concatenate(train_age)

    MI_loader = DataLoader(dataset=PhysionetECG(database_path=cfg.db_path,
                                                categories_to_filter=["MI"],
                                                normalized=True,
                                                training_class='Test',
                                                all=True,
                                                return_beat_id=False),
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=8)
    MI_metric = []
    MI_gender = []
    MI_age = []
    #MI_AAE_metric = []
    for (ecg_batch, feats_batch) in tqdm(MI_loader, total=len(MI_loader)):
        batch = {'input': ecg_batch.swapaxes(1, 2)[:, conds].cuda().half(), 'feats': feats_batch.cuda().half()}
        output = model(batch)
        #X_input = torch.Tensor(ecg_batch.swapaxes(1, 2)[:, conds].numpy().astype(np.float32)).cuda()
        #X_recon = model_AAE(X_input, None)
        #batch = {'input': X_recon[:, conds].cuda().half(), 'feats': feats_batch.cuda().half()}
        #out_baseline = model(batch)
        #MI_AAE_metric.append(out_baseline['uncertainties'].detach().cpu().numpy())
        MI_metric.append(output['uncertainties'].detach().cpu().numpy())
        MI_gender.append(feats_batch.numpy()[:, 0])
        MI_age.append(feats_batch.numpy()[:, 3]*50+50)
    MI_metric = np.concatenate(MI_metric)
    MI_gender = np.concatenate(MI_gender)
    MI_age = np.concatenate(MI_age)
    #MI_AAE_metric = np.concatenate(MI_AAE_metric)

    gen_metric = []
    gen_gender = []
    gen_age = []
    for (ecg_batch, feats_batch, _) in tqdm(gen_loader, total=len(gen_loader)):
        batch = {'input': ecg_batch.cuda().half(), 'feats': feats_batch.cuda().half()}
        output = model(batch)
        gen_metric.append(output['uncertainties'].detach().cpu().numpy())
        gen_gender.append(feats_batch.numpy()[:, 0])
        gen_age.append(feats_batch.numpy()[:, 3]*50+50)
    gen_metric = np.concatenate(gen_metric)
    gen_gender = np.concatenate(gen_gender)
    gen_age = np.concatenate(gen_age)

    test_metric = []
    test_gender = []
    test_age = []
    test_AAE_metric = []
    for (ecg_batch, feats_batch, _) in tqdm(test_loader, total=len(test_loader)):
        batch = {'input': ecg_batch.cuda().half(), 'feats': feats_batch.cuda().half()}
        output = model(batch)
        #X_input = torch.Tensor(ecg_batch[:, conds].numpy().astype(np.float32)).cuda()
        #X_recon = model_AAE(X_input, None)
        #batch = {'input': X_recon[:, conds].cuda().half(), 'feats': feats_batch.cuda().half()}
        #out_baseline = model(batch)
        #test_AAE_metric.append(out_baseline['uncertainties'].detach().cpu().numpy())
        test_metric.append(output['uncertainties'].detach().cpu().numpy())
        test_gender.append(feats_batch.numpy()[:, 0])
        test_age.append(feats_batch.numpy()[:, 3] * 50 + 50)
    test_metric = np.concatenate(test_metric)
    test_gender = np.concatenate(test_gender)
    test_age = np.concatenate(test_age)
    #test_AAE_metric = np.concatenate(test_AAE_metric)

    MI_cond_metric = []
    MI_cond_gender = []
    MI_cond_age = []
    for (ecg_batch, feats_batch, _) in tqdm(MI_cond_loader, total=len(MI_cond_loader)):
        batch = {'input': ecg_batch.cuda().half(), 'feats': feats_batch.cuda().half()}
        output = model(batch)
        MI_cond_metric.append(output['uncertainties'].detach().cpu().numpy())
        MI_cond_gender.append(feats_batch.numpy()[:, 0])
        MI_cond_age.append(feats_batch.numpy()[:, 3] * 50 + 50)
    MI_cond_metric = np.concatenate(MI_cond_metric)
    MI_cond_gender = np.concatenate(MI_cond_gender)
    MI_cond_age = np.concatenate(MI_cond_age)

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    all_colors = [cmap(k) for k in [0, 0.75, 1., 1.]]
    if 'deeper' in results_path:
        all_colors = [cmap(k) for k in [0, 0.6, 1., 1.]]


    print('ok')

    all_metrics = np.concatenate([
        train_metrics*100,
        test_metric*100,
        gen_metric*100, # MI_cond_metric*100, # gen_metric*100,
        MI_metric*100
    ])
    all_ages = np.concatenate([
        train_age,
        test_age,
        gen_age, #MI_cond_age, # gen_age,
        MI_age
    ])
    all_genders = np.concatenate([
            train_gender,
            test_gender,
            gen_gender, # MI_cond_gender,  # gen_gender,
            MI_gender,
        ])
    all_labels = np.array(['train']*train_metrics.shape[0] + ['test']*test_metric.shape[0] + ['gen']*gen_metric.shape[0]+ ['MI']*MI_metric.shape[0])
    # + ['gen'] * gen_metric.shape[0]
    # ['MI_cond']*MI_cond_metric.shape[0]
    balanced = False
    if balanced:
        kept_inds = get_equlibrated(all_ages, all_labels, all_genders)
    else:
        kept_inds = np.arange(all_ages.shape[0])

    df_boxplot = pd.DataFrame({'metric': all_metrics[kept_inds]/100,
        'sex': all_genders[kept_inds],
    'label': all_labels[kept_inds]
    })

    fig = plt.figure(figsize=(5, 5))
    PROPS = {
        'boxprops': {'edgecolor': 'k'}, # 'facecolor': 'none', alpha=0.5
        # 'medianprops': {'color': 'k'},
        'whiskerprops': {'color': 'k'},
        'capprops': {'color': 'k'}
    }
    alpha = 0.8
    color_bg = np.array(to_rgb('white'))
    color_train = (1 - alpha) * color_bg + alpha * np.array(to_rgb('darkred'))
    color_test = (1 - alpha) * color_bg + alpha * np.array(to_rgb('red'))
    color_gen = (1-alpha)* color_bg + alpha*np.array(to_rgb('gray'))
    color_MI = (1-alpha)* color_bg + alpha*np.array(to_rgb('blue'))
    color_train = all_colors[-1]
    color_test = all_colors[-2]
    color_gen = all_colors[1]
    color_MI = all_colors[0]
    sns.boxplot(data=df_boxplot, x='label', y='metric',showfliers=False,
                order=['train', 'test', 'MI_cond', 'MI'],
                palette=[color_train, color_test, color_gen, color_MI])
    fig.subplots_adjust(top=1., left=0.02, bottom=0.02, right=1)
    fig.axes[0].set_xlabel('')
    fig.axes[0].set_ylabel('')
    # plt.yticks(np.arange(1, 10, 3), np.arange(1, 10, 3))
    # plt.ylim(0, 0.2)
    # legend = plt.legend()
    #legend.remove()
    fig.savefig(os.path.join(results_path, 'uncertainties_boxplot'+balanced*'_balanced'+'_CLASSIC.pdf'))
    plt.show()

    # TODO: maybe draw some roc curves ?
    # =========== ROC curves ==========  #
    fig = plt.figure(figsize=(5, 5))
    lw = 2
    # [cmap(k) for k in [0, 0.75, 1., 1.]]
    dic_color = {'train': cmap(1.), 'test': cmap(0.95), 'MI_cond': cmap(0.75)}#'#7BC8F6'}
    for lab in ['train', 'test', 'gen']: # 'gen']:
        # lab_df = np.array(df_boxplot[(df_boxplot['label'] == lab)&(df_boxplot['sex'] == sex)]['metric'])
        # MI_df = np.array(df_boxplot[(df_boxplot['label'] == 'MI')&(df_boxplot['sex'] == sex)]['metric'])
        lab_df = np.array(df_boxplot[df_boxplot['label'] == lab]['metric'])
        MI_df =  np.array(df_boxplot[df_boxplot['label'] == 'MI']['metric'])
        y_score = np.concatenate([lab_df, MI_df])
        y_true = [0]*len(lab_df)+[1]*len(MI_df)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color=dic_color[lab],
            lw=lw,
            label=f"{lab} ({roc_auc:.2f})",
        )
    plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.025])
    plt.xticks([0.5, 1], [ '0.5', '1'])
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
    #plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    #plt.xlabel("False Positive Rate")
    #plt.ylabel("True Positive Rate")
    #plt.title("Receiver operating characteristic example")
    #plt.legend(loc="lower right")
    fig.subplots_adjust(top=1, left=0.11, bottom=0.07, right=1)

    # ========================
    plt.plot(stats['epoch'], stats['train loss'], label='train')
    plt.plot(stats['epoch'], stats['test loss'], label='test')
    plt.plot(stats['epoch'], stats['gen loss'], label='gen')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    fig.savefig(os.path.join(results_path, 'uncertainties_ROC'+balanced*'_balanced'+'_MI_cond.pdf'))
    plt.show()
    print('ok')

    '''
    # ====================== simple NN to predict if an ECG has been generated or is coming from the data ============= #
    for piste in range(2,9):
        results_path = os.path.join(npz_path, f'piste{piste}')
        os.makedirs(results_path, exist_ok=True)
        test_loader = DataLoader(DiscriminatorDataset(npz_test, piste), batch_size=1024, shuffle=True, num_workers=8)
        train_loader = DataLoader(DiscriminatorDataset(npz_train, piste), batch_size=1024, shuffle=True, num_workers=8)
        model = resnetSmall(input_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

        criterion = nn.BCEWithLogitsLoss()
        model_path = os.path.join(results_path, 'epoch.pth')

        all_train_loss, all_train_auc = [], []
        all_val_loss, all_val_auc = [], []
        for epoch in range(50):
            train_loss, train_auc = train(train_loader, model, criterion, epoch, optimizer, device, model_path)
            val_loss, val_auc = evaluate(test_loader, model, criterion, device)
            all_train_loss.append(train_loss)
            all_train_auc.append(train_auc)
            all_val_loss.append(val_loss)
            all_val_auc.append(val_auc)
            df_metrics = pd.DataFrame({'train_loss': all_train_loss, 'train_auc': all_train_auc, 'val_loss': all_val_loss, 'val_auc': all_val_auc})
            df_metrics.to_csv(os.path.join(results_path, 'metrics.csv'))
    '''
if __name__ == '__main__':
    metrics = main()
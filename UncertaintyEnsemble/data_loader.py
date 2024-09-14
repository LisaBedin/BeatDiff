import numpy as np
import torch
from torch.utils.data import Dataset


class UncertaintyDataset(Dataset):
    def __init__(self, npz_data, piste, prefix='target'):
        self.prefix = prefix # target or generated
        self.piste = piste
        self.ecgs = npz_data[f'{prefix}_samples']
        n_train = self.ecgs.shape[0]
        labels = np.ones(n_train)*int(prefix=='generated')
        self.labels = labels
        self.feats = npz_data['class_features']

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, item):
        ECG = torch.Tensor(self.ecgs[item].T)
        if type(self.piste)==np.ndarray:
            ECG = ECG[self.piste]
        elif type(self.piste) == int and self.piste >= 0 and self.piste<9:
            ECG = ECG[self.piste].unsqueeze(0)
        if bool(self.labels[item]):
            lab = torch.Tensor([0, 1])
        else:
            lab = torch.Tensor([1, 0])
        feats = torch.Tensor(self.feats[item])
        return ECG, feats, lab
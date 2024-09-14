import torch
from torch.utils.data.distributed import DistributedSampler
from .sc import SpeechCommands
from .mel2samp import Mel2Samp
from .physionet import PhysionetECG, PhysionetNPZ


def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True, shuffle=True):

    dataset_name = dataset_cfg.pop("_name_")
    if dataset_name == "sc09":
        assert unconditional
        dataset = SpeechCommands(dataset_cfg.data_path)
    elif dataset_name == "ljspeech":
        assert not unconditional
        dataset = Mel2Samp(**dataset_cfg)
    elif dataset_name == "physionet" or dataset_name == 'physionet_arrhythmia':
        dataset = PhysionetECG(**dataset_cfg)
    elif dataset_name == "physionet_centered":
        dataset = PhysionetNPZ(**dataset_cfg)

    dataset_cfg["_name_"] = dataset_name  # Restore

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
        shuffle=shuffle
    )
    return trainloader

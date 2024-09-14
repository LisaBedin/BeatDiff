import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}
add = lambda dic_of_lst, dic: {k: v_lst+[dic[k]] for (k, v_lst) in dic_of_lst.items()}

def run_batches(model, batches, batch_type, optimizer_step=None):
    #stats = stats or StatsLogger(('loss', 'correct'))
    all_loss, all_uncertainties, all_correct = [], [], []
    all_gender = []
    model.train(batch_type=='train')
    for (ecg_batch, feats_batch, labels_batch) in tqdm(batches, total=len(batches)):
        batch = {'input': ecg_batch.cuda().half(), 'feats': feats_batch.cuda().half()}
        output = model(batch)
        all_gender.append(feats_batch.numpy()[:, 0])
        # stats.append(output)
        if batch_type == 'train':
            output['loss'].sum().backward()
            optimizer_step()
            model.zero_grad()
        all_loss.append(output['loss'].detach().cpu().numpy())
        all_uncertainties.append(output['uncertainties'].detach().cpu().numpy())
        all_correct.append(output['correct'].detach().cpu().numpy())
    all_loss = np.concatenate(all_loss)
    all_uncertainties = np.concatenate(all_uncertainties)
    all_correct = np.concatenate(all_correct)
    all_gender = np.concatenate(all_gender)
    stat_dic = {
        f'{batch_type} loss': all_loss.mean(),
        f'{batch_type} metric': all_uncertainties.mean(),
        f'{batch_type} acc': all_correct.mean(),
    }
    print(' | '.join([f'{key}: {val:.2f}' for key, val in stat_dic.items()]))
    if batch_type == 'train':
        info_dic = None
    else:
        info_dic = {
            f'{batch_type}_loss': all_loss,
            f'{batch_type}_metric': all_uncertainties,
            f'{batch_type}_acc': all_correct,
            f'{batch_type}_gender': all_gender
        }
    return model, stat_dic, info_dic


def train_epoch(model, train_batches, test_batches, gen_batches, optimizer_step):
    train_stats, _ = model, run_batches(model, train_batches, 'train', optimizer_step)
    _, test_stats, test_info = run_batches(model, test_batches, 'test')
    gen_stats, gen_info = run_batches(model, gen_batches, 'gen')
    stats = union(train_stats, test_stats, gen_stats)
    infos = union(test_info, gen_info)
    return model, stats, infos


def train_uncertainty(model, optimizer, train_batches, test_batches, gen_batches, epochs, results_path):
    stats = union({'epoch': [], 'lr': []}, *[{
        f'{batch_type} loss': [],
        f'{batch_type} metric': [],
        f'{batch_type} acc': [],
    } for batch_type in ['train', 'test', 'gen']])
    for epoch in range(epochs):
        print(f'epoch={epoch}/{epochs}')
        model, epoch_stats, epoch_infos = train_epoch(model, train_batches, test_batches, gen_batches, optimizer.step)
        if hasattr(optimizer, "param_values"):
            lr = optimizer.param_values()['lr']
        elif hasattr(optimizer, "param_groups"):
            lr = optimizer.param_groups[0]['lr']
        summary = union({'epoch': epoch + 1, 'lr': lr * train_batches.batch_size}, epoch_stats)
        stats = add(stats, summary)
        pd.DataFrame(stats).to_csv(os.path.join(results_path, 'uncertainty_stats.csv'), index=False)
        pd.DataFrame(epoch_infos).to_csv(os.path.join(results_path, 'uncertainty_gen_test.csv'), index=False)

        torch.save(model, os.path.join(results_path, "uncertainty_model.pth"))
    return stats
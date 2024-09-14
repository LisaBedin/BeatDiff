import numpy as np
import matplotlib.pyplot as plt
from functools import partial


limbs_setting = {
    'path': '/mnt/data/lisa/ecg_results/inpainting_samples_20/safe_NSR/NSR_SB_STach_SA.npz',
    'tracks_to_consider': list(range(3, 9)),
    'competitors': ['dowers_reconstruction', 'ekgan_limb']
}
smart_watch_setting = {
    'path': '/mnt/data/lisa/ecg_results/inpainting_samples_20/safe_NSR/NSR_SB_STach_SA_1.npz',
    'tracks_to_consider': list(range(1, 9)),
    'competitors': ['ekgan_I',]
}


def r2_score(samples, tracks_to_consider, ref_data):
    residue = np.linalg.norm(samples[..., tracks_to_consider] - ref_data[..., tracks_to_consider], ord=2,
                             axis=(1, 2)) ** 2
    residue_mean = np.linalg.norm(
        np.mean(ref_data[..., tracks_to_consider], axis=(1, 2))[:, None, None] - ref_data[..., tracks_to_consider],
        ord=2, axis=(1, 2)) ** 2
    return (1 - residue / residue_mean)


def rmse(samples, tracks_to_consider, ref_data):
    residue = (samples[..., tracks_to_consider] - ref_data[..., tracks_to_consider])**2
    return residue.mean(axis=(1, 2))**.5

def mae(samples, tracks_to_consider, ref_data):
    residue = np.abs(samples[..., tracks_to_consider] - ref_data[..., tracks_to_consider])
    return residue.mean(axis=(1, 2))

def conf_interval(samples):
    return f'{np.mean(samples):.2g} +/- {1.96 * samples.std() / (samples.shape[0] ** .5):.2g}'



def eval_data(path, tracks_to_consider, competitors):
    data = np.load(path)
    ref_data = data['ground_truth'].astype(np.float32)
    stats = {}
    for metric_name, metric_fun in [('R2', partial(r2_score, ref_data=ref_data, tracks_to_consider=tracks_to_consider)),
                                    ('RMSE', partial(rmse, ref_data=ref_data, tracks_to_consider=tracks_to_consider)),
                                    ('MAE', partial(mae, ref_data=ref_data, tracks_to_consider=tracks_to_consider))]:
        stats[metric_name] = {"Beat Diff": conf_interval(metric_fun(data['posterior_samples'].mean(axis=1)))}
        for comp in competitors:
            stats[metric_name][comp] = conf_interval(metric_fun(data[comp]))

    return stats


print("LIMBS")
print(eval_data(**limbs_setting))
print("\n")

print("Smart Watch")
print(eval_data(**smart_watch_setting))
print("\n")


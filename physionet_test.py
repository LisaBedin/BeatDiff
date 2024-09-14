import numpy as np
import matplotlib.pyplot as plt
import hydra
import torch
import os
from ecg_inpainting.ipwdp.generative_models import ScoreModel
from ecg_inpainting.ipwdp.optimal_particle_filter import particle_filter
from ecg_inpainting.models.utils import FlattenScoreModel
from omegaconf import OmegaConf
from ecg_inpainting.models import construct_model
from sqlalchemy import create_engine, text
from neurokit2 import ecg_process


def load_net(cfg,
             ckpt_iter=132000,
             local_path='data/denoising_model'):
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    model_cfg = cfg.model
    model_cfg['L'] = 2048
    net = construct_model(model_cfg).cuda()

    # load checkpoint
    print('ckpt_iter', ckpt_iter)
    ckpt_path = local_path
    ckpt_iter = int(ckpt_iter)

    try:
        model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    return net

def read_memmap_from_position(filename,
                              shape,
                              batch_interval):
    byte_size = 2
    n_elements_before_start = batch_interval[0]*shape[1]
    fp = np.memmap(filename=filename,
                   mode='r',
                   dtype='float16',
                   shape=(batch_interval[1] - batch_interval[0], shape[1]),
                   offset=n_elements_before_start*byte_size)
    data = np.asarray(fp).T
    return torch.tensor(data)

def ecgs_only_mask(ecgs):
    r_peaks = ecg_process(ecgs[5], sampling_rate=500)[1]
    mask = torch.zeros_like(ecgs)
    for r_peak in r_peaks['ECG_R_Peaks']:
        mask[:, r_peak - 50:r_peak + 50] = 1
    return mask

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg):
    device = 'cuda:0'
    base_net = load_net(cfg, ckpt_iter=74000)
    betas = torch.linspace(cfg.diffusion.beta_0,
                           cfg.diffusion.beta_T,
                           cfg.diffusion.T)
    alphas = (1 - betas)
    timesteps = torch.linspace(0, cfg.diffusion.T - 1, 25).long()
    eta = 1
    base_model = torch.nn.DataParallel(FlattenScoreModel(
        base_module=base_net,
        shape=(9, 2048))
    )
    base_model.to(device)

    score_model = ScoreModel(net=base_model,
                             alphas_cumprod=torch.cat((torch.ones((1,)), torch.cumprod(alphas, dim=0)), dim=0).to(device)[:-1],
                             device=device
                             )
    database_path = '/mnt/data/gabriel/ps_sedm/self_supervised_nn/data'
    engine = create_engine(f'sqlite:///{database_path}/database.db')
    with engine.connect() as conn:
        query_string = "(target_classes = '20')"  # '(' + ' or '.join(["target_classes like '%"+ i + "%'" for i in categories_to_filter]) + ')'
        ids = conn.execute(text(
            "select dataset_name, dataset_id, record_total_length from records where partition_attribution not like 'Training%' and " + query_string + "group by dataset_name, dataset_id, record_total_length order by RANDOM()")).first()


    ecgs = read_memmap_from_position(f'/mnt/data/gabriel/ps_sedm/self_supervised_nn/data/{ids[0]}/{ids[1]}_ecg.npy',
                                     shape=(2048, 12),
                                     batch_interval=(0, 2048))[3:]
    #mask = ecgs_only_mask(ecgs)
    mask = torch.ones_like(ecgs)
    mask[5:8] = 0
    # mask[:, 500:] = 0
    #mask[7] = 0
    measurement = ecgs.flatten()[mask.flatten() == 1]
    diag = torch.ones(size=(int(mask.sum().item()),))
    coordinate_mask = mask.flatten()
    n_particles = 1000
    initial_particles = torch.randn(n_particles, len(coordinate_mask))
    var_observation = 1e-2

    particles = particle_filter(
        initial_particles=initial_particles.cpu(),
        observation=measurement.cpu(),
        score_model=score_model,
        coordinates_mask=coordinate_mask.cpu(),
        likelihood_diagonal=diag.cpu(),
        var_observation=var_observation,
        timesteps=timesteps,
        eta=eta,
        n_samples_per_gpu_inference=n_particles,
        gaussian_var=1e-4
    )
    for j, ecg in enumerate(ecgs):
        plt.plot(ecg + j, color='blue')
    for part in particles[:100]:
        for j, x in enumerate(part.reshape(-1, 2048)):
            plt.plot(x + j, color='red', alpha=.02)

    plt.show()


if __name__ == '__main__':
    with torch.no_grad():
        main()


# Leveraging an ECG Beat Diffusion Model for Morphological Reconstruction from Indirect Signals

## Preparing the data
* download [Physionet](https://physionet.org/content/challenge-2021/1.0.3/)
* initialize database: ```python beat_db/create_database.py <path to save preprocessed data>```
* process single beats: ```python beat_db/generate_db.py <path of Physionet raw data> <path to save preprocessed beats> <number of processes>```
* (optional) process 10s ECGs: ```python beat_db/generate_db_long_term.py <path of Physionet raw data> <path to save preprocessed ECGs> <number of processes>```

## Training the diffusion model
* training on single beats:
```python beat_net/train.py dataset=pysionet_ecg train=baseline dataset.database_path=<path of preprocessed beats> hydra.run.dir=<path to save the model>```
* (optional) training on 10s ECGs:
```python beat_net/train.py dataset=pysionet_ecg_long_term train=baseline_long_term dataset.database_path=<path of preprocessed 10s ECGs> hydra.run.dir=<path to save the model>```

## Post-training tasks
### Baselines:
* DeScoD for denoising: ```results/models/DeScoD``` obtained with ```Score-based-ECG-Denoising``` trained on the preprocessed beat physionet database corrupted with bw from nstdb
* EkGAN for inpainting from lead I: ```results/models/Ekgan_leadI```
* EkGAN for inpainting from limb leads: ```results/models/Ekgan_limb_leads```
* AAE for anomaly detection: ```results/models/AAE``` obtained with ```python baseline_anomaly.py paths.db_path=<path of preprocessed beats db>```

### Denoising on MIT-BIH Noise Stress Test database
* download [nstdb](https://physionet.org/content/nstdb/1.0.0/)
* EM-beatdiff for baseline-wander removal: ```python EMbeat_diff_denoising.py denoising=bw paths.db_path=<path of preprocessed beats db> paths.noise_path=<path of nstdb> paths.results_path=<folder for storing results>```
* EM-beatdiff for electrode-motion removal: ```python EMbeat_diff_denoising.py denoising=em paths.db_path=<path of preprocessed beats db> paths.noise_path=<path of nstdb> paths.results_path=<folder for storing results>```

### Inpainting
* EM-beatdiff for reconstructing healthy heartbeats from leadI, limb leads, qrs, st-segment: ```python EMbeat_diff_inpainting.py --multirun inpainting=leadI,limbs,qrs,st paths.db_path=<path of preprocessed beats db> paths.results_path=<folder for storing results>```
* for anomaly detection perform the inpatining on different heart conditions: ```python EMbeat_diff_inpainting.py --multirun inpainting.labels=['MI'],['LQTS'] inpainting=leadI,limbs,qrs,st paths.db_path=<path of preprocessed beats db> paths.results_path=<folder for storing results>```

## Citation
If you use this code, please cite the following
```
@article{bedin2024beatdiff,
      title={Leveraging an ECG Beat Diffusion Model for Morphological Reconstruction from Indirect Signals}, 
      author={Lisa Bedin and Gabriel Cardoso and Josselin Duchateau and Remi Dubois and Eric Moulines},
      year={2024},
      volume={37},
      journal={Advances in Neural Information Processing Systems},
}
```

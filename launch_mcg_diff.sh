#!/bin/bash
#process_id=$1
# checkpoint=$1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.1/bin \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python missing_lead.py

CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python baseline_anomaly.py --multirun \
baseline.model_type=AE baseline.adversarial=false baseline.denoising=true

CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python baseline_anomaly.py --multirun \
baseline.model_type=AE baseline.adversarial=true baseline.denoising=false baseline.TCN_opt=true

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python baseline_anomaly.py --multirun \
baseline.model_type=VAE baseline.adversarial=true baseline.denoising=false

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python baseline_anomaly.py --multirun \
baseline.model_type=VAE baseline.adversarial=false baseline.denoising=false baseline.kl=1















CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python mcg_diff.py --multirun \
mcg_diff.setting=Sfinal \
mcg_diff.labels=['SA']  # ,'S2','S3','S4' cfg.mcg_diff.max_patients

CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python discriminator_eval.py # checkpoint=$checkpoint
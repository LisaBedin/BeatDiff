#!/bin/bash
#process_id=$1

#CUDA_VISIBLE_DEVICES=$process_id LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
#PATH=/usr/local/cuda-12.1/bin \
#/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python old_mcg_diff.py \
#net_config.generate.process_idx=$process_id

CUDA_VISIBLE_DEVICES=4,5,6,7 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python qrs_rr.py

CUDA_VISIBLE_DEVICES=1,2,3,4,5 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python mcg_diff.py --multirun \
net_config.generate_eval.n_samples=20 \
net_config.generate_eval.n_samples_per_filter=64 \
net_config.generate_eval.n_gpu=5 \
net_config.generate_eval.n_condition_per_diseases=2 \
'net_config.generate_eval.conditioning_pistes=[0, 1, 2]' \
'net_config.generate_eval.initial_seed=[0, 0]'
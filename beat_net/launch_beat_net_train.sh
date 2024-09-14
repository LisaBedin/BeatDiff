#!/bin/bash
#process_id=$1

#CUDA_VISIBLE_DEVICES=$process_id LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
#PATH=/usr/local/cuda-12.1/bin \
#/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python old_mcg_diff.py \
#net_config.generate.process_idx=$process_id


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
PATH=/usr/local/cuda-12.2/bin \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 \
/mnt/data/lisa/.virtualenvs/ecg_inpainting/bin/python train.py model=deeper
#!/bin/bash

export DEVICE_ID=1

python train_vqvae.py \
  --use_discriminator True \
  --dataset_name video \
  --data_path /disk3/katekong/magvit/datasets/ucf101/ \
  --num_frames 16 \
  --crop_size 128 \
  --num_parallel_workers 1 \
  --drop_overflow_update False \
  --batch_size 1 \
  --mode 0
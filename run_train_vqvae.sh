#!/bin/bash

export DEVICE_ID=7

nohup python train_vqvae.py \
  --use_discriminator True \
  --use_ema True \
  --dataset_name video \
  --data_path /disk3/katekong/magvit/datasets/ucf101/rec_train/ \
  --num_frames 16 \
  --crop_size 128 \
  --num_parallel_workers 1 \
  --drop_overflow_update True \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --clip_grad True \
  --base_learning_rate 1.0e-07 \
  --scale_lr False \
  --init_loss_scale 256 \
  --loss_scaler_type dynamic \
  --dtype fp16 \
  --mode 0 \
  > train.log 2>&1 &
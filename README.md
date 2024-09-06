# MAGVIT-v2: Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation

This folder contains the Mindspore implementation of [MAGVIT-v2](https://arxiv.org/pdf/2310.05737).

## Features

- [x] Lookup-Free-Quantization (LFQ)
- [x] VQVAE-2d Training
- [x] VQVAE-3d Training
- [ ] VQGAN Training
- [ ] MAGVIT-v2 Transformers
- [ ] MAGVIT-v2 Training

## Requirements

1. install Mindspore 2.3 according to
2. For Ascend users, please install CANN version:

```
pip install -r requirements.txt
```

## Datasets

We use UCF-101 as an example.

## Training

### 1. VQVAE

The training of VQVAE can be divided into two stages: VQVAE-2d and VQVAE-3d, where VQVAE-2d is the initialization of VQVAE-3d.

#### 1.1 VQVAE-2d

For the pretraining of VQVAE-2d, we provide a pretrained model weights as follow:

| Model | Dataset | Image Size | Weights |
|-------| ------- | -----------| -- |
| VQVAE-2d | ImageNet | 128x128 | |

If you would like you pretrain your weights, you can:
- 1) Prepare datasets
 We take ImageNet as an example

- 2) Run the training script as below:

 ```
 bash scripts/run_train_vqvae_2d.sh
 ```

- 3) Inflate 2d to 3d
 We provide a script for inflation, you can run the command:
 ```
 python tools/inflate_vae2d_to_3d.py --src VQVAE_2D_MODEL_PATH --target INFALTED_MODEL_PATH
 ```

#### 1.2 VQVAE-3d

Modify the path of pretrained VQVAE-2d model in [run_train_vqvae.sh](./scripts/run_eval_vqvae.sh)

Run the training script as below:

 ```
 bash scripts/run_train_vqvae.sh
 ```

### 2. MAGVIT-v2

The training script of MAGVIT-v2 is still under development, so stay tuned!


## Evaluation
We provide two common evaluation metrics in our implementations: PSNR and SSIM.
To run the evaluations, you can use the following command:

```
bash scripts/run_eval_vqvae.sh
```
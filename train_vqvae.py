"""Trainer for VQVAE"""

import logging
import os
import shutil
import sys
import time

import yaml
from omegaconf import OmegaConf

from videogvt.config.vqgan3d_magvit_v2_config import get_config
from videogvt.config.vqvae_train_args import parse_args
from videogvt.data.loader import create_dataloader
from videogvt.models.vqvae import VQVAE3D, VQVAEOpenSora, StyleGANDiscriminator
from videogvt.models.vqvae.net_with_loss import DiscriminatorWithLoss, GeneratorWithLoss

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.env import init_train_env
from mindone.trainers.callback import (
    EvalSaveCallback,
    OverflowMonitor,
    ProfilerCallback,
)
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler

# from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindcv.optim import create_optimizer

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)


def create_loss_scaler(
    loss_scaler_type, init_loss_scale, loss_scale_factor=2, scale_window=1000
):
    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale,
            scale_factor=args.loss_scale_factor,
            scale_window=args.scale_window,
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    return loss_scaler


def main(args):
    # 1. init
    # ascend_config={"precision_mode": "allow_fp32_to_fp16"}
    ascend_config = {"precision_mode": "allow_mix_precision_bf16"}
    device_id, rank_id, device_num = init_train_env(
        args.mode,
        device_target=args.device_target,
        seed=args.seed,
        distributed=args.use_parallel,
        ascend_config=ascend_config,
    )
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=rank_id,
        log_level=eval(args.log_level),
    )

    # 2. build models
    #  vqvae (G)
    model_config = get_config("B")
    dtype = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
    model_class = {"magvit": VQVAE3D, "opensora": VQVAEOpenSora}[args.model_class]
    vqvae = model_class(
        model_config,
        quantization="lfq",
        is_training=True,
        video_contains_first_frame=args.contains_first_frame,
        separate_first_frame_encoding=args.separate_first_frame_encoding,
        dtype=dtype,
    )
    if args.pretrained is not None:
        logger.info(f"Loading vqvae from {args.pretrained}")
        ms.load_checkpoint(args.pretrained, vqvae, filter_prefix=None)

    # discriminator (D)
    use_discriminator = args.use_discriminator and (
        model_config.lr_configs.disc_weight > 0.0
    )

    if args.use_discriminator and (model_config.lr_configs.disc_weight <= 0.0):
        logging.warning("use_discriminator is True but disc_weight is 0.")

    if use_discriminator:
        crop_size = int(args.crop_size)
        frame_size = int(args.num_frames)
        disc = StyleGANDiscriminator(
            model_config, crop_size, crop_size, frame_size, dtype=dtype
        )
    else:
        disc = None

    # mixed precision
    # TODO: set softmax, sigmoid computed in FP32. manually set inside network since they are ops, instead of layers whose precision will be set by AMP level.
    if args.dtype not in ["fp32", "bf16"]:
        amp_level = "O2"
        vqvae = auto_mixed_precision(vqvae, amp_level, dtype)
        if use_discriminator:
            disc = auto_mixed_precision(disc, amp_level, dtype)
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")
    else:
        amp_level = "O0"

    # 3. build net with loss (core)
    # G with loss
    vqvae_with_loss = GeneratorWithLoss(
        vqvae, discriminator=disc, **model_config.lr_configs, dtype=dtype
    )
    disc_start = model_config.lr_configs.disc_start

    # D with loss
    if use_discriminator:
        disc_with_loss = DiscriminatorWithLoss(vqvae, disc, disc_start)

    tot_params, trainable_params = count_params(vqvae_with_loss)
    logger.info(
        "Total params {:,}; Trainable params {:,}".format(tot_params, trainable_params)
    )

    # 4. build dataset
    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        random_crop=args.random_crop,
        flip=args.flip,
    )
    if args.dataset_name == "video":
        ds_config.update(
            dict(
                sample_stride=args.frame_stride,
                sample_n_frames=args.num_frames,
                return_image=False,
            )
        )
        assert not (
            # model_config.generator.params.ddconfig.split_time_upsample
            args.num_frames % 2 == 0
            and False
        ), "num of frames must be odd if split_time_upsample is True"
    else:
        ds_config.update(dict(expand_dim_t=args.expand_dim_t))
    dataset = create_dataloader(
        ds_config=ds_config,
        batch_size=args.batch_size,
        ds_name=args.dataset_name,
        num_parallel_workers=args.num_parallel_workers,
        shuffle=args.shuffle,
        device_num=device_num,
        rank_id=rank_id,
    )
    dataset_size = dataset.get_dataset_size()

    # 5. build training utils
    # torch scale lr by: model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    if args.scale_lr:
        learning_rate = (
            args.base_learning_rate
            * args.batch_size
            * args.gradient_accumulation_steps
            * device_num
        )
    else:
        learning_rate = args.base_learning_rate

    if not args.decay_steps:
        args.decay_steps = max(1, args.epochs * dataset_size - args.warmup_steps)

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    update_logvar = False  # in torch, vqvae_with_loss.logvar  is not updated.
    if update_logvar:
        vqvae_params_to_update = [
            vqvae_with_loss.vqvae.trainable_params(),
            vqvae_with_loss.logvar,
        ]
    else:
        vqvae_params_to_update = vqvae_with_loss.vqvae.trainable_params()

    optim_vqvae = create_optimizer(
        vqvae_params_to_update,
        opt=args.optim,
        weight_decay=args.weight_decay,
        lr=lr,
        eps=1e-08,
        beta1=0.5,
        beta2=0.99,
    )

    loss_scaler_vqvae = create_loss_scaler(
        args.loss_scaler_type,
        args.init_loss_scale,
        args.loss_scale_factor,
        args.scale_window,
    )

    if use_discriminator:
        optim_disc = create_optimizer(
            disc_with_loss.discriminator.trainable_params(),
            betas=args.betas,
            name=args.optim,
            lr=lr,  # since lr is a shared list
            group_strategy=args.group_strategy,
            weight_decay=args.weight_decay,
        )

        loss_scaler_disc = create_loss_scaler(
            args.loss_scaler_type,
            args.init_loss_scale,
            args.loss_scale_factor,
            args.scale_window,
        )

    ema = (
        EMA(
            vqvae_with_loss.vqvae,
            ema_decay=args.ema_decay,
            offloading=False,
            dtype=dtype
        ).to_float(dtype)
        if args.use_ema
        else None
    )

    # resume training states
    # TODO: resume Discriminator if used
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    if args.resume:
        resume_ckpt = (
            os.path.join(ckpt_dir, "train_resume.ckpt")
            if isinstance(args.resume, bool)
            else args.resume
        )

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            vqvae_with_loss, optim_vqvae, resume_ckpt
        )
        loss_scaler_vqvae.loss_scale_value = loss_scale
        loss_scaler_vqvae.cur_iter = cur_iter
        loss_scaler_vqvae.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    # training step
    training_step_vqvae = TrainOneStepWrapper(
        vqvae_with_loss,
        optimizer=optim_vqvae,
        scale_sense=loss_scaler_vqvae,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    if use_discriminator:
        training_step_disc = TrainOneStepWrapper(
            disc_with_loss,
            optimizer=optim_disc,
            scale_sense=loss_scaler_disc,
            drop_overflow_update=args.drop_overflow_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm,
            ema=None,  # No ema for disriminator
        )

    if rank_id == 0:
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"amp level: {amp_level}",
                f"dtype: {args.dtype}",
                f"Data path: {args.data_path}",
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Rescale size: {args.size}",
                f"Crop size: {args.crop_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

    # 6. training process


    logger.info("Start training...")
    # backup config files
    # shutil.copyfile(args.config, os.path.join(args.output_path, os.path.basename(args.config)))

    # with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
    #     yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    if rank_id == 0:
        ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)

    # output_numpy=True ?
    ds_iter = dataset.create_dict_iterator(args.epochs - start_epoch)
    avg_loss = 0.0

    for epoch in range(start_epoch, args.epochs):
        start_time_e = time.time()
        for step, data in enumerate(ds_iter):
            start_time_s = time.time()
            x = data[args.dataset_name]

            global_step = epoch * dataset_size + step
            global_step = ms.Tensor(global_step, dtype=ms.int64)

            # NOTE: inputs must match the order in GeneratorWithLoss.construct
            loss_vqvae_t, overflow, scaling_sens = training_step_vqvae(x)

            if use_discriminator:
                loss_disc_t, overflow_d, scaling_sens_d = training_step_disc(x)

            cur_global_step = (
                epoch * dataset_size + step + 1
            )  # starting from 1 for logging
            if overflow:
                logger.warning(f"Overflow occurs in step {cur_global_step}")

            # log
            loss_vqvae = float(loss_vqvae_t.asnumpy())
            avg_loss += loss_vqvae
            step_time = time.time() - start_time_s
            if (step+1) % args.log_interval == 0:
                avg_loss /= float(args.log_interval)
                logger.info(
                    f"E: {epoch+1}, S: {step+1}, Loss vqvae avg: {avg_loss:.4f}, Step time: {step_time*1000:.2f}ms"
                )
                avg_loss = 0.0

                if use_discriminator:
                    loss_disc = float(loss_disc_t.asnumpy())
                    logger.info(f"Loss disc: {loss_disc:.4f}")

        epoch_cost = time.time() - start_time_e
        per_step_time = epoch_cost / dataset_size
        cur_epoch = epoch + 1
        logger.info(
            f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], "
            f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time*1000:.2f}ms, "
        )
        if rank_id == 0:
            if (cur_epoch % args.ckpt_save_interval == 0) or (
                cur_epoch == args.epochs
            ):
                ckpt_name = f"vqvae_cb_f8-e{cur_epoch}.ckpt"
                if ema is not None:
                    ema.swap_before_eval()

                ckpt_manager.save(
                    vqvae, None, ckpt_name=ckpt_name, append_dict=None
                )
                if ema is not None:
                    ema.swap_after_eval()

        # TODO: eval while training


if __name__ == "__main__":
    args = parse_args()
    main(args)

import os
from mindspore import nn, ops
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size, init

from mindone.utils.amp import auto_mixed_precision

from videogvt.config.vqgan3d_magvit_v2_config import get_config
from videogvt.config.vqvae_train_args import parse_args
from videogvt.models.vqvae import VQVAE3D, StyleGANDiscriminator
from videogvt.data.loader import create_dataloader


def l2_loss(y_true, y_pred):
    diff = y_true - y_pred
    return ops.mean(ops.square(diff))


def create_dataset(args, n_epochs):
    rank_id = int(os.getenv("RANK_ID", "0")) # int(os.getenv("RANK_ID", "0")) # get_rank() # int(os.getenv("RANK_ID", "0"))
    rank_size = 1 # get_group_size()

    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        random_crop=args.random_crop,
        flip=args.flip,
    )

    data_set = create_dataloader(
        ds_config=ds_config,
        batch_size=1,
        ds_name=args.dataset_name,
        num_parallel_workers=1,
        shuffle=True,
        device_num=rank_size,
        rank_id=rank_id,
    )

    return data_set.create_dict_iterator(n_epochs)


if __name__ == "__main__":
    args = parse_args()

    device_id = int(os.getenv('DEVICE_ID', "0"))
    ms.set_context(device_id=device_id)
    ms.set_context(mode=ms.GRAPH_MODE)
    # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    # init()
    ms.set_seed(1)

    model_config = get_config("B")
    bs = 1
    in_channels = 3
    crop_size = 256
    frame_size = 16
    net = VQVAE3D(model_config, lookup_free_quantization=True, is_training=True, dtype=ms.float16)
    disc = StyleGANDiscriminator(model_config, crop_size, crop_size, frame_size, dtype=ms.float16)

    net = auto_mixed_precision(net, "O2", ms.float16)
    disc = auto_mixed_precision(disc, "O2", ms.float16)

    n_epochs = 3
    data_set = create_dataset(args, n_epochs)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean") # nn.L1Loss(reduction="none")
    loss_l1 = nn.L1Loss(reduction="mean")
    optimizer = nn.SGD(net.trainable_params(), 1e-5)

    def forward_fn(data):
        _, _, logits, aux_loss = net(data)
        aux_loss = aux_loss.astype(ms.float32)
        logits_fake = disc(logits)
        logits_fake = logits_fake.astype(ms.float32)
        g_loss = -ops.mean(logits_fake)
        rec_loss = loss_l1(logits, ops.stop_gradient(data)).astype(ms.float32)
        loss = aux_loss + 5.0 * rec_loss
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
    # grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

    for epoch in range(n_epochs):
        i = 0
        for data in data_set:
            disc.set_grad(False)
            disc.set_train(False)
            x = data[args.dataset_name]
            x = x.astype(ms.float16)
            (loss, _), grads = grad_fn(x)
            # grads = grad_reducer(grads)
            optimizer(grads)
            if i % n_epochs == 0:
                print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss))
            i += 1

import mindspore as ms

from .vqvae import VQVAE_2D, VQVAE_3D
from .discriminator import StyleGANDiscriminator


def build_model(model_name, model_config, is_training=True, dtype=ms.float32):
    if model_name == "vqvae-2d":
        model = VQVAE_2D(
            model_config,
            is_training=is_training,
            dtype=dtype,
        )
    elif model_name == "vqvae-3d":
        model = VQVAE_3D(
            model_config,
            is_training=is_training,
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")

    if model_config.from_pretrained is not None:
        param_dict = ms.load_checkpoint(model_config.from_pretrained)
        ms.load_param_into_net(model, param_dict)

    return model

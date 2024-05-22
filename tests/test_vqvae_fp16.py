import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.models.vqvae import VQVAE3D, StyleGANDiscriminator
from videogvt.config.vqgan3d_magvit_v2_config import get_config
from mindone.utils.amp import auto_mixed_precision

# GRAPH_MODE - 0
# PYNATIVE_MODE - 1

context.set_context(
    mode=0, device_target="Ascend", device_id=1
)

config = get_config("B")
bs = 1
in_channels = 3
crop_size = 256
frame_size = 16
model = VQVAE3D(config, lookup_free_quantization=True, is_training=True, dtype=ms.float16)
discriminator = StyleGANDiscriminator(config, crop_size, crop_size, frame_size, dtype=ms.float16)

amp_level = "O2"
dtype = ms.float16
model = auto_mixed_precision(model, amp_level, dtype)
discriminator = auto_mixed_precision(discriminator, amp_level, dtype)

x = ms.Tensor(np.random.rand(bs, in_channels, frame_size, crop_size, crop_size), ms.float16)

z_e, z_q, x_hat, aux_loss = model(x)

# embedding_loss, x_hat, z_e, z_q = model(x)
logit_true = discriminator(x)
logit_fake = discriminator(x_hat)

print("original input shape: ", x.shape)
print("encoded shape: ", z_e.shape)
print("quantized shape: ", z_q.shape)
print("reconstructed shape:", x_hat.shape)
print("quantization loss: ", aux_loss)
# print("x_discriminate shape: ", x_dis.shape)
print("logits of true sample: ", logit_true)
print("logits of fake sample: ", logit_fake)

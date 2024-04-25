import mindspore as ms
import numpy as np
from mindspore import context

from videogvt.models.vqvae import VQVAE3D, StyleGANDiscriminator
from videogvt.config.vqgan3d_magvit_v2_config import get_config

# GRAPH_MODE
# PYNATIVE_MODE

context.set_context(
    mode=context.GRAPH_MODE, device_target="Ascend", device_id=1
)

config = get_config("B")
model = VQVAE3D(config)
discriminator = StyleGANDiscriminator(config, 128, 128, 16)

x = ms.Tensor(np.random.rand(2, 3, 16, 128, 128), ms.float32)

embedding_loss, x_hat, perplexity, z_e, z_q = model(x)
logit_true = discriminator(x)
logit_fake = discriminator(x_hat)

print("original input shape: ", x.shape)
print("encoded shape: ", z_e.shape)
print("quantized shape: ", z_q.shape)
print("reconstructed shape:", x_hat.shape)
print("quantization loss: ", embedding_loss)
# print("x_discriminate shape: ", x_dis.shape)
print("logits of true sample: ", logit_true)
print("logits of fake sample: ", logit_fake)

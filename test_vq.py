import mindspore as ms
import numpy as np
from mindspore import context

from videogvt.models.vqvae import VQVAE3D
from videogvt.config.vqgan3d_magvit_v2_config import get_config

# GRAPH_MODE
# PYNATIVE_MODE

context.set_context(
    mode=context.GRAPH_MODE, device_target="Ascend", device_id=1
)

config = get_config("B")
model = VQVAE3D(config)

x = ms.Tensor(np.random.rand(2, 3, 16, 128, 128), ms.float32)

embedding_loss, x_hat, perplexity, z_e, z_q = model(x)

print("original input shape: ", x.shape)
print("encoded shape: ", z_e.shape)
print("quantized shape: ", z_q.shape)
print("reconstructed shape:", x_hat.shape)
print("quantization loss: ", embedding_loss)
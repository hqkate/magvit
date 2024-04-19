import mindspore as ms
import numpy as np
from mindspore import context

from videogvt.models.vqvae import VQVAE

# GRAPH_MODE
# PYNATIVE_MODE

context.set_context(
    mode=context.GRAPH_MODE, device_target="Ascend", device_id=1
)

model = VQVAE(128, 32, 2, 512, 64, 0.25)

x = ms.Tensor(np.random.rand(2, 3, 225, 225), ms.float32)

res = model(x)

print(res)
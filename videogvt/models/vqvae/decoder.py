import mindspore as ms
from mindspore import nn, ops
import numpy as np

from videogvt.models.vqvae.residual import ResidualStack


class Decoder(nn.Cell):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.SequentialCell(
            nn.Conv2dTranspose(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1, pad_mode='pad'),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.Conv2dTranspose(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2dTranspose(h_dim//2, 3, kernel_size=kernel,
                               stride=stride, padding=1, pad_mode='pad')
        )

    def construct(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = ms.Tensor(x, ms.float32)

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)

import mindspore as ms
from mindspore import nn, ops
import numpy as np


class ResidualLayer(nn.Cell):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, pad_mode='pad', has_bias=False),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, has_bias=False)
        )

    def construct(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Cell):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.CellList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def construct(self, x):
        for layer in self.stack:
            x = layer(x)
        x = ops.relu(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = ms.Tensor(x, ms.float32)
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
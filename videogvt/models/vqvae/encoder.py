import mindspore as ms
from mindspore import nn, ops
import numpy as np

from videogvt.models.vqvae.model_utils import (
    ResnetBlock3D,
    ResidualStack,
    ResidualNet,
    Downsample,
    CausalConv3d,
    GroupNormExtend,
    _get_selected_flags,
)


class Encoder(nn.Cell):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.SequentialCell(
            nn.Conv2d(
                in_dim,
                h_dim // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                pad_mode="pad",
            ),
            nn.ReLU(),
            nn.Conv2d(
                h_dim // 2,
                h_dim,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                pad_mode="pad",
            ),
            nn.ReLU(),
            nn.Conv2d(
                h_dim,
                h_dim,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
                pad_mode="pad",
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def construct(self, x):
        return self.conv_stack(x)


class Encoder3D(nn.Cell):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, config):
        super(Encoder3D, self).__init__()

        self.config = config

        self.in_channels = 3
        self.out_channels = 18
        self.init_dim = self.config.vqvae.filters  # 128
        self.input_conv_kernel_size = (3, 3, 3)  # (7, 7, 7)
        self.output_conv_kernel_size = (1, 1, 1)

        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_enc_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.temporal_downsample = self.config.vqvae.temporal_downsample
        if isinstance(self.temporal_downsample, int):
            self.temporal_downsample = _get_selected_flags(
                len(self.channel_multipliers) - 1, self.temporal_downsample, False
            )
        self.embedding_dim = self.config.vqvae.embedding_dim
        self.conv_downsample = self.config.vqvae.conv_downsample
        self.custom_conv_padding = self.config.vqvae.get("custom_conv_padding")
        self.norm_type = self.config.vqvae.norm_type
        self.num_remat_block = self.config.vqvae.get("num_enc_remat_blocks", 0)
        if self.config.vqvae.activation_fn == "relu":
            self.activation_fn = ops.relu
        elif self.config.vqvae.activation_fn == "elu":
            self.activation_fn = ops.elu
        else:
            raise NotImplementedError

        dim_gp = self.filters * self.channel_multipliers[-1]
        self.conv_in = CausalConv3d(
            self.in_channels, self.init_dim, self.input_conv_kernel_size, padding=1, dtype=ms.float16
        )
        self.conv_out = CausalConv3d(
            dim_gp, self.out_channels, self.output_conv_kernel_size, padding=0, dtype=ms.float16
        )
        self.residual_stack = nn.SequentialCell()
        self.norm = GroupNormExtend(dim_gp, dim_gp)

        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]

            if i == 0:
                dim_in = self.filters
                t_stride = (1, 2, 2)
            else:
                dim_in = self.filters * self.channel_multipliers[i - 1]
                t_stride = (2, 2, 2)

            self.residual_stack.append(ResnetBlock3D(dim_in, filters))

            for _ in range(self.num_res_blocks - 1):
                self.residual_stack.append(ResnetBlock3D(filters, filters))

            if self.temporal_downsample[i]:
                self.residual_stack.append(
                    CausalConv3d(
                        filters,
                        filters,
                        kernel_size=(3, 3, 3),
                        stride=t_stride,
                        padding=1,
                        dtype=ms.float16,
                    )
                )

    def construct(self, x):
        x = self.conv_in(x)
        x = self.residual_stack(x)
        x = self.norm(x)
        x = self.activation_fn(x)
        x = self.conv_out(x)
        return x


class Encoder_v2(nn.Cell):
    def __init__(self, config):
        super(Encoder_v2, self).__init__()

        self.config = config
        self.in_channels = 3
        self.init_dim = 64
        self.input_conv_kernel_size = (7, 7, 7)
        self.output_conv_kernel_size = (3, 3, 3)
        self.layers = ("residual", "residual", "residual")

        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_enc_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.temporal_downsample = self.config.vqvae.temporal_downsample
        if isinstance(self.temporal_downsample, int):
            self.temporal_downsample = _get_selected_flags(
                len(self.channel_multipliers) - 1, self.temporal_downsample, False
            )
        self.embedding_dim = self.config.vqvae.embedding_dim
        self.conv_downsample = self.config.vqvae.conv_downsample
        self.custom_conv_padding = self.config.vqvae.get("custom_conv_padding")
        self.norm_type = self.config.vqvae.norm_type
        self.num_remat_block = self.config.vqvae.get("num_enc_remat_blocks", 0)
        if self.config.vqvae.activation_fn == "relu":
            self.activation_fn = ops.relu
        elif self.config.vqvae.activation_fn == "elu":
            self.activation_fn = ops.elu
        else:
            raise NotImplementedError

        self.conv_in = CausalConv3d(
            self.in_channels, self.init_dim, self.input_conv_kernel_size, dtype=ms.float16
        )
        self.conv_in_first_frame = nn.Identity()
        self.conv_out_first_frame = nn.Identity()
        self.conv_out = CausalConv3d(
            self.init_dim, self.in_channels, self.output_conv_kernel_size, dtype=ms.float16
        )

        dim = self.init_dim
        dim_out = dim
        residual_conv_kernel_size = 3

        self.enc_layers = nn.CellList()
        for layer in self.layers:
            self.enc_layers.append(ResidualNet(dim, residual_conv_kernel_size))
        self.norm = nn.LayerNorm(dim_out)

    def construct(self, x):
        x = self.conv_in(x)

        for layer in self.enc_layers:
            x = layer(x)

        # rearrange
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = ms.Tensor(x, ms.float32)

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print("Encoder out shape:", encoder_out.shape)

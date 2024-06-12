import mindspore as ms
from mindspore import nn, ops
import numpy as np

from videogvt.models.vqvae.model_utils import (
    ResnetBlock3D,
    ResidualStack,
    CausalConv3d,
    GroupNormExtend,
    SpatialDownsample2x,
    TimeDownsample2x,
    nonlinearity,
    _get_selected_flags,
)
from videogvt.models.vqvae.atten import AttentionResidualBlock


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

    def __init__(self, config, dtype=ms.float32):
        super(Encoder3D, self).__init__()

        self.config = config

        self.in_channels = self.config.vqvae.channels  # 3
        self.out_channels = self.config.vqvae.middle_channels  # 18
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
        self.downsample = self.config.vqvae.get("downsample", "time+spatial")
        self.custom_conv_padding = self.config.vqvae.get("custom_conv_padding")
        self.norm_type = self.config.vqvae.norm_type
        self.num_remat_block = self.config.vqvae.get("num_enc_remat_blocks", 0)

        dim_gp = self.filters * self.channel_multipliers[-1]

        self.conv_out = CausalConv3d(
            dim_gp,
            self.out_channels,
            self.output_conv_kernel_size,
            padding=0,
            dtype=dtype,
        )
        self.residual_stack = nn.SequentialCell()
        self.norm = GroupNormExtend(num_groups=32, num_channels=dim_gp, dtype=dtype)

        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]

            if i == 0:
                dim_in = self.filters
                t_stride = (1, 2, 2)
            else:
                dim_in = self.filters * self.channel_multipliers[i - 1]
                t_stride = (2, 2, 2)

            self.residual_stack.append(ResnetBlock3D(dim_in, filters, dtype=dtype))

            for _ in range(self.num_res_blocks - 1):
                self.residual_stack.append(ResnetBlock3D(filters, filters, dtype=dtype))

            if self.temporal_downsample[i]:
                if self.downsample == "conv":
                    self.residual_stack.append(
                        CausalConv3d(
                            filters,
                            filters,
                            kernel_size=(3, 3, 3),
                            stride=t_stride,
                            padding=1,
                            dtype=dtype,
                        )
                    )
                elif self.downsample == "time+spatial":
                    if t_stride[0] > 1:
                        self.residual_stack.append(
                            TimeDownsample2x(filters, filters, dtype=dtype)
                        )
                        self.residual_stack.append(nn.ReLU())
                    self.residual_stack.append(
                        SpatialDownsample2x(filters, filters, dtype=dtype)
                    )
                    self.residual_stack.append(nn.ReLU())
                else:
                    raise NotImplementedError(f"Unknown downsampler: {self.downsample}")

    def construct(self, x):
        # x = self.conv_in(x)
        x = self.residual_stack(x)
        x = self.norm(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return x


class EncoderOpenSora(nn.Cell):
    def __init__(self, config, dtype=ms.float32):
        super(EncoderOpenSora, self).__init__()

        self.config = config
        self.dtype = dtype

        n_in = config.vqvae.filters
        n_hiddens = 224
        n_res_layers = 4

        spatial_downsample = 3
        self.spatial_conv = nn.CellList()
        for i in range(spatial_downsample):
            in_channels = n_in if i == 0 else n_hiddens
            conv = SpatialDownsample2x(in_channels, n_hiddens, dtype=dtype)
            self.spatial_conv.append(conv)
        self.spatial_res_stack = nn.SequentialCell(
            *[
                AttentionResidualBlock(n_hiddens, dtype=dtype)
                for _ in range(n_res_layers)
            ],
            GroupNormExtend(
                num_groups=1, num_channels=n_hiddens, dtype=dtype
            ),  # nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )
        time_downsample = 2
        self.time_conv = nn.CellList()
        for i in range(time_downsample):
            conv = TimeDownsample2x(n_hiddens, n_hiddens, dtype=dtype)
            self.time_conv.append(conv)
        self.time_res_stack = nn.SequentialCell(
            *[
                AttentionResidualBlock(n_hiddens, dtype=dtype)
                for _ in range(n_res_layers)
            ],
            GroupNormExtend(
                num_groups=1, num_channels=n_hiddens, dtype=dtype
            ),  # nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )

    def construct(self, x):
        h = x
        for conv in self.spatial_conv:
            h = ops.relu(conv(h))
        h = self.spatial_res_stack(h)
        for conv in self.time_conv:
            h = ops.relu(conv(h))
        h = self.time_res_stack(h)
        return h


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = ms.Tensor(x, ms.float32)

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print("Encoder out shape:", encoder_out.shape)

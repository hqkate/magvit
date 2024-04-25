import mindspore as ms
from mindspore import nn, ops
import numpy as np

from videogvt.models.vqvae.model_utils import (
    ResnetBlock3D,
    ResidualStack,
    Upsample3D,
    CausalConv3d,
    GroupNormExtend,
    _get_selected_flags,
)


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
                in_dim,
                h_dim,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
                pad_mode="pad",
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.Conv2dTranspose(
                h_dim,
                h_dim // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                pad_mode="pad",
            ),
            nn.ReLU(),
            nn.Conv2dTranspose(
                h_dim // 2,
                3,
                kernel_size=kernel,
                stride=stride,
                padding=1,
                pad_mode="pad",
            ),
        )

    def construct(self, x):
        return self.inverse_conv_stack(x)


class Decoder3D(nn.Cell):
    """Decoder Blocks."""

    def __init__(self, config, mode="all", output_dim=3):
        super(Decoder3D, self).__init__()

        self.config = config

        self.in_channels = 18
        self.out_channels = 3
        self.input_conv_kernel_size = (3, 3, 3)  # (7, 7, 7)
        self.output_conv_kernel_size = (1, 1, 1)

        self.mode = mode
        self.output_dim = output_dim
        self.filters = self.config.vqvae.filters
        self.num_res_blocks = self.config.vqvae.num_dec_res_blocks
        self.channel_multipliers = self.config.vqvae.channel_multipliers
        self.temporal_downsample = self.config.vqvae.temporal_downsample
        if isinstance(self.temporal_downsample, int):
            self.temporal_downsample = _get_selected_flags(
                len(self.channel_multipliers) - 1, self.temporal_downsample, False
            )
        self.upsample = self.config.vqvae.get("upsample", "nearest+conv")
        self.custom_conv_padding = self.config.vqvae.get("custom_conv_padding")
        self.norm_type = self.config.vqvae.norm_type
        self.num_remat_block = self.config.vqvae.get("num_dec_remat_blocks", 0)
        if self.config.vqvae.activation_fn == "relu":
            self.activation_fn = ops.elu
        elif self.config.vqvae.activation_fn == "elu":
            self.activation_fn = ops.elu
        else:
            raise NotImplementedError

        init_dim = self.filters * self.channel_multipliers[-1]
        self.conv_in = CausalConv3d(
            self.in_channels, init_dim, self.input_conv_kernel_size, padding=1
        )
        self.conv_out = CausalConv3d(
            self.filters, self.out_channels, kernel_size=(3, 3, 3), padding=1
        )
        self.norm = GroupNormExtend(self.filters, self.filters)
        self.residual_stack = nn.SequentialCell()

        num_blocks = len(self.channel_multipliers)

        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]

            if i == num_blocks - 1:
                dim_in = self.filters * self.channel_multipliers[-1]
            else:
                dim_in = self.filters * self.channel_multipliers[i + 1]

            self.residual_stack.append(ResnetBlock3D(dim_in, filters))

            for _ in range(self.num_res_blocks - 1):
                self.residual_stack.append(ResnetBlock3D(filters, filters))

            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride = 2 if i > 1 else 1
                    if self.upsample == "deconv":
                        assert self.custom_conv_padding is None, (
                            "Custom padding not " "implemented for " "ConvTranspose"
                        )
                        self.residual_stack.append(
                            nn.Conv3dTranspose(
                                filters,
                                filters,
                                kernel_size=(3, 3, 3),
                                stride=(t_stride, 2, 2),
                            )
                        )
                    elif self.upsample == "nearest+conv":
                        scales = (float(t_stride), 2.0, 2.0)
                        self.residual_stack.append(
                            Upsample3D(filters, self.temporal_downsample[i - 1], scales)
                        )
                        self.residual_stack.append(
                            nn.Conv3d(filters, filters, kernel_size=(3, 3, 3))
                        )
                    else:
                        raise NotImplementedError(f"Unknown upsampler: {self.upsample}")

                # Adaptive GroupNorm
                self.residual_stack.append(GroupNormExtend(filters, filters))

    def construct(self, x):
        x = self.conv_in(x)
        x = self.residual_stack(x)
        x = self.norm(x)
        x = self.activation_fn(x)
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = ms.Tensor(x, ms.float32)

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print("Dncoder out shape:", decoder_out.shape)

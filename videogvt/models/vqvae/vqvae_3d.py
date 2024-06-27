import mindspore as ms
from mindspore import nn, ops
import numpy as np
from typing import Tuple, Union

from .utils import DiagonalGaussianDistribution
from videogvt.models.quantization import LFQ


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = [(0, 0)] * dims_from_right
    pad_op = ops.Pad(tuple(zeros + [pad] + [(0, 0)] * 2))
    return pad_op(t)


def exists(v):
    return v is not None


class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class CausalConv3d(nn.Cell):
    """
    Temporal padding: Padding with the first frame, by repeating K_t-1 times.
    Spatial padding: follow standard conv3d, determined by pad mode and padding
    Ref: opensora plan

    Args:
        kernel_size: order (T, H, W)
        stride: order (T, H, W)
        padding: int, controls the amount of spatial padding applied to the input on both sides
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides=None,
        pad_mode="valid",
        dtype=ms.float32,
        **kwargs,
    ):
        super().__init__()

        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        # pad temporal dimension by k-1, manually
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (
            (0, 0),
            (0, 0),
            (time_pad, 0),
            (height_pad, height_pad),
            (width_pad, width_pad),
        )

        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)

        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            pad_mode=pad_mode,
            dtype=dtype,
            **kwargs,
        ).to_float(dtype)

    def construct(self, x):
        # x: (bs, Cin, T, H, W )
        op_pad = ops.Pad(self.time_causal_padding)
        x = op_pad(x)
        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Cell):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size=3,
        stride=1,
        dtype=ms.float32,
    ):
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(
            dim, dim_out, kernel_size, stride=stride, pad_mode="valid", dtype=dtype
        ).to_float(dtype)

    def construct(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(-1, c, t)

        x = ops.pad(x, self.time_causal_padding)
        x = self.conv(x)

        x = x.reshape(b, h, w, c, -1)
        x = x.permute(0, 3, 4, 1, 2)

        return x


class TimeUpsample2x(nn.Cell):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size=3,
        dtype=ms.float32,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            dim, dim_out * 2, kernel_size, dtype=dtype
        ).to_float(dtype)
        self.activate = nn.SiLU()

    def construct(self, x):

        b, c, t, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(-1, c, t)

        x = self.conv(x)

        x = x.reshape(b, h, w, -1, t * 2)
        x = ops.permute(x, (0, 3, 4, 1, 2))

        x = self.activate(x)

        return x


class ResBlock(nn.Cell):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
        dtype=ms.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = GroupNormExtend(num_groups, in_channels, dtype=dtype)
        self.conv1 = conv_fn(
            in_channels, self.filters, kernel_size=(3, 3, 3), has_bias=False, dtype=dtype
        )
        self.norm2 = GroupNormExtend(num_groups, self.filters, dtype=dtype)
        self.conv2 = conv_fn(
            self.filters, self.filters, kernel_size=(3, 3, 3), has_bias=False, dtype=dtype
        )
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(
                    in_channels,
                    self.filters,
                    kernel_size=(3, 3, 3),
                    has_bias=False,
                    dtype=dtype,
                )
            else:
                self.conv3 = conv_fn(
                    in_channels,
                    self.filters,
                    kernel_size=(1, 1, 1),
                    has_bias=False,
                    dtype=dtype,
                )

    def construct(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual)
        return x + residual


def get_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU
    elif activation == "swish":
        activation_fn = nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn


class Encoder(nn.Cell):
    """Encoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,  # num channels for latent vector
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        spatial_downsample=(True, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        dtype=ms.flaot32,
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
            dtype=dtype,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            filters,
            kernel_size=(3, 3, 3),
            has_bias=False,
            dtype=dtype,
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = nn.CellList([])
        self.conv_blocks = nn.CellList([])

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = nn.CellList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.spatial_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 2
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters,
                            filters,
                            kernel_size=(3, 3, 3),
                            strides=(t_stride, s_stride, s_stride),
                        )
                    )
                    prev_filters = filters  # update in_channels
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append(nn.Identity(prev_filters))  # Identity
                    prev_filters = filters  # update in_channels

        # last layer res block
        self.res_blocks = nn.CellList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

        # MAGVIT uses Group Normalization
        self.norm1 = GroupNormExtend(self.num_groups, prev_filters, dtype=dtype)

        self.conv2 = self.conv_fn(
            prev_filters,
            self.embedding_dim,
            kernel_size=(1, 1, 1),
            padding="same",
            dtype=dtype,
        )

    def construct(self, x):
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Cell):
    """Decoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        spatial_downsample=(True, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        dtype=ms.float32,
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 2

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
            dtype=dtype,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(
            self.embedding_dim, filters, kernel_size=(3, 3, 3), has_bias=True, dtype=dtype
        )

        # last layer res block
        self.res_blocks = nn.CellList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))

        # ResBlocks and conv upsample
        self.block_res_blocks = nn.CellList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.CellList([])
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.CellList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.spatial_downsample[i - 1]:
                    t_stride = 2 if self.spatial_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters,
                            prev_filters * t_stride * self.s_stride * self.s_stride,
                            kernel_size=(3, 3, 3),
                            dtype=dtype,
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(prev_filters),
                    )

        self.norm1 = GroupNormExtend(self.num_groups, prev_filters, dtype=dtype)

        self.conv_out = self.conv_fn(filters, in_out_channels, 3, dtype=dtype)

    def construct(self, x):
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                b, c, t, h, w = x.shape
                x = x.reshape(b, -1, t * t_stride, h * self.s_stride, w * self.s_stride)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class VAE_3D(nn.Cell):
    def __init__(
        self,
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        spatial_downsample=(True, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        dtype=ms.float32,
    ):
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        self.out_channels = in_out_channels

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            dtype=dtype,
        )
        self.quant_conv = CausalConv3d(2 * latent_embed_dim, 2 * embed_dim, 1, dtype=dtype)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1, dtype=dtype)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            dtype=dtype,
        )

    def encode(self, x):
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature).to(x.dtype)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, num_frames=None):
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x

    def construct(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon_video = self.decode(z, num_frames=x.shape[2])
        return recon_video, posterior, z


class VQVAE_3D(nn.Cell):
    def __init__(
        self,
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        spatial_downsample=(True, True, True),
        num_groups=32,  # for nn.GroupNorm
        num_frames=17,
        activation_fn="swish",
        is_training=True,
        dtype=ms.float32,
    ):
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        self.out_channels = in_out_channels
        self.num_frames = num_frames

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            dtype=dtype,
        )
        self.quant_conv = CausalConv3d(latent_embed_dim, embed_dim, 1, dtype=dtype)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1, dtype=dtype)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            dtype=dtype,
        )

        codebook_size = pow(2, embed_dim)
        entropy_loss_weight = 0.1
        commitment_loss_weight = 0.25

        self.quantizer = LFQ(
            dim=embed_dim,
            codebook_size=codebook_size,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            inv_temperature=10.0,
            cosine_sim_project_in=False,
            return_loss_breakdown=False,
            is_video=True,
            is_training=is_training,
            dtype=dtype,
        )

    def encode(self, x):
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        encoded_feature = self.encoder(x)
        z_e = self.quant_conv(encoded_feature).to(x.dtype)
        return z_e

    def decode(self, z):
        time_padding = (
            0
            if (self.num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - self.num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x

    def construct(self, x, entropy_weight=0.1):
        # encode
        z_e = self.encode(x)

        # quantization
        z_q, indices, aux_loss = self.quantizer(z_e, entropy_weight)

        # decode
        recon_video = self.decode(z_q)

        return z_e, z_q, recon_video, aux_loss


def VAE_KL_3D(from_pretrained=None, dtype=ms.float32, **kwargs):
    model = VAE_3D(
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        spatial_downsample=(True, True, True),
        dtype=dtype,
        **kwargs,
    )
    if from_pretrained is not None:
        param_dict = ms.load_checkpoint(from_pretrained)
        ms.load_param_into_net(model, param_dict)
    return model


def VQVAE_Magvit2_3D(from_pretrained=None, dtype=ms.float32, **kwargs):
    model = VQVAE_3D(
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        spatial_downsample=(True, True, True),
        dtype=dtype,
        **kwargs,
    )
    if from_pretrained is not None:
        param_dict = ms.load_checkpoint(from_pretrained)
        ms.load_param_into_net(model, param_dict)
    return model

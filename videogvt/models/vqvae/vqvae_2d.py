import mindspore as ms
from mindspore import nn, ops
import numpy as np

from .utils import DiagonalGaussianDistribution
from videogvt.models.quantization import LFQ


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


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
            in_channels, self.filters, kernel_size=(3, 3), has_bias=False, dtype=dtype
        )
        self.norm2 = GroupNormExtend(num_groups, self.filters, dtype=dtype)
        self.conv2 = conv_fn(
            self.filters, self.filters, kernel_size=(3, 3), has_bias=False, dtype=dtype
        )
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(
                    in_channels,
                    self.filters,
                    kernel_size=(3, 3),
                    has_bias=False,
                    dtype=dtype,
                )
            else:
                self.conv3 = conv_fn(
                    in_channels,
                    self.filters,
                    kernel_size=(1, 1),
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
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = nn.Conv2d
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
            kernel_size=(3, 3),
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
                    s_stride = 2
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters,
                            filters,
                            kernel_size=(3, 3),
                            strides=(s_stride, s_stride),
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
            kernel_size=(1, 1),
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
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 2

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = nn.Conv2d
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
            self.embedding_dim, filters, kernel_size=(3, 3), has_bias=True, dtype=dtype
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
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters,
                            prev_filters * self.s_stride * self.s_stride,
                            kernel_size=(3, 3),
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
                x = self.conv_blocks[i - 1](x)
                b, c, h, w = x.shape
                x = x.reshape(b, -1, h * self.s_stride, w * self.s_stride)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class VAE_KL_Spatial(nn.Cell):
    def __init__(
        self,
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        spatial_downsample=(True, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        dtype=ms.float32,
    ):
        super().__init__()

        self.out_channels = in_out_channels

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            spatial_downsample=spatial_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            dtype=dtype,
        )
        self.quant_conv = nn.Conv2d(2 * latent_embed_dim, 2 * embed_dim, 1, dtype=dtype)

        self.post_quant_conv = nn.Conv2d(embed_dim, latent_embed_dim, 1, dtype=dtype)
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

    def encode(self, x):
        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature).to(x.dtype)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

    def construct(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon_video = self.decode(z)
        return recon_video, posterior, z


class VQVAE_Spatial(nn.Cell):
    def __init__(
        self,
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        spatial_downsample=(True, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        is_training=True,
        dtype=ms.float32,
    ):
        super().__init__()

        self.out_channels = in_out_channels

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
        self.quant_conv = nn.Conv2d(latent_embed_dim, embed_dim, 1, dtype=dtype)

        self.post_quant_conv = nn.Conv2d(embed_dim, latent_embed_dim, 1, dtype=dtype)
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
            is_video=False,
            is_training=is_training,
            dtype=dtype,
        )

    def encode(self, x):
        encoded_feature = self.encoder(x)
        z_e = self.quant_conv(encoded_feature).to(x.dtype)
        return z_e

    def decode(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

    def construct(self, x, entropy_weight=0.1):
        # encode
        z_e = self.encode(x)

        # quantization
        z_q, indices, aux_loss = self.quantizer(z_e, entropy_weight)

        # decode
        recon_video = self.decode(z_q)

        return z_e, z_q, recon_video, aux_loss


def VAE_KL_2D(from_pretrained=None, dtype=ms.float32, **kwargs):
    model = VAE_KL_Spatial(
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        spatial_downsample=(True, True, True),
        dtype=dtype,
        **kwargs,
    )
    if from_pretrained is not None:
        param_dict = ms.load_checkpoint(from_pretrained)
        ms.load_param_into_net(model, param_dict)
    return model


def VQVAE_Magvit2_2D(from_pretrained=None, dtype=ms.float32, **kwargs):
    model = VQVAE_Spatial(
        in_out_channels=3,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        spatial_downsample=(True, True, True),
        dtype=dtype,
        **kwargs,
    )
    if from_pretrained is not None:
        param_dict = ms.load_checkpoint(from_pretrained)
        ms.load_param_into_net(model, param_dict)
    return model

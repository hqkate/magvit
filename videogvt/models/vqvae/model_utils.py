import functools
import numpy as np
from typing import Tuple, Union

import mindspore as ms
from mindspore import nn, ops
from mindspore.common import initializer as init


def _get_selected_flags(total_len: int, select_len: int, suffix: bool):
    assert select_len <= total_len
    selected = np.zeros(total_len, dtype=bool)
    if not suffix:
        selected[:select_len] = True
    else:
        selected[-select_len:] = True
    return selected


def get_norm_layer(norm_type, dtype):
    if norm_type == "LN":
        norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
    elif norm_type == "GN":
        norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
    elif norm_type is None:
        norm_fn = lambda: (lambda x: x)
    else:
        raise NotImplementedError(f"norm_type: {norm_type}")
    return norm_fn


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def nonlinearity(x, upcast=False):
    # swish
    ori_dtype = x.dtype
    if upcast:
        return x * (ops.sigmoid(x.astype(ms.float32))).astype(ori_dtype)
    else:
        return x * (ops.sigmoid(x))


def default(v, d):
    return v if v is not None else d


class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class BlurPooling(nn.Cell):
    """
    Blur Pooling
    """
    def __init__(self):
        super().__init__()


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
        padding: int = 0,
        dtype=ms.float32,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(padding, int)
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        stride = cast_tuple(stride, 3)  # (stride, 1, 1)
        dilation = cast_tuple(dilation, 3)  # (dilation, 1, 1)

        """
        if isinstance(padding, str):
            if padding == 'same':
                height_pad = height_kernel_size // 2
                width_pad = width_kernel_size // 2
            elif padding == 'valid':
                height_pad = 0
                width_pad = 0
            else:
                raise ValueError
        else:
            padding = list(cast_tuple(padding, 3))
        """

        # pad temporal dimension by k-1, manually
        self.time_pad = dilation[0] * (time_kernel_size - 1) + (1 - stride[0])
        if self.time_pad >= 1:
            self.temporal_padding = True
        else:
            self.temporal_padding = False

        # pad h,w dimensions if used, by conv3d API
        # diff from torch: bias, pad_mode

        # TODO: why not use HeUniform init?
        weight_init_value = 1.0 / (np.prod(kernel_size) * chan_in)
        if padding == 0:
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=stride,
                dilation=dilation,
                has_bias=True,
                pad_mode="same",
                weight_init=weight_init_value,
                bias_init="zeros",
                **kwargs,
            ).to_float(dtype)
        else:
            # axis order (t0, t1, h0 ,h1, w0, w2)
            padding = list(cast_tuple(padding, 6))
            padding[0] = 0
            padding[1] = 0
            padding = tuple(padding)
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                kernel_size,
                stride=stride,
                dilation=dilation,
                has_bias=True,
                pad_mode="pad",
                padding=padding,
                weight_init=weight_init_value,
                bias_init="zeros",
                **kwargs,
            ).to_float(dtype)

    def construct(self, x):
        # x: (bs, Cin, T, H, W )
        if self.temporal_padding:
            first_frame = x[:, :, :1, :, :]
            first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)
            x = ops.concat((first_frame_pad, x), axis=2)

        return self.conv(x)


class CausalConv3dZeroPad(nn.Cell):
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = kernel_size[0]
        stride = kwargs.pop("stride", (1, 1, 1))
        # stride = (stride, 1, 1)
        total_pad = tuple([k - s for k, s in zip(kernel_size[1:], stride[1:])])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        pad_input += (0, 0)
        self.padding = pad_input
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, pad_mode="pad", **kwargs
        )

    def construct(self, x):
        x = ops.pad(x, self.padding)
        first_frame_pad = x[:, :, :1, :, :].repeat(self.time_kernel_size - 1, axis=2)
        x = ops.cat((first_frame_pad, x), axis=2)
        return self.conv(x)


def Normalize(in_channels, num_groups=32, extend=False, dtype=ms.float32):
    if extend:
        return GroupNormExtend(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype
        )
    else:
        return nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype
        )


def Avgpool3d(x):
    # ops.AvgPool3D(strides=(2, 2, 2))
    b, c, h, w, d = x.shape
    x = x.reshape(b*c, h, w, d)
    x = ops.AvgPool(kernel_size=1, strides=2)(x)
    x = ops.permute(x, (0, 2, 3, 1))
    x = ops.AvgPool(kernel_size=1, strides=(1, 2))(x)
    x = ops.permute(x, (0, 3, 1, 2))
    h, w, d = x.shape[-3:]
    x = x.reshape(b, c, h, w, d)
    return x


class Upsample3D(nn.Cell):
    def __init__(self, in_channels, with_conv, scale_factor, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        self.scale_factor = scale_factor
        if self.with_conv:
            self.conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                has_bias=True,
            ).to_float(self.dtype)

    def construct(self, x):
        in_shape = x.shape[-2:]
        out_shape = tuple(2 * x for x in in_shape)
        # x = ops.ResizeNearestNeighbor(out_shape)(x)
        x = ops.interpolate(x, scale_factor=self.scale_factor)

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        include_t_dim: bool = True,
        factor: int = 2,
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.with_conv = with_conv
        self.include_t_dim = include_t_dim
        self.factor = factor
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                pad_mode="valid",
                padding=0,
                has_bias=True,
            ).to_float(self.dtype)

    def construct(self, x):
        if self.with_conv:
            pad = ((0, 0), (0, 0), (0, 1), (0, 1))
            x = nn.Pad(paddings=pad)(x)
            x = self.conv(x)
        else:
            t_factor = self.factor if self.include_t_dim else 1
            shape = (t_factor, self.factor, self.factor)
            x = ops.AvgPool3D(kernel_size=shape, strides=shape)(x)
        return x


class SpatialDownsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
        dtype=ms.float32,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        # TODO: no need to use CausalConv3d, can reshape to spatial (bt, c, h, w) and use conv 2d
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=0,
        )

        # no asymmetric padding, must do it ourselves
        # order (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        # self.padding = (0,1,0,1,0,0) # not compatible for ms2.2
        self.pad = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 1), (0, 1)))

    def construct(self, x):
        # x shape: (b c t h w)
        # x = ops.pad(x, self.padding, mode="constant", value=0)
        x = self.pad(x)
        x = self.conv(x)
        return x


class SpatialUpsample2x(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
        dtype=ms.float32,
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=1,
        )

    def construct(self, x):
        b, c, t, h, w = x.shape

        # x = rearrange(x, "b c t h w -> b (c t) h w")
        x = ops.reshape(x, (b, c * t, h, w))

        hw_in = x.shape[-2:]
        scale_factor = 2
        hw_out = tuple(scale_factor * s_ for s_ in hw_in)
        x = ops.ResizeNearestNeighbor(hw_out)(x)

        # x = ops.interpolate(x, scale_factor=(2.,2.), mode="nearest") # 4D not supported
        # x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = ops.reshape(x, (b, c, t, h * scale_factor, w * scale_factor))

        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Cell):
    def __init__(
        self,
        kernel_size: int = 3,
        replace_avgpool3d: bool = True,  # FIXME: currently, ms+910b does not support nn.AvgPool3d
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.replace_avgpool3d = replace_avgpool3d
        if not replace_avgpool3d:
            self.conv = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))
        else:
            self.conv = nn.AvgPool2d((kernel_size, 1), stride=(2, 1))
        # print('D--: replace avgpool3d', replace_avgpool3d)
        self.time_pad = self.kernel_size - 1

    def construct(self, x):
        first_frame = x[:, :, :1, :, :]
        first_frame_pad = ops.repeat_interleave(first_frame, self.time_pad, axis=2)
        x = ops.concat((first_frame_pad, x), axis=2)

        if not self.replace_avgpool3d:
            return self.conv(x)
        else:
            # FIXME: only work when h, w stride is 1
            b, c, t, h, w = x.shape
            x = ops.reshape(x, (b, c, t, h * w))
            x = self.conv(x)
            x = ops.reshape(x, (b, c, -1, h, w))
            return x


class TimeUpsample2x(nn.Cell):
    def __init__(self, exclude_first_frame=True):
        super().__init__()
        self.exclude_first_frame = exclude_first_frame

    def construct(self, x):
        if x.shape[2] > 1:
            if self.exclude_first_frame:
                x, x_ = x[:, :, :1], x[:, :, 1:]
                # FIXME: ms2.2.10 cannot support trilinear on 910b
                x_ = ops.interpolate(x_, scale_factor=(2.0, 1.0, 1.0), mode="trilinear")
                x = ops.concat([x, x_], axis=2)
            else:
                x = ops.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="trilinear")

        return x


class SqueezeExcite(nn.Cell):
    # global context network - attention-esque squeeze-excite variant (https://arxiv.org/abs/2012.13375)

    def __init__(self, dim, *, dim_out=None, dim_hidden_min=16, init_bias=-10):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_k = nn.Conv2d(dim, 1, 1)
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        self.net = nn.SequentialCell(
            nn.Conv2d(dim, dim_hidden, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_hidden, dim_out, 1),
            nn.Sigmoid(),
        )

        # nn.init.zeros_(self.net[-2].weight)
        # nn.init.constant_(self.net[-2].bias, init_bias)

        self.net[-2].weight.set_data(
            init.initializer(
                init.Zero(), self.net[-2].weight.shape, self.net[-2].weight.dtype
            )
        )
        self.net[-2].bias.set_data(
            init.initializer(
                init.Constant(init_bias),
                self.net[-2].bias.shape,
                self.net[-2].bias.dtype,
            )
        )

    def construct(self, x):
        orig_input, batch = x, x.shape[0]
        is_video = x.ndim == 5

        if is_video:
            # x = rearrange(x, 'b c f h w -> (b f) c h w')
            b, c, f, h, w = x.shape
            x = x.reshape(b * f, c, h, w)

        context = self.to_k(x)

        # context = rearrange(context, 'b c h w -> b c (h w)').softmax(dim = -1)
        b, c, h, w = context.shape
        context = context.reshape(b, c, -1).softmax(axis=-1)

        # spatial_flattened_input = rearrange(x, 'b c h w -> b c (h w)')
        b, c, h, w = x.shape
        spatial_flattened_input = x.reshape(b, c, -1)

        # out = einsum('b i n, b c n -> b c i', context, spatial_flattened_input)
        out = ops.MatMul(transpose_a=True)(context, spatial_flattened_input)
        # out = rearrange(out, '... -> ... 1')
        out = out.unsqueeze(-1)
        gates = self.net(out)

        if is_video:
            # gates = rearrange(gates, '(b f) c h w -> b c f h w', b = batch)
            _, c, h, w = gates.shape
            gates = gates.reshape(batch, c, -1, h, w)

        return gates * orig_input


class ResidualNet(nn.Cell):
    def __init__(self, dim, kernel_size):
        self.net = nn.SequentialCell(
            CausalConv3d(dim, dim, kernel_size, padding=1),
            nn.ELU(),
            nn.Conv3d(dim, dim, 1),
            nn.ELU(),
            SqueezeExcite(dim),
        )

    def construct(self, x):
        return self.net(x) + x


class ResnetBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.1,
        dtype=ms.float32,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.upcast_sigmoid = upcast_sigmoid

        # FIXME: GroupNorm precision mismatch with PT.
        self.norm1 = Normalize(in_channels, extend=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1, dtype=dtype)
        self.norm2 = Normalize(out_channels, extend=True)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1, dtype=dtype)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(
                    in_channels, out_channels, 3, padding=1, dtype=dtype
                )
            else:
                self.nin_shortcut = CausalConv3d(
                    in_channels, out_channels, 1, padding=0, dtype=dtype
                )

    def construct(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h, self.upcast_sigmoid)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


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
            nn.Conv2d(
                in_dim,
                res_h_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, has_bias=False),
        )

    def construct(self, x):
        x = x + self.res_block(x)
        return x


class ResidualLayer3D(nn.Cell):
    """
    One residual layer inputs:
    - dim : the input dimension
    - kernal_size : the kernal size
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, dim, h_dim, res_h_dim):
        super(ResidualLayer3D, self).__init__()
        self.res_block = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv3d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=False,
            ),
            nn.ReLU(),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, has_bias=False),
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
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def construct(self, x):
        for layer in self.stack:
            x = layer(x)
        x = ops.relu(x)
        return x


class ResidualStack3D(nn.Cell):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack3D, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.CellList(
            [ResidualLayer3D(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

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
    print("Res Layer out shape:", res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print("Res Stack out shape:", res_stack_out.shape)

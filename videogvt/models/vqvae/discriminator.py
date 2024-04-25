# Copyright 2023 The videogvt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D StyleGAN discriminator."""

from typing import Any
import math
import ml_collections

import mindspore as ms
from mindspore import nn, ops

from videogvt.models.vqvae.model_utils import Normalize


class ResBlockDown(nn.Cell):
    """3D StyleGAN ResBlock for D."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, (3, 3, 3))
        self.norm1 = Normalize(self.out_channels, extend=True)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3))
        self.norm2 = Normalize(self.out_channels, extend=True)
        self.activation2 = nn.LeakyReLU()
        self.avgpool = nn.AvgPool3d(stride=(2, 2, 2))
        # self.dropout = nn.Dropout(p=dropout)

        self.avgpool_shortcut = nn.AvgPool3d(stride=(2, 2, 2))
        self.conv_shortcut = nn.Conv3d(
            self.in_channels, self.out_channels, (1, 1, 1), has_bias=False
        )

    def construct(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.avgpool(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        residual = self.avgpool_shortcut(residual)
        residual = self.conv_shortcut(residual)

        out = (residual + x) / ops.sqrt(ms.Tensor(2, ms.float32))
        return out


class StyleGANDiscriminator(nn.Cell):
    """StyleGAN Discriminator."""

    def __init__(
        self, config: ml_collections.ConfigDict, height: int, width: int, depth: int
    ):
        super().__init__()
        self.config = config
        self.in_channles = 3
        self.input_size = self.config.image_size
        self.filters = self.config.discriminator.filters
        self.channel_multipliers = self.config.discriminator.channel_multipliers

        self.conv_in = nn.Conv3d(self.in_channles, self.filters, kernel_size=(3, 3, 3))
        self.activation1 = nn.LeakyReLU()
        self.resnet_stack = nn.SequentialCell()

        num_blocks = len(self.channel_multipliers)
        sampling_rate = math.pow(2, num_blocks)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]

            if i == 0:
                dim_in = self.filters
            else:
                dim_in = self.filters * self.channel_multipliers[i - 1]

            self.resnet_stack.append(ResBlockDown(dim_in, filters))

        dim_out = self.filters * self.channel_multipliers[-1]
        self.norm2 = Normalize(dim_out, extend=True)
        self.conv_out = nn.Conv3d(dim_out, dim_out, (3, 3, 3))
        self.activation2 = nn.LeakyReLU()

        dim_dense = int(
            dim_out
            * max(1, height // sampling_rate)
            * max(1, width // sampling_rate)
            * max(1, depth // sampling_rate)
        )
        self.linear1 = nn.Dense(dim_dense, 512)
        self.activation3 = nn.LeakyReLU()
        self.linear2 = nn.Dense(512, 1)

    def construct(self, x):
        # x = self.norm(x)
        x = self.conv_in(x)
        x = self.activation1(x)
        x = self.resnet_stack(x)
        x = self.conv_out(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = x.reshape((x.shape[0], -1))
        x = self.linear1(x)
        x = self.activation3(x)
        x = self.linear2(x)
        return x

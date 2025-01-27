# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv, Pool

from torch.nn import functional as F
import logging

class ScaledStdConv3d(nn.Conv3d):
    """Conv2d layer with Scaled Weight Standardization.
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    Adapted from timm: https://github.com/rwightman/pytorch-image-models/blob/4ea593196414684d2074cbb81d762f3847738484/timm/models/layers/std_conv.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, gain=True, gamma=1.0, eps=1e-5, use_layernorm=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(
            self.out_channels, 1, 1, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3, 4], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv3d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

class WSConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(
            self.weight.size(0), requires_grad=True))

    def standardize_weight(self, eps):
        mean = torch.mean(self.weight, dim=(1, 2, 3, 4), keepdims=True)
        var = torch.std(self.weight, dim=(1, 2, 3, 4), keepdims=True) ** 2
        fan_in = torch.prod(torch.tensor(self.weight.shape[1:]))

        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input, eps=1e-4):
        weight = self.standardize_weight(eps)
        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    For example:

    .. code-block:: python

        from monai.networks.blocks import Convolution

        conv = Convolution(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="ADN",
            act=("prelu", {"init": 0.2}),
            dropout=0.1,
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(conv)

    output::

        Convolution(
          (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (A): PReLU(num_parameters=1)
            (D): Dropout(p=0.1, inplace=False)
            (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
          )
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
        conv_type: str = "standard"
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        self.conv_type = conv_type
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_layer_class = self.get_conv_layer()

        conv: nn.Module
        if is_transposed:
            raise NotImplementedError("Transposed convolution is not supported yet.")
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_layer_class(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_layer_class(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return

        norm_dim = None if norm is None else self.spatial_dims
        logging.info(f'norm_dim: {norm_dim}')

        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=norm_dim,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )

    def get_conv_layer(self):
        logging.info(f"Conv type: {self.conv_type}")
        if self.conv_type == "standard":
            return Conv[Conv.CONVTRANS if self.is_transposed else Conv.CONV, self.spatial_dims]
        elif self.conv_type == "WSConv":
            return WSConv
        elif self.conv_type == "ScaledStdConv":
            return ScaledStdConv3d
        else:
            raise ValueError(f"Unsupported Conv type: {self.conv_type}")


class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="AN",
            act=("prelu", {"init": 0.2}),
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(convs)

    output::

        ResidualUnit(
          (conv): Sequential(
            (unit0): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
            (unit1): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (residual): Identity()
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        conv_type: str = "standard"
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
                conv_type=conv_type
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            # NOW residual path is just avg pool
            print("Using avg pool residual path with concat on overlap with convolution path")
            self.residual = nn.AvgPool3d(kernel_size=rkernel_size,stride=strides,padding=rpadding,)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations

        cx[:,:res.shape[1]] = cx[:,:res.shape[1]] + res
        return cx  # add the residual to the output

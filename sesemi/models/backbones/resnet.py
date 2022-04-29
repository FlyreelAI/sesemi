#
# Copyright 2021, Flyreel. All Rights Reserved.
# =============================================#
"""Specialized residual networks."""
from typing import Optional
from sesemi.utils import freeze_module
from torch import Tensor, nn

import torch.nn.functional as F

from .base import Backbone


class CIFARResidualBlock(nn.Module):
    """A residual block for the CIFARResNet backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
    ):
        """Initializes the residual block.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels. Defaults to the number of
                inputs.
            downsamples: Whether to downsample the inputs using a stride of 2.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        residual = self.bn2(conv2)

        if self.downsample:
            xi = F.interpolate(
                inputs, size=(residual.shape[2], residual.shape[3]), mode="nearest"
            )
        else:
            xi = inputs

        channel_difference = self.out_channels - self.in_channels
        if channel_difference > 0:
            xi = F.pad(
                xi,
                (0, 0, 0, 0, 0, channel_difference, 0, 0),
                mode="constant",
                value=0,
            )

        y = residual + xi
        outputs = self.relu(y)
        return outputs


class CIFARDeepResidualBlock(nn.Module):
    """A deep residual block for the CIFARResNet backbone."""

    def __init__(
        self,
        num_residual_blocks: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
    ):
        """Initializes the deep residual block.

        Args:
            num_residual_blocks: The number of residual block to chain together.
            in_channels: The number of input channels.
            out_channels: The number of output channels. Defaults to the number of
                inputs.
            downsamples: Whether to downsample the inputs using a stride of 2.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.num_residual_blocks = num_residual_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        block_layers = [
            CIFARResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=downsample,
            )
        ]
        for _ in range(num_residual_blocks - 1):
            block_layers.append(CIFARResidualBlock(in_channels=out_channels))

        self.block = nn.Sequential(*block_layers)

    def forward(self, inputs):
        return self.block(inputs)


class CIFARResNet(Backbone):
    """A specialized ResNet designed for CIFAR10 and described in the referenced paper.

    @article{He2016DeepRL,
        title={Deep Residual Learning for Image Recognition},
        author={Kaiming He and X. Zhang and Shaoqing Ren and Jian Sun},
        journal={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
        pages={770-778}
    }

    See section 4.2 for the CIFAR-10 network design and analysis.
    """

    def __init__(
        self,
        out_features: Optional[int] = None,
        n: int = 18,
        drop_rate: float = 0.0,
        freeze: bool = False,
    ):
        """Initializes the ResNet.

        Args:
            out_features: An optional specification for an additional linear layer
                to use at the end of the network. Otherwise the outputs are 64d and no
                linear layer is applied.
            n: This parameter defines a multiplier for the number of layers in use. The
                total layers not counting any added linear linear is 6 * n + 1. The
                default value for n would then result in 109 (6 * 18 + 1) layers. With
                an added linear layer this is 110.
            drop_rate: The dropout rate to use on the output features.
            freeze: Whether to freeze the model's parameters.
        """
        super().__init__()

        self.in_features = 3
        self.out_features = 64 if out_features is None else out_features

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = CIFARDeepResidualBlock(n, in_channels=16)
        self.block2 = CIFARDeepResidualBlock(
            n, in_channels=16, out_channels=32, downsample=True
        )
        self.block3 = CIFARDeepResidualBlock(
            n, in_channels=32, out_channels=64, downsample=True
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)

        if out_features is not None:
            self.linear_block: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(64, out_features, bias=False),
                nn.BatchNorm1d(out_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.linear_block = None

        self.dropout = nn.Dropout(p=drop_rate, inplace=True)

        if freeze:
            freeze_module(self)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        block1 = self.block1(relu1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        features = self.pooling(block3)[..., 0, 0]

        if self.linear_block is not None:
            features = self.linear_block(features)

        outputs = self.dropout(features)
        return outputs


class WRNBasicBlock(nn.Module):
    """A Wide-ResNet basic block."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, drop_rate: float = 0.0
    ):
        """Initializes the Wide-ResNet basic block.

        Args:
            in_channels: The input convolutional channels.
            out_channels: The output convolutional channels.
            stride: The stride to use for the first block.
            drop_rate: The drop rate for dropout.
        """
        super().__init__()
        self.preprocessing_layers = nn.Sequential()
        self.block_layers = nn.Sequential()

        no_use_shortcut = in_channels == out_channels and stride == 1
        if no_use_shortcut:
            self.preprocessing_layers.append(nn.BatchNorm2d(in_channels))
            self.preprocessing_layers.append(nn.ReLU(inplace=True))
        else:
            self.block_layers.append(nn.BatchNorm2d(in_channels))
            self.block_layers.append(nn.ReLU(inplace=True))
        self.block_layers.append(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        )

        self.block_layers.append(nn.BatchNorm2d(out_channels))
        self.block_layers.append(nn.ReLU(inplace=True))
        self.block_layers.append(nn.Dropout(p=drop_rate))
        self.block_layers.append(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

        self.shortcut = (
            nn.Identity()
            if no_use_shortcut
            else nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.preprocessing_layers(inputs)
        outputs = self.block_layers(x) + self.shortcut(x)
        return outputs


class WRNBlock(nn.Module):
    """A Wide-ResNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels,
        stride: int,
        num_basic_blocks: int,
        drop_rate: float = 0.0,
    ):
        """Initializes the Wide-ResNet Block.

        Args:
            in_channels: The input convolutional channels.
            out_channels: The output convolutional channels.
            stride: The stride to use for the first block.
            num_basic_blocks: The number of basic blocks to use.
            drop_rate: The drop rate for dropout.
        """
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.append(
            WRNBasicBlock(in_channels, out_channels, stride, drop_rate=drop_rate)
        )
        for _ in range(num_basic_blocks - 1):
            self.layers.append(
                WRNBasicBlock(out_channels, out_channels, 1, drop_rate=drop_rate)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class WRN(Backbone):
    """A Wide-ResNet implementation following:

    @article{Zagoruyko2016WideRN,
        title={Wide Residual Networks},
            author={Sergey Zagoruyko and Nikos Komodakis},
            journal={ArXiv},
            year={2016},
            volume={abs/1605.07146}
    }

    This contains the variants composed of B(3, 3) blocks
    that were shown to perform best out of those tried in the paper.

    Adapted from the original code:
    https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
    """

    def __init__(
        self, depth: int, k: int, drop_rate: float = 0.0, freeze: bool = False
    ):
        """Initializes the Wide-ResNet (WRN).

        Args:
            depth: The depth of the resnet including a final linear layer. This
                parameter matches those used in the literature, however, as this is only
                the backbone, it does not contain the linear layer.
            k: The widening factor.
            drop_rate: The drop rate for dropout.
            freeze: Whether or not to freeze the backbone.
        """
        super().__init__()
        self.layers = nn.Sequential()

        assert ((depth - 4) % 6) == 0, "depth must be decomposable into 6n+4"
        num_basic_blocks = (depth - 4) // 6

        n_stages = [16, 16 * k, 32 * k, 64 * k]

        # 1 layer
        self.layers.append(nn.Conv2d(3, n_stages[0], 3, 1, 1, bias=False))

        # num_basic_blocks * 2 + 1 layers (shortcut)
        self.layers.append(
            WRNBlock(
                n_stages[0],
                n_stages[1],
                1,
                num_basic_blocks,
                drop_rate=drop_rate,
            )
        )

        # num_basic_blocks * 2 + 1 layers (shortcut)
        self.layers.append(
            WRNBlock(
                n_stages[1],
                n_stages[2],
                2,
                num_basic_blocks,
                drop_rate=drop_rate,
            )
        )

        # num_basic_blocks * 2 + 1 layers (shortcut)
        self.layers.append(
            WRNBlock(
                n_stages[2],
                n_stages[3],
                2,
                num_basic_blocks,
                drop_rate=drop_rate,
            )
        )

        self.layers.append(nn.AdaptiveAvgPool2d(1))

        # The total sum of layers for the backbone is:
        #   1 + (num_basic_blocks * 2 + 1) +
        #       (num_basic_blocks * 2 + 1) +
        #       (num_basic_blocks * 2 + 1)
        #   = num_basic_blocks * 6 + 4

        self.out_features = n_stages[3]

        if freeze:
            freeze_module(self)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)[..., 0, 0]

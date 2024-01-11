import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    KERNEL_SIZE = 3

    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super(BasicBlock, self).__init__()
        assert stride >= 1, "stride must be equal or bigger than 1"
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.downsample = None

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.KERNEL_SIZE,
            padding=1,
            stride=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=self.KERNEL_SIZE,
            padding=1,
            stride=stride,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU()

    def forward(self, x: Tensor):
        identity = self.downsample(x) if self.downsample is not None else x

        x = self.activ(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += identity

        x = self.activ(x)

        return x


class ResNet(nn.Module):
    conv2_x: nn.Module
    conv3_x: nn.Module
    conv4_x: nn.Module
    conv5_x: nn.Module

    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet,
            self,
        ).__init__()
        self.conv1 = nn.Conv2d(
            kernel_size=7, in_channels=in_channels, out_channels=64, stride=2, padding=5
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.flatten = nn.Flatten()

    def _make_layer(
        self, in_channels: int, out_channels: int, stride: int, n_blocks: int
    ):
        layers = []

        layers.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )
        for i in range(1, n_blocks):
            layers.append(
                BasicBlock(
                    in_channels=out_channels, out_channels=out_channels, stride=1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.flatten(self.avg_pool(x))
        return self.fc(x)


class ResNet18(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(
            ResNet18,
            self,
        ).__init__(in_channels, num_classes)
        self.conv2_x = self._make_layer(
            in_channels=64, out_channels=128, stride=2, n_blocks=2
        )
        self.conv3_x = self._make_layer(
            in_channels=128, out_channels=256, stride=2, n_blocks=2
        )
        self.conv4_x = self._make_layer(
            in_channels=256, out_channels=512, stride=2, n_blocks=2
        )
        self.conv5_x = self._make_layer(
            in_channels=512, out_channels=512, stride=2, n_blocks=2
        )

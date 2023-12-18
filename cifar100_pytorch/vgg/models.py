from abc import ABC
from torch import nn


class VGGConvStack(nn.Module):
    """Convolutional stack for VGGNet"""

    CONV_KERNEL_SIZE = 3
    CONV_PADDING = 1
    CONV_STRIDE = 1

    POOL_KERNEL_SIZE = 2
    POOL_STRIDE = 2
    POOL_PADDING = 0

    def __init__(
        self, in_channels, out_channels, num_layers, use_batchnorm, last_layer_1x1=False
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_layers: Number of convolutional layers
            use_batchnorm: Whether to use batch normalization
            last_layer_1x1: Whether to use a 1x1 kernel for the last layer instead of the default 3x3
        """

        super(VGGConvStack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.layers = []
        for i in range(num_layers):
            kernel_size = (
                1 if (i == num_layers - 1) and last_layer_1x1 else self.CONV_KERNEL_SIZE
            )
            padding = (
                0 if (i == num_layers - 1) and last_layer_1x1 else self.CONV_PADDING
            )

            self.layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=self.CONV_STRIDE,
                    padding=padding,
                )
            )
            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            in_channels = out_channels

        self.layers.append(
            nn.MaxPool2d(
                kernel_size=self.POOL_KERNEL_SIZE,
                stride=self.POOL_STRIDE,
                padding=self.POOL_PADDING,
            )
        )

        self.layers = nn.Sequential(*self.layers)

    def __call__(self, x):
        return self.layers(x)


class VGGFCStack(nn.Module):
    """Fully connected stack for VGGNet"""

    def __init__(self, num_classes):
        super(VGGFCStack, self).__init__()
        self.num_classes = num_classes
        self.input_size = 512 * 7 * 7

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def __call__(self, x):
        return self.layers(x)


class VGGNet(nn.Module, ABC):
    """Abstract VGGNet class"""

    CONV_INIT_NORMAL_MEAN = 0
    CONV_INIT_NORMAL_STD = 0.01
    CONV_INIT_BIAS = 0

    FC_INIT_NORMAL_MEAN = 0
    FC_INIT_BIAS = 0

    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv_stacks = nn.Sequential()
        self.fc_stack = None

    def forward(self, x):
        x = self.conv_stacks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize Conv2d weights with normal distribution (mean=0, std=0.01)
                nn.init.normal_(
                    m.weight,
                    mean=self.CONV_INIT_NORMAL_MEAN,
                    std=self.CONV_INIT_NORMAL_STD,
                )
                # Initialize Conv2d biases with zeros
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Initialize Linear weights with normal distribution (mean=0, std=sqrt(2/n_in))
                nn.init.normal_(
                    m.weight,
                    mean=self.FC_INIT_NORMAL_MEAN,
                    std=(2.0 / m.weight.size(1)) ** 0.5,
                )
                # Initialize Linear biases with zeros
                nn.init.constant_(m.bias, self.FC_INIT_BIAS)


class VGGNetA(VGGNet):
    """Configuration A of the VGGNet paper"""

    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNetA, self).__init__()
        self.conv_stacks = nn.Sequential(
            VGGConvStack(in_channels, 64, 1, True),
            VGGConvStack(64, 128, 1, True),
            VGGConvStack(128, 256, 2, True),
            VGGConvStack(256, 512, 2, True),
            VGGConvStack(512, 512, 2, True),
        )
        self.fc_stack = VGGFCStack(num_classes)
        self._init_weights()


class VGGNetB(VGGNet):
    """Configuration B of the VGGNet paper"""

    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNetB, self).__init__()
        self.conv_stacks = nn.Sequential(
            VGGConvStack(in_channels, 64, 2, True),
            VGGConvStack(64, 128, 2, True),
            VGGConvStack(128, 256, 2, True),
            VGGConvStack(256, 512, 2, True),
            VGGConvStack(512, 512, 2, True),
        )
        self.fc_stack = VGGFCStack(num_classes)
        self._init_weights()


class VGGNetC(VGGNet):
    """Configuration C of the VGGNet paper"""

    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNetC, self).__init__()
        self.conv_stacks = nn.Sequential(
            VGGConvStack(in_channels, 64, 2, True),
            VGGConvStack(64, 128, 2, True),
            VGGConvStack(128, 256, 3, True, last_layer_1x1=True),
            VGGConvStack(256, 512, 3, True, last_layer_1x1=True),
            VGGConvStack(512, 512, 3, True, last_layer_1x1=True),
        )
        self.fc_stack = VGGFCStack(num_classes)
        self._init_weights()


class VGGNetD(VGGNet):
    """Configuration D of the VGGNet paper"""

    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNetD, self).__init__()
        self.conv_stacks = nn.Sequential(
            VGGConvStack(in_channels, 64, 2, True),
            VGGConvStack(64, 128, 2, True),
            VGGConvStack(128, 256, 3, True),
            VGGConvStack(256, 512, 3, True),
            VGGConvStack(512, 512, 3, True),
        )
        self.fc_stack = VGGFCStack(num_classes)
        self._init_weights()


class VGGNetE(VGGNet):
    """Configuration E of the VGGNet paper"""

    def __init__(self, in_channels=3, num_classes=10):
        super(VGGNetE, self).__init__()
        self.conv_stacks = nn.Sequential(
            VGGConvStack(in_channels, 64, 2, True),
            VGGConvStack(64, 128, 2, True),
            VGGConvStack(128, 256, 4, True),
            VGGConvStack(256, 512, 4, True),
            VGGConvStack(512, 512, 4, True),
        )
        self.fc_stack = VGGFCStack(num_classes)
        self._init_weights()

from torch import nn, Tensor


class AlexNet(nn.Module):
    """AlexNet model

    Characteristics (follows the paper):
        - Response normalization layers follow the first and second convolutional layers.
        - Max-pooling layers follow both response normalization layers as well as the fifth convolutional layer.
        - The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.
        - We use dropout in the first two fully-connected layers.

    The model expects input of size N x 3 x 224 x 224 and returns output of size N x num_classes.
    """

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()

        # Features: N x 3 x 224 x 224 -> N x 256 x 6 x 6
        self.features = nn.Sequential(
            # Block 1: N x 3 x 224 x 224 -> N x 96 x 55 x 55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2: N x 96 x 55 x 55 -> N x 256 x 27 x 27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 3: N x 256 x 27 x 27 -> N x 384 x 13 x 13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 4: N x 384 x 13 x 13 -> N x 384 x 13 x 13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 5: N x 384 x 13 x 13 -> N x 256 x 6 x 6
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classifier: N x 256 x 6 x 6 -> N x num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the model.
        As described in the paper, the weights are initialized from a Gaussian distribution with mean 0 and standard deviation 0.01.
        The biases in the 2nd, 4th, and 5th convolutional layers as well as in the fully-connected layers are initialized with the constant 1.
        The biases in the remaining layers are initialized with the constant 0.
        """
        for idx, module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                if idx in [1, 4, 6, 7, 8]:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    nn.init.constant_(module.bias, 1)
                else:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.constant_(module.bias, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # Flattens the tensor from N x 256 x 6 x 6 to N x 9216
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

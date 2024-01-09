from alexnet.model import AlexNet
import torch
import pytest


class TestAlexNet:
    @pytest.mark.parametrize("num_classes", [10, 100])
    def test_forward(self, num_classes: int):
        model = AlexNet(num_classes=num_classes)
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, num_classes)

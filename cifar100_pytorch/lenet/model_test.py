from .model import LeNet5
import torch
import pytest


class TestLeNet5:
    @pytest.mark.parametrize("num_classes", [10, 100])
    def test_forward(self, num_classes: int):
        model = LeNet5(in_channels=3, num_classes=num_classes)
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, num_classes)

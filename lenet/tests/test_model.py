from .model import LeNet5
import torch


class TestLeNet5:
    def test_forward(self):
        num_classes = 10
        model = LeNet5()
        x = torch.randn(1, 32, 32)
        y = model(x)
        assert y.shape == (1, num_classes)

from .models import ResNet18, ResNet34, ResNet50
import torch

if __name__ == "__main__":
    model = ResNet18()

    test_input = torch.randn(1, 3, 224, 224)

    test_output = model(test_input)

    print(test_output)

    model = ResNet34()

    test_input = torch.randn(1, 3, 224, 224)

    test_output = model(test_input)

    print(test_output)

    model = ResNet50()

    test_input = torch.randn(1, 3, 224, 224)

    test_output = model(test_input)

    print(test_output)

import torch

from common.util import get_mnist_dataloader_test
from common.config import DEVICE
from lenet.model import LeNet5
from common.util import load_model
from common.util import TestResults, save_test_results
import argparse
from torchvision import transforms


def main(args):
    data_loader = get_mnist_dataloader_test(
        additional_transforms=[
            transforms.Resize((32, 32), antialias=True),
        ]
    )
    # Load the model according to the configuration and weights path
    net = load_model(LeNet5(), args.model_path).to(DEVICE)

    total = 0
    correct = 0
    correct_top5 = 0
    correct_top10 = 0

    net.eval()

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_top5 += (
                torch.eq(labels.unsqueeze(1), torch.topk(outputs, 5).indices)
                .sum()
                .item()
            )
            correct_top10 += (
                torch.eq(labels.unsqueeze(1), torch.topk(outputs, 10).indices)
                .sum()
                .item()
            )

    print("Accuracy of the model on the test images: {}%".format(100 * correct / total))
    results = {
        "accuracy": 100 * correct / total,
        "top5": 100 * correct_top5 / total,
        "top10": 100 * correct_top10 / total,
        "model_name": net.__class__.__name__,
        "total_classes": len(data_loader.dataset.classes),
    }

    results = TestResults(
        model_name=net.__class__.__name__,
        top_1_accuracy=100 * correct / total,
        top_5_accuracy=100 * correct_top5 / total,
        top_10_accuracy=100 * correct_top10 / total,
        n_classes=len(data_loader.dataset.classes),
        dataset_name="CIFAR100",
    )

    save_test_results(results, "test_results_{}.json".format(net.__class__.__name__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model weights",
    )
    main(parser.parse_args())

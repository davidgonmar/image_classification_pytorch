import torch
import argparse
from util import load_model

from cifar100_pytorch.common import get_cifar100_dataloader, DEVICE
from cifar100_pytorch.vgg.util import config_to_net, get_transform


def main(args):
    data_loader = get_cifar100_dataloader(
        batch_size=128,
        train=False,
        transforms=get_transform(),
    )

    # Load the model according to the configuration and weights path
    net = load_model(
        config_to_net(args.config, len(data_loader.dataset.classes)), args.model_path
    )

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

    import json

    with open("results{}.json".format(net.__class__.__name__), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to model", required=True)
    parser.add_argument(
        "--config",
        type=str,
        choices=["A", "B", "C", "D", "E"],
        help="Configuration of the network to use (A, B, C, D, or E)",
        required=True,
    )
    main(parser.parse_args())

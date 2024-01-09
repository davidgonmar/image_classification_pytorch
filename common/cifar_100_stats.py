from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch import Tensor

import json

from common.config import DATA_PATH
import os


dataset = CIFAR100(
    DATA_PATH, train=False, download=True, transform=transforms.ToTensor()
)
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)


def _compute_mean_std(dataset):
    mean = Tensor([0.0, 0.0, 0.0])
    std = Tensor([0.0, 0.0, 0.0])
    for images, _ in tqdm(data_loader):
        curr_batch_size = images.shape[
            0
        ]  # The last batch may not be of the same size as the others
        images = images.view(
            curr_batch_size, images.shape[1], -1
        )  # Flatten the images but keep batch size and channels
        batch_mean = images.mean(2).sum(0)
        batch_std = images.std(2).sum(0)

        mean += batch_mean
        std += batch_std

    mean /= len(data_loader.dataset)
    std /= len(data_loader.dataset)

    return {"mean": mean, "std": std}


def _save_stats(mean, std):
    with open(os.path.join(DATA_PATH, "cifar100_mean_std.json"), "w") as f:
        if isinstance(mean, Tensor):
            mean = mean.tolist()
        if isinstance(std, Tensor):
            std = std.tolist()
        json.dump({"mean": mean, "std": std}, f)


def get_mean_std():
    """Returns the mean and standard deviation of the CIFAR100 dataset. If it is not on disk, it computes it and saves it to disk."""
    try:
        with open(os.path.join(DATA_PATH, "cifar100_mean_std.json"), "r") as f:
            data = json.load(f)
            return {
                "mean": Tensor(data["mean"]),
                "std": Tensor(data["std"]),
            }
    except FileNotFoundError:
        print("Could not find mean and std on disk. Computing them...")
        res = _compute_mean_std(dataset)
        _save_stats(res["mean"], res["std"])
        std = res["std"]
        mean = res["mean"]
        print("Done!. Results: mean={}, std={}".format(mean, std))
        return {"mean": mean, "std": std}


if __name__ == "__main__":
    res = _compute_mean_std(dataset)
    print(res)
    _save_stats(res["mean"], res["std"])

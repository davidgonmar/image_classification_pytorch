import torch
from torch import nn
from pathlib import Path
from typing import Union

from torchvision.datasets import CIFAR100, MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from cifar100_pytorch.common.config import DATA_PATH, SAVED_PATH


def load_model(model: nn.Module, path: Union[str, Path]) -> nn.Module:
    """
    Loads the state dict of a model from a path

    Args:
        model: The model to load the state dict into
        path: The path to the state dict

    Returns:
        The model with the loaded state dict
    """
    model.load_state_dict(torch.load(path))
    return model


def save_model(model: nn.Module, path: Union[str, Path]) -> None:
    """
    Saves the state dict of a model to the '/saved' directory + the given path

    Args:
        model: The model to save the state dict of
        path: The path to save the state dict to
    """
    path = Path(SAVED_PATH) / path
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    torch.save(model.state_dict(), path)


def get_cifar100_dataloader(
    *,
    batch_size,
    train: bool,
    transforms: transforms.Compose = None,
    shuffle: bool = True
) -> DataLoader:
    """
    Returns a dataloader for the CIFAR100 dataset

    Args:
        batch_size: The batch size to use
        train: Whether to use the training or testing dataset
        transforms: The transforms to apply to the dataset

    Returns:
        The dataloader
    """

    dataset = CIFAR100(DATA_PATH, train=train, download=True, transform=transforms)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_mnist_dataloader(
    *,
    batch_size,
    train: bool,
    transforms: transforms.Compose = None,
    shuffle: bool = True
) -> DataLoader:
    """
    Returns a dataloader for the MNIST dataset

    Args:
        batch_size: The batch size to use
        train: Whether to use the training or testing dataset
        transforms: The transforms to apply to the dataset

    Returns:
        The dataloader
    """

    dataset = MNIST(DATA_PATH, train=train, download=True, transform=transforms)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

from cifar100_pytorch.vgg.models import (
    VGGNetA,
    VGGNetB,
    VGGNetC,
    VGGNetD,
    VGGNetE,
    VGGNet,
)
from typing import Union, Literal
from torchvision import transforms
from cifar100_pytorch.common import get_mean_std, DATA_PATH
from torchvision import datasets
from torch.utils.data import DataLoader


def config_to_net(
    config: Union[Literal["A"], Literal["B"], Literal["C"], Literal["D"], Literal["E"]],
    in_channels: int,
    num_classes: int,
) -> VGGNet:
    """
    Converts a configuration to a VGGNet

    Args:
        config: The configuration to convert to a VGGNet

    Returns:
        The VGGNet instance corresponding to the configuration
    """
    str_to_net = {
        "A": VGGNetA,
        "B": VGGNetB,
        "C": VGGNetC,
        "D": VGGNetD,
        "E": VGGNetE,
    }
    if not str_to_net.get(config):
        raise ValueError(f"Invalid config: {config}")

    return str_to_net.get(config)(in_channels, num_classes)


def _get_transform():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=get_mean_std().get("mean"), std=get_mean_std().get("std")
            ),
        ]
    )
    return transform


def get_transformed_dataloader(train: bool, batch_size, shuffle) -> DataLoader:
    """Returns a dataloader for the CIFAR100 dataset"""

    transform = _get_transform()
    dataset = datasets.CIFAR100(
        root=DATA_PATH, train=train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

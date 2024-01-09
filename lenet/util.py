from torchvision import transforms
from common.util import get_mnist_dataloader
from torch.utils.data import DataLoader


def _get_transform():
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    return transform


def get_transformed_dataloader(train: bool, batch_size, shuffle) -> DataLoader:
    """Returns a dataloader for the MNIST dataset ready to be used in LeNet5"""

    return get_mnist_dataloader(
        batch_size=batch_size,
        train=train,
        transforms=_get_transform(),
        shuffle=shuffle,
    )

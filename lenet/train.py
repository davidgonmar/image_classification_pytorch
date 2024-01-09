import torch
from common.util import get_mnist_dataloader_train
from tqdm import tqdm
from lenet.model import LeNet5
from common.util import save_model
from torchvision import transforms

BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10


def main():
    """Main function for training"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training model LeNet5 on device {}".format(device))

    net = LeNet5().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    train_loader, _ = get_mnist_dataloader_train(
        additional_transforms=[
            transforms.Resize((32, 32), antialias=False),
        ],
        split=1.0,  # No validation set
        train_batch_size=BATCH_SIZE,
    )

    for epoch in range(EPOCHS):
        net.train()
        train_loss = 0.0

        train_progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training epoch {}".format(epoch + 1),
            leave=True,
            position=0,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        for i, (inputs, labels) in train_progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_progress.set_postfix(
                {
                    "loss": "{:.6f}".format(train_loss / (i + 1)),
                }
            )

    save_model(net, net.__class__.__name__ + ".pth")

    print(
        "Finished training. Saved model to {}".format(net.__class__.__name__ + ".pth")
    )


if __name__ == "__main__":
    main()

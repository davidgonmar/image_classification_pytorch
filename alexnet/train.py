import torch
from vgg.util import get_transformed_dataloader
from alexnet.model import AlexNet
from common.util import save_model
from tqdm import tqdm

BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
INITIAL_LEARNING_RATE = 0.01
EPOCHS = 100


def main():
    """Main function for training"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training model AlexNet on device {}".format(device))

    net = AlexNet(num_classes=100).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=INITIAL_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    train_dloader = get_transformed_dataloader(
        train=True, batch_size=BATCH_SIZE, shuffle=True
    )

    valid_dloader = get_transformed_dataloader(
        train=False, batch_size=BATCH_SIZE, shuffle=False
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, verbose=True
    )

    for epoch in range(EPOCHS):
        net.train()
        train_loss = 0.0

        train_progress = tqdm(
            enumerate(train_dloader),
            total=len(train_dloader),
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

        train_loss /= len(train_dloader)

        valid_loss = 0.0
        valid_correct = 0

        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_dloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)

                loss = criterion(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                valid_correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_dloader)

        valid_accuracy = 100 * valid_correct / len(valid_dloader.dataset)

        scheduler.step(valid_loss)

        print(
            "Epoch {} - Train Loss: {:.6f}, Valid Loss: {:.6f}, Valid Accuracy: {:.2f}%".format(
                epoch + 1, train_loss, valid_loss, valid_accuracy
            )
        )

    save_model(net, net.__class__.__name__ + ".pth")


if __name__ == "__main__":
    main()

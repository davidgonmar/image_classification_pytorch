import json
from common.util import save_model
import torch
from vgg.util import get_transformed_dataloader
from tqdm import tqdm
import argparse
from vgg.util import config_to_net


BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-5
INITIAL_LEARNING_RATE = 1e-2
NUM_EPOCHS = 100


def main(args):
    """Main function for training"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "Training model VGGNet with configuration {} on device {}".format(
            args.config, device
        )
    )
    train_dloader = get_transformed_dataloader(
        train=True, batch_size=BATCH_SIZE, shuffle=True
    )

    valid_dloader = get_transformed_dataloader(
        train=False, batch_size=BATCH_SIZE, shuffle=False
    )

    net = config_to_net(args.config, in_channels=3, num_classes=100)

    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, verbose=True
    )

    for epoch in range(NUM_EPOCHS):
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

        for i, (features, labels) in train_progress:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dloader)

        # validate
        net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for features, labels in valid_dloader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = net(features)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        valid_loss = valid_loss / len(valid_dloader)

        scheduler.step(valid_loss)

        print(
            "Epoch {}/{}, train_loss: {:.3f}, valid_loss: {:.3f}".format(
                epoch + 1, NUM_EPOCHS, train_loss, valid_loss
            )
        )

        save_model(net, net.__class__.__name__ + ".pth")

        # append state in json file
        with open("results.json", "w") as f:
            _dict = {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "model_name": net.__class__.__name__,
                "total_classes": len(train_dloader.dataset.classes),
                "epoch": epoch,
            }
            json.dump(_dict, f)

    save_model(net, net.__class__.__name__ + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        choices=["A", "B", "C", "D", "E"],
        help="Configuration of the network to use (A, B, C, D, or E)",
        required=True,
    )
    main(parser.parse_args())

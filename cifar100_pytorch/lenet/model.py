from torch import nn
import torch


class LeNet5(nn.Module):
    NUM_CLASSES = 10

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=self.NUM_CLASSES)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = self.avgpool1(self.tanh(self.conv1(x)))
        x = self.avgpool2(self.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.softmax(self.fc3(x))

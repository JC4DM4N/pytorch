import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # output-size 6 x 28 x 28
        x = self.pool(x)            # output-size 6 x 14 x 14
        x = F.relu(self.conv2(x))   # output-size 16 x 10 x 10
        x = self.pool(x)            # output-size 16 x 5 x 5
        x = x.view(-1, 16 * 5 * 5)  # output-size 1 x 400
        x = F.relu(self.fc1(x))     # output-size 1 x 120
        x = F.relu(self.fc2(x))     # output-size 1 x 84
        x = self.fc3(x)             # output-size 1 x 10
        return x

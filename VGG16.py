import torch.nn as nn
import torch.nn.functional as F


class ConvNetVGG16(nn.Module):
    def __init__(self):
        super(ConvNetVGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        # pool
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        # pool
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        # pool
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        # pool
        self.fc1 = nn.Linear(
            in_features=512 * 1 * 1, out_features=256
        )  # altered from original
        self.fc2 = nn.Linear(in_features=256, out_features=64)  # altered from original
        self.fc3 = nn.Linear(in_features=64, out_features=10)  # altered from original

    def forward(self, x):
        x = F.relu(self.conv1(x))  # output-size = 64 x 32 x 32
        x = F.relu(self.conv2(x))  # output-size = 64 x 32 x 32
        x = self.pool(x)  # output-size = 64 x 16 x 16
        x = F.relu(self.conv3(x))  # output-size = 128 x 16 x 16
        x = F.relu(self.conv4(x))  # output-size = 128 x 16 x 16
        x = self.pool(x)  # output-size = 128 x 8 x 8
        x = F.relu(self.conv5(x))  # output-size = 256 x 8 x 8
        x = F.relu(self.conv6(x))  # output-size = 256 x 8 x 8
        x = F.relu(self.conv7(x))  # output-size = 256 x 8 x 8
        x = self.pool(x)  # output-size = 256 x 4 x 4
        x = F.relu(self.conv8(x))  # output-size = 512 x 4 x 4
        x = F.relu(self.conv9(x))  # output-size = 512 x 4 x 4
        x = F.relu(self.conv10(x))  # output-size = 512 x 4 x 4
        x = self.pool(x)  # output-size = 512 x 2 x 2
        x = F.relu(self.conv11(x))  # output-size = 512 x 2 x 2
        x = F.relu(self.conv12(x))  # output-size = 512 x 2 x 2
        x = F.relu(self.conv13(x))  # output-size = 512 x 2 x 2
        x = self.pool(x)  # output-size = 512 x 1 x 1
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))  # output-size = 1 x 256
        x = F.relu(self.fc2(x))  # output-size = 1 x 64
        x = F.relu(self.fc3(x))  # output-size = 1 x 10
        x = F.softmax(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

num_epochs = 4
batch_size = 4
learning_rate = 1e-4
device = torch.device("cpu")
MODEL_OUTPUT_PATH = './cnn.pth'

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ]
)

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

classes = train_data.classes

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                     
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)           
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # output-size 6 x 28 x 28
        x = self.pool(x)          # output-size 6 x 14 x 14
        x = F.relu(self.conv2(x)) # output-size 16 x 10 x 10
        x = self.pool(x)          # output-size 16 x 5 x 5
        x = x.view(-1, 16*5*5)    # output-size 1 x 400
        x = F.relu(self.fc1(x))   # output-size 1 x 120
        x = F.relu(self.fc2(x))   # output-size 1 x 84
        x = self.fc3(x)           # output-size 1 x 10
        return x

class ConvNetVGG16(nn.Module):
    def __init__(self):
        super(ConvNetVGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # pool
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #pool
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # pool
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # pool
        self.fc1 = nn.Linear(in_features=512*1*1, out_features=256) # altered from original
        self.fc2 = nn.Linear(in_features=256, out_features=64)      # altered from original
        self.fc3 = nn.Linear(in_features=64, out_features=10)       # altered from original


    def forward(self, x):
        x = F.relu(self.conv1(x))  # output-size = 64 x 32 x 32
        x = F.relu(self.conv2(x))  # output-size = 64 x 32 x 32
        x = self.pool(x)           # output-size = 64 x 16 x 16
        x = F.relu(self.conv3(x))  # output-size = 128 x 16 x 16
        x = F.relu(self.conv4(x))  # output-size = 128 x 16 x 16
        x = self.pool(x)           # output-size = 128 x 8 x 8
        x = F.relu(self.conv5(x))  # output-size = 256 x 8 x 8
        x = F.relu(self.conv6(x))  # output-size = 256 x 8 x 8
        x = F.relu(self.conv7(x))  # output-size = 256 x 8 x 8
        x = self.pool(x)           # output-size = 256 x 4 x 4
        x = F.relu(self.conv8(x))  # output-size = 512 x 4 x 4
        x = F.relu(self.conv9(x))  # output-size = 512 x 4 x 4
        x = F.relu(self.conv10(x)) # output-size = 512 x 4 x 4
        x = self.pool(x)           # output-size = 512 x 2 x 2
        x = F.relu(self.conv11(x)) # output-size = 512 x 2 x 2
        x = F.relu(self.conv12(x)) # output-size = 512 x 2 x 2
        x = F.relu(self.conv13(x)) # output-size = 512 x 2 x 2
        x = self.pool(x)           # output-size = 512 x 1 x 1
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))    # output-size = 1 x 256
        x = F.relu(self.fc2(x))    # output-size = 1 x 64
        x = F.relu(self.fc3(x))    # output-size = 1 x 10
        x = F.softmax(x)
        return x

model = ConvNetVGG16().to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# TRAIN MODEL
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = loss_func(outputs, labels)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.4f}")

print("Finished")

# save output model
torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

model = ConvNet()
model.load_state_dict(torch.load(MODEL_OUTPUT_PATH))

# EVALUATE MODEL
correct = 0
incorrect = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        correct += (predictions == labels).sum().item()
        incorrect += (predictions != labels).sum().item()

print(f"Correct predictions: {correct}, Incorrect predictions: {incorrect}")

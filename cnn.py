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
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

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

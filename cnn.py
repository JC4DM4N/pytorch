import torch
import torchvision
import torchvision.transforms as transforms

num_epochs = 1
batch_size = 4

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

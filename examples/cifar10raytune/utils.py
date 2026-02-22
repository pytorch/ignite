import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return train_set, test_set


def get_data_loaders(batch_size, data_dir="./data", num_workers=8):
    train_set, _ = load_data(data_dir)

    train_size = int(len(train_set) * 0.8)
    val_size = len(train_set) - train_size

    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader


def get_test_loader(batch_size, data_dir="./data", num_workers=8):
    _, test_set = load_data(data_dir)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader

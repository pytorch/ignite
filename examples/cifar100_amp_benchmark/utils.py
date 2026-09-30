import random

from torch.utils.data import DataLoader, Subset
from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import Compose, Normalize, Pad, RandomCrop, RandomErasing, RandomHorizontalFlip, ToTensor


def get_train_eval_loaders(path, batch_size=256):
    """Setup the dataflow:
        - load CIFAR100 train and test datasets
        - setup train/test image transforms
            - horizontally flipped randomly and augmented using cutout.
            - each mini-batch contained 256 examples
        - setup train/test data loaders

    Returns:
        train_loader, test_loader, eval_train_loader
    """
    train_transform = Compose(
        [
            Pad(4),
            RandomCrop(32),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(),
        ]
    )

    test_transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CIFAR100(root=path, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100(root=path, train=False, transform=test_transform, download=False)

    train_eval_indices = [random.randint(0, len(train_dataset) - 1) for i in range(len(test_dataset))]
    train_eval_dataset = Subset(train_dataset, train_eval_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=12, shuffle=True, drop_last=True, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=12, shuffle=False, drop_last=False, pin_memory=True
    )

    eval_train_loader = DataLoader(
        train_eval_dataset, batch_size=batch_size, num_workers=12, shuffle=False, drop_last=False, pin_memory=True
    )

    return train_loader, test_loader, eval_train_loader

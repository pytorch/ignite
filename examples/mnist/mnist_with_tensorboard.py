"""
 MNIST example with training and validation monitoring using Tensorboard.
 Requirements:
    TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    or PyTorch >= 1.2 which supports Tensorboard
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python mnist_with_tensorboard.py --log_dir=/tmp/tensorboard_logs
    ```
"""

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ModuleNotFoundError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda)."
        )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=train_batch_size, shuffle=True
    )

    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False), batch_size=val_batch_size, shuffle=False
    )
    return train_loader, val_loader


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    writer = SummaryWriter(log_dir=log_dir)
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    model.to(device)  # Move model before creating optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.NLLLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}/{len(train_loader)}] "
            f"Loss: {engine.state.output:.2f}"
        )
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        print(
            f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--log_dir", type=str, default="tensorboard_logs", help="log directory for Tensorboard log output"
    )

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval, args.log_dir)

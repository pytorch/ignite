from __future__ import print_function
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.contrib.handlers import VisdomLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


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

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    visdom_logger = VisdomLogger()

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    #
    # Setup the optimizer with learning rate scheduler
    #
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        param_name='lr',
        start_value=lr,
        end_value=lr / 10,
        cycle_size=epochs,
        save_history=True
    )

    #
    # Setup the training/validation engines
    #
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    ProgressBar().attach(trainer, output_transform=lambda x: {"Loss": x})
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    metrics = {'accuracy': Accuracy(),
               'nll': Loss(F.nll_loss)}

    train_evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device
    )

    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device
    )

    loss_win = visdom_logger.create_window(
        window_title="Training Loss",
        xlabel="Iteration",
        ylabel="Loss"
    )
    loss_win.attach(
        engine=trainer,
        update_period=log_interval,
        plot_event=Events.ITERATION_COMPLETED,
        output_transform=lambda x: {"loss": x},
    )

    visdom_logger.create_window(
        window_title="Learning Rate"
    ).attach(
        engine=trainer,
        param_history=True
    )

    avg_loss_win = visdom_logger.create_window(
        window_title="Average Loss",
        ylabel="Loss",
        show_legend=True
    )
    avg_loss_win.attach(
        engine=train_evaluator,
        metric_names={'train': 'nll'}
    )
    avg_loss_win.attach(
        engine=evaluator,
        metric_names={'val': 'nll'}
    )

    avg_acc_win = visdom_logger.create_window(
        window_title="Average Acc",
        ylabel="Accuracy",
        show_legend=True
    )
    avg_acc_win.attach(
        engine=train_evaluator,
        metric_names={'train': 'accuracy'}
    )
    avg_acc_win.attach(
        engine=evaluator,
        metric_names={'val': 'accuracy'}
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_file", type=str, default=None, help="log file to log output to")

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)

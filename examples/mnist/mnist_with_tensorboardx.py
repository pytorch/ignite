"""
 MNIST example with training and validation monitoring using TensorboardX and Tensorboard.
 Requirements:
    TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python mnist_with_tensorboardx.py --log_dir=/tmp/tensorboard_logs
    ```
"""
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
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.metrics import RunningAverage

LOG_INTERVAL = 10


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


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)

    main_pb = ProgressBar(bar_format=None, persist=True)
    main_pb.attach(trainer, output_transform=lambda x: x)

    train_evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(),
                 'nll': Loss(F.nll_loss)},
        device=device
    )
    ProgressBar(desc="Train Evaluation", bar_format=None).attach(train_evaluator)

    validation_evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(),
                 'nll': Loss(F.nll_loss)},
        device=device
    )
    ProgressBar(desc="Evaluation", bar_format=None).attach(validation_evaluator)

    tbLogger = TensorboardLogger(log_dir=log_dir)

    tbLogger.attach(
        engine=trainer,
        name="training",
        plot_event=Events.ITERATION_COMPLETED,
        update_period=LOG_INTERVAL,
        output_transform=lambda x: x,
    )

    tbLogger.attach(
        engine=train_evaluator,
        name="train_evaluation",
        metric_names=['accuracy', 'nll'],
        model=model,
        histogram_freq=1,
        write_grads=True,
    )

    tbLogger.attach(
        engine=validation_evaluator,
        name="evaluation",
        metric_names=['accuracy', 'nll']
    )

    @trainer.on(Events.STARTED)
    def write_graph(engine):
        tbLogger.write_graph(model, train_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        main_pb.log_message("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))

        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        main_pb.log_message("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))

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
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir)

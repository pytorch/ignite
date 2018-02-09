from __future__ import print_function
from argparse import ArgumentParser
import logging

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

try:
    import visdom
except ImportError:
    raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")

from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from ignite.engine import Events
from ignite.handlers.evaluate import Evaluate
from ignite.handlers.logging import log_simple_moving_average
import numpy as np


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
        return F.log_softmax(x)


def get_plot_training_loss_handler(vis, plot_every):
    train_loss_plot_window = vis.line(X=np.array([1]), Y=np.array([np.nan]),
                                      opts=dict(
                                          xlabel='# Iterations',
                                          ylabel='Loss',
                                          title='Training Loss')
                                      )

    def plot_training_loss_to_visdom(trainer):
        if trainer.current_iteration % plot_every == 0:
            vis.line(X=np.array([trainer.current_iteration]),
                     Y=np.array([trainer.history.simple_moving_average(window_size=100)]),
                     win=train_loss_plot_window,
                     update='append')

    return plot_training_loss_to_visdom


def get_plot_validation_accuracy_handler(vis):
    val_accuracy_plot_window = vis.line(X=np.array([1]), Y=np.array([np.nan]),
                                        opts=dict(
                                            xlabel='# Epochs',
                                            ylabel='Accuracy',
                                            title='Validation Accuracy')
                                        )

    def plot_val_accuracy_to_visdom(evaluator, trainer):
        accuracy = sum([accuracy for (loss, accuracy) in evaluator.history])
        accuracy = (accuracy * 100.) / len(evaluator.dataloader.dataset)
        vis.line(X=np.array([trainer.current_epoch]),
                 Y=np.array([accuracy]),
                 win=val_accuracy_plot_window,
                 update='append')

    return plot_val_accuracy_to_visdom


def get_log_validation_loss_and_accuracy_handler(logger):
    def log_validation_loss_and_accuracy(evaluator):
        avg_loss = np.mean([loss for (loss, accuracy) in evaluator.history])
        accuracy = sum([accuracy for (loss, accuracy) in evaluator.history])
        logger('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, accuracy, len(evaluator.dataloader.dataset),
            (accuracy * 100.) / len(evaluator.dataloader.dataset)
        ))

    return log_validation_loss_and_accuracy


def run(batch_size, val_batch_size, epochs, lr, momentum, log_interval, logger):
    vis = visdom.Visdom()
    if not vis.check_connection():
        raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)

    model = Net()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    def training_update_function(batch):
        model.train()
        optimizer.zero_grad()
        data, target = Variable(batch[0]), Variable(batch[1])
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        return loss.data[0]

    def validation_inference_function(batch):
        model.eval()
        data, target = Variable(batch[0], volatile=True), Variable(batch[1])
        output = model(data)
        loss = F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        return loss, correct

    trainer = Trainer(training_update_function)
    evaluator = Evaluator(validation_inference_function)

    # trainer event handlers
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              log_simple_moving_average,
                              window_size=100,
                              metric_name="NLL",
                              should_log=lambda trainer: trainer.current_iteration % log_interval == 0,
                              logger=logger)
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              get_plot_training_loss_handler(vis, plot_every=log_interval))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluate(evaluator, val_loader, epoch_interval=1))

    # evaluator event handlers
    evaluator.add_event_handler(Events.COMPLETED, get_log_validation_loss_and_accuracy_handler(logger))
    evaluator.add_event_handler(Events.COMPLETED, get_plot_validation_accuracy_handler(vis), trainer)

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

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler())
        logger = logger.info
    else:
        logger = print

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, logger)

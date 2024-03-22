"""
 MNIST example with training and validation monitoring using ClearML.

 Requirements:
    ClearML: `pip install clearml`

 Usage:

    Run the example:
    ```bash
    python mnist_with_clearml_logger.py
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
from ignite.handlers import Checkpoint

from ignite.handlers.clearml_logger import (
    ClearMLLogger,
    ClearMLSaver,
    global_step_from_engine,
    GradsHistHandler,
    GradsScalarHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
)
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger


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


def run(train_batch_size, val_batch_size, epochs, lr, momentum):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    model.to(device)  # Move model before creating optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger("Trainer")

    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)

    clearml_logger = ClearMLLogger(project_name="examples", task_name="ignite")

    clearml_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
    )

    for tag, evaluator in [("training metrics", train_evaluator), ("validation metrics", validation_evaluator)]:
        clearml_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["loss", "accuracy"],
            global_step_transform=global_step_from_engine(trainer),
        )

    clearml_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=100), optimizer=optimizer
    )

    clearml_logger.attach(
        trainer,
        log_handler=WeightsScalarHandler(model, whitelist=["fc1"]),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    def is_conv(n, _):
        return "conv" in n

    clearml_logger.attach(
        trainer,
        log_handler=WeightsHistHandler(model, whitelist=is_conv),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    clearml_logger.attach(
        trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED(every=100)
    )

    clearml_logger.attach(
        trainer,
        log_handler=GradsHistHandler(model, whitelist=["fc2.weight"]),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    handler = Checkpoint(
        {"model": model},
        ClearMLSaver(),
        n_saved=1,
        score_function=lambda e: e.state.metrics["accuracy"],
        score_name="val_acc",
        filename_prefix="best",
        global_step_transform=global_step_from_engine(trainer),
    )
    validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    clearml_logger.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum)

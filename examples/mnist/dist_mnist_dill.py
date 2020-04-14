import os
import dill
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm


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

    train_ds = MNIST(download=True, root=".", transform=data_transform, train=True)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, sampler=train_sampler)

    val_ds = MNIST(download=False, root=".", transform=data_transform, train=False)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, sampler=val_sampler)

    return train_loader, val_loader


def prepare_and_save(train_batch_size, val_batch_size, lr, momentum):

    size = dist.get_world_size()
    rank = dist.get_rank()

    # consistante print
    for r in range(size):
        if rank == r:
            print("run on {}/{}".format(rank, size))
        dist.barrier()

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    device = "cpu"

    # if torch.cuda.is_available():
    #     device = "cuda"

    model.to(device)  # Move model before creating optimizer
    model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "nll": Loss(F.nll_loss)}, device=device
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        if dist.get_rank() == 0:
            metrics = evaluator.state.metrics
            avg_accuracy = metrics["accuracy"]
            avg_nll = metrics["nll"]
            print(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_nll
                )
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        if dist.get_rank() == 0:
            avg_accuracy = metrics["accuracy"]
            avg_nll = metrics["nll"]
            print(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_nll
                )
            )

    with open("trainer.{}.pkl".format(rank), "wb") as file:
        dill.dump(trainer, file)

    with open("train_loader.{}.pkl".format(rank), "wb") as file:
        dill.dump(train_loader, file)


def run(epochs):

    rank = dist.get_rank()

    with open("trainer.{}.pkl".format(rank), "rb") as file:
        trainer = dill.load(file)

    with open("train_loader.{}.pkl".format(rank), "rb") as file:
        train_loader = dill.load(file)

    trainer.run(train_loader, max_epochs=epochs)


def init_process_for_prepare(args, backend="gloo"):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    prepare_and_save(args.batch_size, args.val_batch_size, args.lr, args.momentum)


def init_process_for_run(args, backend="gloo"):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(args.epochs)


if __name__ == "__main__":

    print("argument parsing")
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument("--np", type=int, default=2, help="Number of process")

    parser.add_argument("--dill", type=int, default=0, help="0: prepare 1: run")

    args = parser.parse_args()

    print("run multiprocessing with {} process".format(args.np))

    size = args.np
    processes = []

    if args.dill == 0:
        target = init_process_for_prepare
    else:
        target = init_process_for_run

    for rank in range(size):
        p = Process(target=target, args=(args,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

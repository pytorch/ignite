from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data
import torch.distributed
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets, transforms

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
    train_dataset = datasets.MNIST("../data", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    val_dataset = datasets.MNIST("../data", train=False, download=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_batch_size, shuffle=(train_sampler is None))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    return train_loader, val_loader, train_sampler


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    train_loader, val_loader, train_sampler = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()

    if torch.cuda.is_available():
        device = "cuda"
        model.cuda(args.gpu)
    else:
        device = "cpu"

    if args.distributed:
        model = DistributedDataParallel(model, [args.gpu])

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={"accuracy": Accuracy(),
                                                     "nll": Loss(F.nll_loss)},
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            train_sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if args.distributed:
            train_sampler.set_epoch(iter + 1)

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=1000,
                        help="input batch size for validation (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--dist_method", default="file:///home/user/tmp.dat", type=str,
                        help="url or file path used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--world_size", default=1, type=int, help="Number of GPUs to use.")
    parser.add_argument("--rank", default=0, type=int, help="Used for multi-process training.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU number to use.")
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(args.dist_backend,
                                             init_method=args.dist_method,
                                             world_size=args.world_size, rank=args.rank)

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)

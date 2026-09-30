import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, ToTensor

import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy

in_colab = "COLAB_TPU_ADDR" in os.environ
with_torchrun = "WORLD_SIZE" in os.environ

train_transform = Compose(
    [
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225)),
    ]
)

test_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225)),])


def get_train_test_datasets(path):
    # - Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    train_ds = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

    if idist.get_rank() == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    return train_ds, test_ds


def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    return fn(num_classes=10)


def get_dataflow(config):

    train_dataset, test_dataset = get_train_test_datasets(config.get("data_path", "."))

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config.get("batch_size", 512),
        num_workers=config.get("num_workers", 8),
        shuffle=True,
        drop_last=True,
    )
    config["num_iters_per_epoch"] = len(train_loader)

    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=2 * config.get("batch_size", 512),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
    )
    return train_loader, test_loader


def initialize(config):
    model = get_model(config["model"])
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("learning_rate", 0.1),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 1e-5),
        nesterov=True,
    )
    optimizer = idist.auto_optim(optimizer)
    criterion = nn.CrossEntropyLoss().to(idist.device())

    le = config["num_iters_per_epoch"]
    lr_scheduler = StepLR(optimizer, step_size=le, gamma=0.9)

    return model, optimizer, criterion, lr_scheduler


# slide 1 ####################################################################


def create_trainer(model, optimizer, criterion, lr_scheduler, config):

    # Define any training logic for iteration update
    def train_step(engine, batch):
        x, y = batch[0].to(idist.device()), batch[1].to(idist.device())

        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return loss.item()

    # Define trainer engine
    trainer = Engine(train_step)

    if idist.get_rank() == 0:
        # Add any custom handlers
        @trainer.on(Events.ITERATION_COMPLETED(every=200))
        def save_checkpoint():
            fp = Path(config.get("output_path", "output")) / "checkpoint.pt"
            torch.save(model.state_dict(), fp)

        # Add progress bar showing batch loss value
        ProgressBar().attach(trainer, output_transform=lambda x: {"batch loss": x})

    return trainer


# slide 2 ####################################################################


def training(local_rank, config):

    # Setup dataflow and
    train_loader, val_loader = get_dataflow(config)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    # Setup model trainer and evaluator
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, config)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()}, device=idist.device())

    # Run model evaluation every 3 epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def evaluate_model():
        state = evaluator.run(val_loader)
        if idist.get_rank() == 0:
            print(state.metrics)

    # Setup tensorboard experiment tracking
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            config.get("output_path", "output"), trainer, optimizer, evaluators={"validation": evaluator},
        )

    trainer.run(train_loader, max_epochs=config.get("max_epochs", 3))

    if idist.get_rank() == 0:
        tb_logger.close()


# slide 3 ####################################################################

# Simply run everything on your infrastructure


# --- Single computation device ---
# $ python main.py
#
if __name__ == "__main__" and not (in_colab or with_torchrun):

    backend = None
    nproc_per_node = None
    config = {
        "model": "resnet18",
        "dataset": "cifar10",
    }

    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, config)


# --- Multiple GPUs ---
# $ torchrun --nproc_per_node=2 main.py
#
if __name__ == "__main__" and with_torchrun:

    backend = "nccl"  # or "nccl", "gloo", ...
    nproc_per_node = None
    config = {
        "model": "resnet18",
        "dataset": "cifar10",
    }

    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, config)

# --- Multiple TPUs ---
# In Colab
#
if in_colab:

    backend = "xla-tpu"
    nproc_per_node = 8
    config = {
        "model": "resnet18",
        "dataset": "cifar10",
    }

    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(training, config)


# Full featured CIFAR10 example:
# https://github.com/pytorch/ignite/tree/master/examples/cifar10

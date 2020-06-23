from typing import Mapping

import torch

import ignite.distributed as idist
from ignite.engine import Events
from ignite.utils import manual_seed, setup_logger

import utils
from .trainer import create_trainer

try:
    from apex import amp

    has_apex_amp = True
except ImportError:
    has_apex_amp = False


def training(local_rank: int, config: Mapping):

    rank = idist.get_rank()
    logger = setup_logger("FixMatch Training", distributed_rank=rank)

    if rank == 0:
        logger.info(repr(config))

    manual_seed(config["seed"] + rank)
    device = idist.device()

    model, optimizer, criterion, lr_scheduler = utils.initialize(config)

    if has_apex_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config["amp_opt_level"])

    train_loader, val_loader = utils.get_dataflow(config)

    def train_step(engine, batch):

        # train_loader's batch is a tuple : (sample, target)
        x, y = batch[0].to(device), batch[1].to(device)

        model.train()
        optimizer.zero_grad()

        y_pred = model(x)
        batch_loss = criterion(y_pred, y)

        if has_apex_amp:
            with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            batch_loss.backward()

        optimizer.step()

        return {
            "batch_loss": batch_loss.item(),
        }

    trainer = create_trainer(train_step, ...)

    num_epochs = config["num_epochs"]

    try:
        trainer.run(train_loader, max_epochs=num_epochs)
    except Exception as e:
        logger.exception("")
        raise e


def get_default_config():
    return {
        "seed": 22,
        "model": "resnet18",
        "dataset": "cifar10",
        # Use AMP if installed
        "amp_opt_level": "O1",
    }


if __name__ == "__main__":

    # Setup distributed computation backend
    # backend = "nccl"  # or None to disable
    backend = "gloo"  # or None to disable

    config = get_default_config()

    with idist.Parallel(backend=backend) as parallel:
        parallel.run(training, config)

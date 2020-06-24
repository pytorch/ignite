from typing import Mapping

import torch
import ignite.distributed as idist
from ignite.utils import manual_seed, setup_logger

from utils import initialize, get_dataflow
from trainer import create_trainer

try:
    from apex import amp

    has_apex_amp = True and torch.cuda.is_available()
except ImportError:
    has_apex_amp = False


def training(local_rank: int, config: Mapping):

    rank = idist.get_rank()
    logger = setup_logger("Training", distributed_rank=rank)

    if torch.cuda.is_available() and torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    if rank == 0:
        logger.info(repr(config))

    manual_seed(config["seed"] + rank)
    device = idist.device()

    model, optimizer, criterion, lr_scheduler = initialize(config)

    if has_apex_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config["amp_opt_level"])

    train_loader, val_loader = get_dataflow(config)

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

    trainer = create_trainer(
        train_step,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
    )

    num_epochs = config["num_epochs"]

    try:
        trainer.run(train_loader, max_epochs=num_epochs)
    except Exception as e:
        logger.exception("")
        raise e


def get_default_config():
    return {
        # Global configs
        "seed": 22,
        "output_path": "/tmp/output",
        "batch_size": 8,
        "num_workers": 4,
        # Model configs
        "num_classes": 10,
        "model": "resnet18",
        # Solver configs
        "learning_rate": 0.01,
        "step_size": 20,
        "gamma": 0.3,
        "num_epochs": 50,
        # number of evaluations to tolerate if no improvement before stopping the training. Set None to disable.
        "early_stopping_patience": 3,
        # Trainer custom configs
        "checkpoint_every": 30,  # checkpoint training every 30 iterations
        "validate_every": 3,  # run model validation every 3 epochs
        "amp_opt_level": "O1",  # Use AMP if installed
        "resume_from": None,  # path to checkpoint to resume the training from
    }


if __name__ == "__main__":

    # Setup distributed computation backend
    backend = "nccl"  # or "gloo" or None to disable

    config = get_default_config()

    with idist.Parallel(backend=backend) as parallel:
        parallel.run(training, config)

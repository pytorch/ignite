from typing import Callable, Mapping

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import ignite.distributed as idist
import ignite.contrib.engines.common as common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import RunningAverage, Accuracy, Precision, Recall
from ignite.contrib.handlers import ProgressBar


def create_trainer(
    train_step: Callable,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lr_scheduler: _LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Mapping,
):

    # Define trainer engine
    trainer = Engine(train_step)

    # In distributed configuration we need resample distributed sampler on every epoch
    if idist.get_world_size() > 1:

        assert hasattr(train_loader.sampler, "set_epoch")

        @trainer.on(Events.EPOCH_STARTED)
        def resample():
            train_loader.sampler.set_epoch(trainer.state.epoch - 1)

    # Setup training checkpointer to store locally training ingredients
    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "lr_scheduler": lr_scheduler,
    }
    checkpointer = Checkpoint(
        to_save, save_handler=DiskSaver(config["output_path"], require_empty=False), filename_prefix="training"
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=config["checkpoint_every"]), checkpointer)

    # Add running average mean metric computed on batch loss
    output_name = "batch_loss"  # key of train_step's dictionary
    RunningAverage(output_transform=lambda output: output[output_name]).attach(trainer, output_name)

    # Add training progress bar
    if idist.get_rank() == 0:
        ProgressBar().attach(trainer, metric_names="all")

    # Setup model validation.
    device = idist.device()

    val_metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average=True),
        "recall": Recall(average=True),
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    # Run validation every n-th epochs and in the end of the training
    @trainer.on(Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED)
    def run_validation():
        evaluator.run(val_loader)

    # Save 3 best models according to validation accuracy
    metric_name = "accuracy"
    common.save_best_model_by_val_score(
        config["output_path"],
        evaluator=evaluator,
        model=model,
        metric_name=metric_name,
        n_saved=3,
        trainer=trainer,
        tag="val",
    )

    # Setup early stopping
    if config["early_stopping_patience"] is not None:
        common.add_early_stopping_by_val_score(
            config["early_stopping_patience"], evaluator=evaluator, trainer=trainer, metric_name=metric_name
        )

    # Add tensorboard experiment tracking logger
    # Other available loggers can be added in the same way with `common.setup_*_logging`
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            config["output_path"], trainer, optimizer, evaluators={"validation": evaluator}, log_every_iters=10
        )
        trainer.add_event_handler(Events.COMPLETED | Events.EXCEPTION_RAISED, lambda _: tb_logger.close())

    return trainer

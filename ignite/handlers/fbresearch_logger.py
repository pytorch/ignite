"""FBResearch logger and its helper handlers."""

import datetime

import torch

from ignite.engine import Engine, Events
from ignite.handlers import Timer


MB = 1024.0 * 1024.0


class FBResearchLogger:
    """Logs training and validation metrics for research purposes.

    This logger is designed to attach to an Ignite Engine and log various metrics
    and system stats at configurable intervals, including learning rates, iteration
    times, and GPU memory usage.

    Args:
        logger (logging.Logger): The logger to use for output.
        delimiter (str): The delimiter to use between metrics in the log output.
        show_output (bool): Flag to enable logging of the output from the engine's process function.

    Examples:
        .. code-block:: python

            import logging
            from ignite.handlers.fbresearch_logger import *

            logger = FBResearchLogger(logger=logging.Logger(__name__), show_output=True)
            logger.attach(trainer, name="Train", every=10, optimizer=my_optimizer)
    """

    def __init__(self, logger, delimiter="  ", show_output=False):
        self.delimiter = delimiter
        self.logger = logger
        self.iter_timer = None
        self.data_timer = None
        self.show_output = show_output

    def attach(self, engine: Engine, name: str, every: int = 1, optimizer=None):
        """Attaches all the logging handlers to the given engine.

        Args:
            engine (Engine): The engine to attach the logging handlers to.
            name (str): The name of the engine (e.g., "Train", "Validate") to include in log messages.
            every (int): Frequency of iterations to log information. Logs are generated every 'every' iterations.
            optimizer: The optimizer used during training to log current learning rates.
        """
        engine.add_event_handler(Events.EPOCH_STARTED, self.log_epoch_started, engine, name)
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=every), self.log_every, engine, optimizer=optimizer)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch_completed, engine, name)
        engine.add_event_handler(Events.COMPLETED, self.log_completed, engine, name)

        self.iter_timer = Timer(average=True)
        self.iter_timer.attach(
            engine,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )
        self.data_timer = Timer(average=True)
        self.data_timer.attach(
            engine,
            start=Events.EPOCH_STARTED,
            resume=Events.GET_BATCH_STARTED,
            pause=Events.GET_BATCH_COMPLETED,
            step=Events.GET_BATCH_COMPLETED,
        )

    def log_every(self, engine, optimizer=None):
        """
        Logs the training progress at regular intervals.

        Args:
            engine (Engine): The training engine.
            optimizer (torch.optim.Optimizer, optional): The optimizer used for training. Defaults to None.
        """
        cuda_max_mem = ""
        if torch.cuda.is_available():
            cuda_max_mem = f"GPU Max Mem: {torch.cuda.max_memory_allocated() / MB:.0f} MB"

        current_iter = engine.state.iteration % (engine.state.epoch_length + 1)
        iter_avg_time = self.iter_timer.value()

        eta_seconds = iter_avg_time * (engine.state.epoch_length - current_iter)

        outputs = []
        if self.show_output:
            output = engine.state.output
            if isinstance(output, dict):
                outputs += [f"{k}: {v:.4f}" for k, v in output.items()]
            else:
                outputs += [f"{v:.4f}" if isinstance(v, float) else f"{v}" for v in output]

        lrs = ""
        if optimizer is not None:
            if len(optimizer.param_groups) == 1:
                lrs += f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            else:
                for i, g in enumerate(optimizer.param_groups):
                    lrs += f"lr [g{i}]: {g['lr']:.5f}"

        msg = self.delimiter.join(
            [
                f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]",
                f"[{current_iter}/{engine.state.epoch_length}]:",
                f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}",
                f"{lrs}",
            ]
            + outputs
            + [
                f"Iter time: {iter_avg_time:.4f} s",
                f"Data prep time: {self.data_timer.value():.4f} s",
                cuda_max_mem,
            ]
        )
        self.logger.info(msg)

    def log_epoch_started(self, engine, name):
        """
        Logs the start of an epoch.

        Args:
            engine (Engine): The engine object.
            name (str): The name of the epoch.

        """
        msg = f"{name}: start epoch [{engine.state.epoch}/{engine.state.max_epochs}]"
        self.logger.info(msg)

    def log_epoch_completed(self, engine, name):
        """
        Logs the completion of an epoch.

        Args:
            engine (Engine): The engine object.
            name (str): The name of the epoch.

        """
        epoch_time = engine.state.times[Events.EPOCH_COMPLETED.name]
        epoch_info = f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]" if engine.state.max_epochs > 1 else ""
        msg = self.delimiter.join(
            [
                f"{name}: {epoch_info}",
                f"Total time: {datetime.timedelta(seconds=int(epoch_time))}",
                f"({epoch_time / engine.state.epoch_length:.4f} s / it)",
            ]
        )
        self.logger.info(msg)

    def log_completed(self, engine, name):
        """
        Logs the completion of a run.

        Args:
            engine (Engine): The engine object.
            name (str): The name of the run.

        """
        if engine.state.max_epochs > 1:
            total_time = engine.state.times[Events.COMPLETED.name]
            msg = self.delimiter.join(
                [
                    f"{name}: run completed",
                    f"Total time: {datetime.timedelta(seconds=int(total_time))}",
                ]
            )
            self.logger.info(msg)

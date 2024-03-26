"""FBResearch logger and its helper handlers."""

import datetime
from typing import Any, Optional

# from typing import Any, Dict, Optional, Union

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
        logger: The logger to use for output.
        delimiter: The delimiter to use between metrics in the log output.
        show_output: Flag to enable logging of the output from the engine's process function.

    Examples:
        .. code-block:: python

            import logging
            from ignite.handlers.fbresearch_logger import *

            logger = FBResearchLogger(logger=logging.Logger(__name__), show_output=True)
            logger.attach(trainer, name="Train", every=10, optimizer=my_optimizer)
    """

    def __init__(self, logger: Any, delimiter: str = "  ", show_output: bool = False):
        self.delimiter = delimiter
        self.logger: Any = logger
        self.iter_timer: Timer = Timer(average=True)
        self.data_timer: Timer = Timer(average=True)
        self.show_output: bool = show_output

    def attach(
        self, engine: Engine, name: str, every: int = 1, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """Attaches all the logging handlers to the given engine.

        Args:
            engine: The engine to attach the logging handlers to.
            name: The name of the engine (e.g., "Train", "Validate") to include in log messages.
            every: Frequency of iterations to log information. Logs are generated every 'every' iterations.
            optimizer: The optimizer used during training to log current learning rates.
        """
        engine.add_event_handler(Events.EPOCH_STARTED, self.log_epoch_started, engine, name)
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=every), self.log_every, engine, optimizer=optimizer)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch_completed, engine, name)
        engine.add_event_handler(Events.COMPLETED, self.log_completed, engine, name)

        self.iter_timer.reset()
        self.iter_timer.attach(
            engine,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )
        self.data_timer.reset()
        self.data_timer.attach(
            engine,
            start=Events.EPOCH_STARTED,
            resume=Events.GET_BATCH_STARTED,
            pause=Events.GET_BATCH_COMPLETED,
            step=Events.GET_BATCH_COMPLETED,
        )

    def log_every(self, engine: Engine, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Logs the training progress at regular intervals.

        Args:
            engine: The training engine.
            optimizer: The optimizer used for training. Defaults to None.
        """
        assert engine.state.epoch_length is not None
        cuda_max_mem = ""
        if torch.cuda.is_available():
            cuda_max_mem = f"GPU Max Mem: {torch.cuda.max_memory_allocated() / MB:.0f} MB"

        current_iter = engine.state.iteration % (engine.state.epoch_length + 1)
        iter_avg_time = self.iter_timer.value()

        eta_seconds = iter_avg_time * (engine.state.epoch_length - current_iter)

        outputs = []
        if self.show_output and engine.state.output is not None:
            output = engine.state.output
            if isinstance(output, dict):
                outputs += [f"{k}: {v:.4f}" for k, v in output.items()]
            else:
                outputs += [f"{v:.4f}" if isinstance(v, float) else f"{v}" for v in output]  # type: ignore

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

    def log_epoch_started(self, engine: Engine, name: str) -> None:
        """
        Logs the start of an epoch.

        Args:
            engine: The engine object.
            name: The name of the epoch.

        """
        msg = f"{name}: start epoch [{engine.state.epoch}/{engine.state.max_epochs}]"
        self.logger.info(msg)

    def log_epoch_completed(self, engine: Engine, name: str) -> None:
        """
        Logs the completion of an epoch.

        Args:
            engine: The engine object that triggered the event.
            name: The name of the event.

        Returns:
            None
        """
        epoch_time = engine.state.times[Events.EPOCH_COMPLETED.name]
        epoch_info = (
            f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]"
            if engine.state.max_epochs > 1  # type: ignore
            else ""
        )
        msg = self.delimiter.join(
            [
                f"{name}: {epoch_info}",
                f"Total time: {datetime.timedelta(seconds=int(epoch_time))}",  # type: ignore
                f"({epoch_time / engine.state.epoch_length:.4f} s / it)",  # type: ignore
            ]
        )
        self.logger.info(msg)

    def log_completed(self, engine: Engine, name: str) -> None:
        """
        Logs the completion of a run.

        Args:
            engine: The engine object representing the training/validation loop.
            name: The name of the run.

        """
        if engine.state.max_epochs and engine.state.max_epochs > 1:
            total_time = engine.state.times[Events.COMPLETED.name]
            assert total_time is not None
            msg = self.delimiter.join(
                [
                    f"{name}: run completed",
                    f"Total time: {datetime.timedelta(seconds=int(total_time))}",
                ]
            )
            self.logger.info(msg)

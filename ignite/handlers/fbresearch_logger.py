"""FBResearch logger and its helper handlers."""

import datetime
from typing import Any, Callable, List, Optional

import torch

from ignite import utils
from ignite.engine import Engine, Events
from ignite.handlers import Timer

MB = 1024.0 * 1024.0

__all__ = ["FBResearchLogger"]


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

            import torch
            import torch.nn as nn
            import torch.optim as optim

            from ignite.engine import create_supervised_trainer, Events
            from ignite.handlers.fbresearch_logger import FBResearchLogger
            from ignite.utils import setup_logger

            model = nn.Linear(10, 5)
            opt = optim.SGD(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            data = [(torch.rand(4, 10), torch.randint(0, 5, size=(4, ))) for _ in range(100)]

            trainer = create_supervised_trainer(
                model, opt, criterion, output_transform=lambda x, y, y_pred, loss: {"total_loss": loss.item()}
            )

            logger = setup_logger("trainer", level=logging.INFO)
            logger = FBResearchLogger(logger=logger, show_output=True)
            logger.attach(trainer, name="Train", every=20, optimizer=opt)

            trainer.run(data, max_epochs=4)

        Output:

        .. code-block:: text

            2024-04-22 12:05:47,843 trainer INFO: Train: start epoch [1/4]
            ... Epoch [1/4]  [20/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.5999  Iter time: 0.0008 s  Data prep ..
            ... Epoch [1/4]  [40/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9297  Iter time: 0.0008 s  Data prep ..
            ... Epoch [1/4]  [60/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9985  Iter time: 0.0008 s  Data prep ..
            ... Epoch [1/4]  [80/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9785  Iter time: 0.0008 s  Data prep ..
            ... Epoch [1/4]  [100/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.6211  Iter time: 0.0008 s  Data prep .
            ... Train: Epoch [1/4]  Total time: 0:00:00  (0.0008 s / it)
            ... Train: start epoch [2/4]
            ... Epoch [2/4]  [19/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.5981  Iter time: 0.0009 s  Data prep ..
            ... Epoch [2/4]  [39/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9013  Iter time: 0.0008 s  Data prep ..
            ... Epoch [2/4]  [59/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9811  Iter time: 0.0008 s  Data prep ..
            ... Epoch [2/4]  [79/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9434  Iter time: 0.0008 s  Data prep ..
            ... Epoch [2/4]  [99/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.6116  Iter time: 0.0008 s  Data prep ..
            ... Train: Epoch [2/4]  Total time: 0:00:00  (0.0009 s / it)
            ... Train: start epoch [3/4]
            ... Epoch [3/4]  [18/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.5972  Iter time: 0.0008 s  Data prep ..
            ... Epoch [3/4]  [38/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.8753  Iter time: 0.0008 s  Data prep ..
            ... Epoch [3/4]  [58/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9657  Iter time: 0.0009 s  Data prep ..
            ... Epoch [3/4]  [78/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9112  Iter time: 0.0008 s  Data prep ..
            ... Epoch [3/4]  [98/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.6035  Iter time: 0.0008 s  Data prep ..
            ... Train: Epoch [3/4]  Total time: 0:00:00  (0.0009 s / it)
            ... Train: start epoch [4/4]
            ... Epoch [4/4]  [17/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.5969  Iter time: 0.0008 s  Data prep ..
            ... Epoch [4/4]  [37/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.8516  Iter time: 0.0008 s  Data prep ..
            ... Epoch [4/4]  [57/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.9521  Iter time: 0.0008 s  Data prep ..
            ... Epoch [4/4]  [77/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.8816  Iter time: 0.0008 s  Data prep ..
            ... Epoch [4/4]  [97/100]:  ETA: 0:00:00  lr: 0.00100  total_loss: 1.5966  Iter time: 0.0009 s  Data prep ..
            ... Train: Epoch [4/4]  Total time: 0:00:00  (0.0009 s / it)
            ... Train: run completed  Total time: 0:00:00
    """

    def __init__(self, logger: Any, delimiter: str = "  ", show_output: bool = False):
        self.delimiter = delimiter
        self.logger: Any = logger
        self.iter_timer: Timer = Timer(average=True)
        self.data_timer: Timer = Timer(average=True)
        self.show_output: bool = show_output

    def attach(
        self,
        engine: Engine,
        name: str,
        every: int = 1,
        output_transform: Optional[Callable] = None,
        state_attributes: Optional[List[str]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Attaches all the logging handlers to the given engine.

        Args:
            engine: The engine to attach the logging handlers to.
            name: The name of the engine (e.g., "Train", "Validate") to include in log messages.
            every: Frequency of iterations to log information. Logs are generated every 'every' iterations.
            output_transform: A function to select the value to log.
            state_attributes: A list of attributes to log.
            optimizer: The optimizer used during training to log current learning rates.
        """
        self.name = name
        self.output_transform = output_transform
        self.state_attributes = state_attributes
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
            if self.output_transform is not None:
                output = self.output_transform(output)
            outputs = utils._to_str_list(output)

        lrs = ""
        if optimizer is not None:
            if len(optimizer.param_groups) == 1:
                lrs += f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            else:
                for i, g in enumerate(optimizer.param_groups):
                    lrs += f"lr [g{i}]: {g['lr']:.5f}"

        state_attrs = []
        if self.state_attributes is not None:
            state_attrs = utils._to_str_list(
                {name: getattr(engine.state, name, None) for name in self.state_attributes}
            )
        msg = self.delimiter.join(
            [
                f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]",
                f"[{current_iter}/{engine.state.epoch_length}]:",
                f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}",
                f"{lrs}",
            ]
            + outputs
            + [" ".join(state_attrs)]
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

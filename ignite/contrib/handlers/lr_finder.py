# coding: utf-8
import contextlib
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ignite.contrib.handlers.param_scheduler import LRScheduler, PiecewiseLinear
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint


class FastaiLRFinder:
    """Learning rate finder handler for supervised trainers.

    While attached, the handler increases the learning rate in between two
    boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning
    rates and what can be an optimal learning rate.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers import FastaiLRFinder

        trainer = ...
        model = ...
        optimizer = ...

        lr_finder = FastaiLRFinder()
        to_save = {"model": model, "optimizer": optimizer}

        with lr_finder.attach(trainer, to_save=to_save) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(dataloader)

        # Get lr_finder results
        lr_finder.get_results()

        # Plot lr_finder results (requires matplotlib)
        lr_finder.plot()

        # get lr_finder suggestion for lr
        lr_finder.lr_suggestion()


    Note:
        When context manager is exited all LR finder's handlers are removed.

    Note:
        Please, also keep in mind that all other handlers attached the trainer will be executed during LR finder's run.

    Note:
        This class may require `matplotlib` package to be installed to plot learning rate range test:

        .. code-block:: bash

            pip install matplotlib


    References:

        Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186

        fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self) -> None:
        self._diverge_flag = False
        self._history = {}  # type: Dict[str, List[Any]]
        self._best_loss = None
        self._lr_schedule = None  # type: Optional[Union[LRScheduler, PiecewiseLinear]]
        self.logger = logging.getLogger(__name__)

    def _run(
        self,
        trainer: Engine,
        optimizer: Optimizer,
        output_transform: Callable,
        num_iter: int,
        end_lr: float,
        step_mode: str,
        smooth_f: float,
        diverge_th: float,
    ) -> None:

        self._history = {"lr": [], "loss": []}
        self._best_loss = None
        self._diverge_flag = False

        # attach LRScheduler to trainer.
        if num_iter is None:
            num_iter = trainer.state.epoch_length * trainer.state.max_epochs
        else:
            max_iter = trainer.state.epoch_length * trainer.state.max_epochs  # type: ignore[operator]
            if num_iter > max_iter:
                warnings.warn(
                    f"Desired num_iter {num_iter} is unreachable with the current run setup of {max_iter} iteration "
                    f"({trainer.state.max_epochs} epochs)",
                    UserWarning,
                )

        if not trainer.has_event_handler(self._reached_num_iterations):
            trainer.add_event_handler(Events.ITERATION_COMPLETED, self._reached_num_iterations, num_iter)

        # attach loss and lr logging
        if not trainer.has_event_handler(self._log_lr_and_loss):
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED, self._log_lr_and_loss, output_transform, smooth_f, diverge_th
            )

        self.logger.debug(f"Running LR finder for {num_iter} iterations")
        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            self._lr_schedule = LRScheduler(_ExponentialLR(optimizer, end_lr, num_iter))
        else:
            start_lr = optimizer.param_groups[0]["lr"]
            self._lr_schedule = PiecewiseLinear(
                optimizer, param_name="lr", milestones_values=[(0, start_lr), (num_iter, end_lr)]
            )
        if not trainer.has_event_handler(self._lr_schedule):
            trainer.add_event_handler(Events.ITERATION_COMPLETED, self._lr_schedule, num_iter)

    def _reset(self, trainer: Engine) -> None:
        self.logger.debug("Completed LR finder run")
        trainer.remove_event_handler(self._lr_schedule, Events.ITERATION_COMPLETED)  # type: ignore[arg-type]
        trainer.remove_event_handler(self._log_lr_and_loss, Events.ITERATION_COMPLETED)
        trainer.remove_event_handler(self._reached_num_iterations, Events.ITERATION_COMPLETED)

    def _log_lr_and_loss(self, trainer: Engine, output_transform: Callable, smooth_f: float, diverge_th: float) -> None:
        output = trainer.state.output
        loss = output_transform(output)
        lr = self._lr_schedule.get_param()  # type: ignore[union-attr]
        self._history["lr"].append(lr)
        if trainer.state.iteration == 1:
            self._best_loss = loss
        else:
            if smooth_f > 0:
                loss = smooth_f * loss + (1 - smooth_f) * self._history["loss"][-1]
            if loss < self._best_loss:
                self._best_loss = loss
        self._history["loss"].append(loss)

        # Check if the loss has diverged; if it has, stop the trainer
        if self._history["loss"][-1] > diverge_th * self._best_loss:  # type: ignore[operator]
            self._diverge_flag = True
            self.logger.info("Stopping early, the loss has diverged")
            trainer.terminate()

    def _reached_num_iterations(self, trainer: Engine, num_iter: int) -> None:
        if trainer.state.iteration > num_iter:
            trainer.terminate()

    def _warning(self, _: Any) -> None:
        if not self._diverge_flag:
            warnings.warn(
                "Run completed without loss diverging, increase end_lr, decrease diverge_th or look"
                " at lr_finder.plot()",
                UserWarning,
            )

    def _detach(self, trainer: Engine) -> None:
        """
        Detaches lr_finder from trainer.

        Args:
            trainer: the trainer to detach form.
        """

        if trainer.has_event_handler(self._run, Events.STARTED):
            trainer.remove_event_handler(self._run, Events.STARTED)
        if trainer.has_event_handler(self._warning, Events.COMPLETED):
            trainer.remove_event_handler(self._warning, Events.COMPLETED)
        if trainer.has_event_handler(self._reset, Events.COMPLETED):
            trainer.remove_event_handler(self._reset, Events.COMPLETED)

    def get_results(self) -> Dict[str, List[Any]]:
        """
        Returns: dictionary with loss and lr logs fromm the previous run
        """
        return self._history

    def plot(self, skip_start: int = 10, skip_end: int = 5, log_lr: bool = True) -> None:
        """Plots the learning rate range test.

        This method requires `matplotlib` package to be installed:

        .. code-block:: bash

            pip install matplotlib

        Args:
            skip_start: number of batches to trim from the start.
                Default: 10.
            skip_end: number of batches to trim from the start.
                Default: 5.
            log_lr: True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise RuntimeError(
                "This method requires matplotlib to be installed. "
                "Please install it with command: \n pip install matplotlib"
            )

        if not self._history:
            raise RuntimeError("learning rate finder didn't run yet so results can't be plotted")

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected

        lrs = self._history["lr"]
        losses = self._history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()

    def lr_suggestion(self) -> Any:
        """
        Returns: learning rate at the minimum numerical gradient
        """
        if not self._history:
            raise RuntimeError("learning rate finder didn't run yet so lr_suggestion can't be returned")
        loss = self._history["loss"]
        grads = torch.tensor([loss[i] - loss[i - 1] for i in range(1, len(loss))])
        min_grad_idx = grads.argmin() + 1
        return self._history["lr"][int(min_grad_idx)]

    @contextlib.contextmanager
    def attach(
        self,
        trainer: Engine,
        to_save: Mapping,
        output_transform: Callable = lambda output: output,
        num_iter: Optional[int] = None,
        end_lr: float = 10.0,
        step_mode: str = "exp",
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
    ) -> Any:
        """Attaches lr_finder to a given trainer. It also resets model and optimizer at the end of the run.

        Usage:

        .. code-block:: python

            to_save = {"model": model, "optimizer": optimizer}
            with lr_finder.attach(trainer, to_save=to_save) as trainer_with_lr_finder:
                trainer_with_lr_finder.run(dataloader)

        Args:
            trainer: lr_finder is attached to this trainer. Please, keep in mind that all attached handlers
                will be executed.
            to_save: dictionary with optimizer and other objects that needs to be restored after running
                the LR finder. For example, `to_save={'optimizer': optimizer, 'model': model}`. All objects should
                implement `state_dict` and `load_state_dict` methods.
            output_transform: function that transforms the trainer's `state.output` after each
                iteration. It must return the loss of that iteration.
            num_iter: number of iterations for lr schedule between base lr and end_lr. Default, it will
                run for `trainer.state.epoch_length * trainer.state.max_epochs`.
            end_lr: upper bound for lr search. Default, 10.0.
            step_mode: "exp" or "linear", which way should the lr be increased from optimizer's initial
                lr to `end_lr`. Default, "exp".
            smooth_f: loss smoothing factor in range `[0, 1)`. Default, 0.05
            diverge_th: Used for stopping the search when `current loss > diverge_th * best_loss`.
                Default, 5.0.

        Returns:
            trainer_with_lr_finder (trainer used for finding the lr)

        Note:
            lr_finder cannot be attached to more than one trainer at a time.
        """
        if not isinstance(to_save, Mapping):
            raise TypeError(f"Argument to_save should be a mapping, but given {type(to_save)}")

        Checkpoint._check_objects(to_save, "state_dict")
        Checkpoint._check_objects(to_save, "load_state_dict")

        if "optimizer" not in to_save:
            raise ValueError("Mapping to_save should contain 'optimizer' key")

        if not isinstance(to_save["optimizer"], torch.optim.Optimizer):
            raise TypeError(
                f"Object to_save['optimizer'] should be torch optimizer, but given {type(to_save['optimizer'])}"
            )

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")
        if diverge_th < 1:
            raise ValueError("diverge_th should be larger than 1")
        if step_mode not in ["exp", "linear"]:
            raise ValueError(f"step_mode should be 'exp' or 'linear', but given {step_mode}")
        if num_iter is not None:
            if not isinstance(num_iter, int):
                raise TypeError(f"if provided, num_iter should be an integer, but give {num_iter}")
            if num_iter <= 0:
                raise ValueError(f"if provided, num_iter should be positive, but give {num_iter}")

        # store to_save
        with tempfile.TemporaryDirectory() as tmpdirname:
            obj = {k: o.state_dict() for k, o in to_save.items()}
            # add trainer
            obj["trainer"] = trainer.state_dict()
            cache_filepath = Path(tmpdirname) / "ignite_lr_finder_cache.pt"
            torch.save(obj, cache_filepath.as_posix())

            optimizer = to_save["optimizer"]
            # Attach handlers
            if not trainer.has_event_handler(self._run):
                trainer.add_event_handler(
                    Events.STARTED,
                    self._run,
                    optimizer,
                    output_transform,
                    num_iter,
                    end_lr,
                    step_mode,
                    smooth_f,
                    diverge_th,
                )
            if not trainer.has_event_handler(self._warning):
                trainer.add_event_handler(Events.COMPLETED, self._warning)
            if not trainer.has_event_handler(self._reset):
                trainer.add_event_handler(Events.COMPLETED, self._reset)

            yield trainer
            self._detach(trainer)
            # restore to_save and reset trainer's state
            obj = torch.load(cache_filepath.as_posix())
            trainer.load_state_dict(obj["trainer"])
            for k, o in obj.items():
                if k in to_save:
                    to_save[k].load_state_dict(o)


class _ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Args:
        optimizer: wrapped optimizer.
        end_lr: the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter: the number of iterations over which the test
            occurs. Default: 100.
        last_epoch: the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        curr_iter = self.last_epoch + 1  # type: ignore[attr-defined]
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]  # type: ignore[attr-defined]

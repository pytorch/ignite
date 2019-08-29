# coding: utf-8

from ignite.engine import Events, Engine
from ignite.contrib.handlers import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import torch
import copy
import os
import logging
import weakref
import warnings


class FastaiLRFinder(object):
    """
    Learning rate finder handler for supervised trainers.

    While attached, the handler increases the learning rate in between two
    boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning
    rates and what is the optimal learning rate.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use, the defined
        optimizer learning rate is assumed to be the lower boundary of the range
        test.
        num_iter (int): number of iterations for lr schedule between base lr
            and end_lr. If `None` it will run for
            `len(dataloader) * trainer.state.max_epochs`
        output_transform (callable, optional): function that transform the
            engine's state.output after each iteration. It must return the loss
            of that iteration.
        end_lr (float): upper bound for lr search.
        step_mode (str): "exp" or "linear", which way should the lr be increased
            from optimizer's initial lr to end_lr
        smooth_f (float): loss smoothing factor in range [0, 1)
        diverge_th (float): Used for stopping the search when
            `current loss > diverge_th * best_loss`
        memory_cache (bool): if this flag is set to True, `state_dict` of model
            and optimizer will be cached in memory. Otherwise, they will be
            saved to files under the `cache_dir`.
        cache_dir (str): path for storing temporary files. If no path is
            specified, system-wide temporary directory is used. Notice that this
             parameter will be ignored if `memory_cache` is True.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers import LRFinder

            # Create a lr_finder
            lr_finder = LRFinder(model, optimizer)

            # Attach the lr_finder to the trainer
            lr_finder.attach(trainer)

            # Run trainer
            trainer.run(dataloader)

            # Detach lr_finder
            lr_finder.detach()

            # Get lr_finder results
            lr_finder.get_results()

            # Plot lr_finder results (requires matplotlib)
            lr_finder.plot()

            # get lr_finder suggestion for lr
            lr_finder.lr_suggestion()

        It is recommended to use the lr_finder.attach as context manager:

        .. code-block:: python

            from ignite.contrib.handlers import LRFinder

            lr_finder = LRFinder(model, optimizer)

            with lr_finder.attach(trainer):
                trainer.run(dataloader)

            lr_finder.plot()


    References:
        Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186

        fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self, model, optimizer, num_iter=None, output_transform=lambda output: output, end_lr=10,
                 step_mode="exp", smooth_f=0.05, diverge_th=5, memory_cache=True, cache_dir=None):

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")
        if diverge_th < 1:
            raise ValueError("diverge_th should be larger than 1")
        if step_mode not in ["exp", "linear"]:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))
        if isinstance(num_iter, int) and num_iter <= 0:
            raise ValueError("if provided, num_iter should be a poitive int, got {}".format(num_iter))

        self._model = model
        self._optimizer = optimizer
        self._output_transform = output_transform
        self.num_iter = num_iter
        self._end_lr = end_lr
        self._step_mode = step_mode
        self._smooth_f = smooth_f
        self._diverge_th = diverge_th
        self._memory_cache = memory_cache
        self._cache_dir = cache_dir

        self._engine = None
        self._diverge_flag = False
        self._history = None
        self._best_loss = None
        self._lr_schedule = None

        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(logging.NullHandler())

    def _run(self, engine):
        engine.state.state_cache = _StateCacher(self._memory_cache, cache_dir=self._cache_dir)
        engine.state.state_cache.store("model", self._model.state_dict())
        engine.state.state_cache.store("optimizer", self._optimizer.state_dict())

        self._history = {"lr": [], "loss": []}
        self._best_loss = None
        self._diverge_flag = False

        # attach loss and lr logging
        if not engine.has_event_handler(self._log_lr_and_loss):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_lr_and_loss)

        # attach LRScheduler to engine.
        if self.num_iter is None:
            num_iter = len(engine.state.dataloader) * engine.state.max_epochs
        else:
            dataloader_len = len(engine.state.dataloader)
            required_epochs = np.ceil(self.num_iter / dataloader_len)
            if engine.state.max_epochs < required_epochs:
                warnings.warn("to reach the desired num_iter {} with current dataloader length {}, you mudt run "
                              "trainer for {} epochs".format(self.num_iter, dataloader_len, required_epochs),
                              NotEnoughIterationWarning)
            num_iter = self.num_iter

        self._logger.debug("Running LR finder for {} iterations".format(num_iter))
        # Initialize the proper learning rate policy
        if self._step_mode.lower() == "exp":
            self._lr_schedule = LRScheduler(_ExponentialLR(self._optimizer, self._end_lr, num_iter))
        else:
            self._lr_schedule = LRScheduler(_LinearLR(self._optimizer, self._end_lr, num_iter))
        if not engine.has_event_handler(self._lr_schedule):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._lr_schedule, num_iter)

        if not engine.has_event_handler(self._reached_num_iterations):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._reached_num_iterations, num_iter)

    # Reset model and optimizer, delete state cache and remove handlers
    def _reset(self, engine):
        self._model.load_state_dict(engine.state.state_cache.retrieve("model"))
        self._optimizer.load_state_dict(engine.state.state_cache.retrieve("optimizer"))
        del engine.state.state_cache
        self._logger.debug("Completed LR finder run, resets model & optimizer")

        # Clean up; remove event handlers added during run
        engine.remove_event_handler(self._lr_schedule, Events.ITERATION_COMPLETED)
        engine.remove_event_handler(self._log_lr_and_loss, Events.ITERATION_COMPLETED)
        engine.remove_event_handler(self._reached_num_iterations, Events.ITERATION_COMPLETED)

    def _log_lr_and_loss(self, engine):
        output = engine.state.output
        loss = self._output_transform(output)
        lr = self._lr_schedule.lr_scheduler.get_lr()[0]
        self._history["lr"].append(lr)
        if engine.state.iteration == 1:
            self._best_loss = loss
        else:
            if self._smooth_f > 0:
                loss = self._smooth_f * loss + (1 - self._smooth_f) * self._history["loss"][-1]
            if loss < self._best_loss:
                self._best_loss = loss
        self._history["loss"].append(loss)

        # Check if the loss has diverged; if it has, stop the trainer
        if self._history["loss"][-1] > self._diverge_th * self._best_loss:
            engine.terminate()
            self._diverge_flag = True
            self._logger.info("Stopping early, the loss has diverged")

    def _reached_num_iterations(self, engine, num_iter):
        if engine.state.iteration >= num_iter:
            engine.terminate()

    def _warning(self, engine):
        if not self._diverge_flag:
            warnings.warn("Run completed without loss diverging, increase end_lr, decrease diverge_th or look"
                          " at lr_finder.plot()")

    def attach(self, engine: Engine):
        """
        Attaches lr_finder to engine.
        It is recommended to use `with lr_finder.attach(engine)` instead of
        explicitly detaching using `lr_finder.detach()`
        Args:
            engine: lr_finder is attached to this engine

        Notes:
            lr_finder cannot be attached to more than one engine at a time

        Returns:
            self
        """
        if self._engine:
            raise AlreadyAttachedError("This LRFinder is already attached. create a new one or use lr_finder.detach()")
        self._engine = weakref.ref(engine)
        if not engine.has_event_handler(self._run):
            engine.add_event_handler(Events.STARTED, self._run)
        if not engine.has_event_handler(self._warning):
            engine.add_event_handler(Events.COMPLETED, self._warning)
        if not engine.has_event_handler(self._reset):
            engine.add_event_handler(Events.COMPLETED, self._reset)

        return self

    def detach(self):
        """
        Detaches lr_finder from engine.
        """
        if self._engine:
            engine = self._engine()
            self._engine = None
            if engine.has_event_handler(self._run, Events.STARTED):
                engine.remove_event_handler(self._run, Events.STARTED)
            if engine.has_event_handler(self._warning, Events.COMPLETED):
                engine.remove_event_handler(self._warning, Events.COMPLETED)
            if engine.has_event_handler(self._reset, Events.COMPLETED):
                engine.remove_event_handler(self._reset, Events.COMPLETED)
        else:
            warnings.warn("This LRFinder isn't attached, this action has no effect")

    def get_results(self):
        """
        Returns: dictionary with loss and lr logs fromm the previous run
        """
        return self._history

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.

        Args:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        try:
            import matplotlib.pyplot as plt

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

        except ModuleNotFoundError:
            self._logger.warning("matplotlib not found, can't plot result")

    def lr_suggestion(self):
        """
        Returns: learning rate at the minimum numerical gradient
        """
        loss = self._history["loss"]
        grads = [loss[i] - loss[i - 1] for i in range(1, len(loss))]
        min_grad_idx = np.argmin(grads) + 1
        return self._history["lr"][int(min_grad_idx)]

    def __enter__(self):
        return self

    def __exit__(self, type, val, tb):
        self.detach()


class AlreadyAttachedError(Exception):
    """
    Exception class to raise if trying to attach an already attached LRFinder.
    """


class NotEnoughIterationWarning(Warning):
    """
    Warning class to raise when engine run has les iterations than what is
    specified in th elr_finder.
    """


class _LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class _ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class _StateCacher(object):

    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given `cache_dir` is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError('Failed to load state in {}. File does not exist anymore.'.format(fn))
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""
        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])

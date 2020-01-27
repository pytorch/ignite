# coding: utf-8

from ignite.engine import Events, Engine
from ignite.contrib.handlers import LRScheduler, PiecewiseLinear
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import contextlib
import torch
import copy
import os
import logging
import warnings


class FastaiLRFinder(object):
    """
    Learning rate finder handler for supervised trainers.

    While attached, the handler increases the learning rate in between two
    boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning
    rates and what can be an optimal learning rate.

    This class may require `matplotlib` package to be installed to plot learning rate range test:

    .. code-block:: bash

        pip install matplotlib

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers import LRFinder

            # Create a lr_finder
            lr_finder = LRFinder()

            # Attach the lr_finder to the trainer
            with lr_finder.attach(trainer, model, optimizer) as trainer_with_lr_finder:
                trainer_with_lr_finder.run(dataloader)

            # Get lr_finder results
            lr_finder.get_results()

            # Plot lr_finder results (requires matplotlib)
            lr_finder.plot()

            # get lr_finder suggestion for lr
            lr_finder.lr_suggestion()

    References:

        Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186

        fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self):

        self._diverge_flag = False
        self._history = None
        self._best_loss = None
        self._lr_schedule = None
        self._reset_params()

        self.logger = logging.getLogger(__name__)

    def _run(self, engine, num_iter, end_lr, step_mode, smooth_f, diverge_th):
        engine.state.state_cache = _StateCacher(self._memory_cache, cache_dir=self._cache_dir)
        engine.state.state_cache.store("model", self._model.state_dict())
        engine.state.state_cache.store("optimizer", self._optimizer.state_dict())

        self._history = {"lr": [], "loss": []}
        self._best_loss = None
        self._diverge_flag = False

        # attach LRScheduler to engine.
        if num_iter is None:
            num_iter = len(engine.state.dataloader) * engine.state.max_epochs
        else:
            dataloader_len = len(engine.state.dataloader)
            required_epochs = np.ceil(num_iter / dataloader_len)
            if engine.state.max_epochs < required_epochs:
                warnings.warn("to reach the desired num_iter {} with current dataloader length {}, you must run "
                              "trainer for {} epochs".format(num_iter, dataloader_len, required_epochs),
                              UserWarning)

        if not engine.has_event_handler(self._reached_num_iterations):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._reached_num_iterations, num_iter)

        # attach loss and lr logging
        if not engine.has_event_handler(self._log_lr_and_loss):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_lr_and_loss, smooth_f, diverge_th)

        self._logger.debug("Running LR finder for {} iterations".format(num_iter))
        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            self._lr_schedule = LRScheduler(_ExponentialLR(self._optimizer, end_lr, num_iter))
        else:
            start_lr = self._optimizer.param_groups[0]["lr"]
            self._lr_schedule = PiecewiseLinear(self._optimizer, "lr", [(0, start_lr), (num_iter, end_lr)])
        if not engine.has_event_handler(self._lr_schedule):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._lr_schedule, num_iter)

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

    def _reset_params(self):
        self._model = None
        self._optimizer = None
        self._output_transform = None
        self._num_iter = None
        self._end_lr = None
        self._step_mode = None
        self._smooth_f = None
        self._diverge_th = None

    def _log_lr_and_loss(self, engine, smooth_f, diverge_th):
        output = engine.state.output
        loss = self._output_transform(output)
        lr = self._lr_schedule.get_param()
        self._history["lr"].append(lr)
        if engine.state.iteration == 1:
            self._best_loss = loss
        else:
            if smooth_f > 0:
                loss = smooth_f * loss + (1 - smooth_f) * self._history["loss"][-1]
            if loss < self._best_loss:
                self._best_loss = loss
        self._history["loss"].append(loss)

        # Check if the loss has diverged; if it has, stop the trainer
        if self._history["loss"][-1] > diverge_th * self._best_loss:
            self._diverge_flag = True
            self._logger.info("Stopping early, the loss has diverged")
            engine.terminate()

    def _reached_num_iterations(self, engine, num_iter):
        if engine.state.iteration > num_iter:
            engine.terminate()

    def _warning(self, engine):
        if not self._diverge_flag:
            warnings.warn("Run completed without loss diverging, increase end_lr, decrease diverge_th or look"
                          " at lr_finder.plot()", UserWarning)

    def _setup(self, model, optimizer, output_transform=lambda output: output, num_iter=None, end_lr=10,
               step_mode="exp", smooth_f=0.05, diverge_th=5):

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")
        if diverge_th < 1:
            raise ValueError("diverge_th should be larger than 1")
        if step_mode not in ["exp", "linear"]:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))
        if num_iter is not None and (not isinstance(num_iter, int) or num_iter <= 0):
            raise ValueError("if provided, num_iter should be a positive int, got {}".format(num_iter))

        self._model = model
        self._optimizer = optimizer
        self._output_transform = output_transform
        self._num_iter = num_iter
        self._end_lr = end_lr
        self._step_mode = step_mode
        self._smooth_f = smooth_f
        self._diverge_th = diverge_th

        return self

    def _detach(self, engine):
        """
        Detaches lr_finder from engine.

        Args:
            engine: the engine to detach form.
        """

        if engine.has_event_handler(self._run, Events.STARTED):
            engine.remove_event_handler(self._run, Events.STARTED)
        if engine.has_event_handler(self._warning, Events.COMPLETED):
            engine.remove_event_handler(self._warning, Events.COMPLETED)
        if engine.has_event_handler(self._reset, Events.COMPLETED):
            engine.remove_event_handler(self._reset, Events.COMPLETED)

        self._reset_params()

    def get_results(self):
        """
        Returns: dictionary with loss and lr logs fromm the previous run
        """
        return self._history

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.

        This method requires `matplotlib` package to be installed:

        .. code-block:: bash

            pip install matplotlib

        Args:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise RuntimeError("This method requires matplotlib to be installed. "
                               "Please install it with command: \n pip install matplotlib")

        if self._history is None:
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

    def lr_suggestion(self):
        """
        Returns: learning rate at the minimum numerical gradient
        """
        if self._history is None:
            raise RuntimeError("learning rate finder didn't run yet so lr_suggestion can't be returned")
        loss = self._history["loss"]
        grads = [loss[i] - loss[i - 1] for i in range(1, len(loss))]
        min_grad_idx = np.argmin(grads) + 1
        return self._history["lr"][int(min_grad_idx)]

    @contextlib.contextmanager
    def attach(self, engine, model, optimizer, output_transform=lambda output: output, num_iter=None, end_lr=10.,
               step_mode="exp", smooth_f=0.05, diverge_th=5.):
        """
        Attaches lr_finder to a given trainer. It also resets model and
        optimizer at the end of the run.
        It is used with:
        `with lr_finder.attach(engine, model, optimizer) as
        trainer_with_lr_finder:
            trainer_with_lr_finder.run(dataloader)`
        Args:
            engine (Engine): lr_finder is attached to this engine
            model (torch.nn.Module): the model to train.
            optimizer (torch.optim.Optimizer): the optimizer to use, the defined optimizer learning rate is assumed to
                be the lower boundary of the range test.
            output_transform (callable, optional): function that transforms the engine's `state.output` after each
                iteration. It must return the loss of that iteration.
            num_iter (int, optional): number of iterations for lr schedule between base lr and end_lr. Default, it will
                run for `len(dataloader) * trainer.state.max_epochs`.
            end_lr (float, optional): upper bound for lr search. Default, 10.0.
            step_mode (str, optional): "exp" or "linear", which way should the lr be increased from optimizer's initial lr
                to `end_lr`. Default, "exp".
            smooth_f (float, optional): loss smoothing factor in range `[0, 1)`. Default, 0.05
            diverge_th (float, optional): Used for stopping the search when `current loss > diverge_th * best_loss`.
                Default, 5.0.

        Notes:
            lr_finder cannot be attached to more than one engine at a time

        Returns:
            trainer_with_lr_finder: trainer used for finding the lr
        """
        # create new engine:
        copy_engine = Engine(engine._process_function)

        self._setup(model, optimizer, output_transform, num_iter, end_lr, step_mode, smooth_f, diverge_th)

        # Attach handlers
        if not engine.has_event_handler(self._run):
            engine.add_event_handler(Events.STARTED, self._run, self._num_iter, self._end_lr, self._step_mode,
                                     self._smooth_f, self._diverge_th)
        if not engine.has_event_handler(self._warning):
            engine.add_event_handler(Events.COMPLETED, self._warning)
        if not engine.has_event_handler(self._reset):
            engine.add_event_handler(Events.COMPLETED, self._reset)

        yield engine
        self._detach(engine)


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


# class _StateCacher(object):
#
#     def __init__(self, in_memory, cache_dir=None):
#         self.in_memory = in_memory
#         self.cache_dir = cache_dir
#         self.cached = {}
#
#         if self.cache_dir is None:
#             import tempfile
#             self.cache_dir = tempfile.gettempdir()
#         else:
#             if not os.path.isdir(self.cache_dir):
#                 raise ValueError('Given `cache_dir` is not a valid directory.')
#
#     def store(self, key, state_dict):
#         if self.in_memory:
#             self.cached.update({key: copy.deepcopy(state_dict)})
#         else:
#             fn = os.path.join(self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
#             self.cached.update({key: fn})
#             torch.save(state_dict, fn)
#
#     def retrieve(self, key):
#         if key not in self.cached:
#             raise KeyError('Target {} was not cached.'.format(key))
#
#         if self.in_memory:
#             return self.cached.get(key)
#         else:
#             fn = self.cached.get(key)
#             if not os.path.exists(fn):
#                 raise RuntimeError('Failed to load state in {}. File does not exist anymore.'.format(fn))
#             state_dict = torch.load(fn, map_location=lambda storage, location: storage)
#             return state_dict
#
#     def __del__(self):
#         """Check whether there are unused cached files existing in `cache_dir` before
#         this instance being destroyed."""
#         if self.in_memory:
#             return
#
#         for k in self.cached:
#             if os.path.exists(self.cached[k]):
#                 os.remove(self.cached[k])

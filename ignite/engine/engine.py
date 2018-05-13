import inspect
import logging
import sys
import time
from collections import defaultdict
from enum import Enum

import torch
from torch.utils.data import DataLoader

from ignite._utils import _to_hours_mins_secs, RewindableBatchSampler


IS_PYTHON2 = sys.version_info[0] < 3


class Events(Enum):
    """Events that are fired by the :class:`ignite.engine.Engine` during execution"""
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State(object):
    """An object that is used to pass internal and user-defined state between event handlers"""
    def __init__(self, **kwargs):
        self.iteration = 0
        self.output = None
        self.batch = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = "State:\n"
        for attr, value in self.__dict__.items():
            if attr in ("batch", "output"):
                value = type(value)
            s += "\t{}: {}\n".format(attr, value)
        return s


class Engine(object):
    """Runs a given process_function over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (Callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state

    Example usage:

    .. code-block:: python

        def train_and_store_loss(engine, batch):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        engine = Engine(train_and_store_loss)
        engine.run(data_loader)

        # Loss value is now stored in `engine.state.output`.

    """
    def __init__(self, process_function):
        self._event_handlers = defaultdict(list)
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self._process_function = process_function
        self.should_terminate = False
        self.state = None
        self.dataloader = None

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run")

        self._check_signature(process_function, 'process_function', None)

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired

        Args:
            event_name (Events): event from ignite.engine.Events to attach the handler to
            handler (Callable): the callable event handler that should be invoked
            *args: optional args to be passed to `handler`
            **kwargs: optional keyword args to be passed to `handler`

        Notes:
              The handler function's first argument will be `self`, the `Engine` object it was bound to.

              Note that other arguments can be passed to the handler in addition to the `*args` and `**kwargs`
              passed here, for example during `Events.EXCEPTION_RAISED`.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print("Epoch: {}".format(engine.state.epoch))

            engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

        """
        if event_name not in Events.__members__.values():
            self._logger.error("attempt to add event handler to an invalid event %s ", event_name)
            raise ValueError("Event {} is not a valid event for this Engine".format(event_name))

        event_args = (Exception(), ) if event_name == Events.EXCEPTION_RAISED else ()
        self._check_signature(handler, 'handler', *(event_args + args), **kwargs)

        self._event_handlers[event_name].append((handler, args, kwargs))
        self._logger.debug("added handler for event %s ", event_name)

    def _check_signature(self, fn, fn_description, *args, **kwargs):
        exception_msg = None

        if IS_PYTHON2:
            try:
                callable_ = fn if hasattr(fn, '__name__') else fn.__call__
                inspect.getcallargs(callable_, self, *args, **kwargs)
            except TypeError as exc:
                spec = inspect.getargspec(callable_)
                fn_params = list(spec.args)
                exception_msg = str(exc)
        else:
            signature = inspect.signature(fn)
            try:
                signature.bind(self, *args, **kwargs)
            except TypeError as exc:
                fn_params = list(signature.parameters)
                exception_msg = str(exc)

        if exception_msg:
            passed_params = [self] + list(args) + list(kwargs)
            raise ValueError("Error adding {} '{}': "
                             "takes parameters {} but will be called with {} "
                             "({})".format(
                                 fn, fn_description, fn_params, passed_params, exception_msg))

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler

        Args:
            event_name (Events): event to attach the handler to
            *args: optional args to be passed to `handler`
            **kwargs: optional keyword args to be passed to `handler`

        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _fire_event(self, event_name, *event_args):
        if event_name in self._event_handlers.keys():
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                func(self, *(event_args + args), **kwargs)

    def terminate(self):
        """Sends terminate signal to the engine, so that it terminates after the current iteration
        """
        self._logger.info("Terminate signaled. Engine will stop after current iteration is finished")
        self.should_terminate = True

    def _run_once_on_dataset(self):
        start_time = time.time()

        try:
            for batch in self.dataloader:
                self.state.batch = batch
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                if self.should_terminate:
                    break

        except BaseException as e:
            self._logger.error("Current run is terminating due to exception: %s", str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)

        return hours, mins, secs

    def _handle_exception(self, e):
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    def run(self, data, max_epochs=1, seed=None):
        """Runs the process_function over the passed data.

        Args:
            data (Iterable with length): Collection of batches allowing repeated iteration (e.g., list or DataLoader)
            max_epochs (int, optional): max epochs to run for (default: 1)
            seed (int, optional): random state seed for a reproducible run

        Returns:
            State: output state
        """
        self._check_input_data(data)

        seed = torch.LongTensor(1).random_()[0] if seed is None else seed
        self._logger.info("Engine run starting with max_epochs={}".format(max_epochs))
        self.state = State(epoch=0, max_epochs=max_epochs, metrics={}, seed=seed)
        self._run_with_resume(data)
        return self.state

    def resume(self, data, checkpoint_dirname, to_load):
        """Resume engine run from a checkpoint

        Args:
            data (Iterable with length): Collection of batches allowing repeated iteration (e.g., list or DataLoader)
            checkpoint_dirname (str): path to checkpoint directory with checkpoint.pth.tar file
            to_load (dict): dictionary same as `to_save` used with `EngineCheckpoint`, contains string keys and object
                values.
        """
        self._check_input_data(data)
        assert isinstance(to_load, dict), "Argument `to_load` should be a dictionary"

        from ignite.handlers.checkpoint import EngineCheckpoint

        EngineCheckpoint.check_objects(to_load)
        checkpoint = EngineCheckpoint.load(checkpoint_dirname)
        EngineCheckpoint.load_objects(to_load, checkpoint)

        self.load_state_dict(checkpoint['engine'])
        self._logger.info("Engine run resuming from epoch {} and iteration {} with max_epochs={}"
                          .format(self.state.epoch, self.state.iteration, self.state.max_epochs))

        self.state.metrics = {}
        self._run_with_resume(data)
        return self.state

    def state_dict(self):
        """Returns a dictionary containing a whole state of the module.

        Returns:
            dict:
                a dictionary containing a whole state of the module
        """
        state = self.state if self.state is not None else State(epoch=0, max_epochs=0, seed=None)
        return {
            'epoch': state.epoch,
            'max_epochs': state.max_epochs,
            'seed': state.seed,
            'iteration': state.iteration
        }

    def load_state_dict(self, state_dict):
        """Copies Engine parameters from :attr:`state_dict` into this Engine.

        Arguments:
            state_dict (dict): a dict containing Engine parameters.

        """
        assert isinstance(state_dict, dict), "Argument state_dict should be a dictionary"
        if self.state is None:
            self.state = State(**state_dict)
        else:
            for k in state_dict:
                setattr(self.state, k, state_dict[k])

    def _check_input_data(self, data):
        if not hasattr(data, "__len__"):
            raise TypeError("Data should have built-in __len__ method")
        assert len(data) > 0, "Data length should be positive"

    def _run_with_resume(self, data):
        try:
            # We need to copy torch DataLoader to replace batch sampler by a rewindable, reproducible batch sampler
            self.dataloader = self._copy_dataloader(data) if isinstance(data, DataLoader) else data
            start_time = time.time()
            self._fire_event(Events.STARTED)

            start_batch_index = self.state.iteration % len(self.dataloader)
            if start_batch_index > 0:
                self._resume_from_batch_index(start_batch_index)

            # Continue normally
            if not self.should_terminate:
                self._run_epochs()

            self._fire_event(Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info("Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._logger.error("Engine run is terminating due to exception: %s", str(e))
            self._handle_exception(e)

    def _resume_from_batch_index(self, start_batch_index):
        self._manual_seed(self.state.seed, self.state.epoch)

        if hasattr(self.dataloader, 'original_batch_sampler') and isinstance(self.dataloader, DataLoader):
            # Change batch sampler to a rewindable and reproducible batch sampler
            self.dataloader.batch_sampler = RewindableBatchSampler(self.dataloader.original_batch_sampler,
                                                                   start_batch_index=start_batch_index)
        else:
            # We need to advance self.dataloader until start_batch_index
            pass

        hours, mins, secs = self._run_once_on_dataset()
        self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)
        self._fire_event(Events.EPOCH_COMPLETED)

    def _run_epochs(self):

        while self.state.epoch < self.state.max_epochs and not self.should_terminate:
            self.state.epoch += 1
            self._fire_event(Events.EPOCH_STARTED)
            self._manual_seed(self.state.seed, self.state.epoch)

            if hasattr(self.dataloader, 'original_batch_sampler') and isinstance(self.dataloader, DataLoader):
                # Change batch sampler to a rewindable and reproducible batch sampler
                self.dataloader.batch_sampler = RewindableBatchSampler(self.dataloader.original_batch_sampler,
                                                                       start_batch_index=0)
            hours, mins, secs = self._run_once_on_dataset()
            self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)
            if self.should_terminate:
                break
            self._fire_event(Events.EPOCH_COMPLETED)

    def _manual_seed(self, seed, epoch):
        torch.manual_seed(seed + epoch)

    def _copy_dataloader(self, loader):
        # Copy data if a dataloader and and new attribute original_batch_sampler
        output = DataLoader(loader.dataset, batch_sampler=loader.batch_sampler,
                            num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                            pin_memory=loader.pin_memory, timeout=loader.timeout,
                            worker_init_fn=loader.worker_init_fn)
        output.original_batch_sampler = loader.batch_sampler
        return output

import inspect
import logging
import sys
import time
from collections import defaultdict
from enum import Enum

from ignite._utils import _to_hours_mins_secs

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
        self.should_terminate_single_epoch = False
        self.state = None

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
        """Sends terminate signal to the engine, so that it terminates completely the run after the current iteration
        """
        self._logger.info("Terminate signaled. Engine will stop after current iteration is finished")
        self.should_terminate = True

    def terminate_epoch(self):
        """Sends terminate signal to the engine, so that it terminates the current epoch after the current iteration
        """
        self._logger.info("Terminate current epoch is signaled. "
                          "Current epoch iteration will stop after current iteration is finished")
        self.should_terminate_single_epoch = True

    def _run_once_on_dataset(self):
        start_time = time.time()

        try:
            for batch in self.state.dataloader:
                self.state.batch = batch
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                if self.should_terminate or self.should_terminate_single_epoch:
                    self.should_terminate_single_epoch = False
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

    def run(self, data, max_epochs=1):
        """Runs the process_function over the passed data.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or DataLoader)
            max_epochs (int, optional): max epochs to run for (default: 1)

        Returns:
            State: output state
        """

        self.state = State(dataloader=data, epoch=0, max_epochs=max_epochs, metrics={})

        try:
            self._logger.info("Engine run starting with max_epochs={}".format(max_epochs))
            start_time = time.time()
            self._fire_event(Events.STARTED)
            while self.state.epoch < max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(Events.EPOCH_STARTED)
                hours, mins, secs = self._run_once_on_dataset()
                self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED)

            self._fire_event(Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info("Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._logger.error("Engine run is terminating due to exception: %s", str(e))
            self._handle_exception(e)

        return self.state

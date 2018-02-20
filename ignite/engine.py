import logging
import time
from abc import ABCMeta, abstractmethod
from enum import Enum

from ignite._utils import _to_hours_mins_secs


class Events(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State(object):
    def __init__(self, **kwargs):
        self.iteration = 0
        self.terminate = False
        self.output = None
        self.batch = None
        for k, v in kwargs.items():
            setattr(self, k, v)


class Engine(object):
    __metaclass__ = ABCMeta

    """
    Abstract Engine class that is the super class of the Trainer and Evaluator engines.

    Parameters
    ----------
    process_function : callable
        A function receiving the current training batch in each iteration, outputing data to be stored in the state

    """
    def __init__(self, process_function):
        self._event_handlers = {}
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self._process_function = process_function

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run")

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """
        Add an event handler to be executed when the specified event is fired

        Parameters
        ----------
        event_name: Events
            event from ignite.Engine.Events to attach the
            handler to
        handler: Callable
            the callable event handler that should be invoked
        args:
            optional args to be passed to `handler`
        kwargs:
            optional keyword args to be passed to `handler`

        Returns
        -------
        None
        """
        if event_name not in Events.__members__.values():
            self._logger.error("attempt to add event handler to an invalid event %s ", event_name)
            raise ValueError("Event {} is not a valid event for this Engine".format(event_name))

        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []

        self._event_handlers[event_name].append((handler, args, kwargs))
        self._logger.debug("added handler for event % ", event_name)

    def on(self, event_name, *args, **kwargs):
        """
        Decorator shortcut for add_event_handler

        Parameters
        ----------
        event_name: enum
            event to attach the handler to
        args:
            optional args to be passed to `handler`
        kwargs:
            optional keyword args to be passed to `handler`

        Returns
        -------
        None
        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _fire_event(self, event_name, state, *event_args):
        if event_name in self._event_handlers.keys():
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                func(state, *(event_args + args), **kwargs)

    def _run_once_on_dataset(self, state):
        try:
            start_time = time.time()
            for batch in state.dataloader:
                state.batch = batch
                state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED, state)
                state.output = self._process_function(batch)
                self._fire_event(Events.ITERATION_COMPLETED, state)
                if state.terminate:
                    break

            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            return hours, mins, secs
        except BaseException as e:
            self._logger.error("Current run is terminating due to exception: %s", str(e))
            self._handle_exception(state, e)

    def _handle_exception(self, state, e):
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, state, e)
        else:
            raise e

    @abstractmethod
    def run(self, data, **kwargs):
        """
        Train the model, evaluate the validation set and update best parameters if the validation loss
        improves.
        In the event that the validation set is not run (or doesn't exist), the training loss is used
        to update the best parameters.

        Parameters
        ----------
        data : Iterable
            Collection of batches allowing for the engine to iterate over(e.g., list or DataLoader)
        **kwargs: optional
            Any additional kwargs

        Returns
        -------
        None
        """
        raise NotImplementedError("This method should be implemented by a subclass")

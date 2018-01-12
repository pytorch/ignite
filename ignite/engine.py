import logging
from abc import ABCMeta, abstractmethod


class Engine(object):
    __metaclass__ = ABCMeta

    """
    Abstract Engine class that is the super class of the Trainer and Evaluator engines.

    Parameters
    ----------
    training_update_function : callable
        Update function receiving the current training batch in each iteration

    validation_inference_function : callable
        Function receiving data and performing a feed forward without update
    """
    def __init__(self, valid_events):
        """
        Add an event handler to be executed when the specified event is fired

        Parameters
        ----------
        valid_events: enum
            Events that this engine allows handlers to be registered against
        """
        self._valid_events = valid_events
        self._event_handlers = {}
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self.should_terminate = False

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """
        Add an event handler to be executed when the specified event is fired

        Parameters
        ----------
        event_name: enum
            event from ignite.trainer.TrainingEvents to attach the
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
        if event_name not in self._valid_events.__members__.values():
            self._logger.error("attempt to add event handler to non-existent event %s ",
                               event_name)
            raise ValueError("Event {} not a valid training event".format(event_name))

        if event_name not in self._event_handlers.keys():
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

    def _fire_event(self, event_name):
        if event_name in self._event_handlers.keys():
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                func(self, *args, **kwargs)

    def terminate(self):
        """
        Sends terminate signal to the engine, so that it terminates after the current iteration
        """
        self._logger.info("Terminate signaled. Engine will stop after current iteration is finished")
        self.should_terminate = True

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

from __future__ import division

import logging
import time
from collections import Iterable

from enum import Enum


class TrainingEvents(Enum):
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    TRAINING_EPOCH_STARTED = "training_epoch_started"
    TRAINING_EPOCH_COMPLETED = "training_epoch_completed"
    VALIDATION_STARTING = "validation_starting"
    VALIDATION_COMPLETED = "validation_completed"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_ITERATION_STARTED = "training_iteration_started"
    TRAINING_ITERATION_COMPLETED = "training_iteration_completed"
    VALIDATION_ITERATION_STARTED = "validation_iteration_started"
    VALIDATION_ITERATION_COMPLETED = "validation_iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


class Trainer(object):
    """
    Generic trainer class.

    Training update and validation functions receive batches of data and return values which will
    be stored in the `training_history` and `validation_history`. The trainer defines multiple
    events in `TrainingEvents` for which the user can attach event handlers to. The events get
    passed the trainer, so they can access the training/validation history


    Parameters
    ----------
    training_data : Iterable
        Collection of training batches allowing repeated iteration (e.g., list or DataLoader)

    training_update_function : callable
        Update function receiving the current training batch in each iteration

    validation_data : Iterable
        Collection of validation batches allowing repeated iteration

    validation_inference_function : callable
        Function receiving data and performing a feed forward without update
    """

    def __init__(
        self,
        training_data,
        training_update_function,
        validation_data=None,
        validation_inference_function=None):

        self._logger = self._get_logger()

        self._training_data = training_data
        self._validation_data = validation_data
        self._training_update_function = training_update_function
        self._validation_inference_function = validation_inference_function
        self._event_handlers = {}

        self.training_history = []
        self.validation_history = []
        self.current_iteration = 0
        self.current_validation_iteration = 0
        self.current_epoch = 0
        self.should_terminate = False

    def _get_logger(self):
        logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        logger.addHandler(logging.NullHandler())
        return logger

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
        if event_name not in TrainingEvents.__members__.values():
            self._logger.error("attempt to add event handler to non-existent event %s ",
                               event_name)
            raise ValueError("Event {} not a valid training event".format(event_name))

        if event_name not in self._event_handlers.keys():
            self._event_handlers[event_name] = []

        self._event_handlers[event_name].append((handler, args, kwargs))
        self._logger.debug("added handler for event % ", event_name)

    def _fire_event(self, event_name):
        if event_name in self._event_handlers.keys():
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                func(self, *args, **kwargs)

    def _train_one_epoch(self):
        self._fire_event(TrainingEvents.TRAINING_EPOCH_STARTED)
        start_time = time.time()

        self.epoch_losses = []
        for _, batch in enumerate(self._training_data, 1):
            self._fire_event(TrainingEvents.TRAINING_ITERATION_STARTED)

            training_step_result = self._training_update_function(batch)
            if training_step_result is not None:
                self.training_history.append(training_step_result)

            self.current_iteration += 1

            self._fire_event(TrainingEvents.TRAINING_ITERATION_COMPLETED)
            if self.should_terminate:
                return

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self._logger.info("Epoch[%s] Complete. Time taken: %d:%d:%d", self.current_epoch, hours,
                          mins, secs)

        self._fire_event(TrainingEvents.TRAINING_EPOCH_COMPLETED)

    def validate(self):
        """ Evaluates the validation set"""
        if self._validation_data is None:
            return

        self.current_validation_iteration = 0
        self._fire_event(TrainingEvents.VALIDATION_STARTING)
        start_time = time.time()

        for _, batch in enumerate(self._validation_data, 1):
            self._fire_event(TrainingEvents.VALIDATION_ITERATION_STARTED)
            validation_step_result = self._validation_inference_function(batch)
            if validation_step_result is not None:
                self.validation_history.append(validation_step_result)

            self.current_validation_iteration += 1
            self._fire_event(TrainingEvents.VALIDATION_ITERATION_COMPLETED)
            if self.should_terminate:
                break

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self._logger.info("Validation Complete. Time taken: %d:%d:%d", hours, mins, secs)

        self._fire_event(TrainingEvents.VALIDATION_COMPLETED)

    def terminate(self):
        """
        Sends terminate signal to trainer, so that training terminates after the current iteration
        """
        self._logger.info("Terminate signaled to trainer. " +
                          "Training will stop after current iteration is finished")
        self.should_terminate = True

    def run(self, max_epochs=1, validate_every_epoch=True):
        """
        Train the model, evaluate the validation set and update best parameters if the validation loss
        improves.
        In the event that the validation set is not run (or doesn't exist), the training loss is used
        to update the best parameters.

        Parameters
        ----------
        max_epochs: int, optional
            max epochs to train for [default=1]
        validate_every_epoch: bool, optional
            evaluate the validation set at the end of every epoch [default=True]

        Returns
        -------
        None
        """
        try:
            self._logger.info("Training starting with params max_epochs={} "
                              "validate_every_epoch={}".format(max_epochs, validate_every_epoch))

            start_time = time.time()

            self._fire_event(TrainingEvents.TRAINING_STARTED)
            while self.current_epoch < max_epochs and not self.should_terminate:
                self._fire_event(TrainingEvents.EPOCH_STARTED)
                self._train_one_epoch()
                if self.should_terminate:
                    break
                if validate_every_epoch:
                    self.validate()
                    if self.should_terminate:
                        break

                self._fire_event(TrainingEvents.EPOCH_COMPLETED)
                self.current_epoch += 1

            self._fire_event(TrainingEvents.TRAINING_COMPLETED)
            time_taken = time.time() - start_time
            mins, secs = divmod(time_taken, 60)
            hours, mins = divmod(mins, 60)
            self._logger.info("Training complete. Time taken %d:%d:%d " % (hours, mins, secs))
        except BaseException as e:
            self._logger.error("Training is terminating due to exception: %s", str(e))
            self._fire_event(TrainingEvents.EXCEPTION_RAISED)
            raise e

from __future__ import division

import logging
import time
from collections import Iterable

from enum import Enum
from torch.autograd import Variable
from ignite.trainer.history import History

__all__ = ["TrainingEvents", "Trainer", "create_supervised"]


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
    training_update_function : callable
        Update function receiving the current training batch in each iteration

    validation_inference_function : callable
        Function receiving data and performing a feed forward without update
    """

    def __init__(self, training_update_function, validation_inference_function=None):

        self._logger = self._get_logger()
        self._training_update_function = training_update_function
        self._validation_inference_function = validation_inference_function
        self._event_handlers = {}

        self.training_history = History()
        self.validation_history = History()
        self.current_iteration = 0
        self.current_validation_iteration = 0
        self.current_epoch = 0
        self.max_epochs = 0
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

    def _train_one_epoch(self, training_data):
        self._fire_event(TrainingEvents.TRAINING_EPOCH_STARTED)
        start_time = time.time()

        self.epoch_losses = []
        for _, batch in enumerate(training_data, 1):
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
        self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.current_epoch, hours,
                          mins, secs)

        self._fire_event(TrainingEvents.TRAINING_EPOCH_COMPLETED)

    def validate(self, validation_data):
        """ Evaluates the validation set"""
        if self._validation_inference_function is None:
            raise ValueError("Trainer must have a validation_inference_function in order to validate")

        self.current_validation_iteration = 0
        self._fire_event(TrainingEvents.VALIDATION_STARTING)
        start_time = time.time()

        for _, batch in enumerate(validation_data, 1):
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
        self._logger.info("Validation Complete. Time taken: %02d:%02d:%02d", hours, mins, secs)

        self._fire_event(TrainingEvents.VALIDATION_COMPLETED)

    def terminate(self):
        """
        Sends terminate signal to trainer, so that training terminates after the current iteration
        """
        self._logger.info("Terminate signaled to trainer. " +
                          "Training will stop after current iteration is finished")
        self.should_terminate = True

    def run(self, training_data, max_epochs=1):
        """
        Train the model, evaluate the validation set and update best parameters if the validation loss
        improves.
        In the event that the validation set is not run (or doesn't exist), the training loss is used
        to update the best parameters.

        Parameters
        ----------
        training_data : Iterable
            Collection of training batches allowing repeated iteration (e.g., list or DataLoader)
        max_epochs: int, optional
            max epochs to train for [default=1]

        Returns
        -------
        None
        """

        try:
            self._logger.info("Training starting with max_epochs={}".format(max_epochs))

            self.max_epochs = max_epochs

            start_time = time.time()

            self._fire_event(TrainingEvents.TRAINING_STARTED)
            while self.current_epoch < max_epochs and not self.should_terminate:
                self._fire_event(TrainingEvents.EPOCH_STARTED)
                self._train_one_epoch(training_data)
                if self.should_terminate:
                    break

                self._fire_event(TrainingEvents.EPOCH_COMPLETED)
                self.current_epoch += 1

            self._fire_event(TrainingEvents.TRAINING_COMPLETED)
            time_taken = time.time() - start_time
            mins, secs = divmod(time_taken, 60)
            hours, mins = divmod(mins, 60)
            self._logger.info("Training complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))
        except BaseException as e:
            self._logger.error("Training is terminating due to exception: %s", str(e))
            self._fire_event(TrainingEvents.EXCEPTION_RAISED)
            raise e


def create_supervised(model, optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Trainer: a trainer instance with supervised update and inference functions
    """
    def _prepare_batch(batch):
        x, y = batch
        if cuda:
            x, y = x.cuda(), y.cuda()
        return Variable(x), Variable(y)

    def _update(batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data.cpu()[0]

    def _inference(batch):
        model.eval()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        return y_pred.data.cpu(), y.data.cpu()

    return Trainer(_update, _inference)

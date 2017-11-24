"""
Module for trainer classes to manage training
"""

from __future__ import division

from collections import Iterable
import logging
import time

from enum import Enum
import numpy as np


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
  BEST_LOSS_UPDATED = "best_loss_updated"
  EXCEPTION_RAISED = "exception_raised"


def _to_hours_mins_secs(time_taken):
  mins, secs = divmod(time_taken, 60)
  hours, mins = divmod(mins, 60)
  return hours, mins, secs


class Trainer(object):
  """
  Generic trainer class.

  Training update and validation functions receive batches of data and return loss values.
  Each function may return several losses (e.g., cross-entropy and classification accuracy).
  This class will keep track of the best parameter setting achieved during training, but will only
  consider the first loss if multiple losses are returned by training or validation functions.

  Parameters
  ----------
  training_data : Iterable
      Collection of training batches allowing repeated iteration (e.g., list or DataLoaderManager)

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
    self._best_model_parameter_loss = np.inf
    self._training_update_function = training_update_function
    self._validation_inference_function = validation_inference_function
    self._event_listeners = {}

    self.training_losses = []
    self.avg_training_loss_per_epoch = []
    self.best_training_loss = [np.inf]
    self.avg_validation_loss = []
    self.best_validation_loss = [np.inf]
    self.current_iteration = 0
    self.current_validation_iteration = 0
    self.current_epoch = 0
    self.should_terminate = False

  def _get_logger(self):
    logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
    logger.addHandler(logging.NullHandler())
    return logger

  def add_event_listener(self, event_name, func, *args, **kwargs):
    """
    Add an event listener to be executed when the specified event is fired

    Parameters
    ----------
    event_name: enum
        event from twitter.magicpony.common.training.trainer.TrainingEvents to attach the
        listener to
    func: Callable
        the callable event listener that should be invoked
    args:
        optional args to be passed to func
    kwargs:
        optional keyword args to be passed to func

    Returns
    -------
    None
    """
    if event_name not in TrainingEvents.__members__.values():
      self._logger.error("attempt to add event listener to non-existent event %s ", event_name)
      raise ValueError("Event {} not a valid training event".format(event_name))

    if event_name not in self._event_listeners.keys():
      self._event_listeners[event_name] = []

    self._event_listeners[event_name].append((func, args, kwargs))
    self._logger.debug("added handler for event % ", event_name)

  def _fire_event(self, event_name):
    if event_name in self._event_listeners.keys():
      self._logger.debug("firing handlers for event %s ", event_name)
      for func, args, kwargs in self._event_listeners[event_name]:
        func(self, *args, **kwargs)

  def _train_one_epoch(self):
    self._fire_event(TrainingEvents.TRAINING_EPOCH_STARTED)
    start_time = time.time()

    self.epoch_losses = []
    for _, batch in enumerate(self._training_data,
                              1):  # enumerate used to support PyTorch DataLoader
      self._fire_event(TrainingEvents.TRAINING_ITERATION_STARTED)
      loss = self._training_update_function(batch)

      if not isinstance(loss, Iterable):
        loss = [loss]

      if np.any(np.equal(loss, None)):
        raise ValueError("The loss contains None values.")

      self.epoch_losses.append(loss)
      self.training_losses.append(loss)
      self.current_iteration += 1

      self._fire_event(TrainingEvents.TRAINING_ITERATION_COMPLETED)
      if self.should_terminate:
        return

    if len(self.epoch_losses) == 0:
      raise ValueError("There are no iteration losses. \
            Likely this was caused by an empty training data iterable.")
    else:
      avg_loss = np.mean(self.epoch_losses, 0)

    if not isinstance(avg_loss, Iterable):
      avg_loss = [avg_loss]

    self.avg_training_loss_per_epoch.append(avg_loss)
    if avg_loss[0] < self.best_training_loss[0]:
      self.best_training_loss = avg_loss

    time_taken = time.time() - start_time
    hours, mins, secs = _to_hours_mins_secs(time_taken)
    self._logger.info("Epoch[%s]: Avg. Loss: %s \t Best Loss: %s \t Time Taken: %d:%d:%d",
                      self.current_epoch,
                      avg_loss, self.best_training_loss, hours, mins, secs)

    self._fire_event(TrainingEvents.TRAINING_EPOCH_COMPLETED)

    # don't serialize epoch-wise losses
    del self.epoch_losses

    return avg_loss

  def validate(self):
    """ Evaluates the validation set and updates the best model loss """
    if self._validation_data is None:
      self._update_best_model_loss()
      return

    self.current_validation_iteration = 0
    self._fire_event(TrainingEvents.VALIDATION_STARTING)
    start_time = time.time()
    losses = []
    for _, batch in enumerate(self._validation_data,
                              1):  # enumerate used to support PyTorch DataLoader
      self._fire_event(TrainingEvents.VALIDATION_ITERATION_STARTED)
      loss = self._validation_inference_function(batch)

      if not isinstance(loss, Iterable):
        loss = [loss]

      if np.any(np.equal(loss, None)):
        raise ValueError("The loss contains None values.")

      losses.append(loss)
      self.current_validation_iteration += 1
      self._fire_event(TrainingEvents.VALIDATION_ITERATION_COMPLETED)
      if self.should_terminate:
        break

    if len(losses) == 0:
      raise ValueError("There are no iteration losses. \
            Likely this was caused by an empty validation data iterable.")
    else:
      avg_loss = np.mean(losses, 0)

    if not isinstance(avg_loss, Iterable):
      avg_loss = [avg_loss]

    self.avg_validation_loss.append(avg_loss)
    if avg_loss[0] < self.best_validation_loss[0]:
      self.best_validation_loss = avg_loss

    time_taken = time.time() - start_time
    hours, mins, secs = _to_hours_mins_secs(time_taken)
    self._logger.info("Validation: Avg. Loss: %s \t Best Loss: %s \t Time Taken: %d:%d:%d",
                      avg_loss,
                      self.best_validation_loss, hours, mins, secs)

    self._update_best_model_loss()
    self._fire_event(TrainingEvents.VALIDATION_COMPLETED)

    return avg_loss

  def _update_best_model_loss(self):
    """
    If the loss has improved, stores the current best loss.

    Uses the validation loss if validation data is available, otherwise uses the training loss.
    """

    # obtain best guess of loss corresponding to current set of parameters
    current_loss = np.inf
    if self._validation_data:
      if self.avg_validation_loss:
        current_loss = self.avg_validation_loss[-1]
    elif self.avg_training_loss_per_epoch:
      current_loss = self.avg_training_loss_per_epoch[-1]
    # use first loss when multiple losses present
    if isinstance(current_loss, Iterable):
      current_loss = current_loss[0]

    if current_loss < self._best_model_parameter_loss:
      self._logger.info("Updating best model loss")
      self._fire_event(TrainingEvents.BEST_LOSS_UPDATED)
      self._best_model_parameter_loss = current_loss

  def terminate(self):
    """
    Sends terminate signal to trainer, so that training terminates after the current epoch finishes
    """
    self._logger.info(
      "Terminate signaled to trainer. Training will stop after current epoch is finished")
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
        else:
          self._update_best_model_loss()
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

  def __getstate__(self):
    return {key: self.__dict__[key] for key in self.__dict__.keys() if key != "_logger"}

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._logger = self._get_logger()

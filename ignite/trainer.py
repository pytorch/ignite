from __future__ import division

import time
from collections import Iterable
from torch.autograd import Variable

from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs, to_variable

__all__ = ["Trainer", "create_supervised_trainer"]


class Trainer(Engine):
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
    """

    def __init__(self, training_update_function):
        super(Trainer, self).__init__(training_update_function)
        self.current_epoch = 0
        self.max_epochs = 0

    def _train_one_epoch(self, training_data):
        start_time = time.time()
        self._run_once_on_dataset(training_data)
        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.current_epoch, hours,
                          mins, secs)

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
        self.dataset = training_data
        self.current_iteration = 0
        self.current_epoch = 0

        try:
            self._logger.info("Training starting with max_epochs={}".format(max_epochs))

            self.max_epochs = max_epochs

            start_time = time.time()

            self._fire_event(Events.STARTED)
            while self.current_epoch < max_epochs and not self.should_terminate:
                self._fire_event(Events.EPOCH_STARTED)
                self._train_one_epoch(training_data)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED)
                self.current_epoch += 1

            self._fire_event(Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info("Training complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._logger.error("Training is terminating due to exception: %s", str(e))
            self._fire_event(Events.EXCEPTION_RAISED)
            raise e


def create_supervised_trainer(model, optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Trainer: a trainer instance with supervised update function
    """

    def _prepare_batch(batch):
        x, y = batch
        x = to_variable(x, cuda=cuda)
        y = to_variable(y, cuda=cuda)
        return x, y

    def _update(batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data.cpu()[0]

    return Trainer(_update)

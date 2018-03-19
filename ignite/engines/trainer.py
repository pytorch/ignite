from __future__ import division

import time
from collections import Iterable

from ignite._utils import _to_hours_mins_secs, to_variable
from ignite.engines import Engine, Events, State

__all__ = ["Trainer", "create_supervised_trainer"]


class Trainer(Engine):
    def run(self, data, initial_epoch=0, max_epochs=1):
        """
        Train the model, evaluate the validation set and update best parameters if the validation loss
        improves.
        In the event that the validation set is not run (or doesn't exist), the training loss is used
        to update the best parameters.

        Parameters
        ----------
        data : Iterable
            Collection of training batches allowing repeated iteration (e.g., list or DataLoader)
        initial_epoch : int, optional
            epoch to start counting at [default=0]
        max_epochs: int, optional
            max epochs to train for [default=1]

        Returns
        -------
        None
        """
        state = State(dataloader=data,
                      epoch=initial_epoch,
                      max_epochs=max_epochs)

        try:
            self._logger.info("Training starting with max_epochs={}".format(max_epochs))
            start_time = time.time()
            self._fire_event(Events.STARTED, state)
            while state.epoch < max_epochs and not self.should_terminate:
                state.epoch += 1
                self._fire_event(Events.EPOCH_STARTED, state)
                hours, mins, secs = self._run_once_on_dataset(state)
                self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", state.epoch, hours, mins, secs)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED, state)

            self._fire_event(Events.COMPLETED, state)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info("Training complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._logger.error("Training is terminating due to exception: %s", str(e))
            self._handle_exception(state, e)

        return state


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

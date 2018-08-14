# coding: utf-8

from enum import Enum

import torch

from ignite._utils import convert_tensor, apply_to_tensor
from ignite.engine import Engine


class Tbptt_Events(Enum):
    """Aditional tbptt events.

    Additional events for truncated backpropagation throught time dedicated
    trainer.
    """

    TIME_ITERATION_STARTED = "time_iteration_started"
    TIME_ITERATION_COMPLETED = "time_iteration_completed"


def _prepare_tbptt_batch(batch, tbptt_step, dim=0, device=None):
    """Prepare batch for tbptt trainer.

    Batch come from the dataloader. It is split in chunks along the time
    dimension and fed to the truncated backpropagation throught time trainer.
    """
    x, y = batch
    x = convert_tensor(x, device=device)
    y = convert_tensor(y, device=device)
    return zip(x.split(tbptt_step, dim=dim), y.split(tbptt_step, dim=dim))


def _detach_hidden(hidden):
    """Cut backpropagation graph.

    Auxillary function to cut the backpropagation graph by detaching the hidden
    vector.
    """
    return apply_to_tensor(hidden, torch.Tensor.detach)


def create_supervised_tbptt_trainer(
    model,
    optimizer,
    loss_fn,
    tbtt_step,
    dim=0,
    device=None
):
    """Create a trainer for truncated backprop through time supervised models.

    Training recurrent model on long sequences is computationally intensive as
    it requires to process the whole sequence before getting a gradient.
    However, when the training loss is computed over many outputs
    ([X to many](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)),
    there is an opportunity to compute a gradient over a subsequence. This is
    known as
    [truncated backpropagation through time](
    https://machinelearningmastery.com/gentle-introduction-backpropagation-time/
    ).
    This supervised trainer apply gradient optimization step every `tbtt_step`
    time steps of the sequence, while backpropagating through the same
    `tbtt_step` time steps.

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        tbtt_step (int): the length of time chunks (last one may be smaller)
        dim (int): axis representing the time dimension
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function

    """
    if device:
        model.to(device)

    def _update(engine, batch):
        loss_list = []
        hidden = None

        # Batches split in time chunks
        batch_splits = _prepare_tbptt_batch(
            batch, tbtt_step, dim=dim, device=device
        )
        for x_t, y_t in batch_splits:
            # Fire event for start of iteration
            engine.fire_event(Tbptt_Events.TIME_ITERATION_STARTED)
            # Forward, backward and
            model.train()
            optimizer.zero_grad()
            if hidden is None:
                y_pred_t, hidden = model(x_t)
            else:
                hidden = _detach_hidden(hidden)
                y_pred_t, hidden = model(x_t)
            loss_t = loss_fn(y_pred_t, y_t)
            loss_t.backward()
            optimizer.step()

            # Setting state of engine for consistent behaviour
            engine.state.output = loss_t.item()
            loss_list.append(loss_t.item())

            # Fire event for end of iteration
            engine.fire_event(Tbptt_Events.TIME_ITERATION_COMPLETED)

        # return average loss over the time splits
        return sum(loss_list) / len(loss_list)

    engine = Engine(_update)
    engine.register_events(*Tbptt_Events)
    return engine

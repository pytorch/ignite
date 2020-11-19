# coding: utf-8
import collections.abc as collections
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ignite.engine import Engine, EventEnum, _prepare_batch
from ignite.utils import apply_to_tensor


class Tbptt_Events(EventEnum):
    """Aditional tbptt events.

    Additional events for truncated backpropagation throught time dedicated
    trainer.
    """

    TIME_ITERATION_STARTED = "time_iteration_started"
    TIME_ITERATION_COMPLETED = "time_iteration_completed"


def _detach_hidden(
    hidden: Union[torch.Tensor, Sequence, Mapping, str, bytes]
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Cut backpropagation graph.

    Auxillary function to cut the backpropagation graph by detaching the hidden
    vector.
    """
    return apply_to_tensor(hidden, torch.Tensor.detach)


def create_supervised_tbptt_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    tbtt_step: int,
    dim: int = 0,
    device: Optional[str] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
) -> Engine:
    """Create a trainer for truncated backprop through time supervised models.

    Training recurrent model on long sequences is computationally intensive as
    it requires to process the whole sequence before getting a gradient.
    However, when the training loss is computed over many outputs
    (`X to many <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_),
    there is an opportunity to compute a gradient over a subsequence. This is
    known as
    `truncated backpropagation through time <https://machinelearningmastery.com/
    gentle-introduction-backpropagation-time/>`_.
    This supervised trainer apply gradient optimization step every `tbtt_step`
    time steps of the sequence, while backpropagating through the same
    `tbtt_step` time steps.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        tbtt_step (int): the length of time chunks (last one may be smaller).
        dim (int): axis representing the time dimension.
        device (str, optional): device type specification (default: None).
            Applies to batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU,
            the copy may occur asynchronously with respect to the host. For other cases,
            this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`,
            `non_blocking` and outputs tuple of tensors `(batch_x, batch_y)`.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        * `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        * `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: a trainer engine with supervised update function.

    """

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> float:
        loss_list = []
        hidden = None

        x, y = batch
        for batch_t in zip(x.split(tbtt_step, dim=dim), y.split(tbtt_step, dim=dim)):
            x_t, y_t = prepare_batch(batch_t, device=device, non_blocking=non_blocking)
            # Fire event for start of iteration
            engine.fire_event(Tbptt_Events.TIME_ITERATION_STARTED)
            # Forward, backward and
            model.train()
            optimizer.zero_grad()
            if hidden is None:
                y_pred_t, hidden = model(x_t)
            else:
                hidden = _detach_hidden(hidden)
                y_pred_t, hidden = model(x_t, hidden)
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

# coding: utf-8

from ignite.engine import create_supervised_trainer, Events, Engine, _prepare_batch
from ignite.metrics import Metric
from ignite.contrib.handlers import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import torch
import copy


def create_lr_finder(model, optimizer, loss_fn, end_lr=10, step_mode="exp", smooth_f=0.05, diverge_th=5,
                     device=None, non_blocking=False, prepare_batch=_prepare_batch,
                     output_transform=lambda x, y, y_pred, loss: loss.item()):
    """Factory function for creating a learning rate finder for supervised models.

    based on fastai/lr_find: https://github.com/fastai/fastai

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use, the defined optimizer learning rate is assumed to be
            the lower boundary of the range test.
        loss_fn (torch.nn loss function): the loss function to use.
        end_lr (float): the upper bound of the range test
        step_mode (str): "exp" or "linear", which way should the lr be increased from optimizer's initial lr to end_lr
        smooth_f (float): loss smoothing factor in range [0, 1), 0 for no smoothing
        diverge_th (float): Used for stopping the search when `current loss > diverge_th * best_loss`
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.


    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a lr finder engine with supervised update function.
    """

    if smooth_f < 0 or smooth_f >= 1:
        raise ValueError("smooth_f is outside the range [0, 1]")
    if step_mode not in ["exp", "linear"]:
        raise ValueError(f"expected one of (exp, linear), got {step_mode}")

    lr_finder_engine = create_supervised_trainer(model, optimizer, loss_fn, device=device, non_blocking=non_blocking,
                                                 prepare_batch=prepare_batch, output_transform=output_transform)

    def _start(engine):
        engine.state.state_cache = _StateCacher()
        engine.state.state_cache.store("model", model.state_dict())
        engine.state.state_cache.store("optimizer", optimizer.state_dict())

        num_iter = engine.state.max_epochs * len(engine.state.dataloader)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = LRScheduler(ExponentialLR(optimizer, end_lr, num_iter))
        else:
            lr_schedule = LRScheduler(LinearLR(optimizer, end_lr, num_iter))

        engine.add_event_handler(Events.ITERATION_COMPLETED, lr_schedule)

    lr_finder_engine.add_event_handler(Events.STARTED, _start)

    # Reset model and optimizer and delete state cache
    def _reset(engine):
        model.load_state_dict(engine.state.state_cache.retrieve("model"))
        optimizer.load_state_dict(engine.state.state_cache.retrieve("optimizer"))
        del engine.state.state_cache
        # engine._logger.info("Completed LR finder run, resets model & optimizer")

    lr_finder_engine.add_event_handler(Events.COMPLETED, _reset)

    # log the loss at the end of every train iteration

    loss_and_lr = LossAndLR(lr_finder_engine, smooth_f=smooth_f)
    loss_and_lr.attach(lr_finder_engine, name="lr_vs_loss")

    # Check if the loss has diverged. if it has, stop the trainer
    def _loss_diverged(engine: Engine, loss_metric: LossAndLR):
        if loss_metric.history["loss"][-1] > diverge_th * loss_metric.best_loss:
            engine.terminate()
    lr_finder_engine.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: _loss_diverged(engine, loss_and_lr))

    def _warning(engine):
        if not engine.should_terminate:
            raise Warning("Loss didn't diverge by the end of the run, try running for more epochs or increasing end_lr")

    lr_finder_engine.add_event_handler(Events.COMPLETED, _warning)

    return lr_finder_engine


class LossAndLR(Metric):
    """
    metric used for aggregating the loss and lr during the run of the lr finder
    """
    def __init__(self, engine: Engine, smooth_f, output_transform=lambda x: x):
        super().__init__(output_transform)
        self.history = {"lr": [], "loss": []}
        self.suggestion = None
        self.smooth_f = smooth_f
        self.best_loss = None
        self.engine = engine

    def reset(self):
        self.history = {"lr": [], "loss": []}
        self.suggestion = None

    def update(self, output):
        loss = self._output_transform(output)
        lr = self.egnine.state.lr_schedule.lr_scheduler.get_lr()[0]
        self.history["lr"].append(lr)
        if len(self.history["loss"]) != 0:
            if self.smooth_f > 0:
                loss = self.smooth_f * loss + (1 - self.smooth_f) * self.history["loss"][-1]
            if loss < self.best_loss:
                self.best_loss = loss
        else:
            self.best_loss = loss
        self.history["loss"].append(loss)

    def compute(self):
        loss = self.history["loss"]
        grads = [loss[i] - loss[i - 1] for i in range(1, len(loss))]
        min_grad_idx = np.argmin(grads) + 1
        self.history["suggestion"] = self.history["lr"][int(min_grad_idx)]
        return self.history


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class _StateCacher(object):

    def __init__(self):
        self.cached = {}

    def store(self, key, state_dict):
        self.cached.update({key: copy.deepcopy(state_dict)})

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        return self.cached.get(key)

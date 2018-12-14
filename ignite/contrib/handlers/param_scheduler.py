from __future__ import division

import numpy as np


class ParamScheduler(object):
    """An abstract class for updating an optimizer's parameter value during
    training.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        save_history (bool, optional): whether to log the parameter values
            (default=False)
    """
    def __init__(self, optimizer, param_name, save_history=False):

        if isinstance(optimizer, dict):
            self.optimizer_param_groups = [optimizer]
        else:
            self.optimizer_param_groups = optimizer.param_groups
        self.param_name = param_name
        self.save_history = save_history
        self.event_index = 0

    def __call__(self, engine, name=None):

        value = self.get_param()

        for param_group in self.optimizer_param_groups:
            param_group[self.param_name] = value

        if name is None:
            name = self.param_name

        if self.save_history:
            if not hasattr(engine.state, 'param_history'):
                setattr(engine.state, 'param_history', {})
            engine.state.param_history.setdefault(name, [])
            values = [pg[self.param_name] for pg in self.optimizer_param_groups]
            engine.state.param_history[name].append(values)

        self.event_index += 1

    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        raise NotImplementedError()


class CyclicalScheduler(ParamScheduler):
    """An abstract class for updating an optimizer's parameter value over a
    cycle of some size.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        start_value (float): value at start of cycle
        end_value (float) : value at the middle of the cycle
        cycle_size (int) : length of cycle.
        cycle_mult (float, optional) : ratio by which to change the cycle_size
            at the end of each cycle (default=1),
        save_history (bool, optional): whether to log the parameter values
            (default: False)

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.
    """
    def __init__(self,
                 optimizer,
                 param_name,
                 start_value,
                 end_value,
                 cycle_size,
                 cycle_mult=1,
                 save_history=False):
        super(CyclicalScheduler, self).__init__(
            optimizer,
            param_name,
            save_history=save_history
        )
        self.start_value = start_value
        self.end_value = end_value
        self.cycle_size = cycle_size
        self.cycle_mult = cycle_mult
        self.cycle = 0

    def __call__(self, engine, name=None):
        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1

        return super(CyclicalScheduler, self).__call__(engine, name)


class LinearCyclicalScheduler(CyclicalScheduler):
    """Linearly adjusts param value to 'end_value' for a half-cycle, then linearly
    adjusts it back to 'start_value' for a half-cycle.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        start_value (float): value at start of cycle
        end_value (float) : value at the middle of the cycle
        cycle_size (int) : length of cycle.
        cycle_mult (float, optional) : ratio by which to change the cycle_size
            at the end of each cycle (default=1),
        save_history (bool, optional): whether to log the parameter values
            (default: False)

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

        scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1e-3, 1e-1, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        #
        # Linearly increases the learning rate from 1e-3 to 1e-1 and back to 1e-3
        # over the course of 1 epoch
        #
    """
    def get_param(self):
        cycle_progress = self.event_index / self.cycle_size
        return self.end_value + (self.start_value - self.end_value) * abs(cycle_progress - 0.5) * 2


class CosineAnnealingScheduler(CyclicalScheduler):
    """Anneals 'start_value' to 'end_value' over each cycle.

    The annealing takes the form of the first half of a cosine
    wave (as suggested in [Smith17]_).

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        start_value (float): value at start of cycle
        end_value (float) : value at the end of the cycle
        cycle_size (int) : length of cycle.
        cycle_mult (float, optional) : ratio by which to change the cycle_size
            at the end of each cycle (default=1),
        save_history (bool, optional): whether to log the parameter values
            (default: False)

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

        scheduler = CosineAnnealingScheduler(optimizer, 'lr', 1e-1, 1e-3, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        #
        # Anneals the learning rate from 1e-1 to 1e-3 over the course of 1 epoch.
        #

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

        optimizer = SGD(
            [
                {"params": model.base.parameters(), 'lr': 0.001),
                {"params": model.fc.parameters(), 'lr': 0.01),
            ]
        )

        scheduler1 = LinearCyclicalScheduler(optimizer.param_groups[0], 'lr', 1e-7, 1e-5, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler1, "lr (base)")

        scheduler2 = CosineAnnealingScheduler(optimizer.param_groups[1], 'lr', 1e-5, 1e-3, len(train_loader))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler2, "lr (fc)")

    .. [Smith17] Smith, Leslie N. "Cyclical learning rates for training neural networks."
                 Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017
    """
    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        cycle_progress = self.event_index / self.cycle_size
        return self.start_value + ((self.end_value - self.start_value) / 2) * (1 - np.cos(np.pi * cycle_progress))


class ConcatScheduler(ParamScheduler):
    """Concat a list of Schedulers.

    The `ConcatScheduler` cycles through a list of schedulers (given by
    `schedulers_list`). Each element in the list is a tuple whose first
    element is the scheduler class, the second is the parameters used for
    instantiating the scheduler, and the third is the duration of the
    scheduler. If duration is `None` the `ConcatScheduler` will not
    switch to the next scheduler.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        schedulers_list (list): List of three tuple of the order (scheduler_cls,
            scheduler_kwds, duration).
        save_history (bool, optional): whether to log the parameter values
            (default: False)

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import ConcatScheduler
        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

        scheduler = ConcatScheduler(
            optimizer,
            "lr",
            [
                (
                    LinearCyclicalScheduler,
                    dict(
                        start_value=0.1,
                        end_value=0.5,
                        cycle_size=60
                    ),
                    30
                ),
                (
                    CosineAnnealingScheduler,
                    dict(
                        start_value=0.5,
                        end_value=0.01,
                        cycle_size=60
                    ),
                    None
                ),
            ],
        )
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        #
        # Sets the Learning rate linearly from 0.1 to 0.5 over 30 iterations. Then
        # starts an annealing schedule from 0.5 to 0.01 over 60 iterations.
        # The annealing cycles are repeated indefinitely.
        #
    """
    def __init__(self,
                 optimizer,
                 param_name,
                 schedulers_list,
                 save_history=False):
        super(ConcatScheduler, self).__init__(optimizer, param_name, save_history=save_history)
        self._schedulers_list = schedulers_list
        self._schedulers_index = 0
        self._next_scheduler_switch = 0
        self.optimizer = optimizer

    def _next_scheduler(self):
        scheduler_cls, scheduler_kwds, self._next_scheduler_switch = \
            self._schedulers_list[self._schedulers_index]

        kwds = scheduler_kwds.copy()
        kwds.update(
            dict(
                optimizer=self.optimizer,
                param_name=self.param_name,
                save_history=self.save_history
            )
        )

        self._scheduler = scheduler_cls(**kwds)
        self._schedulers_index = (self._schedulers_index + 1) % len(self._schedulers_list)

    def __call__(self, engine, name=None):
        if self._next_scheduler_switch is not None:
            self._next_scheduler_switch -= 1
            if self._next_scheduler_switch <= 0:
                self._next_scheduler()

        return self._scheduler(engine, name)

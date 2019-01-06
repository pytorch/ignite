from __future__ import division

import math


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

    @classmethod
    def simulate_values(cls, num_events, **scheduler_kwargs):
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events (int): number of events during the simulation
            **scheduler_kwargs : parameter scheduler configuration kwargs

        Returns:
            list of pairs: [event_index, value]

        Examples:

        .. code-block:: python

            lr_values = np.array(LinearCyclicalScheduler.simulate_values(num_events=50, param_name='lr',
                                                                         start_value=1e-1, end_value=1e-3,
                                                                         cycle_size=10))

            plt.plot(lr_values[:, 0], lr_values[:, 1], label="learning rate")
            plt.xlabel("events")
            plt.ylabel("values")
            plt.legend()

        """
        keys_to_remove = ['optimizer', 'save_history']
        for key in keys_to_remove:
            if key in scheduler_kwargs:
                del scheduler_kwargs[key]
        values = []
        scheduler = cls(optimizer={}, save_history=False, **scheduler_kwargs)
        for i in range(num_events):
            scheduler(engine=None)
            values.append([i, scheduler.optimizer_param_groups[0][scheduler.param_name]])
        return values


class CyclicalScheduler(ParamScheduler):
    """An abstract class for updating an optimizer's parameter value over a
    cycle of some size.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        start_value (float): value at start of cycle
        end_value (float): value at the middle of the cycle
        cycle_size (int): length of cycle.
        cycle_mult (float, optional): ratio by which to change the cycle_size
            at the end of each cycle (default=1.0),
        start_value_mult (float, optional): ratio by which to change the start value at the
            end of each cycle (default=1.0)
        end_value_mult (float, optional): ratio by which to change the end value at the
            end of each cycle (default=1.0)
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
                 cycle_mult=1.0,
                 start_value_mult=1.0,
                 end_value_mult=1.0,
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
        self.start_value_mult = start_value_mult
        self.end_value_mult = end_value_mult

    def __call__(self, engine, name=None):
        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1
            self.start_value *= self.start_value_mult
            self.end_value *= self.end_value_mult

        return super(CyclicalScheduler, self).__call__(engine, name)


class LinearCyclicalScheduler(CyclicalScheduler):
    """Linearly adjusts param value to 'end_value' for a half-cycle, then linearly
    adjusts it back to 'start_value' for a half-cycle.

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use
        param_name (str): name of optimizer's parameter to update
        start_value (float): value at start of cycle
        end_value (float): value at the middle of the cycle
        cycle_size (int): length of cycle.
        cycle_mult (float, optional): ratio by which to change the cycle_size
            at the end of each cycle (default=1),
        start_value_mult (float, optional): ratio by which to change the start value at the
            end of each cycle (default=1.0)
        end_value_mult (float, optional): ratio by which to change the end value at the
            end of each cycle (default=1.0)
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
        end_value (float): value at the end of the cycle
        cycle_size (int): length of cycle.
        cycle_mult (float, optional): ratio by which to change the cycle_size
            at the end of each cycle (default=1),
        start_value_mult (float, optional): ratio by which to change the start value at the
            end of each cycle (default=1.0)
        end_value_mult (float, optional): ratio by which to change the end value at the
            end of each cycle (default=1.0)
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
        return self.start_value + ((self.end_value - self.start_value) / 2) * (1 - math.cos(math.pi * cycle_progress))


class ConcatScheduler(ParamScheduler):
    """Concat a list of parameter schedulers.

    The `ConcatScheduler` goes through a list of schedulers given by `schedulers`. Duration of each
    scheduler is defined by `durations` list of integers.

    Args:
        schedulers (list of ParamScheduler): list of parameter schedulers
        durations (list of int): list of number of events that lasts a parameter scheduler from schedulers

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import ConcatScheduler
        from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
        from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler

        scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.1, end_value=0.5, cycle_size=60)
        scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.5, end_value=0.01, cycle_size=60)

        combined_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[30, ])
        trainer.add_event_handler(Events.ITERATION_COMPLETED, combined_scheduler)
        #
        # Sets the Learning rate linearly from 0.1 to 0.5 over 30 iterations. Then
        # starts an annealing schedule from 0.5 to 0.01 over 60 iterations.
        # The annealing cycles are repeated indefinitely.
        #
    """

    def __init__(self, schedulers, durations, save_history=False):

        if not isinstance(schedulers, (list, tuple)) or len(schedulers) < 2:
            raise TypeError("Argument schedulers should be list/tuple of more than one parameter schedulers")

        if not isinstance(durations, (list, tuple)) or sorted(list(durations)) != list(durations):
            raise TypeError("Argument durations should be list/tuple of ordered integers")

        if len(schedulers) != len(durations) + 1:
            raise ValueError("Incorrect number schedulers or duration values")

        for i, scheduler in enumerate(schedulers):
            if not isinstance(scheduler, ParamScheduler):
                raise TypeError("Value at index {} of schedulers should be a parameter scheduler, "
                                "but given {}".format(i, type(scheduler)))

        for i, scheduler in enumerate(schedulers):
            if not isinstance(scheduler, ParamScheduler):
                raise TypeError("Value at index {} of schedulers should be a parameter scheduler, "
                                "but given {}".format(i, type(scheduler)))

        super(ConcatScheduler, self).__init__(optimizer={}, param_name="", save_history=save_history)
        self.schedulers = schedulers
        self.durations = list(durations) + [-1, ]
        self._current_scheduler = None
        self._current_duration = None
        self._set_next_scheduler()

    def _set_next_scheduler(self):
        self._current_scheduler = self.schedulers.pop(0)
        self._current_scheduler.save_history = self.save_history
        self._current_duration = self.durations.pop(0)
        self.optimizer_param_groups = self._current_scheduler.optimizer_param_groups
        self.param_name = self._current_scheduler.param_name

    def __call__(self, engine, name=None):
        self._current_scheduler(engine, name)
        self._current_duration -= 1

        if self._current_duration == 0:
            self._set_next_scheduler()

    @classmethod
    def simulate_values(cls, num_events, schedulers, durations, param_name=None):
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events (int): number of events during the simulation
            schedulers (list of ParamScheduler): list of parameter schedulers
            durations (list of int): list of number of events that lasts a parameter scheduler from schedulers
            param_name (str, optional): parameter name to simulate values.
                By default, the first scheduler parameter name is taken.

        Returns:
            list of pairs: [event_index, value]

        """
        values = []
        scheduler = cls(schedulers, durations)
        if param_name is None:
            param_name = scheduler.param_name
        optimizer_param_group = scheduler.optimizer_param_groups[0]
        for i in range(num_events):
            scheduler(engine=None)
            values.append([i, optimizer_param_group[param_name]])
        return values


class LRScheduler(ParamScheduler):
    """A wrapper class to call `torch.optim.lr_scheduler` objects as `ignite` handlers.

    Args:
        lr_scheduler (subclass of `torch.optim.lr_scheduler._LRScheduler`): lr_scheduler object to wrap.
        save_history (bool, optional): whether to log the parameter values (default=False)

    .. code-block:: python

        from ignite.contrib.handlers.param_scheduler import LRScheduler
        from torch.optim.lr_scheduler import StepLR

        step_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        scheduler = LRScheduler(step_scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    """

    def __init__(self, lr_scheduler, save_history=False, **kwds):

        if len(lr_scheduler.optimizer.param_groups) > 1:
            raise ValueError("Optimizer passed to lr_scheduler should have a single param group, "
                             "but currently there are {} param groups".format(len(lr_scheduler.optimizer.param_groups)))

        self.lr_scheduler = lr_scheduler
        super(LRScheduler, self).__init__(
            optimizer=self.lr_scheduler.optimizer,
            param_name='lr',
            save_history=save_history
        )

    def __call__(self, engine, name=None):
        self.lr_scheduler.last_epoch += 1
        super(LRScheduler, self).__call__(engine, name)

    def get_param(self):
        """Method to get current optimizer's parameter value
        """
        lr_list = self.lr_scheduler.get_lr()
        if len(lr_list) > 1:
            raise ValueError("Optimizer passed to lr_scheduler should have a single param group, "
                             "but currently there are {} param groups".format(len(lr_list)))
        return lr_list[0]

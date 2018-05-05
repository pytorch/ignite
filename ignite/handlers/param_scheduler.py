from __future__ import division

import numpy as np


class ParamScheduler(object):
    """
    Updates an optimizer's parameter value during training.
    """
    def __init__(self, optimizer, param_name, save_history=False):
        self.optimizer = optimizer
        self.param_name = param_name
        self.save_history = save_history
        self.event_index = 0

    def __call__(self, engine):
        value = self.get_param()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = value

        if self.save_history:
            if not hasattr(engine.state, 'param_history'):
                setattr(engine.state, 'param_history', {})
            engine.state.param_history.setdefault(self.param_name, [])
            values = [pg[self.param_name] for pg in self.optimizer.param_groups]
            engine.state.param_history[self.param_name].append(values)

        self.event_index += 1

    def get_param(self):
        raise NotImplementedError()


class CyclicalScheduler(ParamScheduler):
    """
    Updates an optimizer's parameter value over a cycle of some size.

    NOTE: If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should usually be
    the number of batches in an epoch.
    """
    def __init__(self,
                 optimizer,
                 param_name,
                 start_value,
                 end_value,
                 cycle_size,
                 cycle_mult=1,
                 save_history=False):
        super(CyclicalScheduler, self).__init__(optimizer, param_name, save_history=save_history)
        self.start_value = start_value
        self.end_value = end_value
        self.cycle_size = cycle_size
        self.cycle_mult = cycle_mult
        self.cycle = 0

    def get_cycle_progress(self):
        """
        Returns float in range [0.0, 1.0] representing the progress of the current cycle as a percentage.
        """
        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1

        return self.event_index / self.cycle_size


class LinearScheduler(CyclicalScheduler):
    """
    Linearly adjusts param value to 'end_value' for a half-cycle, then linearly
    adjusts it back to 'start_value' for a half-cycle.
    """
    def __init__(self, *args, **kwargs):
        super(LinearScheduler, self).__init__(*args, **kwargs)

    def get_param(self):
        cycle_progress = self.get_cycle_progress()
        return self.end_value + (self.start_value - self.end_value) * abs(cycle_progress - 0.5) * 2


class CosineAnnealingScheduler(CyclicalScheduler):
    """
    Anneals 'start_value' to 'end_value' over each cycle.
    """
    def __init__(self, *args, **kwargs):
        super(CosineAnnealingScheduler, self).__init__(*args, **kwargs)

    def get_param(self):
        cycle_progress = self.get_cycle_progress()
        return self.start_value + ((self.end_value - self.start_value) / 2) * (1 + np.cos(np.pi * cycle_progress))

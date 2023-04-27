import numbers
import warnings
from bisect import bisect_right
from typing import Any, List, Sequence, Tuple, Union

from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from ignite.handlers.param_scheduler import BaseParamScheduler


class StateParamScheduler(BaseParamScheduler):
    """An abstract class for updating an engine state parameter values during training.

    Args:
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to ``engine.state.param_history``, (default=False).
        create_new: whether to create ``param_name`` on ``engine.state`` taking into account whether ``param_name``
            attribute already exists or not. Overrides existing attribute by default, (default=False).

    Note:
        Parameter scheduler works independently of the internal state of the attached engine.
        More precisely, whatever the state of the engine (newly created or used by another scheduler) the scheduler
        sets defined absolute values.

    .. versionadded:: 0.4.7

    """

    def __init__(self, param_name: str, save_history: bool = False, create_new: bool = False):
        super(StateParamScheduler, self).__init__(param_name, save_history)
        self.create_new = create_new

    def attach(
        self,
        engine: Engine,
        event: Union[str, Events, CallableEventWithFilter, EventsList] = Events.ITERATION_COMPLETED,
    ) -> None:
        """Attach the handler to the engine. Once the handler is attached, the ``Engine.state`` will have a new
        attribute with the name ``param_name``. Then the current value of the parameter can be retrieved from
        ``Engine.state`` when the engine is running.

        Args:
            engine: trainer to which the handler will be attached.
            event: trigger ``param_name`` value update.

        """
        if hasattr(engine.state, self.param_name):
            if self.create_new:
                raise ValueError(
                    f"Attribute '{self.param_name}' already exists in the engine.state. "
                    f"This may be a conflict between multiple handlers. "
                    f"Please choose another name."
                )
        else:
            if not self.create_new:
                warnings.warn(
                    f"Attribute '{self.param_name}' is not defined in the engine.state. "
                    f"{type(self).__name__} will create it. Remove this warning by setting create_new=True."
                )
            setattr(engine.state, self.param_name, None)

        if self.save_history:
            if not hasattr(engine.state, "param_history") or engine.state.param_history is None:
                setattr(engine.state, "param_history", {})
            engine.state.param_history.setdefault(self.param_name, [])  # type: ignore[attr-defined]

        engine.add_event_handler(event, self)

    def __call__(self, engine: Engine) -> None:
        self.event_index += 1
        value = self.get_param()
        setattr(engine.state, self.param_name, value)
        if self.save_history:
            engine.state.param_history[self.param_name].append(value)  # type: ignore[attr-defined]

    @classmethod
    def simulate_values(cls, num_events: int, **scheduler_kwargs: Any) -> List[List[int]]:
        """Method to simulate scheduled engine state parameter values during `num_events` events.

        Args:
            num_events: number of events during the simulation.
            scheduler_kwargs: parameter scheduler configuration kwargs.

        Returns:
            event_index, value

        Examples:

        .. code-block:: python

            import matplotlib.pyplot as plt
            import numpy as np

            step_state_param_values = np.array(
                StepStateScheduler.simulate_values(
                    num_events=20, param_name="step_scheduled_param", initial_value=10, gamma=0.99, step_size=5
                )
            )

            plt.plot(step_state_param_values[:, 0], step_state_param_values[:, 1], label="learning rate")
            plt.xlabel("events")
            plt.ylabel("values")
            plt.legend()

        """
        for key in ["save_history"]:
            if key in scheduler_kwargs:
                del scheduler_kwargs[key]
        values = []
        scheduler = cls(save_history=False, **scheduler_kwargs)
        engine = Engine(lambda e, b: None)
        for i in range(num_events):
            scheduler(engine=engine)
            values.append([i, getattr(engine.state, scheduler_kwargs["param_name"])])
        return values


class LambdaStateScheduler(StateParamScheduler):
    """Update a parameter during training by using a user defined callable object.
        User defined callable object is taking an event index as input and returns parameter value.

    Args:
        lambda_obj: user defined callable object.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        create_new: whether to create ``param_name`` on
            ``engine.state`` taking into account whether
            ``param_name`` attribute already exists or not.
            Overrides existing attribute by default, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            class LambdaState:
                def __init__(self, initial_value, gamma):
                    self.initial_value = initial_value
                    self.gamma = gamma

                def __call__(self, event_index):
                    return self.initial_value * self.gamma ** (event_index % 9)

            param_scheduler = LambdaStateScheduler(
                param_name="param", lambda_obj=LambdaState(1, 0.9), create_new=True
            )

            # parameter is param, initial_value sets param to 1 and in this example gamma = 1
            # using class 'LambdaState' user defined callable object can be created
            # update a parameter during training by using a user defined callable object
            # user defined callable object is taking an event index as input and returns parameter value
            # in this example, we update as initial_value * gamma ** (event_endex % 9)
            # in every Epoch the parameter is updated as 1 * 0.9 ** (Epoch % 9)
            # In Epoch 3, parameter param = 1 * 0.9 ** (3 % 9) = 0.729
            # In Epoch 10, parameter param = 1 * 0.9 ** (10 % 9) = 0.9

            param_scheduler.attach(default_trainer, Events.EPOCH_COMPLETED)

            @default_trainer.on(Events.EPOCH_COMPLETED)
            def print_param():
                print(default_trainer.state.param)

            default_trainer.run([0], max_epochs=10)

        .. testoutput::

            0.9
            0.81
            0.7290...
            0.6561
            0.5904...
            0.5314...
            0.4782...
            0.4304...
            1.0
            0.9

    .. versionadded:: 0.4.7

    """

    def __init__(self, lambda_obj: Any, param_name: str, save_history: bool = False, create_new: bool = False):
        super(LambdaStateScheduler, self).__init__(param_name, save_history, create_new)

        if not callable(lambda_obj):
            raise ValueError("Expected lambda_obj to be callable.")

        self.lambda_obj = lambda_obj
        self._state_attrs += ["lambda_obj"]

    def get_param(self) -> Union[List[float], float]:
        return self.lambda_obj(self.event_index)


class PiecewiseLinearStateScheduler(StateParamScheduler):
    """Piecewise linear state parameter scheduler.

    Args:
        milestones_values: list of tuples (event index, parameter value)
            represents milestones and parameter values. Milestones should be increasing integers.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        create_new: whether to create ``param_name`` on
            ``engine.state`` taking into account whether
            ``param_name`` attribute already exists or not.
            Overrides existing attribute by default, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            param_scheduler = PiecewiseLinearStateScheduler(
                param_name="param",  milestones_values=[(5, 1.0), (10, 0.8), (15, 0.6)], create_new=True
            )

            # parameter is param, milestone (5, 1.0) sets param to 1.0
            # milestone is (5, 1.0), param=1  for Epoch 1 to 5,
            # next milestone is (10, 0.8), param linearly reduces from 1.0 to 0.8
            # Epoch 10, param = 0.8
            # next milestone is (15,0.6), param linearly reduces from 0.8 to 0.6
            # Epoch 15, param = 0.6

            param_scheduler.attach(default_trainer, Events.EPOCH_COMPLETED)

            @default_trainer.on(Events.EPOCH_COMPLETED)
            def print_param():
                print(default_trainer.state.param)

            default_trainer.run([0], max_epochs=15)

        .. testoutput::

            1.0
            1.0
            1.0
            1.0
            1.0
            0.96
            0.92
            0.88
            0.8400...
            0.8
            0.76
            0.72
            0.68
            0.64
            0.6

    .. versionadded:: 0.4.7
    """

    def __init__(
        self,
        milestones_values: List[Tuple[int, float]],
        param_name: str,
        save_history: bool = False,
        create_new: bool = False,
    ):
        super(PiecewiseLinearStateScheduler, self).__init__(param_name, save_history, create_new)

        if not isinstance(milestones_values, Sequence):
            raise TypeError(
                f"Argument milestones_values should be a list or tuple, but given {type(milestones_values)}"
            )
        if len(milestones_values) < 1:
            raise ValueError(
                f"Argument milestones_values should be with at least one value, but given {milestones_values}"
            )

        values: List[float] = []
        milestones: List[int] = []
        for pair in milestones_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Argument milestones_values should be a list of pairs (milestone, param_value)")
            if not isinstance(pair[0], numbers.Integral):
                raise TypeError(f"Value of a milestone should be integer, but given {type(pair[0])}")
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError(
                    f"Milestones should be increasing integers, but given {pair[0]} is smaller "
                    f"than the previous milestone {milestones[-1]}"
                )
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0
        self._state_attrs += ["values", "milestones", "_index"]

    def _get_start_end(self) -> Tuple[int, int, float, float]:
        if self.milestones[0] > self.event_index:
            return self.event_index - 1, self.event_index, self.values[0], self.values[0]
        elif self.milestones[-1] <= self.event_index:
            return (self.event_index, self.event_index + 1, self.values[-1], self.values[-1])
        elif self.milestones[self._index] <= self.event_index < self.milestones[self._index + 1]:
            return (
                self.milestones[self._index],
                self.milestones[self._index + 1],
                self.values[self._index],
                self.values[self._index + 1],
            )
        else:
            self._index += 1
            return self._get_start_end()

    def get_param(self) -> Union[List[float], float]:
        start_index, end_index, start_value, end_value = self._get_start_end()
        return start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)


class ExpStateScheduler(StateParamScheduler):
    """Update a parameter during training by using exponential function.
    The function decays the parameter value by gamma every step.
    Based on the closed form of ExponentialLR from PyTorch
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html

    Args:
        initial_value: Starting value of the parameter.
        gamma: Multiplicative factor of parameter value decay.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        create_new: whether to create ``param_name`` on
            ``engine.state`` taking into account whether
            ``param_name`` attribute already exists or not.
            Overrides existing attribute by default, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            param_scheduler = ExpStateScheduler(
                param_name="param", initial_value=1, gamma=0.9, create_new=True
            )

            # parameter is param, initial_value sets param to 1, gamma is set as 0.9
            # Epoch 1, param changes from 1 to 1*0.9, param = 0.9
            # Epoch 2, param changes from 0.9 to 0.9*0.9, param = 0.81
            # Epoch 3, param changes from 0.81 to 0.81*0.9, param = 0.729
            # Epoch 4, param changes from 0.81 to 0.729*0.9, param = 0.6561

            param_scheduler.attach(default_trainer, Events.EPOCH_COMPLETED)

            @default_trainer.on(Events.EPOCH_COMPLETED)
            def print_param():
                print(default_trainer.state.param)

            default_trainer.run([0], max_epochs=4)

        .. testoutput::

            0.9
            0.81
            0.7290...
            0.6561

    .. versionadded:: 0.4.7

    """

    def __init__(
        self, initial_value: float, gamma: float, param_name: str, save_history: bool = False, create_new: bool = False
    ):
        super(ExpStateScheduler, self).__init__(param_name, save_history, create_new)
        self.initial_value = initial_value
        self.gamma = gamma
        self._state_attrs += ["initial_value", "gamma"]

    def get_param(self) -> Union[List[float], float]:
        return self.initial_value * self.gamma**self.event_index


class StepStateScheduler(StateParamScheduler):
    """Update a parameter during training by using a step function.
    This function decays the parameter value by gamma every step_size.
    Based on StepLR from PyTorch.
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

    Args:
        initial_value: Starting value of the parameter.
        gamma: Multiplicative factor of parameter value decay.
        step_size: Period of parameter value decay.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        create_new: whether to create ``param_name`` on
            ``engine.state`` taking into account whether
            ``param_name`` attribute already exists or not.
            Overrides existing attribute by default, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            param_scheduler = StepStateScheduler(
                param_name="param", initial_value=1, gamma=0.9, step_size=5, create_new=True
            )

            # parameter is param, initial_value sets param to 1, gamma is set as 0.9
            # Epoch 1 to 4, param does not change as step size is 5,
            # Epoch 5, param changes from 1 to 1*0.9, param = 0.9
            # Epoch 5 to 9, param = 0.9 as step size is 5,
            # Epoch 10, param changes from 0.9 to 0.9*0.9, param = 0.81
            # Epoch 10 to 14, param = 0.81, as step size is 5
            # Epoch 15, param changes from 0.81 to 0.81*0.9, param = 0.729
            # and so on ... the param change at Epoch = 5, 10, 15, 20, . . .

            param_scheduler.attach(default_trainer, Events.EPOCH_COMPLETED)

            @default_trainer.on(Events.EPOCH_COMPLETED(every=5))
            def print_param():
                print(default_trainer.state.param)

            default_trainer.run([0], max_epochs=25)

        .. testoutput::

            0.9
            0.81
            0.7290...
            0.6561
            0.5904...

    .. versionadded:: 0.4.7

    """

    def __init__(
        self,
        initial_value: float,
        gamma: float,
        step_size: int,
        param_name: str,
        save_history: bool = False,
        create_new: bool = False,
    ):
        super(StepStateScheduler, self).__init__(param_name, save_history, create_new)
        self.initial_value = initial_value
        self.gamma = gamma
        self.step_size = step_size
        self._state_attrs += ["initial_value", "gamma", "step_size"]

    def get_param(self) -> Union[List[float], float]:
        return self.initial_value * self.gamma ** (self.event_index // self.step_size)


class MultiStepStateScheduler(StateParamScheduler):
    """Update a parameter during training by using a multi step function.
    The function decays the parameter value by gamma once the number of steps reaches one of the milestones.
    Based on MultiStepLR from PyTorch.
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html

    Args:
        initial_value: Starting value of the parameter.
        gamma: Multiplicative factor of parameter value decay.
        milestones: List of step indices. Must be increasing.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        create_new: whether to create ``param_name`` on
            ``engine.state`` taking into account whether
            ``param_name`` attribute already exists or not.
            Overrides existing attribute by default, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            param_scheduler = MultiStepStateScheduler(
                param_name="param", initial_value=1, gamma=0.9, milestones=[3, 6, 9, 12], create_new=True
            )

            # parameter is param, initial_value sets param to 1, gamma is set as 0.9
            # Epoch 1 to 2, param does not change as milestone is 3
            # Epoch 3, param changes from 1 to 1*0.9, param = 0.9
            # Epoch 3 to 5, param does not change as milestone is 6
            # Epoch 6, param changes from 0.9 to 0.9*0.9, param = 0.81
            # Epoch 6 to 8, param does not change as milestone is 9
            # Epoch 9, param changes from 0.81 to 0.81*0.9, param = 0.729
            # Epoch 9 to 11, param does not change as milestone is 12
            # Epoch 12, param changes from 0.729 to 0.729*0.9, param = 0.6561

            param_scheduler.attach(default_trainer, Events.EPOCH_COMPLETED)

            @default_trainer.on(Events.EPOCH_COMPLETED)
            def print_param():
                print(default_trainer.state.param)

            default_trainer.run([0], max_epochs=12)

        .. testoutput::

            1.0
            1.0
            0.9
            0.9
            0.9
            0.81
            0.81
            0.81
            0.7290...
            0.7290...
            0.7290...
            0.6561

    .. versionadded:: 0.4.7

    """

    def __init__(
        self,
        initial_value: float,
        gamma: float,
        milestones: List[int],
        param_name: str,
        save_history: bool = False,
        create_new: bool = False,
    ):
        super(MultiStepStateScheduler, self).__init__(param_name, save_history, create_new)
        self.initial_value = initial_value
        self.gamma = gamma
        self.milestones = milestones
        self._state_attrs += ["initial_value", "gamma", "milestones"]

    def get_param(self) -> Union[List[float], float]:
        return self.initial_value * self.gamma ** bisect_right(self.milestones, self.event_index)

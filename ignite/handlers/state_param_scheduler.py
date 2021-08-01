import numbers
from bisect import bisect_right
from typing import Any, Callable, List, Tuple, Union

from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from ignite.handlers import BaseParamScheduler


class StateParamScheduler(BaseParamScheduler):
    """An abstract class for updating an engine state parameter values during training.

    Args:
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Note:
        Parameter scheduler works independently of the internal state of the attached engine.
        More precisely, whatever the state of the engine (newly created or used by another scheduler) the scheduler
        sets defined absolute values.

    .. versionadded:: 0.6.0

    """

    def __init__(
        self, param_name: str, save_history: bool = False,
    ):
        super(StateParamScheduler, self).__init__(param_name, save_history)

    def attach(
        self,
        engine: Engine,
        event: Union[str, Events, CallableEventWithFilter, EventsList] = Events.ITERATION_COMPLETED,
    ) -> None:
        """Attach the handler to engine. After the handler is attached, the ``Engine.state`` will add a new attribute
        with name ``param_name``. Then, current parameter value can be retrieved by from ``Engine.state`` when the
        engine runs.

        Args:
            engine: trainer to which the handler will be attached.
            event: trigger ``param_name`` value update.

        """
        if hasattr(engine.state, self.param_name):
            raise ValueError(
                f"Attribute: '{self.param_name}' is already in Engine.state and might be "
                f"overridden by other StateParameterScheduler handlers. Please select another name."
            )

        engine.add_event_handler(event, self)

    def __call__(self, engine: Engine) -> None:
        self.event_index += 1
        value = self.get_param()
        setattr(engine.state, self.param_name, value)
        if self.save_history and engine:
            if not hasattr(engine.state, "param_history") or engine.state.param_history is None:  # type: ignore
                setattr(engine.state, "param_history", {})
            engine.state.param_history.setdefault(self.param_name, [])  # type: ignore[attr-defined]
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
        keys_to_remove = ["save_history"]
        for key in keys_to_remove:
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
    """Update a parameter during training by using a user defined function.
        User defined function is taking an event index as input and returns parameter value.

    Args:
        lambda_fn: user defined parameter update function.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

        .. code-block:: python

            ...
            engine = Engine(train_step)

            initial_value = 10
            gamma = 0.99

            lambda_scheduler = LambdaStateScheduler(
                param_name="lambda",
                lambda_fn=lambda event_index: initial_value * gamma ** (event_index % 9),
            )

            lambda_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)

            # basic handler to print scheduled state parameter
            engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _ : print(engine.state.custom_scheduled_param))

            engine.run([0] * 8, max_epochs=2)

    .. versionadded:: 0.6.0

    """

    def __init__(self, lambda_fn: Callable, param_name: str, save_history: bool = False):
        super(LambdaStateScheduler, self).__init__(param_name, save_history)
        self.lambda_fn = lambda_fn
        self._state_attrs += ["lambda_fn"]

    def get_param(self) -> Union[List[float], float]:
        return self.lambda_fn(self.event_index)


class PwLinearStateScheduler(StateParamScheduler):
    """Piecewise linear state parameter scheduler.

    Args:
        milestones_values: list of tuples (event index, parameter value)
            represents milestones and parameter values. Milestones should be increasing integers.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

        .. code-block:: python

            ...
            engine = Engine(train_step)

            param_scheduler = PwLinearStateScheduler(
                param_name="pw_linear_scheduled_param",
                milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)]
            )

            pwlinear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)

            # basic handler to print scheduled state parameter
            engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _ : print(engine.state.pw_linear_scheduled_param))

            engine.run([0] * 8, max_epochs=40)
            #
            # Sets the state parameter value to 0.5 over the first 10 iterations, then decreases linearly from 0.5 to
            # 0.45 between 10th and 20th iterations. Next there is a jump to 0.3 at the 21st iteration and the state
            # parameter value decreases linearly from 0.3 to 0.1 between 21st and 30th iterations and remains 0.1 until
            # the end of the iterations.
            #

    .. versionadded:: 0.6.0
    """

    def __init__(
        self, milestones_values: List[Tuple[int, float]], param_name: str, save_history: bool = False,
    ):
        super(PwLinearStateScheduler, self).__init__(param_name, save_history)

        values = []  # type: List[float]
        milestones = []  # type: List[int]
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
    Based on the closed form of ExponentialLR from Pytorch
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html

    Args:
        initial_value: Starting value of the parameter.
        gamma: Multiplicative factor of parameter value decay.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

        .. code-block:: python

            ...
            engine = Engine(train_step)

            exp_state_parameter_scheduler = ExpStateScheduler(
                param_name="exp_scheduled_param", initial_value=10, gamma=0.99
            )

            # basic handler to print scheduled state parameter
            engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _ : print(engine.state.exp_scheduled_param))

            exp_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
            engine.run([0] * 8, max_epochs=2)

    .. versionadded:: 0.6.0

    """

    def __init__(
        self, initial_value: float, gamma: float, param_name: str, save_history: bool = False,
    ):
        super(ExpStateScheduler, self).__init__(param_name, save_history)
        self.initial_value = initial_value
        self.gamma = gamma
        self._state_attrs += ["initial_value", "gamma"]

    def get_param(self) -> Union[List[float], float]:
        return self.initial_value * self.gamma ** self.event_index


class StepStateScheduler(StateParamScheduler):
    """Update a parameter during training by using a step function.
    This function decays the parameter value by gamma every step_size.
    Based on StepLR from Pytorch.
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

    Args:
        initial_value: Starting value of the parameter.
        gamma: Multiplicative factor of parameter value decay.
        step_size: Period of parameter value decay.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

        .. code-block:: python

            ...
            engine = Engine(train_step)

            step_state_parameter_scheduler = StepStateScheduler(
                param_name="step_scheduled_param", initial_value=10, gamma=0.99, step_size=5
            )

            # basic handler to print scheduled state parameter
            engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _ : print(engine.state.step_scheduled_param))

            step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
            engine.run([0] * 8, max_epochs=10)

    .. versionadded:: 0.6.0

    """

    def __init__(
        self, initial_value: float, gamma: float, step_size: int, param_name: str, save_history: bool = False,
    ):
        super(StepStateScheduler, self).__init__(param_name, save_history)
        self.initial_value = initial_value
        self.gamma = gamma
        self.step_size = step_size
        self._state_attrs += ["initial_value", "gamma", "step_size"]

    def get_param(self) -> Union[List[float], float]:
        return self.initial_value * self.gamma ** (self.event_index // self.step_size)


class MultiStepStateScheduler(StateParamScheduler):
    """Update a parameter during training by using a multi step function.
    The function decays the parameter value by gamma once the number of steps reaches one of the milestones.
    Based on MultiStepLR from Pytorch.
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html

    Args:
        initial_value: Starting value of the parameter.
        gamma: Multiplicative factor of parameter value decay.
        milestones: List of step indices. Must be increasing.
        param_name: name of parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

        .. code-block:: python

            ...
            engine = Engine(train_step)

            multi_step_state_parameter_scheduler = MultiStepStateScheduler(
                param_name="multistep_scheduled_param", initial_value=10, gamma=0.99, milestones=[3, 6],
            )

            # basic handler to print scheduled state parameter
            engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _ : print(engine.state.multistep_scheduled_param))

            multi_step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
            engine.run([0] * 8, max_epochs=10)

    .. versionadded:: 0.6.0

    """

    def __init__(
        self, initial_value: float, gamma: float, milestones: List[int], param_name: str, save_history: bool = False,
    ):
        super(MultiStepStateScheduler, self).__init__(param_name, save_history)
        self.initial_value = initial_value
        self.gamma = gamma
        self.milestones = milestones
        self._state_attrs += ["initial_value", "gamma", "milestones"]

    def get_param(self) -> Union[List[float], float]:
        return self.initial_value * self.gamma ** bisect_right(self.milestones, self.event_index)

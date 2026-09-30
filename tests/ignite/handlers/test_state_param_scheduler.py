import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from packaging.version import Version

from ignite.engine import Engine, Events
from ignite.handlers.state_param_scheduler import (
    ExpStateScheduler,
    LambdaStateScheduler,
    MultiStepStateScheduler,
    PiecewiseLinearStateScheduler,
    StepStateScheduler,
)

config1 = (3, [(2, 0), (5, 10)], True, [0.0, 0.0, 3.3333333333333335])
expected_hist2 = [0.0] * 10 + [float(i) for i in range(1, 11)] + [10.0] * 10
config2 = (30, [(10, 0), (20, 10)], True, expected_hist2)
config3 = (
    PiecewiseLinearStateScheduler,
    {"param_name": "linear_scheduled_param", "milestones_values": [(3, 12), (5, 10)], "create_new": True},
)
config4 = (
    ExpStateScheduler,
    {"param_name": "exp_scheduled_param", "initial_value": 10, "gamma": 0.99, "create_new": True},
)
config5 = (
    MultiStepStateScheduler,
    {
        "param_name": "multistep_scheduled_param",
        "initial_value": 10,
        "gamma": 0.99,
        "milestones": [3, 6],
        "create_new": True,
    },
)

if Version(torch.__version__) < Version("1.9.0"):
    torch_testing_assert_close = torch.testing.assert_allclose
else:
    torch_testing_assert_close = torch.testing.assert_close


class LambdaState:
    def __init__(self, initial_value, gamma):
        self.initial_value = initial_value
        self.gamma = gamma

    def __call__(self, event_index):
        return self.initial_value * self.gamma ** (event_index % 9)


config6 = (
    LambdaStateScheduler,
    {
        "param_name": "custom_scheduled_param",
        "lambda_obj": LambdaState(initial_value=10, gamma=0.99),
        "create_new": True,
    },
)

config7 = (
    StepStateScheduler,
    {"param_name": "step_scheduled_param", "initial_value": 10, "gamma": 0.99, "step_size": 5, "create_new": True},
)


@pytest.mark.parametrize("max_epochs, milestones_values,  save_history, expected_param_history", [config1, config2])
def test_pwlinear_scheduler_linear_increase_history(
    max_epochs, milestones_values, save_history, expected_param_history
):
    # Testing linear increase
    engine = Engine(lambda e, b: None)
    pw_linear_step_parameter_scheduler = PiecewiseLinearStateScheduler(
        param_name="pwlinear_scheduled_param",
        milestones_values=milestones_values,
        save_history=save_history,
        create_new=True,
    )
    pw_linear_step_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    expected_param_history = expected_param_history
    assert hasattr(engine.state, "param_history")
    state_param = engine.state.param_history["pwlinear_scheduled_param"]
    assert len(state_param) == len(expected_param_history)
    assert state_param == expected_param_history

    state_dict = pw_linear_step_parameter_scheduler.state_dict()
    pw_linear_step_parameter_scheduler.load_state_dict(state_dict)


@pytest.mark.parametrize("max_epochs, milestones_values", [(3, [(3, 12), (5, 10)]), (5, [(10, 12), (20, 10)])])
def test_pwlinear_scheduler_step_constant(max_epochs, milestones_values):
    # Testing step_constant
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = PiecewiseLinearStateScheduler(
        param_name="pwlinear_scheduled_param", milestones_values=milestones_values, create_new=True
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch_testing_assert_close(getattr(engine.state, "pwlinear_scheduled_param"), float(milestones_values[0][1]))

    state_dict = linear_state_parameter_scheduler.state_dict()
    linear_state_parameter_scheduler.load_state_dict(state_dict)


@pytest.mark.parametrize(
    "max_epochs, milestones_values, expected_val",
    [(2, [(0, 0), (3, 10)], 6.666666666666667), (10, [(0, 0), (20, 10)], 5.0)],
)
def test_pwlinear_scheduler_linear_increase(max_epochs, milestones_values, expected_val):
    # Testing linear increase
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = PiecewiseLinearStateScheduler(
        param_name="pwlinear_scheduled_param", milestones_values=milestones_values, create_new=True
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch_testing_assert_close(getattr(engine.state, "pwlinear_scheduled_param"), expected_val, atol=0.001, rtol=0.0)

    state_dict = linear_state_parameter_scheduler.state_dict()
    linear_state_parameter_scheduler.load_state_dict(state_dict)


@pytest.mark.parametrize("max_epochs, milestones_values,", [(3, [(0, 0), (3, 10)]), (40, [(0, 0), (20, 10)])])
def test_pwlinear_scheduler_max_value(max_epochs, milestones_values):
    # Testing max_value
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = PiecewiseLinearStateScheduler(
        param_name="linear_scheduled_param", milestones_values=milestones_values, create_new=True
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch_testing_assert_close(getattr(engine.state, "linear_scheduled_param"), float(milestones_values[-1][1]))

    state_dict = linear_state_parameter_scheduler.state_dict()
    linear_state_parameter_scheduler.load_state_dict(state_dict)


def test_piecewiselinear_asserts():
    with pytest.raises(TypeError, match=r"Argument milestones_values should be a list or tuple"):
        PiecewiseLinearStateScheduler(param_name="linear_scheduled_param", milestones_values=None)

    with pytest.raises(ValueError, match=r"Argument milestones_values should be with at least one value"):
        PiecewiseLinearStateScheduler(param_name="linear_scheduled_param", milestones_values=[])

    with pytest.raises(ValueError, match=r"Argument milestones_values should be a list of pairs"):
        PiecewiseLinearStateScheduler(param_name="linear_scheduled_param", milestones_values=[(0.5,)])

    with pytest.raises(ValueError, match=r"Argument milestones_values should be a list of pairs"):
        PiecewiseLinearStateScheduler(param_name="linear_scheduled_param", milestones_values=[(10, 0.5), (0.6,)])

    with pytest.raises(ValueError, match=r"Milestones should be increasing integers"):
        PiecewiseLinearStateScheduler(param_name="linear_scheduled_param", milestones_values=[(10, 0.5), (5, 0.6)])

    with pytest.raises(TypeError, match=r"Value of a milestone should be integer"):
        PiecewiseLinearStateScheduler(param_name="linear_scheduled_param", milestones_values=[(0.5, 1)])


@pytest.mark.parametrize("max_epochs, initial_value, gamma", [(3, 10, 0.99), (40, 5, 0.98)])
def test_exponential_scheduler(max_epochs, initial_value, gamma):
    engine = Engine(lambda e, b: None)
    exp_state_parameter_scheduler = ExpStateScheduler(
        param_name="exp_scheduled_param", initial_value=initial_value, gamma=gamma, create_new=True
    )
    exp_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch_testing_assert_close(getattr(engine.state, "exp_scheduled_param"), initial_value * gamma**max_epochs)

    state_dict = exp_state_parameter_scheduler.state_dict()
    exp_state_parameter_scheduler.load_state_dict(state_dict)


@pytest.mark.parametrize("max_epochs, initial_value, gamma, step_size", [(3, 10, 0.99, 5), (40, 5, 0.98, 22)])
def test_step_scheduler(max_epochs, initial_value, gamma, step_size):
    engine = Engine(lambda e, b: None)
    step_state_parameter_scheduler = StepStateScheduler(
        param_name="step_scheduled_param",
        initial_value=initial_value,
        gamma=gamma,
        step_size=step_size,
        create_new=True,
    )
    step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch_testing_assert_close(
        getattr(engine.state, "step_scheduled_param"), initial_value * gamma ** (max_epochs // step_size)
    )

    state_dict = step_state_parameter_scheduler.state_dict()
    step_state_parameter_scheduler.load_state_dict(state_dict)


from bisect import bisect_right


@pytest.mark.parametrize(
    "max_epochs, initial_value, gamma, milestones", [(3, 10, 0.99, [3, 6]), (40, 5, 0.98, [3, 6, 9, 10, 11])]
)
def test_multistep_scheduler(max_epochs, initial_value, gamma, milestones):
    engine = Engine(lambda e, b: None)
    multi_step_state_parameter_scheduler = MultiStepStateScheduler(
        param_name="multistep_scheduled_param",
        initial_value=initial_value,
        gamma=gamma,
        milestones=milestones,
        create_new=True,
    )
    multi_step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch_testing_assert_close(
        getattr(engine.state, "multistep_scheduled_param"),
        initial_value * gamma ** bisect_right(milestones, max_epochs),
    )

    state_dict = multi_step_state_parameter_scheduler.state_dict()
    multi_step_state_parameter_scheduler.load_state_dict(state_dict)


def test_custom_scheduler():
    engine = Engine(lambda e, b: None)

    class LambdaState:
        def __init__(self, initial_value, gamma):
            self.initial_value = initial_value
            self.gamma = gamma

        def __call__(self, event_index):
            return self.initial_value * self.gamma ** (event_index % 9)

    lambda_state_parameter_scheduler = LambdaStateScheduler(
        param_name="custom_scheduled_param", lambda_obj=LambdaState(initial_value=10, gamma=0.99), create_new=True
    )
    lambda_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=2)
    torch_testing_assert_close(
        getattr(engine.state, "custom_scheduled_param"), LambdaState(initial_value=10, gamma=0.99)(2)
    )
    engine.run([0] * 8, max_epochs=20)
    torch_testing_assert_close(
        getattr(engine.state, "custom_scheduled_param"), LambdaState(initial_value=10, gamma=0.99)(20)
    )

    state_dict = lambda_state_parameter_scheduler.state_dict()
    lambda_state_parameter_scheduler.load_state_dict(state_dict)


def test_custom_scheduler_asserts():
    class LambdaState:
        def __init__(self, initial_value, gamma):
            self.initial_value = initial_value
            self.gamma = gamma

    with pytest.raises(ValueError, match=r"Expected lambda_obj to be callable."):
        lambda_state_parameter_scheduler = LambdaStateScheduler(
            param_name="custom_scheduled_param", lambda_obj=LambdaState(initial_value=10, gamma=0.99), create_new=True
        )


@pytest.mark.parametrize("scheduler_cls, scheduler_kwargs", [config3, config4, config5, config6])
def test_simulate_and_plot_values(scheduler_cls, scheduler_kwargs):
    import matplotlib

    matplotlib.use("Agg")

    event = Events.EPOCH_COMPLETED
    max_epochs = 2
    data = [0] * 10

    scheduler = scheduler_cls(**scheduler_kwargs)
    trainer = Engine(lambda engine, batch: None)
    scheduler.attach(trainer, event)
    trainer.run(data, max_epochs=max_epochs)

    # launch plot values
    scheduler_cls.plot_values(num_events=len(data) * max_epochs, **scheduler_kwargs)


@pytest.mark.parametrize("save_history", [False, True])
@pytest.mark.parametrize("scheduler_cls, scheduler_kwargs", [config3, config4, config5, config6])
def test_simulate_values(scheduler_cls, scheduler_kwargs, save_history):
    max_epochs = 2
    data = [0] * 10
    scheduler_kwargs["save_history"] = save_history
    scheduler_cls.simulate_values(num_events=len(data) * max_epochs, **scheduler_kwargs)


def test_torch_save_load(dirname):
    lambda_state_parameter_scheduler = LambdaStateScheduler(
        param_name="custom_scheduled_param", lambda_obj=LambdaState(initial_value=10, gamma=0.99), create_new=True
    )

    filepath = Path(dirname) / "dummy_lambda_state_parameter_scheduler.pt"
    torch.save(lambda_state_parameter_scheduler, filepath)
    if Version(torch.__version__) >= Version("1.13.0"):
        kwargs = {"weights_only": False}
    else:
        kwargs = {}
    loaded_lambda_state_parameter_scheduler = torch.load(filepath, **kwargs)

    engine1 = Engine(lambda e, b: None)
    lambda_state_parameter_scheduler.attach(engine1, Events.EPOCH_COMPLETED)
    engine1.run([0] * 8, max_epochs=2)
    torch_testing_assert_close(
        getattr(engine1.state, "custom_scheduled_param"), LambdaState(initial_value=10, gamma=0.99)(2)
    )

    engine2 = Engine(lambda e, b: None)
    loaded_lambda_state_parameter_scheduler.attach(engine2, Events.EPOCH_COMPLETED)
    engine2.run([0] * 8, max_epochs=2)
    torch_testing_assert_close(
        getattr(engine2.state, "custom_scheduled_param"), LambdaState(initial_value=10, gamma=0.99)(2)
    )
    torch_testing_assert_close(
        getattr(engine1.state, "custom_scheduled_param"), getattr(engine2.state, "custom_scheduled_param")
    )


def test_simulate_and_plot_values_no_matplotlib():
    with pytest.raises(ModuleNotFoundError, match=r"This method requires matplotlib to be installed."):
        with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
            event = Events.EPOCH_COMPLETED
            max_epochs = 2
            data = [0] * 10

            kwargs = {
                "param_name": "multistep_scheduled_param",
                "initial_value": 10,
                "gamma": 0.99,
                "milestones": [3, 6],
                "create_new": True,
            }
            scheduler = MultiStepStateScheduler(**kwargs)
            trainer = Engine(lambda engine, batch: None)
            scheduler.attach(trainer, event)
            trainer.run(data, max_epochs=max_epochs)

            # launch plot values
            MultiStepStateScheduler.plot_values(num_events=len(data) * max_epochs, **kwargs)


def test_multiple_scheduler_with_save_history():
    engine_multiple_schedulers = Engine(lambda e, b: None)
    configs = [config3, config4, config5, config6, config7]
    for scheduler, config in configs:
        if "save_history" in config:
            del config["save_history"]
        _scheduler = scheduler(**config, save_history=True)
        _scheduler.attach(engine_multiple_schedulers)

    engine_multiple_schedulers.run([0] * 8, max_epochs=2)

    for scheduler, config in configs:
        engine = Engine(lambda e, b: None)
        _scheduler = scheduler(**config, save_history=True)
        _scheduler.attach(engine)
        engine.run([0] * 8, max_epochs=2)
        torch_testing_assert_close(
            engine_multiple_schedulers.state.param_history[config["param_name"]],
            engine.state.param_history[config["param_name"]],
        )


def test_docstring_examples():
    # LambdaStateScheduler

    engine = Engine(lambda e, b: None)

    class LambdaState:
        def __init__(self, initial_value, gamma):
            self.initial_value = initial_value
            self.gamma = gamma

        def __call__(self, event_index):
            return self.initial_value * self.gamma ** (event_index % 9)

    param_scheduler = LambdaStateScheduler(param_name="param", lambda_obj=LambdaState(10, 0.99), create_new=True)

    param_scheduler.attach(engine, Events.EPOCH_COMPLETED)

    engine.run([0] * 8, max_epochs=2)

    # PiecewiseLinearStateScheduler

    engine = Engine(lambda e, b: None)

    param_scheduler = PiecewiseLinearStateScheduler(
        param_name="param", milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)], create_new=True
    )

    param_scheduler.attach(engine, Events.EPOCH_COMPLETED)

    engine.run([0] * 8, max_epochs=40)

    # ExpStateScheduler

    engine = Engine(lambda e, b: None)

    param_scheduler = ExpStateScheduler(param_name="param", initial_value=10, gamma=0.99, create_new=True)

    param_scheduler.attach(engine, Events.EPOCH_COMPLETED)

    engine.run([0] * 8, max_epochs=2)

    # StepStateScheduler

    engine = Engine(lambda e, b: None)

    param_scheduler = StepStateScheduler(param_name="param", initial_value=10, gamma=0.99, step_size=5, create_new=True)

    param_scheduler.attach(engine, Events.EPOCH_COMPLETED)

    engine.run([0] * 8, max_epochs=10)

    # MultiStepStateScheduler

    engine = Engine(lambda e, b: None)

    param_scheduler = MultiStepStateScheduler(
        param_name="param", initial_value=10, gamma=0.99, milestones=[3, 6], create_new=True
    )

    param_scheduler.attach(engine, Events.EPOCH_COMPLETED)

    engine.run([0] * 8, max_epochs=10)


def test_param_scheduler_attach_exception():
    trainer = Engine(lambda e, b: None)
    param_name = "state_param"

    setattr(trainer.state, param_name, None)

    save_history = True
    create_new = True

    param_scheduler = PiecewiseLinearStateScheduler(
        param_name=param_name,
        milestones_values=[(0, 0.0), (10, 0.999)],
        save_history=save_history,
        create_new=create_new,
    )

    with pytest.raises(
        ValueError,
        match=r"Attribute '" + re.escape(param_name) + "' already exists in the engine.state. "
        r"This may be a conflict between multiple handlers. "
        r"Please choose another name.",
    ):
        param_scheduler.attach(trainer, Events.ITERATION_COMPLETED)


def test_param_scheduler_attach_warning():
    trainer = Engine(lambda e, b: None)
    param_name = "state_param"
    save_history = True
    create_new = False

    param_scheduler = PiecewiseLinearStateScheduler(
        param_name=param_name,
        milestones_values=[(0, 0.0), (10, 0.999)],
        save_history=save_history,
        create_new=create_new,
    )

    with pytest.warns(
        UserWarning,
        match=r"Attribute '" + re.escape(param_name) + "' is not defined in the engine.state. "
        r"PiecewiseLinearStateScheduler will create it. Remove this warning by setting create_new=True.",
    ):
        param_scheduler.attach(trainer, Events.ITERATION_COMPLETED)


def test_param_scheduler_with_ema_handler():
    from ignite.handlers import EMAHandler

    model = nn.Linear(2, 1)
    trainer = Engine(lambda e, b: model(b))
    data = torch.rand(100, 2)

    param_name = "ema_decay"

    ema_handler = EMAHandler(model)
    ema_handler.attach(trainer, name=param_name, event=Events.ITERATION_COMPLETED)

    ema_decay_scheduler = PiecewiseLinearStateScheduler(
        param_name=param_name, milestones_values=[(0, 0.0), (10, 0.999)], save_history=True
    )
    ema_decay_scheduler.attach(trainer, Events.ITERATION_COMPLETED)
    trainer.run(data, max_epochs=20)

from unittest.mock import patch

import pytest
import torch

from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import (
    ExponentialStateParameterScheduler,
    LambdaStateParameterScheduler,
    LinearStateParameterScheduler,
    MultiStepStateParameterScheduler,
    StepStateParameterScheduler,
)


def test_linear_scheduler_linear_increase_history():
    # Testing linear increase
    engine = Engine(lambda e, b: None)
    linear_step_parameter_scheduler = LinearStateParameterScheduler(
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
        save_history=True,
    )
    linear_step_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=3)
    expected_param_history = [0.0, 0.0, 3.3333333333333335]
    assert hasattr(engine.state, "param_history")
    state_param = engine.state.param_history["linear_scheduled_param"]
    assert len(state_param) == len(expected_param_history)
    assert state_param == expected_param_history


def test_linear_scheduler_step_constant():
    # Testing step_constant
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = LinearStateParameterScheduler(
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
        save_history=True,
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(getattr(engine.state, "linear_scheduled_param"), 0.0)


def test_linear_scheduler_linear_increase():
    # Testing linear increase
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = LinearStateParameterScheduler(
        param_name="linear_scheduled_param", initial_value=0, step_constant=2, step_max_value=5, max_value=10,
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=3)
    torch.testing.assert_allclose(getattr(engine.state, "linear_scheduled_param"), 3.333333, atol=0.001, rtol=0.0)


def test_linear_scheduler_max_value():
    # Testing max_value
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = LinearStateParameterScheduler(
        param_name="linear_scheduled_param", initial_value=0, step_constant=2, step_max_value=5, max_value=10,
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(getattr(engine.state, "linear_scheduled_param"), 10)


def test_exponential_scheduler():
    engine = Engine(lambda e, b: None)
    exp_state_parameter_scheduler = ExponentialStateParameterScheduler(
        param_name="exp_scheduled_param", initial_value=10, gamma=0.99
    )
    exp_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(getattr(engine.state, "exp_scheduled_param"), 10 * 0.99 * 0.99)


def test_step_scheduler():
    engine = Engine(lambda e, b: None)
    step_state_parameter_scheduler = StepStateParameterScheduler(
        param_name="step_scheduled_param", initial_value=10, gamma=0.99, step_size=5
    )
    step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(getattr(engine.state, "step_scheduled_param"), 10 * 0.99 * 0.99)


def test_multistep_scheduler():
    engine = Engine(lambda e, b: None)
    multi_step_state_parameter_scheduler = MultiStepStateParameterScheduler(
        param_name="multistep_scheduled_param", initial_value=10, gamma=0.99, milestones=[3, 6],
    )
    multi_step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(getattr(engine.state, "multistep_scheduled_param"), 10 * 0.99 * 0.99)


def test_custom_scheduler():

    initial_value = 10
    gamma = 0.99
    engine = Engine(lambda e, b: None)
    lambda_state_parameter_scheduler = LambdaStateParameterScheduler(
        param_name="custom_scheduled_param", lambda_fn=lambda event_index: initial_value * gamma ** (event_index % 9),
    )
    lambda_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(getattr(engine.state, "custom_scheduled_param"), 10 * 0.99 * 0.99)


def test_simulate_and_plot_values():

    import matplotlib

    matplotlib.use("Agg")

    def _test(scheduler_cls, **scheduler_kwargs):
        event = Events.EPOCH_COMPLETED
        max_epochs = 2
        data = [0] * 10

        scheduler = scheduler_cls(**scheduler_kwargs)
        trainer = Engine(lambda engine, batch: None)
        scheduler.attach(trainer, event)
        trainer.run(data, max_epochs=max_epochs)

        # launch plot values
        scheduler_cls.plot_values(num_events=len(data) * max_epochs, **scheduler_kwargs)

    # LinearStateParameterScheduler
    _test(
        LinearStateParameterScheduler,
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
    )

    # ExponentialStateParameterScheduler
    _test(ExponentialStateParameterScheduler, param_name="exp_scheduled_param", initial_value=10, gamma=0.99)

    # StepStateParameterScheduler
    _test(StepStateParameterScheduler, param_name="step_scheduled_param", initial_value=10, gamma=0.99, step_size=5)

    # MultiStepStateParameterScheduler
    _test(
        MultiStepStateParameterScheduler,
        param_name="multistep_scheduled_param",
        initial_value=10,
        gamma=0.99,
        milestones=[3, 6],
    )

    # LambdaStateParameterScheduler
    initial_value = 10
    gamma = 0.99
    _test(
        LambdaStateParameterScheduler,
        param_name="custom_scheduled_param",
        lambda_fn=lambda event_index: initial_value * gamma ** (event_index % 9),
    )

    with pytest.raises(RuntimeError, match=r"This method requires matplotlib to be installed."):
        with patch.dict("sys.modules", {"matplotlib.pylab": None}):
            _test(
                MultiStepStateParameterScheduler,
                param_name="multistep_scheduled_param",
                initial_value=10,
                gamma=0.99,
                milestones=[3, 6],
            )

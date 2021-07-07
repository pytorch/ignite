import torch
from torch.nn import Module

from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import (
    ExponentialStateParameterScheduler,
    LambdaStateParameterScheduler,
    LinearStateParameterScheduler,
    MultiStepStateParameterScheduler,
    StepStateParameterScheduler,
    _get_fake_param_setter,
)


def test_fake_param_setter():
    param_setter = _get_fake_param_setter()
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 0.0)
    param_setter(5.0)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 5.0)


def test_linear_scheduler_linear_increase_history():
    # Testing linear increase
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearStateParameterScheduler(
        parameter_setter=param_setter,
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
        save_history=True,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=3)
    print(engine.state.param_history)
    expected_param_history = [0.0, 0.0, 3.3333333333333335]
    assert hasattr(engine.state, "param_history")
    state_param = engine.state.param_history["linear_scheduled_param"]
    assert len(state_param) == len(expected_param_history)
    assert state_param == expected_param_history


def test_linear_scheduler_step_constant():
    # Testing step_constant
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearStateParameterScheduler(
        parameter_setter=param_setter,
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
        save_history=True,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 0)


def test_linear_scheduler_linear_increase():
    # Testing linear increase
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearStateParameterScheduler(
        parameter_setter=param_setter,
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=3)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 3.333333, atol=0.001, rtol=0.0)


def test_linear_scheduler_max_value():
    # Testing max_value
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearStateParameterScheduler(
        parameter_setter=param_setter,
        param_name="linear_scheduled_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 10)


def test_exponential_scheduler():
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    exp_any_parameter_scheduler = ExponentialStateParameterScheduler(
        parameter_setter=param_setter, param_name="exp_scheduled_param", initial_value=10, gamma=0.99
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, exp_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 10 * 0.99 * 0.99)


def test_step_scheduler():
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    step_any_parameter_scheduler = StepStateParameterScheduler(
        parameter_setter=param_setter, param_name="step_scheduled_param", initial_value=10, gamma=0.99, step_size=5
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, step_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 10 * 0.99 * 0.99)


def test_multistep_scheduler():
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    multi_step_any_parameter_scheduler = MultiStepStateParameterScheduler(
        parameter_setter=param_setter,
        param_name="multistep_scheduled_param",
        initial_value=10,
        gamma=0.99,
        milestones=[3, 6],
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, multi_step_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 10 * 0.99 * 0.99)


def test_custom_scheduler():
    # def custom_logic(initial_value, gamma, current_step):
    #    return initial_value * gamma ** (current_step % 9)

    initial_value = 10
    gamma = 0.99
    param_setter = _get_fake_param_setter()
    engine = Engine(lambda e, b: None)
    lambda_any_parameter_scheduler = LambdaStateParameterScheduler(
        parameter_setter=param_setter,
        param_name="custom_scheduled_param",
        lambda_fn=lambda event_index: initial_value * gamma ** (event_index % 9),
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, lambda_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(param_setter.__closure__[0].cell_contents, 10 * 0.99 * 0.99)

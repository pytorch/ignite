import torch
from torch.nn import Module

from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import (
    ExponentialAnyParameterScheduler,
    LambdaAnyParameterScheduler,
    LinearAnyParameterScheduler,
    MultiStepAnyParameterScheduler,
    StepAnyParameterScheduler,
)


class ToyNet(Module):
    def __init__(self, value):
        super(ToyNet, self).__init__()
        self.value = value

    def forward(self, input):
        return input

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


def test_linear_scheduler_linear_increase_history():
    # Testing linear increase
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearAnyParameterScheduler(
        parameter_setter=net.set_value,
        param_name="toy_net_param",
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
    state_param = engine.state.param_history["toy_net_param"]
    assert len(state_param) == len(expected_param_history)
    assert state_param == expected_param_history


def test_linear_scheduler_step_constant():
    # Testing step_constant
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearAnyParameterScheduler(
        parameter_setter=net.set_value,
        param_name="toy_net_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
        save_history=True,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(net.get_value(), 0)


def test_linear_scheduler_linear_increase():
    # Testing linear increase
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearAnyParameterScheduler(
        parameter_setter=net.set_value,
        param_name="toy_net_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=3)
    torch.testing.assert_allclose(net.get_value(), 3.333333, atol=0.001, rtol=0.0)


def test_linear_scheduler_max_value():
    # Testing max_value
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    linear_any_parameter_scheduler = LinearAnyParameterScheduler(
        parameter_setter=net.set_value,
        param_name="toy_net_param",
        initial_value=0,
        step_constant=2,
        step_max_value=5,
        max_value=10,
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, linear_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(net.get_value(), 10)


def test_exponential_scheduler():
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    exp_any_parameter_scheduler = ExponentialAnyParameterScheduler(
        parameter_setter=net.set_value, param_name="toy_net_param", initial_value=10, gamma=0.99
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, exp_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(net.get_value(), 10 * 0.99 * 0.99)


def test_step_scheduler():
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    step_any_parameter_scheduler = StepAnyParameterScheduler(
        parameter_setter=net.set_value, param_name="toy_net_param", initial_value=10, gamma=0.99, step_size=5
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, step_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(net.get_value(), 10 * 0.99 * 0.99)


def test_multistep_scheduler():
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    multi_step_any_parameter_scheduler = MultiStepAnyParameterScheduler(
        parameter_setter=net.set_value, param_name="toy_net_param", initial_value=10, gamma=0.99, milestones=[3, 6],
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, multi_step_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=10)
    torch.testing.assert_allclose(net.get_value(), 10 * 0.99 * 0.99)


def test_custom_scheduler():
    # def custom_logic(initial_value, gamma, current_step):
    #    return initial_value * gamma ** (current_step % 9)

    initial_value = 10
    gamma = 0.99
    net = ToyNet(value=-1)
    engine = Engine(lambda e, b: None)
    lambda_any_parameter_scheduler = LambdaAnyParameterScheduler(
        parameter_setter=net.set_value,
        param_name="toy_net_param",
        lambda_fn=lambda event_index: initial_value * gamma ** (event_index % 9),
    )
    engine.add_event_handler(Events.EPOCH_COMPLETED, lambda_any_parameter_scheduler)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(net.get_value(), 10 * 0.99 * 0.99)

from unittest.mock import patch

import pytest
import torch

from ignite.engine import Engine, Events
from ignite.handlers.state_param_scheduler import (
    ExpStateScheduler,
    LambdaStateScheduler,
    LinearStateScheduler,
    MultiStepStateScheduler,
    StepStateScheduler,
)


@pytest.mark.parametrize(
    "max_epochs, initial_value, step_constant, step_max_value, max_value, save_history, expected_param_history",
    [
        (3, 0, 2, 5, 10, True, [0.0, 0.0, 3.3333333333333335]),
        (
            30,
            0,
            10,
            20,
            10,
            True,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ],
        ),
    ],
)
def test_linear_scheduler_linear_increase_history(
    max_epochs, initial_value, step_constant, step_max_value, max_value, save_history, expected_param_history
):
    # Testing linear increase
    engine = Engine(lambda e, b: None)
    linear_step_parameter_scheduler = LinearStateScheduler(
        param_name="linear_scheduled_param",
        initial_value=initial_value,
        step_constant=step_constant,
        step_max_value=step_max_value,
        max_value=max_value,
        save_history=save_history,
    )
    linear_step_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    expected_param_history = expected_param_history
    assert hasattr(engine.state, "param_history")
    state_param = engine.state.param_history["linear_scheduled_param"]
    assert len(state_param) == len(expected_param_history)
    assert state_param == expected_param_history


@pytest.mark.parametrize(
    "max_epochs, initial_value, step_constant, step_max_value, max_value, save_history",
    [(3, 12, 2, 5, 10, True), (30, 12, 10, 20, 10, True),],
)
def test_linear_scheduler_step_constant(
    max_epochs, initial_value, step_constant, step_max_value, max_value, save_history
):
    # Testing step_constant
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = LinearStateScheduler(
        param_name="linear_scheduled_param",
        initial_value=initial_value,
        step_constant=max_epochs,
        step_max_value=step_max_value,
        max_value=max_value,
        save_history=True,
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch.testing.assert_allclose(getattr(engine.state, "linear_scheduled_param"), initial_value)


@pytest.mark.parametrize(
    "max_epochs, initial_value, step_constant, step_max_value, max_value, save_history, expected_val",
    [(2, 0, 0, 3, 10, True, 6.666666666666667), (10, 0, 0, 20, 10, True, 5.0),],
)
def test_linear_scheduler_linear_increase(
    max_epochs, initial_value, step_constant, step_max_value, max_value, save_history, expected_val
):
    # Testing linear increase
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = LinearStateScheduler(
        param_name="linear_scheduled_param",
        initial_value=initial_value,
        step_constant=step_constant,
        step_max_value=step_max_value,
        max_value=max_value,
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch.testing.assert_allclose(getattr(engine.state, "linear_scheduled_param"), expected_val, atol=0.001, rtol=0.0)


@pytest.mark.parametrize(
    "max_epochs, initial_value, step_constant, step_max_value, max_value, save_history",
    [(3, 0, 0, 3, 10, True), (40, 0, 0, 20, 10, True),],
)
def test_linear_scheduler_max_value(
    max_epochs, initial_value, step_constant, step_max_value, max_value, save_history,
):
    # Testing max_value
    engine = Engine(lambda e, b: None)
    linear_state_parameter_scheduler = LinearStateScheduler(
        param_name="linear_scheduled_param",
        initial_value=initial_value,
        step_constant=step_constant,
        step_max_value=step_max_value,
        max_value=max_value,
    )
    linear_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch.testing.assert_allclose(getattr(engine.state, "linear_scheduled_param"), max_value)


@pytest.mark.parametrize(
    "max_epochs, initial_value, gamma, save_history", [(3, 10, 0.99, True), (40, 5, 0.98, True)],
)
def test_exponential_scheduler(
    max_epochs, initial_value, gamma, save_history,
):
    engine = Engine(lambda e, b: None)
    exp_state_parameter_scheduler = ExpStateScheduler(
        param_name="exp_scheduled_param", initial_value=initial_value, gamma=gamma
    )
    exp_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch.testing.assert_allclose(getattr(engine.state, "exp_scheduled_param"), initial_value * gamma ** max_epochs)


@pytest.mark.parametrize(
    "max_epochs, initial_value, gamma, step_size, save_history", [(3, 10, 0.99, 5, True), (40, 5, 0.98, 22, True)],
)
def test_step_scheduler(
    max_epochs, initial_value, gamma, step_size, save_history,
):
    engine = Engine(lambda e, b: None)
    step_state_parameter_scheduler = StepStateScheduler(
        param_name="step_scheduled_param", initial_value=initial_value, gamma=gamma, step_size=step_size
    )
    step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch.testing.assert_allclose(
        getattr(engine.state, "step_scheduled_param"), initial_value * gamma ** (max_epochs // step_size)
    )


from bisect import bisect_right


@pytest.mark.parametrize(
    "max_epochs, initial_value, gamma, milestones, save_history",
    [(3, 10, 0.99, [3, 6], True), (40, 5, 0.98, [3, 6, 9, 10, 11], True)],
)
def test_multistep_scheduler(
    max_epochs, initial_value, gamma, milestones, save_history,
):
    engine = Engine(lambda e, b: None)
    multi_step_state_parameter_scheduler = MultiStepStateScheduler(
        param_name="multistep_scheduled_param", initial_value=initial_value, gamma=gamma, milestones=milestones,
    )
    multi_step_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=max_epochs)
    torch.testing.assert_allclose(
        getattr(engine.state, "multistep_scheduled_param"),
        initial_value * gamma ** bisect_right(milestones, max_epochs),
    )


def test_custom_scheduler():

    initial_value = 10
    gamma = 0.99
    engine = Engine(lambda e, b: None)
    lambda_fn = lambda event_index: initial_value * gamma ** (event_index % 9)
    lambda_state_parameter_scheduler = LambdaStateScheduler(param_name="custom_scheduled_param", lambda_fn=lambda_fn,)
    lambda_state_parameter_scheduler.attach(engine, Events.EPOCH_COMPLETED)
    engine.run([0] * 8, max_epochs=2)
    torch.testing.assert_allclose(getattr(engine.state, "custom_scheduled_param"), lambda_fn(2))
    engine.run([0] * 8, max_epochs=20)
    torch.testing.assert_allclose(getattr(engine.state, "custom_scheduled_param"), lambda_fn(20))


@pytest.mark.parametrize(
    "scheduler_cls,scheduler_kwargs",
    [
        (
            LinearStateScheduler,
            {
                "param_name": "linear_scheduled_param",
                "initial_value": 0,
                "step_constant": 2,
                "step_max_value": 5,
                "max_value": 10,
            },
        ),
        (ExpStateScheduler, {"param_name": "exp_scheduled_param", "initial_value": 10, "gamma": 0.99}),
        (
            StepStateScheduler,
            {"param_name": "step_scheduled_param", "initial_value": 10, "gamma": 0.99, "step_size": 5},
        ),
        (
            MultiStepStateScheduler,
            {"param_name": "multistep_scheduled_param", "initial_value": 10, "gamma": 0.99, "milestones": [3, 6]},
        ),
        (
            LambdaStateScheduler,
            {"param_name": "custom_scheduled_param", "lambda_fn": lambda event_index: 10 * 0.99 ** (event_index % 9),},
        ),
    ],
)
def test_simulate_and_plot_values(scheduler_cls, scheduler_kwargs):

    import matplotlib

    matplotlib.use("Agg")

    def _test(scheduler_cls, scheduler_kwargs):
        event = Events.EPOCH_COMPLETED
        max_epochs = 2
        data = [0] * 10

        scheduler = scheduler_cls(**scheduler_kwargs)
        trainer = Engine(lambda engine, batch: None)
        scheduler.attach(trainer, event)
        trainer.run(data, max_epochs=max_epochs)

        # launch plot values
        scheduler_cls.plot_values(num_events=len(data) * max_epochs, **scheduler_kwargs)

    _test(scheduler_cls, scheduler_kwargs)


def test_simulate_and_plot_values_no_matplotlib():
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

    with pytest.raises(RuntimeError, match=r"This method requires matplotlib to be installed."):
        with patch.dict("sys.modules", {"matplotlib.pylab": None}):
            _test(
                MultiStepStateScheduler,
                param_name="multistep_scheduled_param",
                initial_value=10,
                gamma=0.99,
                milestones=[3, 6],
            )

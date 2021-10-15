from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    LinearCyclicalScheduler,
    LRScheduler,
    ParamGroupScheduler,
    ParamScheduler,
    PiecewiseLinear,
    create_lr_scheduler_with_warmup,
)
from tests.ignite.contrib.handlers import MockFP16DeepSpeedZeroOptimizer

try:
    from torch.optim.lr_scheduler import MultiplicativeLR
except ImportError:
    has_multiplicative_lr = False
else:
    from distutils.version import LooseVersion

    # https://github.com/pytorch/pytorch/issues/32756
    has_multiplicative_lr = LooseVersion(torch.__version__) >= LooseVersion("1.5.0")


class FakeParamScheduler(ParamScheduler):
    def get_param(self):
        return [0]


def test_param_scheduler_asserts():

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([{"params": t1, "lr": 0.1}, {"params": t2, "lr": 0.1}])

    lr_scheduler = FakeParamScheduler(optimizer, "lr")

    with pytest.raises(ValueError, match=r"size of value is different than optimizer_param_groups"):
        lr_scheduler(None)

    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary, but given"):
        lr_scheduler.load_state_dict(None)

    with pytest.raises(ValueError, match=r"Required state attribute 'event_index' is absent in provided state_dict"):
        lr_scheduler.load_state_dict({})

    with pytest.raises(TypeError, match=r"Argument optimizer should be torch.optim.Optimizer"):
        FakeParamScheduler({}, "lr")


def test_linear_scheduler():

    with pytest.raises(TypeError, match=r"Argument optimizer should be torch.optim.Optimizer"):
        LinearCyclicalScheduler({}, "lr", 1, 0, cycle_size=0)

    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.0)

    with pytest.raises(ValueError, match=r"Argument cycle_size should be positive and larger than 1"):
        LinearCyclicalScheduler(optimizer, "lr", 1, 0, cycle_size=0)

    with pytest.raises(ValueError, match=r"Argument cycle_size should be positive and larger than 1"):
        LinearCyclicalScheduler(optimizer, "lr", 1, 0, cycle_size=1)

    scheduler = LinearCyclicalScheduler(optimizer, "lr", 1, 0, 10)
    state_dict = scheduler.state_dict()

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    for _ in range(2):
        lrs = []
        trainer.run([0] * 9, max_epochs=2)

        assert lrs == list(
            map(
                pytest.approx,
                [
                    # Cycle 1
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                    0.2,
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    # Cycle 2
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                    0.2,
                    0.0,
                    0.2,
                    0.4,  # 0.6, 0.8,
                ],
            )
        )
        scheduler.load_state_dict(state_dict)

    optimizer = torch.optim.SGD([tensor], lr=0)
    scheduler = LinearCyclicalScheduler(optimizer, "lr", 1, 0, 10, cycle_mult=2)
    state_dict = scheduler.state_dict()

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    for _ in range(2):
        lrs = []
        trainer.run([0] * 10, max_epochs=3)

        assert lrs == list(
            map(
                pytest.approx,
                [
                    # Cycle 1
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                    0.2,
                    0.0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    # Cycle 2
                    1.0,
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                    0.4,
                    0.3,
                    0.2,
                    0.1,
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ],
            )
        )
        scheduler.load_state_dict(state_dict)

    # With float cycle_size
    optimizer = torch.optim.SGD([tensor], lr=0)
    scheduler = LinearCyclicalScheduler(
        optimizer, "lr", start_value=1.2, end_value=0.2, cycle_size=10.00000012, cycle_mult=1.0
    )
    state_dict = scheduler.state_dict()

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    for _ in range(2):
        lrs = []
        trainer.run([0] * 9, max_epochs=2)
        assert lrs == list(
            map(
                pytest.approx,
                [
                    # Cycle 1
                    1.2,
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    # Cycle 2
                    1.2,
                    1.0,
                    0.8,
                    0.6,
                    0.4,
                    0.2,
                    0.4,
                    0.6,  # 0.8, 1.0,
                ],
            )
        )
        scheduler.load_state_dict(state_dict)


def test_linear_scheduler_cycle_size_two():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = LinearCyclicalScheduler(optimizer, "lr", 1, 0, cycle_size=2)

    data = [0] * 10
    max_epochs = 2
    simulated_values = LinearCyclicalScheduler.simulate_values(
        num_events=len(data) * max_epochs, param_name="lr", start_value=1, end_value=0, cycle_size=2
    )

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    lrs = []
    trainer.run(data, max_epochs=max_epochs)
    assert lrs == list(
        map(
            pytest.approx,
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        )
    )

    assert lrs == pytest.approx([v for i, v in simulated_values])


def test_cosine_annealing_scheduler():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = CosineAnnealingScheduler(optimizer, "lr", 0, 1, 10)
    state_dict = scheduler.state_dict()

    data = [0] * 9
    max_epochs = 2
    simulated_values = CosineAnnealingScheduler.simulate_values(
        num_events=len(data) * max_epochs, param_name="lr", start_value=0, end_value=1, cycle_size=10
    )

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    for _ in range(2):
        lrs = []
        trainer.run(data, max_epochs=max_epochs)

        assert lrs == list(
            map(
                pytest.approx,
                [
                    0.0,
                    0.02447174185242318,
                    0.09549150281252627,
                    0.20610737385376332,
                    0.3454915028125263,
                    0.5,
                    0.6545084971874737,
                    0.7938926261462365,
                    0.9045084971874737,
                    0.9755282581475768,
                    0.0,
                    0.02447174185242318,
                    0.09549150281252627,
                    0.20610737385376332,
                    0.3454915028125263,
                    0.5,
                    0.6545084971874737,
                    0.7938926261462365,  # 0.9045084971874737, 0.9755282581475768
                ],
            )
        )
        scheduler.load_state_dict(state_dict)

        assert lrs == pytest.approx([v for i, v in simulated_values])


def test_concat_scheduler_asserts():

    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
    scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)

    with pytest.raises(TypeError, match=r"Argument schedulers should be a sequence"):
        ConcatScheduler(schedulers=None, durations=[])

    with pytest.raises(ValueError, match=r"Argument schedulers should be of more than one parameter schedulers"):
        ConcatScheduler(schedulers=[], durations=[])

    with pytest.raises(ValueError, match=r"Argument schedulers should be of more than one parameter schedulers"):
        ConcatScheduler(schedulers=[scheduler_1], durations=[10])

    with pytest.raises(TypeError, match=r"Value at index 1 of schedulers should be a parameter scheduler"):
        ConcatScheduler(schedulers=[scheduler_1, 12], durations=[10])

    with pytest.raises(ValueError, match=r"Incorrect number schedulers or duration values"):
        ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[10, 5])

    with pytest.raises(ValueError, match=r"Argument durations should be list/tuple of integers"):
        ConcatScheduler(schedulers=[scheduler_1, scheduler_2, scheduler_2], durations=[15, 12.0])

    with pytest.raises(TypeError, match=r"Argument durations should be list/tuple"):
        ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations="abc")

    with pytest.raises(TypeError, match=r"Argument param_names should be list or tuple"):
        ConcatScheduler.simulate_values(
            num_events=123, schedulers=[scheduler_1, scheduler_2], durations=[15], param_names="abc"
        )

    with pytest.raises(ValueError, match=r"Argument param_names should be list or tuple of strings"):
        ConcatScheduler.simulate_values(
            num_events=123, schedulers=[scheduler_1, scheduler_2], durations=[15], param_names=[1]
        )

    optimizer_2 = torch.optim.SGD([tensor], lr=0)
    scheduler_3 = CosineAnnealingScheduler(optimizer_2, "lr", start_value=0.0, end_value=1.0, cycle_size=10)

    with pytest.raises(ValueError, match=r"schedulers should be related to same optimizer"):
        ConcatScheduler([scheduler_1, scheduler_3], durations=[30,])

    scheduler_4 = CosineAnnealingScheduler(optimizer, "lr2", start_value=0.0, end_value=1.0, cycle_size=10)

    with pytest.raises(ValueError, match=r"schedulers should be related to same param_name"):
        ConcatScheduler([scheduler_1, scheduler_4], durations=[30,])

    with pytest.raises(ValueError, match=r"schedulers should be related to same optimizer"):
        ConcatScheduler.simulate_values(3, [scheduler_1, scheduler_3], durations=[30,])


def test_concat_scheduler_state_dict():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)
    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
    scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)
    durations = [10]
    concat_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=durations, save_history=False)
    state_dict = concat_scheduler.state_dict()

    assert state_dict["durations"] == durations
    assert state_dict["_current_duration"] == durations[0]
    assert state_dict["_scheduler_index"] == 0

    for _ in range(20):
        concat_scheduler(None, None)

    concat_scheduler.load_state_dict(state_dict)
    assert concat_scheduler.durations == durations
    assert concat_scheduler._current_duration == durations[0]
    assert id(concat_scheduler._current_scheduler) == id(scheduler_1)

    with pytest.raises(ValueError, match=r"Required state attribute 'schedulers' is absent in provided state_dict"):
        concat_scheduler.load_state_dict({"a": 1})

    with pytest.raises(ValueError, match=r"Input state_dict contains 0 state_dicts of concatenated schedulers"):
        concat_scheduler.load_state_dict({"schedulers": []})

    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary, but given"):
        concat_scheduler.load_state_dict(None)


def test_concat_scheduler_two_schedulers():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    def _test(duration_vals_as_np_int):
        scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
        scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)

        durations = [10]
        if duration_vals_as_np_int:
            durations = [np.int64(t) for t in durations]

        concat_scheduler = ConcatScheduler(
            schedulers=[scheduler_1, scheduler_2], durations=durations, save_history=True
        )
        state_dict = concat_scheduler.state_dict()

        data = [0] * 10
        max_epochs = 2
        simulated_values = ConcatScheduler.simulate_values(
            num_events=len(data) * max_epochs, schedulers=[scheduler_1, scheduler_2], durations=durations
        )

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]["lr"])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(Events.ITERATION_STARTED, concat_scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

        for _ in range(2):
            lrs = []
            trainer.run(data, max_epochs=max_epochs)

            assert lrs == list(
                map(
                    pytest.approx,
                    [
                        # Cycle 1 of the LinearCyclicalScheduler
                        1.0,
                        0.8,
                        0.6,
                        0.4,
                        0.2,
                        0.0,
                        0.2,
                        0.4,
                        0.6,
                        0.8,
                        # Cycle 1 of the CosineAnnealingScheduler
                        0.0,
                        0.02447174185242318,
                        0.09549150281252627,
                        0.20610737385376332,
                        0.3454915028125263,
                        0.5,
                        0.6545084971874737,
                        0.7938926261462365,
                        0.9045084971874737,
                        0.9755282581475768,
                    ],
                )
            )

            state_lrs = trainer.state.param_history["lr"]
            assert len(state_lrs) == len(lrs)
            # Unpack singleton lists
            assert [group[0] for group in state_lrs] == lrs
            assert lrs == pytest.approx([v for i, v in simulated_values])
            concat_scheduler.load_state_dict(state_dict)

            trainer.state.param_history = None

    _test(duration_vals_as_np_int=False)
    _test(duration_vals_as_np_int=True)


def test_concat_scheduler_two_linear():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.0, end_value=0.1, cycle_size=2)
    scheduler_2 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.2, end_value=1.0, cycle_size=2)

    durations = [5]
    concat_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=durations, save_history=True)
    state_dict = concat_scheduler.state_dict()

    assert concat_scheduler.get_param() == 0.0

    data = [0] * 10
    max_epochs = 2
    simulated_values = ConcatScheduler.simulate_values(
        num_events=len(data) * max_epochs, schedulers=[scheduler_1, scheduler_2], durations=durations
    )

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, concat_scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    for _ in range(2):
        lrs = []
        trainer.run(data, max_epochs=max_epochs)

        assert lrs == list(
            map(
                pytest.approx,
                [
                    # first LinearCyclicalScheduler
                    0.0,
                    0.1,
                    0.0,
                    0.1,
                    0.0,
                    # second LinearCyclicalScheduler
                    0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.2,
                    1.0,
                    0.2,
                ],
            )
        )

        state_lrs = trainer.state.param_history["lr"]
        assert len(state_lrs) == len(lrs)
        # Unpack singleton lists
        assert [group[0] for group in state_lrs] == lrs

        assert lrs == pytest.approx([v for i, v in simulated_values])
        concat_scheduler.load_state_dict(state_dict)

        trainer.state.param_history = None


def test_concat_scheduler_3_schedulers():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.5, cycle_size=20)
    scheduler_2 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.5, end_value=0.45, cycle_size=10)
    scheduler_3 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.5, end_value=0.0, cycle_size=20)
    durations = [10, 5]

    concat_scheduler = ConcatScheduler(
        schedulers=[scheduler_1, scheduler_2, scheduler_3], durations=durations, save_history=True
    )
    state_dict = concat_scheduler.state_dict()

    data = [0] * 10
    max_epochs = 2
    simulated_values = ConcatScheduler.simulate_values(
        num_events=len(data) * max_epochs, schedulers=[scheduler_1, scheduler_2, scheduler_3], durations=durations
    )

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, concat_scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    for _ in range(2):
        lrs = []
        trainer.run(data, max_epochs=max_epochs)

        assert lrs == list(
            map(
                pytest.approx,
                [
                    # Cycle 1 of the first LinearCyclicalScheduler
                    1.0,
                    0.95,
                    0.9,
                    0.85,
                    0.8,
                    0.75,
                    0.7,
                    0.65,
                    0.6,
                    0.55,
                    # Cycle 1 of the second LinearCyclicalScheduler
                    0.5,
                    0.49,
                    0.48,
                    0.47,
                    0.46,
                    # Cycle 1 of the third LinearCyclicalScheduler
                    0.5,
                    0.45,
                    0.4,
                    0.35,
                    0.3,
                ],
            )
        )

        state_lrs = trainer.state.param_history["lr"]
        assert len(state_lrs) == len(lrs)
        # Unpack singleton lists
        assert [group[0] for group in state_lrs] == lrs

        assert lrs == pytest.approx([v for i, v in simulated_values])
        concat_scheduler.load_state_dict(state_dict)

        trainer.state.param_history = None


def test_save_param_history():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = LinearCyclicalScheduler(optimizer, "lr", 1, 0, 10, save_history=True)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    assert not hasattr(trainer.state, "param_history")

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run([0] * 10, max_epochs=2)

    state_lrs = trainer.state.param_history["lr"]
    assert len(state_lrs) == len(lrs)
    # Unpack singleton lists
    assert [group[0] for group in state_lrs] == lrs


def test_lr_scheduler_asserts():

    with pytest.raises(
        TypeError, match=r"Argument lr_scheduler should be a subclass of torch.optim.lr_scheduler._LRScheduler"
    ):
        LRScheduler(123)

    with pytest.raises(
        TypeError, match=r"Argument lr_scheduler should be a subclass of torch.optim.lr_scheduler._LRScheduler"
    ):
        LRScheduler.simulate_values(1, None)


def test_lr_scheduler():
    def _test(torch_lr_scheduler_cls, **kwargs):

        tensor = torch.zeros([1], requires_grad=True)
        optimizer1 = torch.optim.SGD([tensor], lr=0.01)
        optimizer2 = torch.optim.SGD([tensor], lr=0.01)
        opt_state_dict1 = optimizer1.state_dict()
        opt_state_dict2 = optimizer2.state_dict()

        torch_lr_scheduler1 = torch_lr_scheduler_cls(optimizer=optimizer1, **kwargs)
        scheduler = LRScheduler(torch_lr_scheduler1)
        state_dict1 = scheduler.state_dict()

        torch_lr_scheduler2 = torch_lr_scheduler_cls(optimizer=optimizer2, **kwargs)
        state_dict2 = torch_lr_scheduler2.state_dict()

        def dummy_update(engine, batch):
            optimizer1.step()
            optimizer2.step()

        trainer = Engine(dummy_update)

        @trainer.on(Events.ITERATION_STARTED)
        def save_lr(engine):
            lrs.append(optimizer1.param_groups[0]["lr"])

        @trainer.on(Events.ITERATION_STARTED)
        def save_true_lr(engine):
            lrs_true.append(optimizer2.param_groups[0]["lr"])

        @trainer.on(Events.ITERATION_COMPLETED)
        def torch_lr_scheduler_step(engine):
            torch_lr_scheduler2.step()

        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

        for _ in range(2):
            lrs = []
            lrs_true = []
            data = [0] * 10
            max_epochs = 2
            trainer.run(data, max_epochs=max_epochs)
            assert lrs_true == pytest.approx(lrs), f"{_}: {lrs_true} ({len(lrs_true)}) vs {lrs} ({len(lrs)})"
            optimizer1.load_state_dict(opt_state_dict1)
            scheduler.load_state_dict(state_dict1)
            optimizer2.load_state_dict(opt_state_dict2)
            torch_lr_scheduler2.load_state_dict(state_dict2)

        optimizer3 = torch.optim.SGD([tensor], lr=0.01)
        torch_lr_scheduler3 = torch_lr_scheduler_cls(optimizer=optimizer3, **kwargs)

        simulated_values = LRScheduler.simulate_values(
            num_events=len(data) * max_epochs, lr_scheduler=torch_lr_scheduler3
        )
        assert lrs == pytest.approx([v for i, v in simulated_values])

    _test(StepLR, step_size=5, gamma=0.5)
    _test(ExponentialLR, gamma=0.78)
    if has_multiplicative_lr:
        _test(MultiplicativeLR, lr_lambda=lambda epoch: 0.95)


def test_piecewiselinear_asserts():

    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    with pytest.raises(TypeError, match=r"Argument milestones_values should be a list or tuple"):
        PiecewiseLinear(optimizer, "lr", milestones_values=None)

    with pytest.raises(ValueError, match=r"Argument milestones_values should be with at least one value"):
        PiecewiseLinear(optimizer, "lr", milestones_values=[])

    with pytest.raises(ValueError, match=r"Argument milestones_values should be a list of pairs"):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(0.5,)])

    with pytest.raises(ValueError, match=r"Argument milestones_values should be a list of pairs"):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(10, 0.5), (0.6,)])

    with pytest.raises(ValueError, match=r"Milestones should be increasing integers"):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(10, 0.5), (5, 0.6)])

    with pytest.raises(TypeError, match=r"Value of a milestone should be integer"):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(0.5, 1)])


def test_piecewiselinear():
    def _test(milestones_as_np_int):
        tensor = torch.zeros([1], requires_grad=True)
        optimizer = torch.optim.SGD([tensor], lr=0)

        milestones_values = [(5, 0.5), (15, 1.0), (25, 0.0), (35, 1.0), (40, 0.5)]
        if milestones_as_np_int:
            milestones_values = [(np.int64(t), v) for t, v in milestones_values]

        scheduler = PiecewiseLinear(optimizer, "lr", milestones_values=milestones_values)
        state_dict = scheduler.state_dict()

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]["lr"])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

        for _ in range(2):
            lrs = []
            trainer.run([0] * 25, max_epochs=2)

            assert lrs == list(
                map(
                    pytest.approx,
                    [
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.55,
                        0.6,
                        0.65,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9,
                        0.95,
                        1.0,
                        0.9,
                        0.8,
                        0.7,
                        0.6,
                        0.5,
                        0.4,
                        0.3,
                        0.2,
                        0.1,
                        0.0,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                        0.9,
                        0.8,
                        0.7,
                        0.6,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                    ],
                )
            )
            scheduler.load_state_dict(state_dict)

    _test(milestones_as_np_int=True)
    _test(milestones_as_np_int=False)


def test_simulate_and_plot_values():

    import matplotlib

    matplotlib.use("Agg")

    def _test(scheduler_cls, **scheduler_kwargs):

        optimizer = None
        event = Events.ITERATION_STARTED
        if scheduler_cls == LRScheduler:
            optimizer = scheduler_kwargs["lr_scheduler"].optimizer
            event = Events.ITERATION_COMPLETED
        elif scheduler_cls == ConcatScheduler:
            optimizer = scheduler_kwargs["optimizer"]
            del scheduler_kwargs["optimizer"]
        else:
            tensor = torch.zeros([1], requires_grad=True)
            scheduler_kwargs["optimizer"] = torch.optim.SGD([tensor], lr=0.1)
            optimizer = scheduler_kwargs["optimizer"]

        max_epochs = 2
        data = [0] * 10
        # simulated_values = scheduler_cls.simulate_values(num_events=len(data) * max_epochs, **scheduler_kwargs)

        scheduler = scheduler_cls(**scheduler_kwargs)

        lrs = []

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]["lr"])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(event, scheduler)
        trainer.add_event_handler(Events.ITERATION_STARTED, save_lr)
        trainer.run(data, max_epochs=max_epochs)

        # assert lrs == pytest.approx([v for i, v in simulated_values])

        if scheduler_cls == LRScheduler or scheduler_cls == ConcatScheduler:
            # As internal state of torch lr scheduler has been changed the following checks will fail
            return

        # reexecute to check if no internal changes
        # simulated_values = scheduler_cls.simulate_values(num_events=len(data) * max_epochs,
        #                                                  save_history=True,  # this will be removed
        #                                                  **scheduler_kwargs)
        # assert lrs == pytest.approx([v for i, v in simulated_values])

        # launch plot values
        scheduler_cls.plot_values(num_events=len(data) * max_epochs, **scheduler_kwargs)

    # LinearCyclicalScheduler
    _test(LinearCyclicalScheduler, param_name="lr", start_value=1.0, end_value=0.0, cycle_size=10)

    # CosineAnnealingScheduler
    _test(CosineAnnealingScheduler, param_name="lr", start_value=1.0, end_value=0.0, cycle_size=10)

    # LRScheduler
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.1)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

    _test(LRScheduler, lr_scheduler=torch_lr_scheduler)

    # ConcatScheduler = [LinearCyclicalScheduler, CosineAnnealingScheduler]
    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=20)
    scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)
    durations = [10]
    _test(ConcatScheduler, optimizer=optimizer, schedulers=[scheduler_1, scheduler_2], durations=durations)

    # ConcatScheduler = [LinearCyclicalScheduler, LRScheduler]
    tensor = torch.ones([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.001)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1.5)
    scheduler_1 = LRScheduler(torch_lr_scheduler)
    scheduler_2 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.1, end_value=0.0, cycle_size=10)
    durations = [10]
    _test(ConcatScheduler, optimizer=optimizer, schedulers=[scheduler_1, scheduler_2], durations=durations)

    # PiecewiseLinear
    tensor = torch.ones([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.001)
    _test(
        PiecewiseLinear,
        optimizer=optimizer,
        param_name="lr",
        milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)],
    )

    with pytest.raises(RuntimeError, match=r"This method requires matplotlib to be installed."):
        with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
            _test(
                PiecewiseLinear,
                optimizer=optimizer,
                param_name="lr",
                milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)],
            )


def test_create_lr_scheduler_with_warmup():

    with pytest.raises(TypeError, match=r"Argument lr_scheduler should be a subclass of"):
        create_lr_scheduler_with_warmup(12, warmup_start_value=0.0, warmup_end_value=0.1, warmup_duration=10)

    t1 = torch.zeros([1], requires_grad=True)
    # A) opt lr != warmup_end_value
    optimizer = torch.optim.SGD([t1], lr=0.2)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    with pytest.raises(ValueError, match=r"Argument warmup_duration should be at least 2 events"):
        create_lr_scheduler_with_warmup(
            torch_lr_scheduler, warmup_start_value=0.0, warmup_end_value=0.1, warmup_duration=1
        )

    with pytest.raises(TypeError, match=r"Argument warmup_duration should be integer"):
        create_lr_scheduler_with_warmup(
            torch_lr_scheduler, warmup_start_value=0.0, warmup_end_value=0.1, warmup_duration="abc"
        )

    with pytest.raises(TypeError, match=r"Argument output_simulated_values should be a list of None"):
        simulated_values = ()
        create_lr_scheduler_with_warmup(
            torch_lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=0.1,
            warmup_duration=10,
            output_simulated_values=simulated_values,
        )

    def _test(lr_scheduler, optimizer, warmup_start_value, warmup_end_value, warmup_duration, warmup_end_next_value):
        num_iterations = 10
        max_epochs = 20

        simulated_values = [None] * (num_iterations * max_epochs)
        scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler,
            warmup_start_value=warmup_start_value,
            warmup_end_value=warmup_end_value,
            warmup_duration=warmup_duration,
            output_simulated_values=simulated_values,
        )
        if warmup_end_value is None:
            warmup_end_value = optimizer.param_groups[0]["lr"]

        state_dict = scheduler.state_dict()
        trainer = Engine(lambda engine, batch: None)

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        @trainer.on(Events.ITERATION_STARTED)
        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]["lr"])

        data = [0] * num_iterations

        for _ in range(2):
            lrs = []
            trainer.run(data, max_epochs=max_epochs)

            assert lrs == pytest.approx([v for i, v in simulated_values])

            assert lrs[0] == pytest.approx(warmup_start_value), f"lrs={lrs[: warmup_duration + num_iterations]}"
            assert lrs[warmup_duration - 1] == pytest.approx(
                warmup_end_value
            ), f"lrs={lrs[: warmup_duration + num_iterations]}"
            assert lrs[warmup_duration] == pytest.approx(
                warmup_end_next_value
            ), f"lrs={lrs[: warmup_duration + num_iterations]}"
            scheduler.load_state_dict(state_dict)

    t1 = torch.zeros([1], requires_grad=True)
    # A) opt lr != warmup_end_value
    optimizer = torch.optim.SGD([t1], lr=0.2)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    _test(torch_lr_scheduler, optimizer, 0.01, 0.05, 10, 0.2)
    optimizer = torch.optim.SGD([t1], lr=0.2)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    _test(torch_lr_scheduler, optimizer, 0.01, 0.05, 2, 0.2)

    # B) opt lr == warmup_end_value
    optimizer = torch.optim.SGD([t1], lr=0.2)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    _test(torch_lr_scheduler, optimizer, 0.01, 0.2, 10, 0.2 * 0.98)
    optimizer = torch.optim.SGD([t1], lr=0.2)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    _test(torch_lr_scheduler, optimizer, 0.01, 0.2, 2, 0.2 * 0.98)

    # C) lr_scheduler start_value != warmup_end_value
    t1 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([t1], lr=0.0)
    lr_scheduler = LinearCyclicalScheduler(
        optimizer=optimizer, param_name="lr", start_value=0.8, end_value=0.0, cycle_size=10
    )
    _test(lr_scheduler, optimizer, 0.01, 0.05, 10, 0.8)
    optimizer = torch.optim.SGD([t1], lr=0.0)
    lr_scheduler = LinearCyclicalScheduler(
        optimizer=optimizer, param_name="lr", start_value=0.8, end_value=0.0, cycle_size=10
    )
    _test(lr_scheduler, optimizer, 0.01, 0.05, 2, 0.8)

    # D) lr_scheduler start_value == warmup_end_value
    t1 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([t1], lr=0.0)
    lr_scheduler = LinearCyclicalScheduler(
        optimizer=optimizer, param_name="lr", start_value=0.8, end_value=0.0, cycle_size=10
    )
    _test(lr_scheduler, optimizer, 0.01, 0.8, 10, 0.8 - (0.8 / 5.0))
    optimizer = torch.optim.SGD([t1], lr=0.0)
    lr_scheduler = LinearCyclicalScheduler(
        optimizer=optimizer, param_name="lr", start_value=0.8, end_value=0.0, cycle_size=10
    )
    _test(lr_scheduler, optimizer, 0.01, 0.8, 2, 0.8 - (0.8 / 5.0))

    # E) warmup_end_value is None: fall back to case B)
    optimizer = torch.optim.SGD([t1], lr=0.2)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    _test(torch_lr_scheduler, optimizer, 0.01, None, 10, 0.2 * 0.98)


def test_create_lr_scheduler_with_warmup_on_combined_scheduler():
    # Test with a complex scheduler
    def _test(save_history):
        tensor = torch.ones([1], requires_grad=True)
        optimizer = torch.optim.SGD([tensor], lr=0.001)

        max_epochs = 25
        lr_max_value = 0.4
        num_iterations_per_epoch = 128
        num_iterations = max_epochs * num_iterations_per_epoch
        warmup_duration = 5 * num_iterations_per_epoch
        cooldown_duration = 5 * num_iterations_per_epoch

        scheduler_1 = LinearCyclicalScheduler(
            optimizer,
            "lr",
            start_value=lr_max_value,
            end_value=lr_max_value * 0.9,
            cycle_size=(num_iterations - warmup_duration - cooldown_duration) * 2,
        )

        scheduler_2 = LinearCyclicalScheduler(
            optimizer, "lr", start_value=lr_max_value, end_value=0.0, cycle_size=cooldown_duration * 2
        )

        lr_scheduler = ConcatScheduler(
            schedulers=[scheduler_1, scheduler_2],
            durations=[num_iterations - warmup_duration - cooldown_duration],
            save_history=False,
        )
        lr_values = [None] * num_iterations
        scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=lr_max_value,
            warmup_duration=warmup_duration,
            save_history=save_history,
            output_simulated_values=lr_values,
        )
        state_dict = scheduler.state_dict()

        trainer = Engine(lambda engine, batch: None)

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]["lr"])

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        data = [0] * num_iterations_per_epoch

        for _ in range(2):
            lrs = []
            trainer.run(data, max_epochs=max_epochs)

            assert lrs == pytest.approx([v for i, v in lr_values])

            if save_history:
                param_history = trainer.state.param_history["lr"]
                assert lrs == pytest.approx([v[0] for v in param_history])

                trainer.state.param_history = None

            scheduler.load_state_dict(state_dict)

    _test(save_history=False)
    _test(save_history=True)


def test_create_lr_scheduler_with_warmup_with_real_model(dummy_model_factory):

    model = dummy_model_factory(with_grads=False, with_frozen_layer=False)
    init_lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    scaled_lr = 0.02
    warmup_duration = 5
    step_size = 2
    gamma = 0.97

    output_simulated_values = [None] * 50

    create_lr_scheduler_with_warmup(
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        warmup_start_value=0.0,
        warmup_end_value=scaled_lr,
        warmup_duration=warmup_duration,
        output_simulated_values=output_simulated_values,
    )

    assert output_simulated_values[0] == [0, 0.0]
    assert output_simulated_values[warmup_duration - 1] == [warmup_duration - 1, scaled_lr]
    assert output_simulated_values[warmup_duration] == [warmup_duration, init_lr]
    v = [warmup_duration + step_size, init_lr * gamma]
    assert output_simulated_values[warmup_duration + step_size] == v


def test_param_group_scheduler_asserts():

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([{"params": t1, "lr": 0.1}, {"params": t2, "lr": 0.1}])

    lr_scheduler1 = LinearCyclicalScheduler(
        optimizer, "lr", param_group_index=0, start_value=1.0, end_value=0.0, cycle_size=10
    )
    lr_scheduler2 = LinearCyclicalScheduler(
        optimizer, "lr", param_group_index=1, start_value=1.0, end_value=0.0, cycle_size=10
    )

    with pytest.raises(TypeError, match=r"Argument schedulers should be a list/tuple"):
        ParamGroupScheduler(schedulers=None, names=["a", "b", "c"])

    with pytest.raises(ValueError, match=r"Argument schedulers should be a list/tuple of parameter schedulers"):
        ParamGroupScheduler(schedulers=[0, 1, 2], names=["a", "b", "c"])

    with pytest.raises(ValueError, match=r"Argument schedulers should be a list/tuple of parameter schedulers"):
        ParamGroupScheduler(schedulers=[lr_scheduler1, "2"], names=["a", "b"])

    with pytest.raises(TypeError, match=r"Argument names should be a list/tuple"):
        ParamGroupScheduler(schedulers=[lr_scheduler1, lr_scheduler2], names="ab")

    with pytest.raises(ValueError, match=r"Argument names should be a list/tuple of parameter scheduler's names"):
        ParamGroupScheduler(schedulers=[lr_scheduler1, lr_scheduler2], names=[1, 2])

    with pytest.raises(ValueError, match=r"\d should be equal \d"):
        ParamGroupScheduler(schedulers=[lr_scheduler1, lr_scheduler2], names=["a"])

    scheduler = ParamGroupScheduler(schedulers=[lr_scheduler1, lr_scheduler2], names=["a", "b"])
    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary"):
        scheduler.load_state_dict(None)

    with pytest.raises(ValueError, match=r"Required state attribute 'schedulers' is absent in provided state_dict"):
        scheduler.load_state_dict({"a": 1})

    with pytest.raises(ValueError, match=r"Input state_dict contains 0 state_dicts of param group schedulers"):
        scheduler.load_state_dict({"schedulers": []})

    with pytest.raises(ValueError, match=r"Required state attribute 'schedulers' is absent in provided state_dict"):
        scheduler.load_state_dict({})

    with pytest.raises(
        ValueError, match=r"Name of scheduler from input state dict does not " r"correspond to required one"
    ):
        scheduler.load_state_dict({"schedulers": [("a", lr_scheduler1.state_dict()), ("bad_name", {})]})


def test_param_group_scheduler():
    def _test(lr_schedulers, save_lr):
        num_iterations = 10
        max_epochs = 20

        scheduler = ParamGroupScheduler(lr_schedulers, names=[f"s_{i}" for i in range(len(lr_schedulers))])
        state_dict = scheduler.state_dict()

        trainer = Engine(lambda engine, batch: None)

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        data = [0] * num_iterations

        for _ in range(2):
            lrs = []
            trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr, lrs)
            trainer.run(data, max_epochs=max_epochs)
            trainer.remove_event_handler(save_lr, Events.ITERATION_COMPLETED)
            assert [lr[0] for lr in lrs] == pytest.approx([lr[1] for lr in lrs])
            scheduler.load_state_dict(state_dict)

            values = ParamGroupScheduler.simulate_values(max_epochs * num_iterations, lr_schedulers)
            assert [lr[1] for lr in values] == pytest.approx([lr[2] for lr in values])
            assert [lr[0] for lr in lrs] == pytest.approx([lr[1] for lr in values])

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([{"params": t1, "lr": 0.1}, {"params": t2, "lr": 0.1}])

    lr_scheduler1 = LinearCyclicalScheduler(
        optimizer, "lr", param_group_index=0, start_value=1.0, end_value=0.0, cycle_size=10
    )
    lr_scheduler2 = LinearCyclicalScheduler(
        optimizer, "lr", param_group_index=1, start_value=1.0, end_value=0.0, cycle_size=10
    )

    def save_lr_one_optimizer(engine, lrs):
        lrs.append((optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]))

    _test([lr_scheduler1, lr_scheduler2], save_lr_one_optimizer)

    t1 = torch.zeros([1], requires_grad=True)
    optimizer_1 = torch.optim.SGD(params=[t1], lr=0.1)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer_2 = torch.optim.SGD(params=[t2], lr=0.1)

    lr_scheduler1 = LinearCyclicalScheduler(optimizer_1, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
    lr_scheduler2 = LinearCyclicalScheduler(optimizer_2, "lr", start_value=1.0, end_value=0.0, cycle_size=10)

    def save_lr_mutliple_optimizers(engine, lrs):
        lrs.append((optimizer_1.param_groups[0]["lr"], optimizer_2.param_groups[0]["lr"]))

    _test([lr_scheduler1, lr_scheduler2], save_lr_mutliple_optimizers)


def test_scheduler_with_param_groups():
    def _test(lr_scheduler, optimizer):
        num_iterations = 10
        max_epochs = 20

        state_dict = lr_scheduler.state_dict()

        trainer = Engine(lambda engine, batch: None)

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_lr():
            lrs.append((optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]))

        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

        data = [0] * num_iterations

        for _ in range(2):
            lrs = []
            trainer.run(data, max_epochs=max_epochs)
            assert [lr[0] for lr in lrs] == pytest.approx([lr[1] for lr in lrs])
            lr_scheduler.load_state_dict(state_dict)

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([{"params": t1, "lr": 0.1}, {"params": t2, "lr": 0.1}])

    lr_scheduler = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
    _test(lr_scheduler, optimizer)

    lr_scheduler = PiecewiseLinear(
        optimizer, "lr", milestones_values=[(5, 0.5), (15, 1.0), (25, 0.0), (35, 1.0), (40, 0.5)]
    )
    _test(lr_scheduler, optimizer)

    lr_scheduler = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)
    _test(lr_scheduler, optimizer)

    torch_lr_scheduler = ExponentialLR(optimizer, gamma=0.98)
    _test(LRScheduler(torch_lr_scheduler), optimizer)

    torch_lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    _test(LRScheduler(torch_lr_scheduler), optimizer)


def test_lr_scheduling_on_non_torch_optimizers():
    # tests https://github.com/pytorch/ignite/issues/1162
    optimizer = MagicMock()
    optimizer.param_groups = [{"params": 0}]
    FakeParamScheduler(optimizer, "lr")

    tensor = torch.zeros([1], requires_grad=True)
    base_optimizer = torch.optim.SGD([tensor], lr=0)
    optimizer = MockFP16DeepSpeedZeroOptimizer(base_optimizer)

    milestones_values = [(5, 0.5), (15, 1.0)]

    scheduler = PiecewiseLinear(optimizer, "lr", milestones_values=milestones_values)

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]["lr"])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    lrs = []
    trainer.run([0] * 15, max_epochs=1)

    assert lrs == list(
        map(pytest.approx, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,],)
    )

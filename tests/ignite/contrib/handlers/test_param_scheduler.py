import pytest

import torch

from ignite.engine import Engine, Events
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler, CosineAnnealingScheduler
from ignite.contrib.handlers.param_scheduler import ConcatScheduler
from ignite.contrib.handlers.param_scheduler import LRScheduler


def test_simulate_values():

    def _test(scheduler_cls, **scheduler_kwargs):
        tensor = torch.zeros([1], requires_grad=True)
        optimizer = torch.optim.SGD([tensor], lr=0)

        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        lrs = []

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]['lr'])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
        max_epochs = 2
        data = [0] * 10
        trainer.run(data, max_epochs=max_epochs)
        simulated_values = scheduler_cls.simulate_values(num_events=len(data) * max_epochs, **scheduler_kwargs)
        assert lrs == pytest.approx([v for i, v in simulated_values])

    _test(LinearCyclicalScheduler, param_name="lr", start_value=1.0, end_value=0.0, cycle_size=10)
    _test(CosineAnnealingScheduler, param_name="lr", start_value=1.0, end_value=0.0, cycle_size=10)


def test_linear_scheduler():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1, 0, 10)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run([0] * 10, max_epochs=2)

    assert lrs == list(map(pytest.approx, [
        # Cycle 1
        1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.2, 0.4, 0.6, 0.8,
        # Cycle 2
        1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.2, 0.4, 0.6, 0.8,
    ]))

    optimizer = torch.optim.SGD([tensor], lr=0)
    scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1, 0, 10, cycle_mult=2)

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    lrs = []
    trainer.run([0] * 10, max_epochs=3)

    assert lrs == list(map(pytest.approx, [
        # Cycle 1
        1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.2, 0.4, 0.6, 0.8,
        # Cycle 2
        1.0, 0.9, 0.8, 0.7, 0.6,
        0.5, 0.4, 0.3, 0.2, 0.1,
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9,
    ]))


def test_cosine_annealing_scheduler():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = CosineAnnealingScheduler(optimizer, 'lr', 0, 1, 10)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run([0] * 10, max_epochs=2)

    assert lrs == list(map(pytest.approx, [
        0.0, 0.02447174185242318, 0.09549150281252627, 0.20610737385376332, 0.3454915028125263,
        0.5, 0.6545084971874737, 0.7938926261462365, 0.9045084971874737, 0.9755282581475768,
        0.0, 0.02447174185242318, 0.09549150281252627, 0.20610737385376332, 0.3454915028125263,
        0.5, 0.6545084971874737, 0.7938926261462365, 0.9045084971874737, 0.9755282581475768
    ]))


def test_concat_scheduler():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
    scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)
    durations = [10, ]

    concat_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2],
                                       durations=durations, save_history=True)

    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, concat_scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    data = [0] * 10
    max_epochs = 2
    trainer.run(data, max_epochs=max_epochs)

    assert lrs == list(map(pytest.approx, [
        # Cycle 1 of the LinearCyclicalScheduler
        1.0, 0.8, 0.6, 0.4, 0.2,
        0.0, 0.2, 0.4, 0.6, 0.8,
        # Cycle 1 of the CosineAnnealingScheduler
        0.0, 0.02447174185242318, 0.09549150281252627, 0.20610737385376332, 0.3454915028125263,
        0.5, 0.6545084971874737, 0.7938926261462365, 0.9045084971874737, 0.9755282581475768,
    ]))

    state_lrs = trainer.state.param_history['lr']
    assert len(state_lrs) == len(lrs)
    # Unpack singleton lists
    assert [group[0] for group in state_lrs] == lrs

    simulated_values = ConcatScheduler.simulate_values(num_events=len(data) * max_epochs,
                                                       schedulers=[scheduler_1, scheduler_2],
                                                       durations=durations)
    assert lrs == pytest.approx([v for i, v in simulated_values])


def test_save_param_history():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = LinearCyclicalScheduler(optimizer, 'lr', 1, 0, 10, save_history=True)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    assert not hasattr(trainer.state, 'param_history')

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run([0] * 10, max_epochs=2)

    state_lrs = trainer.state.param_history['lr']
    assert len(state_lrs) == len(lrs)
    # Unpack singleton lists
    assert [group[0] for group in state_lrs] == lrs


def test_lr_scheduler():

    def _test(torch_lr_scheduler_cls, **kwargs):

        tensor = torch.zeros([1], requires_grad=True)
        optimizer1 = torch.optim.SGD([tensor], lr=0.1)
        optimizer2 = torch.optim.SGD([tensor], lr=0.1)

        torch_lr_scheduler1 = torch_lr_scheduler_cls(optimizer=optimizer1, **kwargs)
        torch_lr_scheduler2 = torch_lr_scheduler_cls(optimizer=optimizer2, **kwargs)
        scheduler = LRScheduler(torch_lr_scheduler1)

        lrs = []
        lrs_true = []

        trainer = Engine(lambda engine, batch: None)

        @trainer.on(Events.ITERATION_STARTED)
        def torch_lr_scheduler_step(engine):
            torch_lr_scheduler2.step()

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_lr(engine):
            lrs.append(optimizer1.param_groups[0]['lr'])

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_true_lr(engine):
            lrs_true.append(optimizer2.param_groups[0]['lr'])

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        data = [0] * 10
        max_epochs = 2
        trainer.run(data, max_epochs=max_epochs)

        assert lrs_true == pytest.approx(lrs)

        optimizer3 = torch.optim.SGD([tensor], lr=0.1)
        torch_lr_scheduler3 = torch_lr_scheduler_cls(optimizer=optimizer3, **kwargs)

        simulated_values = LRScheduler.simulate_values(num_events=len(data) * max_epochs,
                                                       lr_scheduler=torch_lr_scheduler3)
        assert lrs == pytest.approx([v for i, v in simulated_values])

    _test(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5)
    _test(torch.optim.lr_scheduler.ExponentialLR, gamma=0.78)

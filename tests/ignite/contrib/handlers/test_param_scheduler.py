import pytest

import torch

from ignite.engine import Engine, Events
from ignite.contrib.handlers.param_scheduler import LinearScheduler, CosineAnnealingScheduler


def test_linear_scheduler():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = LinearScheduler(optimizer, 'lr', 1, 0, 10)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
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
    scheduler = LinearScheduler(optimizer, 'lr', 1, 0, 10, cycle_mult=2)

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
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

    scheduler = CosineAnnealingScheduler(optimizer, 'lr', 1, 0, 10)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run([0] * 10, max_epochs=2)

    assert lrs == list(map(pytest.approx, [
        0.0, 0.02447174185242318, 0.09549150281252627, 0.20610737385376332, 0.3454915028125263,
        0.5, 0.6545084971874737, 0.7938926261462365, 0.9045084971874737, 0.9755282581475768,
        0.0, 0.02447174185242318, 0.09549150281252627, 0.20610737385376332, 0.3454915028125263,
        0.5, 0.6545084971874737, 0.7938926261462365, 0.9045084971874737, 0.9755282581475768
    ]))


def test_save_param_history():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler = LinearScheduler(optimizer, 'lr', 1, 0, 10, save_history=True)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    assert not hasattr(trainer.state, 'param_history')

    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run([0] * 10, max_epochs=2)

    state_lrs = trainer.state.param_history['lr']
    assert len(state_lrs) == len(lrs)
    # Unpack singleton lists
    assert [group[0] for group in state_lrs] == lrs

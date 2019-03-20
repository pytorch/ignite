import pytest

import numpy as np

import torch

from ignite.engine import Engine, Events
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler, CosineAnnealingScheduler
from ignite.contrib.handlers.param_scheduler import ConcatScheduler, LRScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.handlers.param_scheduler import ParamGroupScheduler, PiecewiseLinear


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

    # With float cycle_size
    optimizer = torch.optim.SGD([tensor], lr=0)
    scheduler = LinearCyclicalScheduler(optimizer, 'lr',
                                        start_value=1.2, end_value=0.2,
                                        cycle_size=10.00000012, cycle_mult=1.0)

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)

    lrs = []
    trainer.run([0] * 10, max_epochs=2)
    assert lrs == list(map(pytest.approx, [
        # Cycle 1
        1.2, 1.0, 0.8, 0.6, 0.4,
        0.2, 0.4, 0.6, 0.8, 1.0,
        # Cycle 2
        1.2, 1.0, 0.8, 0.6, 0.4,
        0.2, 0.4, 0.6, 0.8, 1.0,
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


def test_concat_scheduler_asserts():

    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
    scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)

    with pytest.raises(ValueError):
        ConcatScheduler(schedulers=[], durations=[])

    with pytest.raises(ValueError):
        ConcatScheduler(schedulers=[scheduler_1, ], durations=[10, ])

    with pytest.raises(TypeError):
        ConcatScheduler(schedulers=[scheduler_1, 12], durations=[10, ])

    with pytest.raises(ValueError):
        ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[10, 5])

    with pytest.raises(ValueError):
        ConcatScheduler(schedulers=[scheduler_1, scheduler_2, scheduler_2], durations=[15, 12.0])

    with pytest.raises(ValueError):
        ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations="abc")

    with pytest.raises(ValueError):
        ConcatScheduler.simulate_values(num_events=123, schedulers=[scheduler_1, scheduler_2],
                                        durations=[15, ], param_names="abc")


def test_concat_scheduler():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    def _test(duration_vals_as_np_int):
        scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.0, cycle_size=10)
        scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", start_value=0.0, end_value=1.0, cycle_size=10)

        durations = [10, ]
        if duration_vals_as_np_int:
            durations = [np.int64(t) for t in durations]

        concat_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2],
                                           durations=durations, save_history=True)

        data = [0] * 10
        max_epochs = 2
        simulated_values = ConcatScheduler.simulate_values(num_events=len(data) * max_epochs,
                                                           schedulers=[scheduler_1, scheduler_2],
                                                           durations=durations)

        lrs = []

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]['lr'])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(Events.ITERATION_STARTED, concat_scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
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

        assert lrs == pytest.approx([v for i, v in simulated_values])

    _test(duration_vals_as_np_int=False)
    _test(duration_vals_as_np_int=True)


def test_concat_scheduler_3_schedulers():
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1.0, end_value=0.5, cycle_size=20)
    scheduler_2 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.5, end_value=0.45, cycle_size=10)
    scheduler_3 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.5, end_value=0.0, cycle_size=20)
    durations = [10, 5]

    concat_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2, scheduler_3],
                                       durations=durations, save_history=True)

    data = [0] * 10
    max_epochs = 2
    simulated_values = ConcatScheduler.simulate_values(num_events=len(data) * max_epochs,
                                                       schedulers=[scheduler_1, scheduler_2, scheduler_3],
                                                       durations=durations)
    lrs = []

    def save_lr(engine):
        lrs.append(optimizer.param_groups[0]['lr'])

    trainer = Engine(lambda engine, batch: None)
    trainer.add_event_handler(Events.ITERATION_STARTED, concat_scheduler)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
    trainer.run(data, max_epochs=max_epochs)

    assert lrs == list(map(pytest.approx, [
        # Cycle 1 of the first LinearCyclicalScheduler
        1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
        # Cycle 1 of the second LinearCyclicalScheduler
        0.5, 0.49, 0.48, 0.47, 0.46,
        # Cycle 1 of the third LinearCyclicalScheduler
        0.5, 0.45, 0.4, 0.35, 0.3,
    ]))

    state_lrs = trainer.state.param_history['lr']
    assert len(state_lrs) == len(lrs)
    # Unpack singleton lists
    assert [group[0] for group in state_lrs] == lrs

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


def test_lr_scheduler_asserts():

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([
        {"params": t1, 'lr': 0.1},
        {"params": t2, 'lr': 0.1},
    ])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    with pytest.raises(ValueError):
        LRScheduler(lr_scheduler)

    with pytest.raises(ValueError):
        LRScheduler.simulate_values(num_events=100, lr_scheduler=lr_scheduler)

    with pytest.raises(TypeError):
        LRScheduler(123)


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

    # test _replicate_lr_scheduler
    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.78)
    init_lr_scheduler_state = dict(lr_scheduler.state_dict())
    copy_lr_scheduler = LRScheduler._replicate_lr_scheduler(lr_scheduler)
    for _ in range(10):
        lr_scheduler.step()

    assert copy_lr_scheduler.state_dict() == init_lr_scheduler_state


def test_piecewiselinear_asserts():

    tensor = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0)

    with pytest.raises(ValueError):
        PiecewiseLinear(optimizer, "lr", milestones_values=[])

    with pytest.raises(ValueError):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(0.5,), ])

    with pytest.raises(ValueError):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(10, 0.5), (0.6,)])

    with pytest.raises(ValueError):
        PiecewiseLinear(optimizer, "lr", milestones_values=[(10, 0.5), (5, 0.6)])


def test_piecewiselinear():

    def _test(milestones_as_np_int):
        tensor = torch.zeros([1], requires_grad=True)
        optimizer = torch.optim.SGD([tensor], lr=0)

        milestones_values = [(5, 0.5),
                             (15, 1.0),
                             (25, 0.0),
                             (35, 1.0),
                             (40, 0.5)]
        if milestones_as_np_int:
            milestones_values = [(np.int64(t), v) for t, v in milestones_values]

        scheduler = PiecewiseLinear(optimizer, 'lr',
                                    milestones_values=milestones_values)
        lrs = []

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]['lr'])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
        trainer.run([0] * 25, max_epochs=2)

        assert lrs == list(map(pytest.approx, [
            0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            1.0, 0.9, 0.8, 0.7, 0.6,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ]))

    _test(milestones_as_np_int=True)
    _test(milestones_as_np_int=False)


def test_simulate_values():
    def _test(scheduler_cls, **scheduler_kwargs):

        optimizer = None
        if scheduler_cls == LRScheduler:
            scheduler_kwargs['optimizer'] = scheduler_kwargs['lr_scheduler'].optimizer
            optimizer = scheduler_kwargs['optimizer']
        elif scheduler_cls == ConcatScheduler:
            optimizer = scheduler_kwargs['optimizer']
            del scheduler_kwargs['optimizer']
        else:
            tensor = torch.zeros([1], requires_grad=True)
            scheduler_kwargs['optimizer'] = torch.optim.SGD([tensor], lr=0.1)
            optimizer = scheduler_kwargs['optimizer']

        max_epochs = 2
        data = [0] * 10
        simulated_values = scheduler_cls.simulate_values(num_events=len(data) * max_epochs, **scheduler_kwargs)

        scheduler = scheduler_cls(**scheduler_kwargs)

        lrs = []

        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]['lr'])

        trainer = Engine(lambda engine, batch: None)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, save_lr)
        trainer.run(data, max_epochs=max_epochs)

        assert lrs == pytest.approx([v for i, v in simulated_values])

        if scheduler_cls == LRScheduler or scheduler_cls == ConcatScheduler:
            # As internal state of torch lr scheduler has been changed the following checks will fail
            return

        # reexecute to check if no internal changes
        simulated_values = scheduler_cls.simulate_values(num_events=len(data) * max_epochs,
                                                         save_history=True,  # this will be removed
                                                         **scheduler_kwargs)
        assert lrs == pytest.approx([v for i, v in simulated_values])

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
    durations = [10, ]
    _test(ConcatScheduler, optimizer=optimizer, schedulers=[scheduler_1, scheduler_2], durations=durations)

    # ConcatScheduler = [LinearCyclicalScheduler, LRScheduler]
    tensor = torch.ones([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.001)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1.5)
    scheduler_1 = LRScheduler(torch_lr_scheduler)
    scheduler_2 = LinearCyclicalScheduler(optimizer, "lr", start_value=0.1, end_value=0.0, cycle_size=10)
    durations = [10, ]
    _test(ConcatScheduler, optimizer=optimizer, schedulers=[scheduler_1, scheduler_2], durations=durations)

    # PiecewiseLinear
    tensor = torch.ones([1], requires_grad=True)
    optimizer = torch.optim.SGD([tensor], lr=0.001)
    _test(PiecewiseLinear, optimizer=optimizer, param_name="lr",
          milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])


def test_create_lr_scheduler_with_warmup():
    with pytest.raises(TypeError):
        create_lr_scheduler_with_warmup(12, warmup_start_value=0.0, warmup_end_value=0.1, warmup_duration=10)

    def _test(lr_scheduler, optimizer):
        num_iterations = 10
        max_epochs = 20

        simulated_values = [None] * (num_iterations * max_epochs)
        scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                    warmup_start_value=0.0, warmup_end_value=0.1, warmup_duration=10,
                                                    output_simulated_values=simulated_values)

        lrs = []
        trainer = Engine(lambda engine, batch: None)

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]['lr'])

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        data = [0] * num_iterations
        trainer.run(data, max_epochs=max_epochs)

        assert lrs == pytest.approx([v for i, v in simulated_values])

    t1 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([t1], lr=0.1)
    torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    _test(torch_lr_scheduler, optimizer)

    t1 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([t1], lr=0.1)
    lr_scheduler = LinearCyclicalScheduler(optimizer=optimizer, param_name='lr',
                                           start_value=1.0, end_value=0.0, cycle_size=10)
    _test(lr_scheduler, optimizer)


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

        scheduler_1 = LinearCyclicalScheduler(optimizer, "lr",
                                              start_value=lr_max_value, end_value=lr_max_value * 0.9,
                                              cycle_size=(num_iterations - warmup_duration - cooldown_duration) * 2)

        scheduler_2 = LinearCyclicalScheduler(optimizer, "lr",
                                              start_value=lr_max_value, end_value=0.0,
                                              cycle_size=cooldown_duration * 2)

        lr_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2],
                                       durations=[num_iterations - warmup_duration - cooldown_duration, ],
                                       save_history=False)
        lr_values = [None] * num_iterations
        scheduler = create_lr_scheduler_with_warmup(
            lr_scheduler,
            warmup_start_value=0.0,
            warmup_end_value=lr_max_value,
            warmup_duration=warmup_duration,
            save_history=save_history,
            output_simulated_values=lr_values
        )

        lrs = []
        trainer = Engine(lambda engine, batch: None)

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_lr(engine):
            lrs.append(optimizer.param_groups[0]['lr'])

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        data = [0] * num_iterations_per_epoch
        trainer.run(data, max_epochs=max_epochs)

        assert lrs == pytest.approx([v for i, v in lr_values])

        if save_history:
            param_history = trainer.state.param_history['lr']
            assert lrs == pytest.approx([v[0] for v in param_history])

    _test(save_history=False)
    _test(save_history=True)


def test_param_group_scheduler_asserts():

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([
        {"params": t1, 'lr': 0.1},
        {"params": t2, 'lr': 0.1},
    ])

    lr_scheduler1 = LinearCyclicalScheduler(optimizer.param_groups[0], "lr",
                                            start_value=1.0, end_value=0.0, cycle_size=10)
    lr_scheduler2 = LinearCyclicalScheduler(optimizer.param_groups[1], "lr",
                                            start_value=1.0, end_value=0.0, cycle_size=10)

    with pytest.raises(ValueError):
        ParamGroupScheduler(schedulers=[0, 1, 2], names=['a', 'b', 'c'])

    with pytest.raises(ValueError):
        ParamGroupScheduler(schedulers=[lr_scheduler1, '2'], names=['a', 'b'])

    with pytest.raises(ValueError):
        ParamGroupScheduler(schedulers=[lr_scheduler1, lr_scheduler2], names='ab')

    with pytest.raises(ValueError):
        ParamGroupScheduler(schedulers=[lr_scheduler1, lr_scheduler2], names=['a', ])


def test_param_group_scheduler():

    def _test(lr_schedulers, optimizer):
        num_iterations = 10
        max_epochs = 20

        scheduler = ParamGroupScheduler(lr_schedulers, names=["s_{}".format(i) for i in range(len(lr_schedulers))])

        lrs = []
        trainer = Engine(lambda engine, batch: None)

        @trainer.on(Events.ITERATION_COMPLETED)
        def save_lr(engine):
            lrs.append(
                (optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
            )

        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        data = [0] * num_iterations
        trainer.run(data, max_epochs=max_epochs)
        assert [lr[0] for lr in lrs] == pytest.approx([lr[1] for lr in lrs])

    t1 = torch.zeros([1], requires_grad=True)
    t2 = torch.zeros([1], requires_grad=True)
    optimizer = torch.optim.SGD([
        {"params": t1, 'lr': 0.1},
        {"params": t2, 'lr': 0.1},
    ])

    lr_scheduler1 = LinearCyclicalScheduler(optimizer.param_groups[0], "lr",
                                            start_value=1.0, end_value=0.0, cycle_size=10)
    lr_scheduler2 = LinearCyclicalScheduler(optimizer.param_groups[1], "lr",
                                            start_value=1.0, end_value=0.0, cycle_size=10)
    _test([lr_scheduler1, lr_scheduler2], optimizer)

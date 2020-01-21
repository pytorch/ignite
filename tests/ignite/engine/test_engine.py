from __future__ import division

import os
import pytest
from unittest.mock import call, MagicMock, Mock

import numpy as np
import torch

from ignite.engine import Engine, Events, State


def test_terminate():
    engine = Engine(lambda e, b: 1)
    assert not engine.should_terminate
    engine.terminate()
    assert engine.should_terminate


def test_invalid_process_raises_with_invalid_signature():
    with pytest.raises(ValueError):
        Engine(None)

    with pytest.raises(ValueError):
        Engine(lambda: None)

    with pytest.raises(ValueError):
        Engine(lambda batch: None)

    with pytest.raises(ValueError):
        Engine(lambda engine, batch, extra_arg: None)


def test_current_epoch_counter_increases_every_epoch(counter_factory):
    engine = Engine(MagicMock(return_value=1))
    max_epochs = 5

    counter = counter_factory('epoch')
    engine.add_event_handler(Events.EPOCH_STARTED, counter)

    state = engine.run([1, 2], max_epochs=max_epochs)
    assert state.epoch == max_epochs
    counter.current_epoch_count = 1
    state = engine.run([1, 2], max_epochs=max_epochs)
    assert state.epoch == max_epochs


def test_current_iteration_counter_increases_every_iteration(counter_factory):
    batches = [1, 2, 3]
    engine = Engine(MagicMock(return_value=1))
    max_epochs = 5

    counter = counter_factory('iter')
    engine.add_event_handler(Events.ITERATION_STARTED, counter)

    state = engine.run(batches, max_epochs=max_epochs)
    assert state.iteration == max_epochs * len(batches)
    counter.current_iteration_count = 1
    state = engine.run(batches, max_epochs=max_epochs)
    assert state.iteration == max_epochs * len(batches)


def test_stopping_criterion_is_max_epochs():
    engine = Engine(MagicMock(return_value=1))
    max_epochs = 5
    state = engine.run([1], max_epochs=max_epochs)
    assert state.epoch == max_epochs


def test_terminate_at_end_of_epoch_stops_run():
    max_epochs = 5
    last_epoch_to_run = 3

    engine = Engine(MagicMock(return_value=1))

    def end_of_epoch_handler(engine):
        if engine.state.epoch == last_epoch_to_run:
            engine.terminate()

    engine.add_event_handler(Events.EPOCH_COMPLETED, end_of_epoch_handler)

    assert not engine.should_terminate

    state = engine.run([1], max_epochs=max_epochs)

    assert state.epoch == last_epoch_to_run
    assert engine.should_terminate


def test_terminate_at_start_of_epoch_stops_run_after_completing_iteration():
    max_epochs = 5
    epoch_to_terminate_on = 3
    batches_per_epoch = [1, 2, 3]

    engine = Engine(MagicMock(return_value=1))

    def start_of_epoch_handler(engine):
        if engine.state.epoch == epoch_to_terminate_on:
            engine.terminate()

    engine.add_event_handler(Events.EPOCH_STARTED, start_of_epoch_handler)

    assert not engine.should_terminate

    state = engine.run(batches_per_epoch, max_epochs=max_epochs)

    # epoch is not completed so counter is not incremented
    assert state.epoch == epoch_to_terminate_on
    assert engine.should_terminate
    # completes first iteration
    assert state.iteration == ((epoch_to_terminate_on - 1) * len(batches_per_epoch)) + 1


def test_terminate_stops_run_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 3
    engine = Engine(MagicMock(return_value=1))

    def start_of_iteration_handler(engine):
        if engine.state.iteration == iteration_to_stop:
            engine.terminate()

    engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = engine.run(data=[None] * num_iterations_per_epoch, max_epochs=3)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert (state.iteration == iteration_to_stop)
    assert state.epoch == np.ceil(iteration_to_stop / num_iterations_per_epoch)  # it starts from 0


def test_terminate_epoch_stops_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 4
    engine = Engine(MagicMock(return_value=1))

    def start_of_iteration_handler(engine):
        if engine.state.iteration == iteration_to_stop:
            engine.terminate_epoch()

    max_epochs = 3
    engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = engine.run(data=[None] * num_iterations_per_epoch, max_epochs=max_epochs)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert state.iteration == num_iterations_per_epoch * (max_epochs - 1) + \
        iteration_to_stop % num_iterations_per_epoch


def _create_mock_data_loader(epochs, batches_per_epoch):
    batches = [MagicMock()] * batches_per_epoch
    data_loader_manager = MagicMock()
    batch_iterators = [iter(batches) for _ in range(epochs)]

    data_loader_manager.__iter__.side_effect = batch_iterators
    data_loader_manager.__len__.return_value = batches_per_epoch

    return data_loader_manager


def test_iteration_events_are_fired():
    max_epochs = 5
    num_batches = 3
    data = _create_mock_data_loader(max_epochs, num_batches)

    engine = Engine(MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    engine.add_event_handler(Events.ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    engine.run(data, max_epochs=max_epochs)

    assert iteration_started.call_count == num_batches * max_epochs
    assert iteration_complete.call_count == num_batches * max_epochs

    expected_calls = []
    for i in range(max_epochs * num_batches):
        expected_calls.append(call.iteration_started(engine))
        expected_calls.append(call.iteration_complete(engine))

    assert mock_manager.mock_calls == expected_calls


def test_last_event_name():
    engine = Engine(MagicMock(return_value=1))
    assert engine.last_event_name is None

    @engine.on(Events.STARTED)
    def _(_engine):
        assert _engine.last_event_name == Events.STARTED

    @engine.on(Events.EPOCH_STARTED)
    def _(_engine):
        assert _engine.last_event_name == Events.EPOCH_STARTED

    @engine.on(Events.ITERATION_STARTED)
    def _(_engine):
        assert _engine.last_event_name == Events.ITERATION_STARTED

    @engine.on(Events.ITERATION_COMPLETED)
    def _(_engine):
        assert _engine.last_event_name == Events.ITERATION_COMPLETED

    @engine.on(Events.EPOCH_COMPLETED)
    def _(_engine):
        assert _engine.last_event_name == Events.EPOCH_COMPLETED

    engine.run([0, 1])
    assert engine.last_event_name == Events.COMPLETED


def test_reset_should_terminate():

    def update_fn(engine, batch):
        pass

    engine = Engine(update_fn)

    @engine.on(Events.ITERATION_COMPLETED)
    def terminate_on_iteration_10(engine):
        if engine.state.iteration == 10:
            engine.terminate()

    engine.run([0] * 20)
    assert engine.state.iteration == 10

    engine.run([0] * 20)
    assert engine.state.iteration == 10


def test_batch_values():

    def _test(data):
        # This test check the content passed to update function
        counter = [0]
        num_iters = len(data)

        def update_fn(engine, batch):
            assert batch == data[counter[0] % num_iters]
            counter[0] += 1

        engine = Engine(update_fn)
        engine.run(data, max_epochs=10)

    data = torch.randint(0, 1000, size=(256, ))
    _test(data)


def test_state_repr():

    data = [0, 1, 2, 3, 4, 5]
    max_epochs = 1
    metrics = {"accuracy": Mock()}
    state = State(dataloader=data, max_epochs=max_epochs, metrics=metrics)
    s = repr(state)
    assert "iteration" in s
    assert "epoch" in s
    assert "max_epochs: 1" in s
    assert "dataloader" in s
    assert "metrics" in s
    assert "output" in s
    assert "batch" in s


def test_alter_batch():

    small_shape = (1, 2, 2)
    large_shape = (1, 3, 3)

    small_loader = torch.randint(0, 256, size=(30, ) + small_shape)
    large_loader = torch.randint(0, 256, size=(20, ) + large_shape)

    switch_iteration = 50

    def should_take_large_img(i):
        return i >= switch_iteration

    def update_fn(engine, batch):
        i = engine.state.iteration
        if i < switch_iteration:
            assert batch.shape == small_shape
            assert (small_loader[(i - 1) % len(small_loader), ...] == batch).all()
        else:
            assert batch.shape == large_shape
            assert (large_loader[(i - switch_iteration) % len(large_loader), ...] == batch).all()

    trainer = Engine(update_fn)

    def cycle(seq):
        while True:
            for i in seq:
                yield i

    small_loader_iter = cycle(small_loader)
    large_loader_iter = cycle(large_loader)

    @trainer.on(Events.ITERATION_STARTED)
    def choose_batch(engine):
        i = engine.state.iteration
        if should_take_large_img(i):
            batch = next(large_loader_iter)
        else:
            batch = next(small_loader_iter)

        engine.state.batch = batch

    num_epochs = 5
    num_iters = 25
    data = list(range(num_iters))
    trainer.run(data, num_epochs)


def test__is_done():
    state = State(iteration=10, epoch=1, max_epochs=100, epoch_length=100)
    assert not Engine._is_done(state)

    state = State(iteration=1000, max_epochs=10, epoch_length=100)
    assert Engine._is_done(state)


def test__setup_engine():
    engine = Engine(lambda e, b: 1)
    engine.state = State(iteration=10, epoch=1, max_epochs=100, epoch_length=100, seed=12)

    data = list(range(100))
    engine.state.dataloader = data
    engine._setup_engine()
    assert engine._dataloader_len == len(data)


def test_run_asserts():
    engine = Engine(lambda e, b: 1)

    with pytest.raises(ValueError, match=r"Input data has zero size. Please provide non-empty data"):
        engine.run([])

    with pytest.raises(ValueError, match=r"Argument `epoch_length` should be defined if `data` is an iterator"):
        engine.run(iter([0, 1, 2, 3]))


def test_state_get_event_attrib_value():
    state = State()
    state.iteration = 10
    state.epoch = 9

    e = Events.ITERATION_STARTED
    assert state.get_event_attrib_value(e) == state.iteration
    e = Events.ITERATION_COMPLETED
    assert state.get_event_attrib_value(e) == state.iteration
    e = Events.EPOCH_STARTED
    assert state.get_event_attrib_value(e) == state.epoch
    e = Events.EPOCH_COMPLETED
    assert state.get_event_attrib_value(e) == state.epoch
    e = Events.STARTED
    assert state.get_event_attrib_value(e) == state.epoch
    e = Events.COMPLETED
    assert state.get_event_attrib_value(e) == state.epoch

    e = Events.ITERATION_STARTED(every=10)
    assert state.get_event_attrib_value(e) == state.iteration
    e = Events.ITERATION_COMPLETED(every=10)
    assert state.get_event_attrib_value(e) == state.iteration
    e = Events.EPOCH_STARTED(once=5)
    assert state.get_event_attrib_value(e) == state.epoch
    e = Events.EPOCH_COMPLETED(once=5)
    assert state.get_event_attrib_value(e) == state.epoch


def _test_run_check_triggered_events():

    def _test(data, max_epochs, epoch_length):
        engine = Engine(lambda e, b: 1)

        events = [Events.STARTED, Events.EPOCH_STARTED, Events.ITERATION_STARTED,
                  Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED, Events.COMPLETED]

        handlers = {e: MagicMock() for e in events}

        for e, handler in handlers.items():
            engine.add_event_handler(e, handler)

        engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        expected_num_calls = {
            Events.STARTED: 1,
            Events.COMPLETED: 1,
            Events.EPOCH_STARTED: max_epochs,
            Events.EPOCH_COMPLETED: max_epochs,
            Events.ITERATION_STARTED: max_epochs * epoch_length,
            Events.ITERATION_COMPLETED: max_epochs * epoch_length,
        }

        for n, handler in handlers.items():
            assert handler.call_count == expected_num_calls[n], \
                "{}: {} vs {}".format(n, handler.call_count, expected_num_calls[n])

    _test(list(range(100)), max_epochs=5, epoch_length=100)
    _test(list(range(100)), max_epochs=5, epoch_length=50)
    _test(list(range(100)), max_epochs=5, epoch_length=150)


def test_run_check_triggered_events():

    _test_run_check_triggered_events()


def _test_run_check_triggered_events_on_iterator():

    def _test(data, max_epochs, epoch_length):
        engine = Engine(lambda e, b: 1)

        events = [Events.STARTED, Events.EPOCH_STARTED, Events.ITERATION_STARTED,
                  Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED, Events.COMPLETED]

        handlers = {e: MagicMock() for e in events}

        for e, handler in handlers.items():
            engine.add_event_handler(e, handler)

        engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        expected_num_calls = {
            Events.STARTED: 1,
            Events.COMPLETED: 1,
            Events.EPOCH_STARTED: max_epochs,
            Events.EPOCH_COMPLETED: max_epochs,
            Events.ITERATION_STARTED: max_epochs * epoch_length,
            Events.ITERATION_COMPLETED: max_epochs * epoch_length,
        }

        for n, handler in handlers.items():
            assert handler.call_count == expected_num_calls[n], \
                "{}: {} vs {}".format(n, handler.call_count, expected_num_calls[n])

    def infinite_data_iterator():
        while True:
            for i in range(100):
                yield i

    _test(infinite_data_iterator(), max_epochs=5, epoch_length=100)
    _test(infinite_data_iterator(), max_epochs=5, epoch_length=50)
    _test(infinite_data_iterator(), max_epochs=5, epoch_length=150)

    def limited_data_iterator():
        for i in range(100):
            yield i

    _test(limited_data_iterator(), max_epochs=1, epoch_length=100)
    _test(limited_data_iterator(), max_epochs=10, epoch_length=10)

    # These tests will fail
    with pytest.raises(AssertionError):
        with pytest.warns(UserWarning, match=r"Data iterator can not provide data anymore"):
            _test(limited_data_iterator(), max_epochs=3, epoch_length=100)

    with pytest.raises(AssertionError):
        with pytest.warns(UserWarning, match=r"Data iterator can not provide data anymore"):
            _test(limited_data_iterator(), max_epochs=3, epoch_length=75)

    with pytest.raises(AssertionError):
        with pytest.warns(UserWarning, match=r"Data iterator can not provide data anymore"):
            _test(limited_data_iterator(), max_epochs=1, epoch_length=101)


def test_run_check_triggered_events_on_iterator():

    _test_run_check_triggered_events_on_iterator()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):
    _test_run_check_triggered_events_on_iterator()
    _test_run_check_triggered_events()


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    _test_run_check_triggered_events_on_iterator()
    _test_run_check_triggered_events()


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    _test_run_check_triggered_events_on_iterator()
    _test_run_check_triggered_events()


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    _test_run_check_triggered_events_on_iterator()
    _test_run_check_triggered_events()

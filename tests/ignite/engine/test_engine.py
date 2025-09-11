import os
import time
from unittest.mock import call, MagicMock, Mock

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events, State
from ignite.engine.deterministic import keep_random_state
from ignite.metrics import Average
from tests.ignite.engine import BatchChecker, EpochCounter, IterationCounter


class RecordedEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.called_events = []

    def _fire_event(self, event_name, *event_args, **event_kwargs):
        self.called_events.append((self.state.epoch, self.state.iteration, event_name.name))
        return super()._fire_event(event_name, *event_args, **event_kwargs)


def _create_mock_data_loader(epochs, batches_per_epoch):
    batches = [MagicMock()] * batches_per_epoch
    data_loader_manager = MagicMock()
    batch_iterators = [iter(batches) for _ in range(epochs)]

    data_loader_manager.__iter__.side_effect = batch_iterators
    data_loader_manager.__len__.return_value = batches_per_epoch

    return data_loader_manager


@pytest.mark.parametrize("interrupt_resume_enabled", [False, True])
class TestEngine:
    @pytest.fixture(autouse=True)
    def set_interrupt_resume_enabled(self, interrupt_resume_enabled):
        Engine.interrupt_resume_enabled = interrupt_resume_enabled

    @pytest.mark.parametrize("skip_completed", [True, False])
    def test_terminate(self, skip_completed):
        engine = Engine(lambda e, b: 1)
        assert not engine.should_terminate

        engine.terminate(skip_completed)

        if skip_completed:
            assert engine.should_terminate == "skip_completed"
        else:
            assert engine.should_terminate == True  # noqa: E712

    def test_invalid_process_raises_with_invalid_signature(self):
        with pytest.raises(ValueError, match=r"Engine must be given a processing function in order to run"):
            Engine(None)

        with pytest.raises(ValueError, match=r"Error adding .+ takes parameters .+ but will be called with"):
            Engine(lambda: None)

        with pytest.raises(ValueError, match=r"Error adding .+ takes parameters .+ but will be called with"):
            Engine(lambda batch: None)

        with pytest.raises(ValueError, match=r"Error adding .+ takes parameters .+ but will be called with"):
            Engine(lambda engine, batch, extra_arg: None)

    def test_invalid_input_data(self):
        engine = Engine(lambda e, b: None)

        def data():
            pass

        with pytest.raises(TypeError, match=r"Argument data should be iterable"):
            engine.run(data)

    @pytest.mark.parametrize("data", [None, [1, 2]])
    def test_current_epoch_counter_increases_every_epoch(self, data):
        engine = Engine(MagicMock(return_value=1))
        max_epochs = 5

        counter = EpochCounter()
        engine.add_event_handler(Events.EPOCH_STARTED, counter)

        state = engine.run(data, max_epochs=max_epochs, epoch_length=2)
        assert state.epoch == max_epochs
        counter.current_epoch_count = 1
        state = engine.run(data, max_epochs=max_epochs, epoch_length=2)
        assert state.epoch == max_epochs

    @pytest.mark.parametrize("data", [None, [1, 2, 3]])
    def test_current_iteration_counter_increases_every_iteration(self, data):
        engine = Engine(MagicMock(return_value=1))
        max_epochs = 5

        counter = IterationCounter()
        engine.add_event_handler(Events.ITERATION_STARTED, counter)

        epoch_length = 3
        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)
        assert state.iteration == max_epochs * epoch_length
        counter.current_iteration_count = 1
        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)
        assert state.iteration == max_epochs * epoch_length

    def test_stopping_criterion_is_max_epochs(self):
        engine = Engine(MagicMock(return_value=1))
        max_epochs = 5
        state = engine.run([1], max_epochs=max_epochs)
        assert state.epoch == max_epochs

    @pytest.mark.parametrize("data", [None, [1, 2]])
    def test_terminate_at_end_of_epoch_stops_run(self, data):
        max_epochs = 5
        last_epoch_to_run = 3

        engine = Engine(MagicMock(return_value=1))

        def end_of_epoch_handler(engine):
            if engine.state.epoch == last_epoch_to_run:
                engine.terminate()

        engine.add_event_handler(Events.EPOCH_COMPLETED, end_of_epoch_handler)

        assert not engine.should_terminate

        state = engine.run(data, max_epochs=max_epochs, epoch_length=2)

        assert state.epoch == last_epoch_to_run
        assert engine.should_terminate
        assert engine._dataloader_iter is None

    @pytest.mark.parametrize("data, epoch_length", [(None, 10), (range(10), None)])
    def test_terminate_at_start_of_epoch(self, data, epoch_length):
        max_epochs = 5
        epoch_to_terminate_on = 3
        real_epoch_length = epoch_length if data is None else len(data)

        engine = Engine(MagicMock(return_value=1))

        def start_of_epoch_handler(engine):
            if engine.state.epoch == epoch_to_terminate_on:
                engine.terminate()

        engine.add_event_handler(Events.EPOCH_STARTED, start_of_epoch_handler)

        assert not engine.should_terminate

        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        # epoch is not completed so counter is not incremented
        assert state.epoch == epoch_to_terminate_on
        assert engine.should_terminate
        assert engine._dataloader_iter is None
        assert state.iteration == ((epoch_to_terminate_on - 1) * real_epoch_length)

        # Engine continue from epoch_to_terminate_on until max_epochs
        first_epoch_iter = [None, None]

        @engine.on(Events.STARTED)
        def check_iter_epoch():
            assert engine.state.epoch == first_epoch_iter[0]
            assert engine.state.iteration == first_epoch_iter[1]

        if data is not None:
            expected_data_iter = iter(data)
            expected_iter = state.iteration

            @engine.on(Events.ITERATION_STARTED)
            def check_iter_and_data():
                nonlocal expected_data_iter, expected_iter

                expected_iter += 1
                assert engine.state.iteration == expected_iter

                try:
                    assert engine.state.batch == next(expected_data_iter)
                except StopIteration:
                    expected_data_iter = iter(data)
                    assert engine.state.batch == next(expected_data_iter)

        first_epoch_iter[0], first_epoch_iter[1] = state.epoch, state.iteration
        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        assert state.epoch == max_epochs
        assert not engine.should_terminate
        assert engine._dataloader_iter is None
        # As terminated epoch is skipped -> iterations are not incremented
        assert state.iteration == real_epoch_length * (max_epochs - 1)

    @pytest.mark.parametrize("data, epoch_length", [(None, 10), (range(10), None)])
    def test_terminate_stops_run_mid_epoch(self, data, epoch_length):
        max_epochs = 5
        iteration_to_stop = 13
        real_epoch_length = epoch_length if data is None else len(data)

        engine = Engine(MagicMock(return_value=1))

        def start_of_iteration_handler(engine):
            if engine.state.iteration == iteration_to_stop:
                engine.terminate()

        @engine.on(Events.EXCEPTION_RAISED)
        def assert_no_exceptions(ee):
            assert False, f"Engine should terminate without raising an exception, got '{type(ee)}'"

        engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)
        # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
        assert state.iteration == iteration_to_stop
        assert state.epoch == np.ceil(iteration_to_stop / real_epoch_length)  # it starts from 0
        assert engine._dataloader_iter is None

        # Engine continue from epoch_to_terminate_on until max_epochs
        first_epoch_iter = [None, None]
        num_calls_check_iter_epoch = 0

        @engine.on(Events.STARTED, first_epoch_iter)
        def check_iter_epoch(first_epoch_iter):
            nonlocal num_calls_check_iter_epoch
            assert engine.state.epoch == first_epoch_iter[0]
            assert engine.state.iteration == first_epoch_iter[1]
            num_calls_check_iter_epoch += 1

        if data is not None:
            expected_iter = state.iteration

            @engine.on(Events.ITERATION_STARTED)
            def check_iter_and_data():
                nonlocal expected_iter

                expected_iter += 1
                assert engine.state.iteration == expected_iter
                assert engine.state.batch == data[(expected_iter - first_epoch_iter[1] - 1) % len(data)]

        first_epoch_iter[0], first_epoch_iter[1] = state.epoch, state.iteration
        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        assert state.epoch == max_epochs
        assert not engine.should_terminate
        assert state.iteration == real_epoch_length * (max_epochs - 1) + (iteration_to_stop % real_epoch_length)
        assert num_calls_check_iter_epoch == 1

    @pytest.mark.parametrize(
        "terminate_event, e, i, skip_completed",
        [
            (Events.STARTED, 0, 0, True),
            (Events.EPOCH_STARTED(once=2), 2, None, True),
            (Events.EPOCH_COMPLETED(once=2), 2, None, True),
            (Events.GET_BATCH_STARTED(once=12), None, 12, True),
            (Events.GET_BATCH_COMPLETED(once=12), None, 12, False),
            (Events.ITERATION_STARTED(once=14), None, 14, True),
            (Events.ITERATION_COMPLETED(once=14), None, 14, True),
            (Events.STARTED, 0, 0, False),
            (Events.EPOCH_STARTED(once=2), 2, None, False),
            (Events.EPOCH_COMPLETED(once=2), 2, None, False),
            (Events.GET_BATCH_STARTED(once=12), None, 12, False),
            (Events.GET_BATCH_COMPLETED(once=12), None, 12, False),
            (Events.ITERATION_STARTED(once=14), None, 14, False),
            (Events.ITERATION_COMPLETED(once=14), None, 14, False),
        ],
    )
    def test_terminate_events_sequence(self, terminate_event, e, i, skip_completed):
        engine = RecordedEngine(MagicMock(return_value=1))
        data = range(10)
        max_epochs = 5

        @engine.on(terminate_event)
        def call_terminate():
            engine.terminate(skip_completed)

        @engine.on(Events.EXCEPTION_RAISED)
        def assert_no_exceptions(ee):
            assert False, f"Engine should terminate without raising an exception, got '{type(ee)}'"

        engine.run(data, max_epochs=max_epochs)

        if i is None:
            if terminate_event == Events.EPOCH_STARTED:
                i = len(data) * (e - 1)
            else:
                i = len(data) * e

        if e is None:
            e = i // len(data) + 1

        if skip_completed:
            assert engine.called_events[-1] == (e, i, Events.TERMINATE)
            assert engine.called_events[-2] == (e, i, terminate_event)
        else:
            assert engine.called_events[-1] == (e, i, Events.COMPLETED)
            assert engine.called_events[-2] == (e, i, Events.TERMINATE)
            assert engine.called_events[-3] == (e, i, terminate_event)

        assert engine.called_events[0] == (0, 0, Events.STARTED)
        assert engine._dataloader_iter is None

    @pytest.mark.parametrize(
        "data, epoch_length, skip_epoch_completed",
        [(None, 10, False), (range(10), None, False), (None, 10, True), (range(10), None, True)],
    )
    def test_terminate_epoch_stops_mid_epoch(self, data, epoch_length, skip_epoch_completed):
        real_epoch_length = epoch_length if data is None else len(data)
        iteration_to_stop = real_epoch_length + 4

        engine = Engine(MagicMock(return_value=1))

        def start_of_iteration_handler(engine):
            if engine.state.iteration == iteration_to_stop:
                engine.terminate_epoch(skip_epoch_completed)

        max_epochs = 3
        engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
        state = engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)
        # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
        true_value = real_epoch_length * (max_epochs - 1) + iteration_to_stop % real_epoch_length
        assert state.iteration == true_value
        assert state.epoch == max_epochs

    @pytest.mark.parametrize(
        "terminate_epoch_event, i, skip_epoch_completed",
        [
            (Events.GET_BATCH_STARTED(once=12), 12, False),
            (Events.GET_BATCH_COMPLETED(once=12), 12, False),
            (Events.ITERATION_STARTED(once=14), 14, False),
            (Events.ITERATION_COMPLETED(once=14), 14, False),
            (Events.GET_BATCH_STARTED(once=12), 12, True),
            (Events.GET_BATCH_COMPLETED(once=12), 12, True),
            (Events.ITERATION_STARTED(once=14), 14, True),
            (Events.ITERATION_COMPLETED(once=14), 14, True),
            (Events.STARTED, 30, False),
            (Events.STARTED, 30, True),
            (Events.EPOCH_STARTED(once=2), 10, False),
            (Events.EPOCH_STARTED(once=2), 10, True),
        ],
    )
    def test_terminate_epoch_events_sequence(self, terminate_epoch_event, i, skip_epoch_completed):
        engine = RecordedEngine(MagicMock(return_value=1))
        data = range(10)
        max_epochs = 3

        # TODO: Bug: Events.GET_BATCH_STARTED(once=12) is called twice !
        # prevent call_terminate_epoch to be called twice
        call_count = 0

        @engine.on(terminate_epoch_event)
        def call_terminate_epoch():
            assert not engine.should_terminate_single_epoch
            nonlocal call_count
            if call_count < 1:
                engine.terminate_epoch(skip_epoch_completed)
                if skip_epoch_completed:
                    assert engine.should_terminate_single_epoch == "skip_epoch_completed"
                else:
                    assert engine.should_terminate_single_epoch == True  # noqa: E712

            call_count += 1

        @engine.on(Events.EPOCH_STARTED)
        def check_skip_reset():
            if terminate_epoch_event != Events.EPOCH_STARTED:
                assert engine.should_terminate_single_epoch == False  # noqa: E712

        @engine.on(Events.TERMINATE_SINGLE_EPOCH)
        def check_previous_events(iter_counter):
            e = i // len(data) + 1
            assert engine.called_events[0] == (0, 0, Events.STARTED)
            assert engine.called_events[-2] == (e, i, terminate_epoch_event)
            assert engine.called_events[-1] == (e, i, Events.TERMINATE_SINGLE_EPOCH)
            if skip_epoch_completed:
                assert engine.should_terminate_single_epoch == "skip_epoch_completed"
            else:
                assert engine.should_terminate_single_epoch == True  # noqa: E712

        @engine.on(Events.EPOCH_COMPLETED)
        def check_previous_events2():
            e = i // len(data) + 1
            if e == engine.state.epoch and i == engine.state.iteration:
                assert not skip_epoch_completed
                assert isinstance(engine.should_terminate_single_epoch, bool)
                assert engine.called_events[-3] == (e, i, terminate_epoch_event)
                assert engine.called_events[-2] == (e, i, Events.TERMINATE_SINGLE_EPOCH)
                assert engine.called_events[-1] == (e, i, Events.EPOCH_COMPLETED)

        if terminate_epoch_event in [Events.STARTED, Events.EPOCH_STARTED]:
            with pytest.raises(RuntimeError):
                engine.run(data, max_epochs=max_epochs)
        else:
            engine.run(data, max_epochs=max_epochs)

            assert engine.state.epoch == max_epochs
            assert (max_epochs - 1) * len(data) < engine.state.iteration < max_epochs * len(data)

            epoch_completed_events = [e for e in engine.called_events if e[2] == Events.EPOCH_COMPLETED.name]
            assert len(epoch_completed_events) == max_epochs - skip_epoch_completed

    @pytest.mark.parametrize("data", [None, "mock_data_loader"])
    def test_iteration_events_are_fired(self, data):
        max_epochs = 5
        num_batches = epoch_length = 3
        if isinstance(data, str) and data == "mock_data_loader":
            data = _create_mock_data_loader(max_epochs, num_batches)
            epoch_length = None

        engine = Engine(MagicMock(return_value=1))

        mock_manager = Mock()
        iteration_started = Mock()
        engine.add_event_handler(Events.ITERATION_STARTED, iteration_started)

        iteration_complete = Mock()
        engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_complete)

        mock_manager.attach_mock(iteration_started, "iteration_started")
        mock_manager.attach_mock(iteration_complete, "iteration_complete")

        engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        assert iteration_started.call_count == num_batches * max_epochs
        assert iteration_complete.call_count == num_batches * max_epochs

        expected_calls = []
        for _ in range(max_epochs * num_batches):
            expected_calls.append(call.iteration_started(engine))
            expected_calls.append(call.iteration_complete(engine))

        assert mock_manager.mock_calls == expected_calls

    @pytest.mark.parametrize("data", [None, [1, 2]])
    def test_last_event_name(self, data):
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

        epoch_length = 2 if data is None else None
        engine.run(data, epoch_length=epoch_length)
        assert engine.last_event_name == Events.COMPLETED

    def test_reset_should_terminate(self):
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

    def test_batch_values(self):
        def _test(data):
            # This test check the content passed to update function
            counter = [0]
            num_iters = len(data)

            def update_fn(_, batch):
                assert batch == data[counter[0] % num_iters]
                counter[0] += 1

            engine = Engine(update_fn)
            engine.run(data, max_epochs=10)

        data = torch.randint(0, 1000, size=(256,))
        _test(data)

    def test_state_repr(self):
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

    def test_alter_batch(self):
        small_shape = (1, 2, 2)
        large_shape = (1, 3, 3)

        small_loader = torch.randint(0, 256, size=(30,) + small_shape)
        large_loader = torch.randint(0, 256, size=(20,) + large_shape)

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
        data = range(num_iters)
        trainer.run(data, num_epochs)

    def test__is_done(self):
        state = State(iteration=10, epoch=1, max_epochs=100, epoch_length=100)
        assert not Engine._is_done(state)

        state = State(iteration=1000, max_epochs=10, epoch_length=100)
        assert Engine._is_done(state)

    def test__setup_engine(self):
        engine = Engine(lambda e, b: 1)
        engine.state = State(iteration=10, epoch=1, max_epochs=100, epoch_length=100)

        data = list(range(100))
        engine.state.dataloader = data
        engine._setup_engine()
        assert engine._init_iter == 10

    def test_run_asserts(self):
        engine = Engine(lambda e, b: 1)

        with pytest.raises(ValueError, match=r"Input data has zero size. Please provide non-empty data"):
            engine.run([])

    def test_state_get_event_attrib_value(self):
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

    @pytest.mark.parametrize(
        "data, max_epochs, epoch_length", [(range(100), 2, 100), (range(200), 2, 100), (range(200), 5, 100)]
    )
    def test_time_stored_in_state(self, data, max_epochs, epoch_length):
        sleep_time = 0.01
        extra_sleep_time = 0.1
        engine = Engine(lambda e, b: time.sleep(sleep_time))

        @engine.on(Events.EPOCH_COMPLETED)
        def check_epoch_time():
            assert engine.state.times[Events.EPOCH_COMPLETED.name] >= sleep_time * epoch_length
            time.sleep(extra_sleep_time)

        @engine.on(Events.COMPLETED)
        def check_completed_time():
            assert (
                engine.state.times[Events.COMPLETED.name] >= (sleep_time * epoch_length + extra_sleep_time) * max_epochs
            )
            time.sleep(extra_sleep_time)

        engine.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

        assert engine.state.times[Events.EPOCH_COMPLETED.name] >= sleep_time * epoch_length + extra_sleep_time
        assert (
            engine.state.times[Events.COMPLETED.name]
            >= (sleep_time * epoch_length + extra_sleep_time) * max_epochs + extra_sleep_time
        )

    def _test_check_triggered_events(self, data, max_epochs, epoch_length, exp_iter_stops=None):
        engine = Engine(lambda e, b: 1)
        events = [
            Events.STARTED,
            Events.EPOCH_STARTED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.EPOCH_COMPLETED,
            Events.COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
            Events.DATALOADER_STOP_ITERATION,
        ]

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
            Events.GET_BATCH_STARTED: max_epochs * epoch_length if data is not None else 0,
            Events.GET_BATCH_COMPLETED: max_epochs * epoch_length if data is not None else 0,
            Events.DATALOADER_STOP_ITERATION: (max_epochs - 1) if exp_iter_stops is None else exp_iter_stops,
        }

        for n, handler in handlers.items():
            assert handler.call_count == expected_num_calls[n], f"{n}: {handler.call_count} vs {expected_num_calls[n]}"

    def _test_run_check_triggered_events(self):
        # tests issue https://github.com/pytorch/ignite/issues/818
        self._test_check_triggered_events(list(range(10)), max_epochs=4, epoch_length=10)
        self._test_check_triggered_events(list(range(100)), max_epochs=5, epoch_length=100)
        self._test_check_triggered_events(list(range(100)), max_epochs=5, epoch_length=50, exp_iter_stops=50 * 5 // 100)
        self._test_check_triggered_events(
            list(range(100)), max_epochs=5, epoch_length=150, exp_iter_stops=150 * 5 // 100
        )
        self._test_check_triggered_events(None, max_epochs=5, epoch_length=150, exp_iter_stops=0)

    def test_run_check_triggered_events_list(self):
        self._test_run_check_triggered_events()

    def _test_run_check_triggered_events_on_iterator(self):
        def infinite_data_iterator():
            while True:
                for i in range(100):
                    yield i

        self._test_check_triggered_events(infinite_data_iterator(), max_epochs=5, epoch_length=100, exp_iter_stops=0)
        self._test_check_triggered_events(infinite_data_iterator(), max_epochs=5, epoch_length=50, exp_iter_stops=0)
        self._test_check_triggered_events(infinite_data_iterator(), max_epochs=5, epoch_length=150, exp_iter_stops=0)

        def limited_data_iterator():
            for i in range(100):
                yield i

        self._test_check_triggered_events(limited_data_iterator(), max_epochs=1, epoch_length=100, exp_iter_stops=0)
        self._test_check_triggered_events(limited_data_iterator(), max_epochs=10, epoch_length=10, exp_iter_stops=0)

        # These tests should fail
        with pytest.raises(AssertionError):
            with pytest.warns(UserWarning, match=r"Data iterator can not provide data anymore"):
                self._test_check_triggered_events(limited_data_iterator(), max_epochs=3, epoch_length=100)

        with pytest.raises(AssertionError):
            with pytest.warns(UserWarning, match=r"Data iterator can not provide data anymore"):
                self._test_check_triggered_events(limited_data_iterator(), max_epochs=3, epoch_length=75)

        with pytest.raises(AssertionError):
            # Below test does not raise "Data iterator can not provide data anymore" warning as the last
            # epoch is equal max_epochs
            # with pytest.warns(UserWarning, match=r"Data iterator can not provide data anymore"):
            self._test_check_triggered_events(limited_data_iterator(), max_epochs=1, epoch_length=101)

    def test_run_check_triggered_events_on_iterator(self):
        self._test_run_check_triggered_events_on_iterator()

    @pytest.mark.distributed
    @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
    @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
    def test_distrib_nccl_gpu(self, distributed_context_single_node_nccl):
        self._test_run_check_triggered_events_on_iterator()
        self._test_run_check_triggered_events()

    @pytest.mark.distributed
    @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
    def test_distrib_gloo_cpu_or_gpu(self, distributed_context_single_node_gloo):
        self._test_run_check_triggered_events_on_iterator()
        self._test_run_check_triggered_events()

    @pytest.mark.multinode_distributed
    @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
    @pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
    def test_multinode_distrib_gloo_cpu_or_gpu(self, distributed_context_multi_node_gloo):
        self._test_run_check_triggered_events_on_iterator()
        self._test_run_check_triggered_events()

    @pytest.mark.multinode_distributed
    @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
    @pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
    def test_multinode_distrib_nccl_gpu(self, distributed_context_multi_node_nccl):
        self._test_run_check_triggered_events_on_iterator()
        self._test_run_check_triggered_events()

    def test_engine_random_state(self):
        def random_data_generator():
            while True:
                yield torch.randint(0, 100, size=(5,))

        def sum_data(_, batch):
            result = torch.sum(batch)
            return result

        def get_engine():
            engine = Engine(sum_data)
            average = Average()
            average.attach(engine, "average")
            return engine

        torch.manual_seed(34)
        engine = get_engine()
        state1 = engine.run(random_data_generator(), max_epochs=2, epoch_length=2)

        torch.manual_seed(34)
        engine = get_engine()
        state2 = engine.run(random_data_generator(), max_epochs=2, epoch_length=2)

        torch.manual_seed(42)
        engine = get_engine()
        state3 = engine.run(random_data_generator(), max_epochs=2, epoch_length=2)

        assert state1.metrics["average"] == pytest.approx(state2.metrics["average"])
        assert state1.metrics["average"] != pytest.approx(state3.metrics["average"])
        assert state2.metrics["average"] != pytest.approx(state3.metrics["average"])

    def test_altered_random_state(self):
        # tests issue https://github.com/pytorch/ignite/issues/795
        size = 1

        def random_train_data_generator(size):
            while True:
                yield torch.randint(0, 100, size=(size,))

        def random_val_data_generator(size):
            while True:
                yield torch.randint(0, 100, size=(size,)) + 100

        train_only_batches = []

        def train_fn(_, batch):
            train_only_batches.append(batch[0].item())

        torch.manual_seed(1)
        epoch_length = 6
        trainer = Engine(train_fn)
        trainer.run(random_train_data_generator(size), max_epochs=4, epoch_length=epoch_length)

        def val_fn(_1, _2):
            pass

        evaluator = Engine(val_fn)
        train_batches = []

        def train_fn2(_, batch):
            train_batches.append(batch[0].item())

        trainer = Engine(train_fn2)

        @trainer.on(Events.EPOCH_COMPLETED)
        @keep_random_state
        def run_evaluation(_):
            evaluator.run(random_val_data_generator(size), epoch_length=4)

        torch.manual_seed(1)
        trainer.run(random_train_data_generator(size), max_epochs=4, epoch_length=epoch_length)

        for i in range(epoch_length):
            assert train_batches[epoch_length + i] != train_batches[2 * epoch_length + i]
            assert train_batches[i] == train_only_batches[i]

    def test_engine_with_dataloader_no_auto_batching(self):
        # tests https://github.com/pytorch/ignite/issues/941
        from torch.utils.data import BatchSampler, DataLoader, RandomSampler

        data = torch.rand(64, 4, 10)
        data_loader = DataLoader(
            data, batch_size=None, sampler=BatchSampler(RandomSampler(data), batch_size=8, drop_last=True)
        )

        counter = [0]

        def foo(e, b):
            counter[0] += 1

        engine = Engine(foo)
        engine.run(data_loader, epoch_length=10, max_epochs=5)

        assert counter[0] == 50

    def test_run_once_finite_iterator_no_epoch_length(self):
        # FR: https://github.com/pytorch/ignite/issues/871

        unknown_size = 11

        def finite_unk_size_data_iter():
            for i in range(unknown_size):
                yield i

        bc = BatchChecker(data=list(range(unknown_size)))

        engine = Engine(lambda e, b: bc.check(b))

        completed_handler = MagicMock()
        engine.add_event_handler(Events.COMPLETED, completed_handler)

        data_iter = finite_unk_size_data_iter()
        engine.run(data_iter)

        assert engine.state.epoch == 1
        assert engine.state.iteration == unknown_size
        assert completed_handler.call_count == 1

    def test_run_finite_iterator_no_epoch_length(self):
        # FR: https://github.com/pytorch/ignite/issues/871
        unknown_size = 11

        def finite_unk_size_data_iter():
            for i in range(unknown_size):
                yield i

        bc = BatchChecker(data=list(range(unknown_size)))

        engine = Engine(lambda e, b: bc.check(b))

        @engine.on(Events.DATALOADER_STOP_ITERATION)
        def restart_iter():
            engine.state.dataloader = finite_unk_size_data_iter()

        data_iter = finite_unk_size_data_iter()
        engine.run(data_iter, max_epochs=5)

        assert engine.state.epoch == 5
        assert engine.state.iteration == unknown_size * 5

    def test_run_finite_iterator_no_epoch_length_2(self):
        # FR: https://github.com/pytorch/ignite/issues/871
        known_size = 11

        def finite_size_data_iter(size):
            for i in range(size):
                yield i

        bc = BatchChecker(data=list(range(known_size)))

        engine = Engine(lambda e, b: bc.check(b))

        @engine.on(Events.ITERATION_COMPLETED(every=known_size))
        def restart_iter():
            engine.state.dataloader = finite_size_data_iter(known_size)

        data_iter = finite_size_data_iter(known_size)
        engine.run(data_iter, max_epochs=5)

        assert engine.state.epoch == 5
        assert engine.state.iteration == known_size * 5

    def test_faq_inf_iterator_with_epoch_length(self):
        # Code snippet from FAQ
        # import torch

        torch.manual_seed(12)

        def infinite_iterator(batch_size):
            while True:
                batch = torch.rand(batch_size, 3, 32, 32)
                yield batch

        def train_step(trainer, batch):
            # ...
            s = trainer.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch.norm():.3f}")

        trainer = Engine(train_step)
        # We need to specify epoch_length to define the epoch
        trainer.run(infinite_iterator(4), epoch_length=5, max_epochs=3)

        assert trainer.state.epoch == 3
        assert trainer.state.iteration == 3 * 5

    def test_faq_inf_iterator_no_epoch_length(self):
        # Code snippet from FAQ
        # import torch

        torch.manual_seed(12)

        def infinite_iterator(batch_size):
            while True:
                batch = torch.rand(batch_size, 3, 32, 32)
                yield batch

        def train_step(trainer, batch):
            # ...
            s = trainer.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch.norm():.3f}")

        trainer = Engine(train_step)

        @trainer.on(Events.ITERATION_COMPLETED(once=15))
        def stop_training():
            trainer.terminate()

        trainer.run(infinite_iterator(4))

        assert trainer.state.epoch == 1
        assert trainer.state.iteration == 15

    def test_faq_fin_iterator_unknw_size(self):
        # Code snippet from FAQ
        # import torch

        torch.manual_seed(12)

        def finite_unk_size_data_iter():
            for i in range(11):
                yield i

        def train_step(trainer, batch):
            # ...
            s = trainer.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}")

        trainer = Engine(train_step)

        @trainer.on(Events.DATALOADER_STOP_ITERATION)
        def restart_iter():
            trainer.state.dataloader = finite_unk_size_data_iter()

        data_iter = finite_unk_size_data_iter()
        trainer.run(data_iter, max_epochs=5)

        assert trainer.state.epoch == 5
        assert trainer.state.iteration == 5 * 11

        # Code snippet from FAQ
        # import torch

        torch.manual_seed(12)

        def finite_unk_size_data_iter():
            for i in range(11):
                yield i

        def val_step(evaluator, batch):
            # ...
            s = evaluator.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}")

        evaluator = Engine(val_step)

        data_iter = finite_unk_size_data_iter()
        evaluator.run(data_iter)

        assert evaluator.state.epoch == 1
        assert evaluator.state.iteration == 1 * 11

    def test_faq_fin_iterator(self):
        # Code snippet from FAQ
        # import torch

        torch.manual_seed(12)

        size = 11

        def finite_size_data_iter(size):
            for i in range(size):
                yield i

        def train_step(trainer, batch):
            # ...
            s = trainer.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}")

        trainer = Engine(train_step)

        @trainer.on(Events.ITERATION_COMPLETED(every=size))
        def restart_iter():
            trainer.state.dataloader = finite_size_data_iter(size)

        data_iter = finite_size_data_iter(size)
        trainer.run(data_iter, max_epochs=5)

        assert trainer.state.epoch == 5
        assert trainer.state.iteration == 5 * size

        # Code snippet from FAQ
        # import torch

        torch.manual_seed(12)

        size = 11

        def finite_size_data_iter(size):
            for i in range(size):
                yield i

        def val_step(evaluator, batch):
            # ...
            s = evaluator.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch:.3f}")

        evaluator = Engine(val_step)

        data_iter = finite_size_data_iter(size)
        evaluator.run(data_iter)

        assert evaluator.state.epoch == 1
        assert evaluator.state.iteration == size

    def test_set_data(self):
        # tests FR https://github.com/pytorch/ignite/issues/833
        from torch.utils.data import DataLoader

        num_iters1 = 10
        num_iters2 = 20
        batch_size = 4

        torch.manual_seed(1)
        data1 = DataLoader(torch.rand(num_iters1 * batch_size, 11), batch_size=batch_size)
        data2 = DataLoader(torch.rand(num_iters2 * batch_size, 22), batch_size=batch_size)

        switch_iteration = 35

        def train_fn(e, batch):
            if e.state.iteration <= switch_iteration:
                assert batch.shape[1] == 11, f"{e.state.iteration}: {batch.shape}"
            else:
                assert batch.shape[1] == 22, f"{e.state.iteration}: {batch.shape}"

        trainer = Engine(train_fn)

        @trainer.on(Events.ITERATION_COMPLETED(once=switch_iteration))
        def switch_dataloader():
            trainer.set_data(data2)

        trainer.run(data1, max_epochs=10)

    def test_run_with_max_iters(self):
        max_iters = 8
        engine = Engine(lambda e, b: 1)
        engine.run([0] * 20, max_iters=max_iters)
        assert engine.state.iteration == max_iters
        assert engine.state.max_iters == max_iters

    def test_run_with_max_iters_greater_than_epoch_length(self):
        max_iters = 73
        engine = Engine(lambda e, b: 1)
        engine.run([0] * 20, max_iters=max_iters)
        assert engine.state.iteration == max_iters

    def test_run_with_invalid_max_iters_and_max_epoch(self):
        max_iters = 12
        max_epochs = 2
        engine = Engine(lambda e, b: 1)
        with pytest.raises(
            ValueError,
            match=r"Arguments max_iters and max_epochs are mutually exclusive."
            "Please provide only max_epochs or max_iters.",
        ):
            engine.run([0] * 20, max_iters=max_iters, max_epochs=max_epochs)

    def test_epoch_events_fired_max_iters(self):
        max_iters = 32
        engine = Engine(lambda e, b: 1)

        @engine.on(Events.EPOCH_COMPLETED)
        def fired_event(engine):
            assert engine.state.iteration % engine.state.epoch_length == 0

        engine.run([0] * 10, max_iters=max_iters)

    def test_is_done_with_max_iters(self):
        state = State(iteration=100, epoch=1, max_epochs=3, epoch_length=100, max_iters=250)
        assert not Engine._is_done(state)

        state = State(iteration=250, epoch=1, max_epochs=3, epoch_length=100, max_iters=250)
        assert Engine._is_done(state)

    @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
    def test_batch_is_released_before_new_one_is_loaded_on_cuda(self):
        torch.cuda.empty_cache()

        engine = Engine(lambda e, b: None)

        def _test():
            mem_consumption = []

            def dataloader():
                for _ in range(4):
                    mem_consumption.append(torch.cuda.memory_allocated())
                    batch = torch.randn(10).cuda()
                    mem_consumption.append(torch.cuda.memory_allocated())
                    yield batch

            engine.run(dataloader(), max_epochs=2, epoch_length=2)
            return mem_consumption

        mem_consumption1 = _test()
        # mem_consumption should look like [0, 512, 512, 512, 512, 512, 512, 512]
        assert len(set(mem_consumption1[1:])) == 1

        mem_consumption2 = _test()
        assert len(set(mem_consumption2[1:])) == 1

        assert mem_consumption1 == mem_consumption2

    @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
    def test_output_is_released_before_new_one_is_assigned_on_cuda(self):
        torch.cuda.empty_cache()

        def _test():
            mem_consumption = []

            def update_fn(engine, batch):
                mem_consumption.append(torch.cuda.memory_allocated())
                output = torch.rand(10).cuda()
                mem_consumption.append(torch.cuda.memory_allocated())
                return output

            engine = Engine(update_fn)
            engine.run([0, 1], max_epochs=2)

            return mem_consumption

        mem_consumption1 = _test()[2:]
        # mem_consumption ~ [0, 512, 0, 512, 0, 512, 0, 512]
        assert len(set(mem_consumption1)) == 2

        mem_consumption2 = _test()[2:]
        assert len(set(mem_consumption2)) == 2

        assert mem_consumption1 == mem_consumption2

    def test_has_registered_events_builtin(self):
        """Test has_registered_events with built-in events."""
        engine = Engine(lambda e, b: None)

        # Built-in events should be registered by default
        assert engine.has_registered_events(Events.STARTED)
        assert engine.has_registered_events(Events.COMPLETED)
        assert engine.has_registered_events(Events.ITERATION_COMPLETED)

        # Non-existent event should return False
        assert not engine.has_registered_events("non_existent_event")

    def test_engine_no_data_asserts(self):
        trainer = Engine(lambda e, b: None)

        with pytest.raises(ValueError, match=r"epoch_length should be provided if data is None"):
            trainer.run(max_epochs=10)

    def test_engine_no_data(self):
        def train_step(engine, batch):
            assert batch is None

        trainer = Engine(train_step)
        trainer.run(max_epochs=10, epoch_length=10)

        assert trainer.state.iteration == 10 * 10
        assert trainer.state.epoch == 10
        assert trainer.state.dataloader is None

        # continue
        trainer.run(max_epochs=20)

        assert trainer.state.iteration == 20 * 10
        assert trainer.state.epoch == 20
        assert trainer.state.dataloader is None

    def test_engine_no_data_events(self):
        # Reproduces the issue https://github.com/pytorch/ignite/issues/3190
        max_epochs = 4
        dataset = range(10)

        def training_step(engine, _):
            assert engine.state.dataloader is None

        trainer = Engine(training_step)
        trainer.state.dataiter = iter(dataset)

        @trainer.on(Events.DATALOADER_STOP_ITERATION)
        @trainer.on(Events.GET_BATCH_STARTED)
        @trainer.on(Events.GET_BATCH_COMPLETED)
        def should_not_be_called():
            assert False, trainer.last_event_name

        trainer.run(max_epochs=max_epochs, epoch_length=4)

    @pytest.mark.parametrize("data, epoch_length", [(None, 10), (range(10), None)])
    def test_engine_run_resume(self, data, epoch_length):
        # https://github.com/pytorch/ignite/wiki/Roadmap#runresume-logic-improvements
        engine = Engine(lambda e, b: None)
        real_epoch_length = len(data) if data is not None else epoch_length

        first_epoch_iter = [None, None]

        @engine.on(Events.STARTED, first_epoch_iter)
        def check_iter_epoch(first_epoch_iter):
            assert engine.state.epoch == first_epoch_iter[0]
            assert engine.state.iteration == first_epoch_iter[1]

        # (re)start from 0 to 5
        first_epoch_iter[0], first_epoch_iter[1] = 0, 0
        # Engine run starting with max_epochs=5 => state.epoch=5
        engine.run(data, max_epochs=5, epoch_length=epoch_length)
        assert engine.state.epoch == 5
        assert engine.state.iteration == 5 * real_epoch_length

        # continue from 5 to 7
        first_epoch_iter[0], first_epoch_iter[1] = 5, 5 * real_epoch_length
        # Engine run resuming from iteration 50, epoch 5 until 7 epochs => state.epoch=7
        engine.run(data, max_epochs=7, epoch_length=epoch_length)
        assert engine.state.epoch == 7
        assert engine.state.iteration == 7 * real_epoch_length

        # error
        with pytest.raises(ValueError, match="Argument max_epochs should be greater than or equal to the start epoch"):
            engine.run(data, max_epochs=4, epoch_length=epoch_length)

        # restart from 0 to 7 (As state.epoch == max_epochs(=7),
        # this should be like that as we always do: evaluator.run(data) without any other instructions)
        first_epoch_iter[0], first_epoch_iter[1] = 0, 0
        # Engine run starting with max_epochs=7 => state.epoch=7
        engine.run(data, max_epochs=7, epoch_length=epoch_length)
        assert engine.state.epoch == 7
        assert engine.state.iteration == 7 * real_epoch_length

        # forced restart from 0 to 5
        engine.state.max_epochs = None
        first_epoch_iter[0], first_epoch_iter[1] = 0, 0
        # Engine run starting with max_epochs=5 => state.epoch=5
        engine.run(data, max_epochs=5, epoch_length=epoch_length)
        assert engine.state.epoch == 5
        assert engine.state.iteration == 5 * real_epoch_length

        # forced restart from 0 to 9, instead of continue from state.epoch=5
        engine.state.max_epochs = None
        first_epoch_iter[0], first_epoch_iter[1] = 0, 0
        # Engine run starting with max_epochs=9 => state.epoch=9
        engine.run(data, max_epochs=9, epoch_length=epoch_length)
        assert engine.state.epoch == 9
        assert engine.state.iteration == 9 * real_epoch_length

        # continue from 9 until 10
        first_epoch_iter[0], first_epoch_iter[1] = 9, 9 * real_epoch_length
        # Engine run resuming from iteration 90, epoch 9 until 10 epochs => state.epoch=10
        engine.run(data, max_epochs=10, epoch_length=epoch_length)
        assert engine.state.epoch == 10
        assert engine.state.iteration == 10 * real_epoch_length

    def test_iterator_state_output(self):
        torch.manual_seed(12)

        batch_size = 4
        finite_map = [torch.rand(batch_size, 3, 32, 32) for _ in range(4)]

        def train_step(trainer, batch):
            s = trainer.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch.norm():.3f}")
            return "flag_value"

        trainer = Engine(train_step)
        trainer.run(iter(finite_map), max_epochs=2)

        assert trainer.state.output == "flag_value"
        assert trainer.state.epoch == 2
        # We don't reset the iterator so only 1 epoch runs
        assert trainer.state.iteration == 1 * 4

    def test_map_state_output(self):
        torch.manual_seed(12)

        batch_size = 4
        finite_map = [torch.rand(batch_size, 3, 32, 32) for _ in range(4)]

        def train_step(trainer, batch):
            s = trainer.state
            print(f"{s.epoch}/{s.max_epochs} : {s.iteration} - {batch.norm():.3f}")
            return "flag_value"

        trainer = Engine(train_step)
        trainer.run(finite_map, max_epochs=2)

        assert trainer.state.output == "flag_value"
        assert trainer.state.epoch == 2
        assert trainer.state.iteration == 2 * 4


@pytest.mark.parametrize(
    "interrupt_event, e, i",
    [
        (Events.EPOCH_STARTED(once=2), 2, None),
        (Events.EPOCH_COMPLETED(once=2), 2, None),
        (Events.GET_BATCH_STARTED(once=12), None, 12),
        (Events.GET_BATCH_COMPLETED(once=12), None, 12),
        (Events.ITERATION_STARTED(once=14), None, 14),
        (Events.ITERATION_COMPLETED(once=14), None, 14),
    ],
)
def test_engine_run_interrupt_resume(interrupt_event, e, i):
    assert Engine.interrupt_resume_enabled

    data = range(10)
    max_epochs = 5

    def check_input_data(e, b):
        i = (e.state.iteration - 1) % len(data)
        assert b == data[i]

    engine = RecordedEngine(check_input_data)
    engine.run(data, max_epochs=max_epochs)

    expected_called_events = list(engine.called_events)
    engine.called_events = []

    @engine.on(interrupt_event)
    def call_interrupt():
        engine.interrupt()

    state = engine.run(data, max_epochs=max_epochs)

    if i is None:
        if interrupt_event == Events.EPOCH_STARTED:
            i = len(data) * (e - 1)
        else:
            i = len(data) * e

    if e is None:
        e = i // len(data) + 1

    # Check the last events
    assert engine.called_events[-1] == (e, i, Events.INTERRUPT)
    assert engine.called_events[-2] == (e, i, interrupt_event)
    assert state.epoch == e
    assert state.iteration == i
    assert not engine.should_interrupt
    # implementation detail check:
    assert engine._dataloader_iter is not None
    assert engine._internal_run_generator is not None

    le = len(engine.called_events)
    # We need to skip the last INTERRUPT event to compare
    assert expected_called_events[: le - 1] == engine.called_events[:-1]

    engine.called_events = []

    @engine.on(Events.STARTED)
    def raise_error():
        raise RuntimeError("Shouldn't be here")

    engine.run(data, max_epochs=max_epochs)

    assert expected_called_events[le - 1 :] == engine.called_events
    # implementation detail check:
    assert engine._dataloader_iter is None
    assert engine._internal_run_generator is None


def test_engine_run_multiple_interrupt_resume():
    assert Engine.interrupt_resume_enabled

    data = range(10)
    max_epochs = 3

    def check_input_data(e, b):
        i = (e.state.iteration - 1) % len(data)
        assert b == data[i]

    engine = Engine(check_input_data)

    can_interrupt = True

    @engine.on(Events.ITERATION_COMPLETED(every=6))
    def call_interrupt():
        if can_interrupt:
            engine.interrupt()

    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == 6 * 1 and state.epoch == 1
    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == 6 * 2 and state.epoch == 2
    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == 6 * 3 and state.epoch == 2
    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == 6 * 4 and state.epoch == 3
    # We did an interruption on the last epoch
    assert state.epoch == max_epochs

    # Run remaining iterations without interruptions
    can_interrupt = False

    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == max_epochs * len(data) and state.epoch == max_epochs
    # Check implementation details
    assert engine._dataloader_iter is None
    assert engine._internal_run_generator is None

    # Rerun the engine from start to end without interruptions
    num_calls_check_iter_epoch = 0

    @engine.on(Events.STARTED)
    def check_iter_epoch():
        nonlocal num_calls_check_iter_epoch
        assert engine.state.epoch == 0
        assert engine.state.iteration == 0
        num_calls_check_iter_epoch += 1

    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == max_epochs * len(data) and state.epoch == max_epochs
    assert num_calls_check_iter_epoch == 1


def test_engine_should_interrupt_error():
    Engine.interrupt_resume_enabled = False

    engine = Engine(lambda e, b: None)

    with pytest.raises(RuntimeError, match="Engine 'interrupt/resume' feature is disabled"):
        engine.interrupt()

    Engine.interrupt_resume_enabled = True


def test_engine_interrupt_restart():
    assert Engine.interrupt_resume_enabled

    data = range(10)
    max_epochs = 3

    def check_input_data(e, b):
        i = (e.state.iteration - 1) % len(data)
        assert b == data[i]

    engine = Engine(check_input_data)
    can_interrupt = True

    @engine.on(Events.ITERATION_COMPLETED(every=11))
    def call_interrupt():
        if can_interrupt:
            engine.interrupt()

    # Run and interrupt
    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == 11 and state.epoch == 2

    num_calls_check_iter_epoch = 0

    @engine.on(Events.STARTED)
    def check_iter_epoch():
        nonlocal num_calls_check_iter_epoch
        assert engine.state.epoch == 0
        assert engine.state.iteration == 0
        num_calls_check_iter_epoch += 1

    # Reset and run with interruption
    state.max_epochs = None
    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == 11 and state.epoch == 2
    assert num_calls_check_iter_epoch == 1

    can_interrupt = False
    num_calls_check_iter_epoch = 0
    # Reset and run without interruption
    state.max_epochs = None
    state = engine.run(data, max_epochs=max_epochs)
    assert state.iteration == max_epochs * len(data) and state.epoch == max_epochs
    assert num_calls_check_iter_epoch == 1

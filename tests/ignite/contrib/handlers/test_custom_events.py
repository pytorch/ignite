import math

import pytest

from ignite.contrib.handlers.custom_events import CustomPeriodicEvent
from ignite.engine import Engine


def test_bad_input():
    with pytest.warns(DeprecationWarning, match=r"CustomPeriodicEvent is deprecated"):
        with pytest.raises(TypeError, match="Argument n_iterations should be an integer"):
            CustomPeriodicEvent(n_iterations="a")
        with pytest.raises(ValueError, match="Argument n_iterations should be positive"):
            CustomPeriodicEvent(n_iterations=0)
        with pytest.raises(TypeError, match="Argument n_iterations should be an integer"):
            CustomPeriodicEvent(n_iterations=10.0)
        with pytest.raises(TypeError, match="Argument n_epochs should be an integer"):
            CustomPeriodicEvent(n_epochs="a")
        with pytest.raises(ValueError, match="Argument n_epochs should be positive"):
            CustomPeriodicEvent(n_epochs=0)
        with pytest.raises(TypeError, match="Argument n_epochs should be an integer"):
            CustomPeriodicEvent(n_epochs=10.0)
        with pytest.raises(ValueError, match="Either n_iterations or n_epochs should be defined"):
            CustomPeriodicEvent()
        with pytest.raises(ValueError, match="Either n_iterations or n_epochs should be defined"):
            CustomPeriodicEvent(n_iterations=1, n_epochs=2)


def test_new_events():
    def update(*args, **kwargs):
        pass

    with pytest.warns(DeprecationWarning, match="CustomPeriodicEvent is deprecated"):
        engine = Engine(update)
        cpe = CustomPeriodicEvent(n_iterations=5)
        cpe.attach(engine)

        assert hasattr(cpe, "Events")
        assert hasattr(cpe.Events, "ITERATIONS_5_STARTED")
        assert hasattr(cpe.Events, "ITERATIONS_5_COMPLETED")

        assert engine._allowed_events[-2] == getattr(cpe.Events, "ITERATIONS_5_STARTED")
        assert engine._allowed_events[-1] == getattr(cpe.Events, "ITERATIONS_5_COMPLETED")

    with pytest.warns(DeprecationWarning, match="CustomPeriodicEvent is deprecated"):
        cpe = CustomPeriodicEvent(n_epochs=5)
        cpe.attach(engine)

        assert hasattr(cpe, "Events")
        assert hasattr(cpe.Events, "EPOCHS_5_STARTED")
        assert hasattr(cpe.Events, "EPOCHS_5_COMPLETED")

        assert engine._allowed_events[-2] == getattr(cpe.Events, "EPOCHS_5_STARTED")
        assert engine._allowed_events[-1] == getattr(cpe.Events, "EPOCHS_5_COMPLETED")


def test_integration_iterations():
    def _test(n_iterations, max_epochs, n_iters_per_epoch):
        def update(*args, **kwargs):
            pass

        engine = Engine(update)
        with pytest.warns(DeprecationWarning, match="CustomPeriodicEvent is deprecated"):
            cpe = CustomPeriodicEvent(n_iterations=n_iterations)
            cpe.attach(engine)
        data = list(range(n_iters_per_epoch))

        custom_period = [0]
        n_calls_iter_started = [0]
        n_calls_iter_completed = [0]

        event_started = getattr(cpe.Events, "ITERATIONS_{}_STARTED".format(n_iterations))

        @engine.on(event_started)
        def on_my_event_started(engine):
            assert (engine.state.iteration - 1) % n_iterations == 0
            custom_period[0] += 1
            custom_iter = getattr(engine.state, "iterations_{}".format(n_iterations))
            assert custom_iter == custom_period[0]
            n_calls_iter_started[0] += 1

        event_completed = getattr(cpe.Events, "ITERATIONS_{}_COMPLETED".format(n_iterations))

        @engine.on(event_completed)
        def on_my_event_ended(engine):
            assert engine.state.iteration % n_iterations == 0
            custom_iter = getattr(engine.state, "iterations_{}".format(n_iterations))
            assert custom_iter == custom_period[0]
            n_calls_iter_completed[0] += 1

        engine.run(data, max_epochs=max_epochs)

        n = len(data) * max_epochs / n_iterations
        nf = math.floor(n)
        assert custom_period[0] == n_calls_iter_started[0]
        assert n_calls_iter_started[0] == nf + 1 if nf < n else nf
        assert n_calls_iter_completed[0] == nf

    _test(3, 5, 16)
    _test(4, 5, 16)
    _test(5, 5, 16)
    _test(300, 50, 1000)


def test_integration_epochs():
    def update(*args, **kwargs):
        pass

    engine = Engine(update)

    n_epochs = 3
    with pytest.warns(DeprecationWarning, match="CustomPeriodicEvent is deprecated"):
        cpe = CustomPeriodicEvent(n_epochs=n_epochs)
        cpe.attach(engine)
    data = list(range(16))

    custom_period = [1]

    @engine.on(cpe.Events.EPOCHS_3_STARTED)
    def on_my_epoch_started(engine):
        assert (engine.state.epoch - 1) % n_epochs == 0
        assert engine.state.epochs_3 == custom_period[0]

    @engine.on(cpe.Events.EPOCHS_3_COMPLETED)
    def on_my_epoch_ended(engine):
        assert engine.state.epoch % n_epochs == 0
        assert engine.state.epochs_3 == custom_period[0]
        custom_period[0] += 1

    engine.run(data, max_epochs=10)

    assert custom_period[0] == 4

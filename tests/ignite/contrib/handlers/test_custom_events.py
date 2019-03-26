import pytest

from ignite.engine import Engine
from ignite.contrib.handlers.custom_events import CustomPeriodicEvent


def test_bad_input():

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_iterations="a")

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_iterations=0)

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_iterations=10.0)

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_epochs="a")

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_epochs=0)

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_epochs=10.0)

    with pytest.raises(ValueError):
        CustomPeriodicEvent()

    with pytest.raises(ValueError):
        CustomPeriodicEvent(n_iterations=1, n_epochs=2)


def test_new_events():

    def update(*args, **kwargs):
        pass

    engine = Engine(update)
    cpe = CustomPeriodicEvent(n_iterations=5)
    cpe.attach(engine)

    assert hasattr(cpe, "Events")
    assert hasattr(cpe.Events, "ITERATIONS_5_STARTED")
    assert hasattr(cpe.Events, "ITERATIONS_5_COMPLETED")

    assert engine._allowed_events[-2] == getattr(cpe.Events, "ITERATIONS_5_STARTED")
    assert engine._allowed_events[-1] == getattr(cpe.Events, "ITERATIONS_5_COMPLETED")

    cpe = CustomPeriodicEvent(n_epochs=5)
    cpe.attach(engine)

    assert hasattr(cpe, "Events")
    assert hasattr(cpe.Events, "EPOCHS_5_STARTED")
    assert hasattr(cpe.Events, "EPOCHS_5_COMPLETED")

    assert engine._allowed_events[-2] == getattr(cpe.Events, "EPOCHS_5_STARTED")
    assert engine._allowed_events[-1] == getattr(cpe.Events, "EPOCHS_5_COMPLETED")


def test_integration_iterations():

    def update(*args, **kwargs):
        pass

    engine = Engine(update)

    n_iterations = 3
    cpe = CustomPeriodicEvent(n_iterations=n_iterations)
    cpe.attach(engine)
    data = list(range(16))

    custom_period = [1]

    @engine.on(cpe.Events.ITERATIONS_3_STARTED)
    def on_my_epoch_started(engine):
        assert (engine.state.iteration - 1) % n_iterations == 0
        assert engine.state.iterations_3 == custom_period[0]

    @engine.on(cpe.Events.ITERATIONS_3_COMPLETED)
    def on_my_epoch_ended(engine):
        assert engine.state.iteration % n_iterations == 0
        assert engine.state.iterations_3 == custom_period[0]
        custom_period[0] += 1

    engine.run(data, max_epochs=2)

    assert custom_period[0] == 11


def test_integration_epochs():

    def update(*args, **kwargs):
        pass

    engine = Engine(update)

    n_epochs = 3
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

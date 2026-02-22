from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine


def test_global_step_from_engine_registered_event():
    engine = Engine(lambda e, b: None)
    engine.state.iteration = 42

    transform = global_step_from_engine(engine)

    step = transform(engine, Events.ITERATION_COMPLETED)

    assert step == 42


def test_global_step_from_engine_fallback_for_unregistered_event():
    engine = Engine(lambda e, b: None)

    engine.state.epoch = 7

    transform = global_step_from_engine(engine)

    class DummyEvent:
        pass

    step = transform(engine, DummyEvent())

    assert step == 7

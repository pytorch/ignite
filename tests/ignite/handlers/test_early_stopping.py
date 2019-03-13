
import pytest

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping


def test_args_validation():

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)

    with pytest.raises(ValueError):
        h = EarlyStopping(patience=-1, score_function=lambda engine: 0, trainer=trainer)

    with pytest.raises(TypeError):
        h = EarlyStopping(patience=2, score_function=12345, trainer=trainer)

    with pytest.raises(TypeError):
        h = EarlyStopping(patience=2, score_function=lambda engine: 0, trainer=None)


def test_simple_early_stopping():

    scores = iter([1.0, 0.8, 0.88])

    def score_function(engine):
        return next(scores)

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)

    h = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
    # Call 3 times and check if stopped
    assert not trainer.should_terminate
    h(None)
    assert not trainer.should_terminate
    h(None)
    assert not trainer.should_terminate
    h(None)
    assert trainer.should_terminate


def test_simple_early_stopping_on_plateau():

    def score_function(engine):
        return 42

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)

    h = EarlyStopping(patience=1, score_function=score_function, trainer=trainer)
    # Call 2 times and check if stopped
    assert not trainer.should_terminate
    h(None)
    assert not trainer.should_terminate
    h(None)
    assert trainer.should_terminate


def test_simple_no_early_stopping():

    scores = iter([1.0, 0.8, 1.2])

    def score_function(engine):
        return next(scores)

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)

    h = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
    # Call 3 times and check if not stopped
    assert not trainer.should_terminate
    h(None)
    h(None)
    h(None)
    assert not trainer.should_terminate


def test_with_engine_early_stopping():

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = iter([1.0, 0.8, 1.2, 1.5, 0.9, 1.0, 0.99, 1.1, 0.9])

    def score_function(engine):
        return next(scores)

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)
    evaluator = Engine(update_fn)
    early_stopping = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert n_epochs_counter.count == 7


def test_with_engine_early_stopping_on_plateau():

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    def score_function(engine):
        return 0.047

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)
    evaluator = Engine(update_fn)
    early_stopping = EarlyStopping(patience=4, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert n_epochs_counter.count == 5


def test_with_engine_no_early_stopping():

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = iter([1.0, 0.8, 1.2, 1.23, 0.9, 1.0, 1.1, 1.253, 1.26, 1.2])

    def score_function(engine):
        return next(scores)

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)
    evaluator = Engine(update_fn)
    early_stopping = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert n_epochs_counter.count == 10

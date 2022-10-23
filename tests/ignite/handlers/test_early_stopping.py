import os

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, NoImprovementHandler


def do_nothing_update_fn(engine, batch):
    pass


@pytest.fixture
def trainer():
    trainer = Engine(do_nothing_update_fn)
    trainer.state_dict_user_keys.append("alpha")
    trainer.state.alpha = 0.1
    return trainer


def test_args_validation_no_improvement_handler():

    trainer = Engine(do_nothing_update_fn)

    with pytest.raises(TypeError, match=r"Argument score_function should be a function."):
        NoImprovementHandler(
            patience=2, score_function=12345, pass_function=12345, stop_function=12345, trainer=trainer
        )

    with pytest.raises(TypeError, match=r"Argument pass_function should be a function."):
        NoImprovementHandler(
            patience=2, score_function=lambda engine: 0, pass_function=12345, stop_function=12345, trainer=trainer
        )

    with pytest.raises(TypeError, match=r"Argument stop_function should be a function."):
        NoImprovementHandler(
            patience=2,
            score_function=lambda engine: 0,
            pass_function=lambda engine: 0,
            stop_function=12345,
            trainer=trainer,
        )

    with pytest.raises(TypeError, match=r"Argument trainer should be an instance of Engine."):
        NoImprovementHandler(
            patience=2,
            score_function=lambda engine: 0,
            pass_function=lambda engine: 0,
            stop_function=lambda engine: 0,
            trainer=None,
        )

    with pytest.raises(ValueError, match=r"Argument patience should be positive integer."):
        NoImprovementHandler(
            patience=-1,
            score_function=lambda engine: 0,
            pass_function=lambda engine: 0,
            stop_function=lambda engine: 0,
            trainer=trainer,
        )

    with pytest.raises(ValueError, match=r"Argument min_delta should not be a negative number."):
        NoImprovementHandler(
            patience=2,
            min_delta=-0.1,
            score_function=lambda engine: 0,
            pass_function=lambda engine: 0,
            stop_function=lambda engine: 0,
            trainer=trainer,
        )


def test_simple_no_improvement_handler(trainer):

    scores = iter([1.0, 0.8, 0.88])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha *= 2

    def pass_function(trainer):
        pass

    h = NoImprovementHandler(patience=2, score_function=score_function, stop_function=stop_function, trainer=trainer)
    # Call 3 times and check if stop_function called
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 0.2


def test_pass_function_no_improvement_handler(trainer):

    scores = iter([1.0, 0.8, 0.88])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha = 10

    def pass_function(trainer):
        trainer.state.alpha *= 2

    h = NoImprovementHandler(
        patience=2,
        score_function=score_function,
        stop_function=stop_function,
        pass_function=pass_function,
        trainer=trainer,
    )
    assert trainer.state.alpha == 0.1
    h(None)
    # Pass function should double the value
    assert trainer.state.alpha == 0.2
    h(None)
    # Pass function should double the value
    assert trainer.state.alpha == 0.4
    h(None)
    # stop function should convert to 10
    assert trainer.state.alpha == 10


def test_repeated_pass_stop_no_improvement_handler(trainer):
    scores = iter([1.0, 0.8, 0.88, 1.1, 0.9, 0.8, 1.2])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha = -1

    def pass_function(trainer):
        trainer.state.alpha *= 2

    h = NoImprovementHandler(
        patience=2,
        score_function=score_function,
        stop_function=stop_function,
        pass_function=pass_function,
        trainer=trainer,
    )
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 0.2
    h(None)
    assert trainer.state.alpha == 0.4
    h(None)
    # Stop function gets called
    assert trainer.state.alpha == -1
    h(None)
    # Pass function resumes
    assert trainer.state.alpha == -2
    h(None)
    assert trainer.state.alpha == -4
    h(None)
    # Stop function gets called again
    assert trainer.state.alpha == -1


def test_state_dict_no_improvement_handler(trainer):

    scores = iter([1.0, 0.8, 0.88])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(patience=2, score_function=score_function, stop_function=stop_function, trainer=trainer)

    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 0.1

    # Swap to new object, but maintain state
    h2 = NoImprovementHandler(patience=2, score_function=score_function, stop_function=stop_function, trainer=trainer)
    h2.load_state_dict(h.state_dict())

    h2(None)
    assert trainer.state.alpha == 0.1
    h2(None)
    assert trainer.state.alpha == 1


def test_no_improvement_handler_on_delta(trainer):

    scores = iter([1.0, 2.0, 2.01, 3.0, 3.01, 3.02])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(
        patience=2, min_delta=0.1, score_function=score_function, stop_function=stop_function, trainer=trainer
    )

    assert trainer.state.alpha == 0.1
    h(None)  # counter == 0
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 1.0; counter == 0
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.01; counter == 1
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.99; counter == 0
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.01; counter == 1
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.01; counter == 2
    assert trainer.state.alpha == 1


def test_no_improvement_handler_on_last_event_delta(trainer):

    scores = iter([0.0, 0.3, 0.6])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(
        patience=2, min_delta=0.4, score_function=score_function, stop_function=stop_function, trainer=trainer
    )

    assert trainer.state.alpha == 0.1
    h(None)  # counter == 0
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.3; counter == 1
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.3; counter == 2
    assert trainer.state.alpha == 1


def test_no_improvement_on_cumulative_delta(trainer):

    scores = iter([0.0, 0.3, 0.6])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(
        patience=2,
        min_delta=0.4,
        cumulative_delta=True,
        score_function=score_function,
        stop_function=stop_function,
        trainer=trainer,
    )

    assert trainer.state.alpha == 0.1
    h(None)  # counter == 0
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.3; counter == 1
    assert trainer.state.alpha == 0.1
    h(None)  # delta == 0.6; counter == 0
    assert trainer.state.alpha == 0.1


def test_simple_no_improvement_on_plateau(trainer):
    def score_function(engine):
        return 42

    def stop_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(patience=1, score_function=score_function, stop_function=stop_function, trainer=trainer)
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 1


def test_simple_no_improvement_on_plateau_then_pass(trainer):

    scores = iter([42, 42, 43, 45])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha = -1

    def pass_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(
        patience=1,
        score_function=score_function,
        pass_function=pass_function,
        stop_function=stop_function,
        trainer=trainer,
    )
    assert trainer.state.alpha == 0.1
    h(None)
    assert trainer.state.alpha == 1
    h(None)
    # Stops here due to plateu
    assert trainer.state.alpha == -1
    # Pass function gets called
    h(None)
    assert trainer.state.alpha == -10
    # Pass function gets called
    h(None)
    assert trainer.state.alpha == -100


def test_simple_not_trigger_no_improvement_handler(trainer):

    scores = iter([1.0, 0.8, 1.2])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.state.alpha *= 10

    h = NoImprovementHandler(patience=2, score_function=score_function, stop_function=stop_function, trainer=trainer)

    # Call 3 times and check if not stopped
    assert trainer.state.alpha == 0.1
    h(None)
    h(None)
    h(None)
    assert trainer.state.alpha == 0.1


def test_with_engine_no_improvement_handler(trainer):
    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = iter([1.0, 0.8, 1.2, 1.5, 0.9, 1.0, 0.99, 1.1, 0.9, 0.9])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.terminate()

    def pass_function(trainer):
        trainer.state.alpha *= 2

    evaluator = Engine(do_nothing_update_fn)

    h = NoImprovementHandler(
        patience=3,
        score_function=score_function,
        pass_function=pass_function,
        stop_function=stop_function,
        trainer=trainer,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, h)
    trainer.run([0], max_epochs=10)
    assert trainer.state.alpha == 6.4
    assert n_epochs_counter.count == 7
    assert trainer.state.epoch == 7


def test_with_engine_no_improvement_on_plateau(trainer):
    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    def score_function(engine):
        return 0.03899

    def stop_function(trainer):
        trainer.state.alpha = -1

    def pass_function(trainer):
        trainer.state.alpha *= 2

    evaluator = Engine(do_nothing_update_fn)
    h = NoImprovementHandler(
        patience=4,
        score_function=score_function,
        pass_function=pass_function,
        stop_function=stop_function,
        trainer=trainer,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, h)
    trainer.run([0], max_epochs=4)
    # Runs 4 times so 0.1* 2^4
    assert trainer.state.alpha == 1.6
    assert n_epochs_counter.count == 4
    assert trainer.state.epoch == 4

    trainer.run([0], max_epochs=10)
    # stop_function should get called for epochs > 4
    assert trainer.state.alpha == -1
    assert n_epochs_counter.count == 10
    assert trainer.state.epoch == 10


def test_with_engine_not_triggering_no_improvement_handler(trainer):
    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = iter([1.0, 0.8, 1.2, 1.23, 0.9, 1.0, 1.1, 1.253, 1.26, 1.2])

    def score_function(engine):
        return next(scores)

    def stop_function(trainer):
        trainer.terminate()

    def pass_function(trainer):
        trainer.state.alpha *= 2

    evaluator = Engine(do_nothing_update_fn)
    h = NoImprovementHandler(
        patience=5,
        score_function=score_function,
        pass_function=pass_function,
        stop_function=stop_function,
        trainer=trainer,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, h)
    trainer.run([0], max_epochs=10)
    # Runs 10 times so 0.1* 2^10
    assert trainer.state.alpha == 102.4
    assert n_epochs_counter.count == 10
    assert trainer.state.epoch == 10


def test_args_validation_early_stopping():

    trainer = Engine(do_nothing_update_fn)

    with pytest.raises(ValueError, match=r"Argument patience should be positive integer."):
        EarlyStopping(patience=-1, score_function=lambda engine: 0, trainer=trainer)

    with pytest.raises(ValueError, match=r"Argument min_delta should not be a negative number."):
        EarlyStopping(patience=2, min_delta=-0.1, score_function=lambda engine: 0, trainer=trainer)

    with pytest.raises(TypeError, match=r"Argument score_function should be a function."):
        EarlyStopping(patience=2, score_function=12345, trainer=trainer)

    with pytest.raises(TypeError, match=r"Argument trainer should be an instance of Engine."):
        EarlyStopping(patience=2, score_function=lambda engine: 0, trainer=None)


def test_simple_early_stopping():

    scores = iter([1.0, 0.8, 0.88])

    def score_function(engine):
        return next(scores)

    trainer = Engine(do_nothing_update_fn)

    h = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
    # Call 3 times and check if stopped
    assert not trainer.should_terminate
    h(None)
    assert not trainer.should_terminate
    h(None)
    assert not trainer.should_terminate
    h(None)
    assert trainer.should_terminate


def test_state_dict():

    scores = iter([1.0, 0.8, 0.88])

    def score_function(engine):
        return next(scores)

    trainer = Engine(do_nothing_update_fn)

    h = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
    # Call 3 times and check if stopped
    assert not trainer.should_terminate
    h(None)
    assert not trainer.should_terminate

    # Swap to new object, but maintain state
    h2 = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
    h2.load_state_dict(h.state_dict())

    h2(None)
    assert not trainer.should_terminate
    h2(None)
    assert trainer.should_terminate


def test_early_stopping_on_delta():

    scores = iter([1.0, 2.0, 2.01, 3.0, 3.01, 3.02])

    trainer = Engine(do_nothing_update_fn)

    h = EarlyStopping(patience=2, min_delta=0.1, score_function=lambda _: next(scores), trainer=trainer)

    assert not trainer.should_terminate
    h(None)  # counter == 0
    assert not trainer.should_terminate
    h(None)  # delta == 1.0; counter == 0
    assert not trainer.should_terminate
    h(None)  # delta == 0.01; counter == 1
    assert not trainer.should_terminate
    h(None)  # delta == 0.99; counter == 0
    assert not trainer.should_terminate
    h(None)  # delta == 0.01; counter == 1
    assert not trainer.should_terminate
    h(None)  # delta == 0.01; counter == 2
    assert trainer.should_terminate


def test_early_stopping_on_last_event_delta():

    scores = iter([0.0, 0.3, 0.6])

    trainer = Engine(do_nothing_update_fn)

    h = EarlyStopping(
        patience=2, min_delta=0.4, cumulative_delta=False, score_function=lambda _: next(scores), trainer=trainer
    )

    assert not trainer.should_terminate
    h(None)  # counter == 0
    assert not trainer.should_terminate
    h(None)  # delta == 0.3; counter == 1
    assert not trainer.should_terminate
    h(None)  # delta == 0.3; counter == 2
    assert trainer.should_terminate


def test_early_stopping_on_cumulative_delta():

    scores = iter([0.0, 0.3, 0.6])

    trainer = Engine(do_nothing_update_fn)

    h = EarlyStopping(
        patience=2, min_delta=0.4, cumulative_delta=True, score_function=lambda _: next(scores), trainer=trainer
    )

    assert not trainer.should_terminate
    h(None)  # counter == 0
    assert not trainer.should_terminate
    h(None)  # delta == 0.3; counter == 1
    assert not trainer.should_terminate
    h(None)  # delta == 0.6; counter == 0
    assert not trainer.should_terminate


def test_simple_early_stopping_on_plateau():
    def score_function(engine):
        return 42

    trainer = Engine(do_nothing_update_fn)

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

    trainer = Engine(do_nothing_update_fn)

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

    trainer = Engine(do_nothing_update_fn)
    evaluator = Engine(do_nothing_update_fn)
    early_stopping = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert n_epochs_counter.count == 7
    assert trainer.state.epoch == 7


def test_with_engine_early_stopping_on_plateau():
    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    def score_function(engine):
        return 0.047

    trainer = Engine(do_nothing_update_fn)
    evaluator = Engine(do_nothing_update_fn)
    early_stopping = EarlyStopping(patience=4, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert n_epochs_counter.count == 5
    assert trainer.state.epoch == 5


def test_with_engine_no_early_stopping():
    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = iter([1.0, 0.8, 1.2, 1.23, 0.9, 1.0, 1.1, 1.253, 1.26, 1.2])

    def score_function(engine):
        return next(scores)

    trainer = Engine(do_nothing_update_fn)
    evaluator = Engine(do_nothing_update_fn)
    early_stopping = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert n_epochs_counter.count == 10
    assert trainer.state.epoch == 10


def _test_distrib_with_engine_no_improvement_handler(device):

    if device is None:
        device = idist.device()
    if isinstance(device, str):
        device = torch.device(device)

    torch.manual_seed(12)

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = torch.tensor([1.0, 0.8, 1.2, 1.5, 0.9, 1.0, 0.99, 1.6, 0.9], requires_grad=False).to(device)

    def score_function(engine):
        i = trainer.state.epoch - 1
        v = scores[i]
        idist.all_reduce(v)
        v /= idist.get_world_size()
        return v.item()

    def stop_function(trainer):
        trainer.state.alpha = -1

    def pass_function(trainer):
        trainer.state.alpha *= 2

    trainer = Engine(do_nothing_update_fn)
    trainer.state_dict_user_keys.append("alpha")
    trainer.state.alpha = 0.1

    evaluator = Engine(do_nothing_update_fn)
    nih = NoImprovementHandler(
        patience=3,
        score_function=score_function,
        stop_function=stop_function,
        pass_function=pass_function,
        trainer=trainer,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, nih)

    # Runs pass_function in this case
    trainer.run([0], max_epochs=6)
    assert trainer.state.alpha == 6.4

    # Terminates and calls stop_function
    trainer.run([0], max_epochs=7)
    assert trainer.state.alpha == -1

    # Unlike EarlyStopping this No Improvement handler doesnt terminate, hence will start comparing conditions again
    # after stopping condition is met once also. pass_function gets called again
    trainer.run([0], max_epochs=8)
    assert trainer.state.alpha == -2


def _test_distrib_integration_engine_no_improvement_handler(device):

    from ignite.metrics import Accuracy

    if device is None:
        device = idist.device()
    if isinstance(device, str):
        device = torch.device(device)
    metric_device = device
    if device.type == "xla":
        metric_device = "cpu"

    rank = idist.get_rank()
    ws = idist.get_world_size()
    torch.manual_seed(12)

    n_epochs = 10
    n_iters = 20

    y_preds = (
        [torch.randint(0, 2, size=(n_iters, ws)).to(device)]
        + [torch.ones(n_iters, ws).to(device)]
        + [torch.randint(0, 2, size=(n_iters, ws)).to(device) for _ in range(n_epochs - 2)]
    )

    y_true = (
        [torch.randint(0, 2, size=(n_iters, ws)).to(device)]
        + [torch.ones(n_iters, ws).to(device)]
        + [torch.randint(0, 2, size=(n_iters, ws)).to(device) for _ in range(n_epochs - 2)]
    )

    def update(engine, _):
        e = trainer.state.epoch - 1
        i = engine.state.iteration - 1
        return y_preds[e][i, rank], y_true[e][i, rank]

    evaluator = Engine(update)
    acc = Accuracy(device=metric_device)
    acc.attach(evaluator, "acc")

    def score_function(engine):
        return engine.state.metrics["acc"]

    def stop_function(trainer):
        trainer.state.alpha = -1

    def pass_function(trainer):
        trainer.state.alpha *= 2

    trainer = Engine(lambda e, b: None)
    trainer.state_dict_user_keys.append("alpha")
    trainer.state.alpha = 0.1

    nih = NoImprovementHandler(
        patience=3,
        score_function=score_function,
        stop_function=stop_function,
        pass_function=pass_function,
        trainer=trainer,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        data = list(range(n_iters))
        evaluator.run(data=data)

    evaluator.add_event_handler(Events.COMPLETED, nih)
    trainer.run([0], max_epochs=4)
    assert trainer.state.alpha == 1.6

    trainer.run([0], max_epochs=5)
    assert trainer.state.alpha == -1


def _test_distrib_with_engine_early_stopping(device):

    if device is None:
        device = idist.device()
    if isinstance(device, str):
        device = torch.device(device)

    torch.manual_seed(12)

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = torch.tensor([1.0, 0.8, 1.2, 1.5, 0.9, 1.0, 0.99, 1.1, 0.9], requires_grad=False).to(device)

    def score_function(engine):
        i = trainer.state.epoch - 1
        v = scores[i]
        idist.all_reduce(v)
        v /= idist.get_world_size()
        return v.item()

    trainer = Engine(do_nothing_update_fn)
    evaluator = Engine(do_nothing_update_fn)
    early_stopping = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        evaluator.run([0])
        n_epochs_counter.count += 1

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert trainer.state.epoch == 7
    assert n_epochs_counter.count == 7


def _test_distrib_integration_engine_early_stopping(device):

    from ignite.metrics import Accuracy

    if device is None:
        device = idist.device()
    if isinstance(device, str):
        device = torch.device(device)
    metric_device = device
    if device.type == "xla":
        metric_device = "cpu"

    rank = idist.get_rank()
    ws = idist.get_world_size()
    torch.manual_seed(12)

    n_epochs = 10
    n_iters = 20

    y_preds = (
        [torch.randint(0, 2, size=(n_iters, ws)).to(device)]
        + [torch.ones(n_iters, ws).to(device)]
        + [torch.randint(0, 2, size=(n_iters, ws)).to(device) for _ in range(n_epochs - 2)]
    )

    y_true = (
        [torch.randint(0, 2, size=(n_iters, ws)).to(device)]
        + [torch.ones(n_iters, ws).to(device)]
        + [torch.randint(0, 2, size=(n_iters, ws)).to(device) for _ in range(n_epochs - 2)]
    )

    def update(engine, _):
        e = trainer.state.epoch - 1
        i = engine.state.iteration - 1
        return y_preds[e][i, rank], y_true[e][i, rank]

    evaluator = Engine(update)
    acc = Accuracy(device=metric_device)
    acc.attach(evaluator, "acc")

    def score_function(engine):
        return engine.state.metrics["acc"]

    trainer = Engine(lambda e, b: None)
    early_stopping = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluation(engine):
        data = list(range(n_iters))
        evaluator.run(data=data)

    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    trainer.run([0], max_epochs=10)
    assert trainer.state.epoch == 5


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)
    _test_distrib_with_engine_no_improvement_handler(device)
    _test_distrib_integration_engine_no_improvement_handler(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_with_engine_early_stopping, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_engine_early_stopping, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_with_engine_no_improvement_handler, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_engine_no_improvement_handler, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)
    _test_distrib_with_engine_no_improvement_handler(device)
    _test_distrib_integration_engine_no_improvement_handler(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)
    _test_distrib_with_engine_no_improvement_handler(device)
    _test_distrib_integration_engine_no_improvement_handler(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)
    _test_distrib_with_engine_no_improvement_handler(device)
    _test_distrib_integration_engine_no_improvement_handler(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)
    _test_distrib_with_engine_no_improvement_handler(device)
    _test_distrib_integration_engine_no_improvement_handler(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

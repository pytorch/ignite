import os
import torch

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

import pytest


def do_nothing_update_fn(engine, batch):
    pass


def test_args_validation():

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

    h = EarlyStopping(patience=2, min_delta=0.4, cumulative_delta=False,
                      score_function=lambda _: next(scores), trainer=trainer)

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

    h = EarlyStopping(patience=2, min_delta=0.4, cumulative_delta=True,
                      score_function=lambda _: next(scores), trainer=trainer)

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


def _test_distrib_with_engine_early_stopping(device):

    import torch.distributed as dist

    torch.manual_seed(12)

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    n_epochs_counter = Counter()

    scores = torch.tensor([1.0, 0.8, 1.2, 1.5, 0.9, 1.0, 0.99, 1.1, 0.9], requires_grad=False).to(device)

    def score_function(engine):
        i = trainer.state.epoch - 1
        v = scores[i]
        dist.all_reduce(v)
        v /= dist.get_world_size()
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

    import torch.distributed as dist
    from ignite.metrics import Accuracy
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.manual_seed(12)

    n_epochs = 10
    n_iters = 20

    y_preds = [
        torch.randint(0, 2, size=(n_iters, ws)).to(device)
    ] + [
        torch.ones(n_iters, ws).to(device)
    ] + [
        torch.randint(0, 2, size=(n_iters, ws)).to(device) for _ in range(n_epochs - 2)
    ]

    y_true = [
        torch.randint(0, 2, size=(n_iters, ws)).to(device)
    ] + [
        torch.ones(n_iters, ws).to(device)
    ] + [
        torch.randint(0, 2, size=(n_iters, ws)).to(device) for _ in range(n_epochs - 2)
    ]

    def update(engine, _):
        e = trainer.state.epoch - 1
        i = engine.state.iteration - 1
        return y_preds[e][i, rank], y_true[e][i, rank]

    evaluator = Engine(update)
    acc = Accuracy(device=device)
    acc.attach(evaluator, "acc")

    def score_function(engine):
        return engine.state.metrics['acc']

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
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = "cuda:{}".format(local_rank)
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)


@pytest.mark.distributed
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_distrib_with_engine_early_stopping(device)
    _test_distrib_integration_engine_early_stopping(device)

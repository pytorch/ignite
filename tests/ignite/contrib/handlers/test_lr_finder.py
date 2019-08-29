import pytest
import torch
from torch import nn
import numpy as np
from ignite.contrib.handlers import FastaiLRFinder
from ignite.contrib.handlers.lr_finder import AlreadyAttachedError, NotEnoughIterationWarning, NotDivergedWarning
from torch.optim import SGD
from ignite.engine import create_supervised_trainer, Events
import copy
import os


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


@pytest.fixture()
def model():
    model = DummyModel()
    yield model


@pytest.fixture()
def optimizer(model):
    yield SGD(model.parameters(), lr=1e-4, momentum=0.9)


@pytest.fixture()
def lr_finder(model, optimizer):
    yield FastaiLRFinder(model, optimizer)


@pytest.fixture()
def dummy_engine(model, optimizer):
    engine = create_supervised_trainer(model, optimizer, nn.MSELoss())
    yield engine


@pytest.fixture()
def dataloader():
    yield torch.rand(100, 2, 1)


def test_with_attach(lr_finder, model, optimizer, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine):
        dummy_engine.run(dataloader, max_epochs=3)
    assert lr_finder._engine is None

    for event in dummy_engine._event_handlers:
        assert len(dummy_engine._event_handlers[event]) == 0


def test_double_attach_error(dummy_engine, lr_finder):
    lr_finder.attach(dummy_engine)
    with pytest.raises(AlreadyAttachedError):
        lr_finder.attach(dummy_engine)


def test_detach_warns(dummy_engine, lr_finder):
    with pytest.warns(Warning):
        lr_finder.detach()
    lr_finder.attach(dummy_engine)
    lr_finder.detach()
    with pytest.warns(Warning):
        lr_finder.detach()


def test_in_memory_model_optimizer_reset(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder(model, optimizer)

    init_optimizer = copy.deepcopy(optimizer.state_dict())
    if isinstance(model, list):
        pass
    else:
        init_model = copy.deepcopy(model.state_dict())

    @dummy_engine.on(Events.EPOCH_COMPLETED)
    def compare_states(engine):
        mid_optimizer = lr_finder._optimizer.state_dict()
        mid_model = lr_finder._model.state_dict()

        assert init_optimizer["param_groups"][0]["params"] == mid_optimizer["param_groups"][0]["params"]

        for v1, v2 in zip(init_model.values(), mid_model.values()):
            assert any(v1 != v2)

    with lr_finder.attach(dummy_engine):
        dummy_engine.run(dataloader)

    end_optimizer = lr_finder._optimizer.state_dict()
    end_model = lr_finder._model.state_dict()
    assert init_optimizer["param_groups"][0]["params"] == end_optimizer["param_groups"][0]["params"]

    for v1, v2 in zip(init_model.values(), end_model.values()):
        assert all(v1 == v2)


def test_in_dir_model_optimizer_reset(tmpdir, model, optimizer, dummy_engine, dataloader):
    tmpdir_num_files = len(os.listdir(tmpdir))
    lr_finder = FastaiLRFinder(model, optimizer, memory_cache=False)

    init_optimizer = copy.deepcopy(optimizer.state_dict())
    init_model = copy.deepcopy(model.state_dict())

    @dummy_engine.on(Events.EPOCH_COMPLETED)
    def compare_states(engine):
        mid_optimizer = lr_finder._optimizer.state_dict()
        mid_model = lr_finder._model.state_dict()

        assert init_optimizer["param_groups"][0]["params"] == mid_optimizer["param_groups"][0]["params"]

        for v1, v2 in zip(init_model.values(), mid_model.values()):
            assert any(v1 != v2)

        assert tmpdir_num_files != len(os.listdir(tmpdir))

    with lr_finder.attach(dummy_engine):
        dummy_engine.run(dataloader)

    end_optimizer = lr_finder._optimizer.state_dict()
    end_model = lr_finder._model.state_dict()
    assert init_optimizer["param_groups"][0]["params"] == end_optimizer["param_groups"][0]["params"]

    for v1, v2 in zip(init_model.values(), end_model.values()):
        assert all(v1 == v2)

    assert tmpdir_num_files == len(os.listdir(tmpdir))


def test_lr_policy(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder(model, optimizer, step_mode="linear")
    with lr_finder.attach(dummy_engine):
        dummy_engine.run(dataloader)
    lr = lr_finder.get_results()["lr"]
    assert all([lr[i - 1] < lr[i] for i in range(1, len(lr))])

    lr_finder = FastaiLRFinder(model, optimizer, step_mode="exp")
    with lr_finder.attach(dummy_engine):
        dummy_engine.run(dataloader)
    lr = lr_finder.get_results()["lr"]
    assert all([lr[i - 1] < lr[i] for i in range(1, len(lr))])


def assert_output_sizes(lr_finder, dummy_engine, dataloader, num_epochs):
    with lr_finder.attach(dummy_engine):
        dummy_engine.run(dataloader, max_epochs=num_epochs)

    iteration = dummy_engine.state.iteration
    lr_finder_results = lr_finder.get_results()
    lr, loss = lr_finder_results["lr"], lr_finder_results["loss"]
    assert len(lr) == len(loss) == iteration


def test_num_iter_is_none(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder(model, optimizer, diverge_th=np.inf)
    for i in range(1, 5):
        assert_output_sizes(lr_finder, dummy_engine, dataloader, i)
        assert dummy_engine.state.iteration == i * len(dataloader)


def test_num_iter_is_enough(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder(model, optimizer, num_iter=50, diverge_th=np.inf)
    assert_output_sizes(lr_finder, dummy_engine, dataloader, 1)
    assert dummy_engine.state.iteration == 50


def test_num_iter_is_not_enough(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder(model, optimizer, num_iter=150, diverge_th=np.inf)
    with pytest.warns(NotEnoughIterationWarning):
        assert_output_sizes(lr_finder, dummy_engine, dataloader, 1)
    assert dummy_engine.state.iteration == len(dataloader)


def test_detach_terminates(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder(model, optimizer, end_lr=100, diverge_th=2)
    with lr_finder.attach(dummy_engine):
        with pytest.warns(None) as record:
            dummy_engine.run(dataloader)
            assert len(record) == 0

    dummy_engine.run(dataloader, max_epochs=3)
    assert dummy_engine.state.epoch == 3

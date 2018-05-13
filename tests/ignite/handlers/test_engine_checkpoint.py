import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
import shutil

from ignite.engine import Engine, Events, create_supervised_trainer, State
from ignite.handlers import EngineCheckpoint


_PREFIX = 'PREFIX'


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


def test_args_validation(dirname):

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.net = nn.Linear(1, 1)

        def forward(self, x):
            return self.net

    existing = os.path.join(dirname, 'existing_dir')
    nonempty = os.path.join(dirname, 'nonempty')

    os.makedirs(existing)
    os.makedirs(nonempty)

    with open(os.path.join(nonempty, '{}_name_0.pth'.format(_PREFIX)), 'w'):
        pass

    # incorrect to_save argument
    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, to_save=DummyModel())

    # incorrect to_save argument
    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, to_save=[1, 2, 3])

    # incorrect to_save argument
    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, to_save={"a": 0, "b": [0, 1, 2]})

    model = DummyModel()
    optim = SGD(model.parameters(), lr=0.1)
    # non empty dir
    with pytest.raises(ValueError):
        h = EngineCheckpoint(nonempty, to_save={"model": DummyModel(), "optimizer": optim})

    # Nonexisting dir with create_dir=False
    with pytest.raises(ValueError):
        h = EngineCheckpoint(os.path.join(dirname, 'non_existing_dir'),
                             to_save={"model": DummyModel(), "optimizer": optim},
                             create_dir=False)


def check_states(state1, state2):
    assert len(state1) == len(state2)
    for key in state1.keys():
        assert key in state2
        value1 = state1[key]
        value2 = state2[key]
        if torch.is_tensor(value1) and torch.is_tensor(value2):
            assert (value1.numpy() == value2.numpy()).all()
        else:
            assert value1 == value2


def check_checkpoints(chkpt1, chkpt2):
    assert len(chkpt1) == len(chkpt2)
    for k1, k2 in zip(chkpt1, chkpt2):
        check_states(chkpt1[k1], chkpt2[k2])


class _TestModel(nn.Module):
    def __init__(self):
        super(_TestModel, self).__init__()
        self.net = nn.Linear(32, 10)

    def forward(self, x):
        return self.net(x)


def test_basic_checkpointing(dirname):

    model = _TestModel()
    optim = SGD(model.parameters(), lr=0.1)
    engine = create_supervised_trainer(model, optim, nn.CrossEntropyLoss())
    engine.state = State(epoch=1, max_epochs=10, seed=12345)
    engine.state.iteration = 12

    h = EngineCheckpoint(dirname, to_save={"model": model, "optimizer": optim}, save_interval=1)

    true_checkpoint = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "engine": engine.state_dict()
    }

    test_checkpoint = h._setup_checkpoint(engine)

    check_checkpoints(true_checkpoint, test_checkpoint)

    h(engine)
    fname = os.path.join(dirname, 'checkpoint.pth.tar')
    assert os.path.exists(fname)

    test_checkpoint = EngineCheckpoint.load(dirname)
    check_checkpoints(true_checkpoint, test_checkpoint)


def test_check_objects():

    model = _TestModel()

    # incorrect to_save_or_load argument
    with pytest.raises(AssertionError):
        EngineCheckpoint.check_objects({"model": model, "b": [1, 2, 3]})

    # incorrect to_save_or_load argument
    with pytest.raises(AssertionError):
        EngineCheckpoint.check_objects({"model": 1234, "b": [1, 2, 3]})


def test_load_objects(dirname):

    model = _TestModel()
    optim = SGD(model.parameters(), lr=0.1)

    engine = create_supervised_trainer(model, optim, nn.CrossEntropyLoss())
    engine.state = State(epoch=1, max_epochs=10, seed=12345)
    engine.state.iteration = 12

    true_checkpoint = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "engine": engine.state_dict()
    }

    path = os.path.join(dirname, "checkpoint.pth.tar")
    torch.save(true_checkpoint, path)

    test_checkpoint = EngineCheckpoint.load(dirname)
    check_checkpoints(true_checkpoint, test_checkpoint)

    to_load = {"model": model, "optimizer": optim}

    EngineCheckpoint.load_objects(to_load, test_checkpoint)

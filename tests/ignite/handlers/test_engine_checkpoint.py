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

    # incorrect models argument
    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, models=DummyModel(), optimizers=None)

    # incorrect models argument
    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, models=[1, 2, 3], optimizers=None)

    # incorrect optimizers argument
    model = DummyModel()
    optim = SGD(model.parameters(), lr=0.1)
    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, models=[DummyModel(), ], optimizers=optim)

    with pytest.raises(AssertionError):
        h = EngineCheckpoint(existing, models=[DummyModel(), ], optimizers=[1, 2, 3])

    # non empty dir
    with pytest.raises(ValueError):
        h = EngineCheckpoint(nonempty, models=[DummyModel(), ], optimizers=[optim, ])

    # Nonexisting dir with create_dir=False
    with pytest.raises(ValueError):
        h = EngineCheckpoint(os.path.join(dirname, 'non_existing_dir'),
                             models=[DummyModel(), ], optimizers=[optim, ],
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

    models_states1 = chkpt1['models']
    models_states2 = chkpt2['models']
    assert len(models_states1) == len(models_states2)

    for s1, s2 in zip(models_states1, models_states2):
        check_states(s1, s2)

    opt_states1 = chkpt1['optimizers']
    opt_states2 = chkpt2['optimizers']
    assert len(opt_states1) == len(opt_states2)
    for s1, s2 in zip(opt_states1, opt_states2):
        check_states(s1, s2)

    assert chkpt1['engine'].iteration == chkpt2['engine'].iteration
    assert chkpt1['engine'].epoch == chkpt2['engine'].epoch
    assert chkpt1['engine'].max_epochs == chkpt2['engine'].max_epochs
    assert chkpt1['engine'].seed == chkpt2['engine'].seed


def test_basic_checkpointing(dirname):

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.net = nn.Linear(32, 10)

        def forward(self, x):
            return self.net(x)

    model = TestModel()
    optim = SGD(model.parameters(), lr=0.1)
    engine = create_supervised_trainer(model, optim, nn.CrossEntropyLoss())
    engine.state = State(epoch=1, max_epochs=10, seed=12345)
    engine.state.iteration = 12

    h = EngineCheckpoint(dirname, models=[model, ], optimizers=[optim, ], save_interval=1)

    true_checkpoint = {
            "models": [model.state_dict()],
            "optimizers": [optim.state_dict()],
            "engine": engine.state
    }

    test_checkpoint = h._setup_checkpoint(engine)

    check_checkpoints(true_checkpoint, test_checkpoint)

    h(engine)
    fname = os.path.join(dirname, 'checkpoint.pth.tar')
    assert os.path.exists(fname)

    test_checkpoint = torch.load(fname)
    check_checkpoints(true_checkpoint, test_checkpoint)

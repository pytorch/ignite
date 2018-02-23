import os
import tempfile

import pytest
import torch

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.trainer import Trainer

_PREFIX = 'PREFIX'


@pytest.fixture
def dirname():
    directory = tempfile.TemporaryDirectory()
    dirname = directory.name

    yield dirname


def test_args_validation(dirname):
    existing = os.path.join(dirname, 'existing_dir')
    nonempty = os.path.join(dirname, 'nonempty')

    os.makedirs(existing)
    os.makedirs(nonempty)

    with open(os.path.join(nonempty, '{}_name_0.pth'.format(_PREFIX)), 'w'):
        pass

    # save_interval & score_func
    with pytest.raises(ValueError):
        h = ModelCheckpoint(existing, _PREFIX,
                            create_dir=False)

    with pytest.raises(FileExistsError):
        h = ModelCheckpoint(existing, _PREFIX, create_dir=True,
                            save_interval=42)

    with pytest.raises(ValueError):
        h = ModelCheckpoint(nonempty, _PREFIX, exist_ok=True,
                            save_interval=42)


def test_simple_recovery(dirname):
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, save_interval=1)
    h(None, {'obj': 42})

    fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'obj', 1))
    assert torch.load(fname) == 42


def test_atomic(dirname):
    serializable = 42
    non_serializable = (42, lambda _: 42)

    def _test_existance(atomic, name, obj, expected):
        h = ModelCheckpoint(dirname, _PREFIX,
                            atomic=atomic,
                            create_dir=False,
                            require_empty=False,
                            save_interval=1)

        try:
            h(None, {name: obj})
        except AttributeError:
            pass

        fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, name, 1))
        assert os.path.exists(fname) == expected

    _test_existance(atomic=False, name='nonatomic_OK', obj=serializable, expected=True)
    _test_existance(atomic=False, name='nonatomic_FAIL', obj=non_serializable, expected=True)

    _test_existance(atomic=True, name='atomic_OK', obj=serializable, expected=True)
    _test_existance(atomic=True, name='atomic_FAIL', obj=non_serializable, expected=False)


def test_last_k(dirname):
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2, save_interval=1)
    to_save = {'name': 42}

    for _ in range(4):
        h(None, to_save)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, 'name', i)
                for i in [3, 4]]

    assert sorted(os.listdir(dirname)) == expected


def test_best_k(dirname):
    scores = iter([1.0, -2., 3.0, -4.0])

    def score_function(engine):
        return next(scores)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False,
                        n_saved=2, score_function=score_function)

    to_save = {'name': 42}
    for _ in range(4):
        h(None, to_save)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, 'name', i)
                for i in [1, 3]]

    assert sorted(os.listdir(dirname)) == expected


def test_with_trainer(dirname):

    def update_fn(batch):
        pass

    name = 'model'
    trainer = Trainer(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False,
                              n_saved=2, save_interval=1)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {name: 42})
    trainer.run([0], max_epochs=4)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, name, i)
                for i in [3, 4]]

    assert sorted(os.listdir(dirname)) == expected

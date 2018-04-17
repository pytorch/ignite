import os
import tempfile

import pytest
import torch
import shutil

from ignite.engines import Engine, Events
from ignite.handlers import ModelCheckpoint

_PREFIX = 'PREFIX'


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


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

    with pytest.raises(ValueError):
        h = ModelCheckpoint(nonempty, _PREFIX, save_interval=42)

    with pytest.raises(ValueError):
        h = ModelCheckpoint(nonempty, _PREFIX,
                            score_function="score",
                            save_interval=42)

    with pytest.raises(ValueError):
        h = ModelCheckpoint(os.path.join(dirname, 'non_existing_dir'), _PREFIX,
                            create_dir=False,
                            save_interval=42)


def test_simple_recovery(dirname):
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, save_interval=1)
    h(None, {'obj': 42})

    fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'obj', 1))
    assert torch.load(fname) == 42


def test_simple_recovery_from_existing_non_empty(dirname):
    previous_fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'obj', 1))
    with open(previous_fname, 'w') as f:
        f.write("test")

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=True, require_empty=False, save_interval=1)
    h(None, {'obj': 42})

    fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'obj', 1))
    assert torch.load(fname) == 42
    assert os.path.exists(previous_fname)


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
        except:
            pass

        fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, name, 1))
        assert os.path.exists(fname) == expected

    _test_existance(atomic=False, name='nonatomic_OK', obj=serializable, expected=True)
    _test_existance(atomic=False, name='nonatomic_FAIL', obj=non_serializable, expected=True)

    _test_existance(atomic=True, name='atomic_OK', obj=serializable, expected=True)
    _test_existance(atomic=True, name='atomic_FAIL', obj=non_serializable, expected=False)


def test_last_k(dirname):
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2, save_interval=2)
    to_save = {'name': 42}

    for _ in range(8):
        h(None, to_save)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, 'name', i)
                for i in [6, 8]]

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


def test_best_k_with_suffix(dirname):
    scores = [0.3456789, 0.1234, 0.4567, 0.134567]
    scores_iter = iter(scores)

    def score_function(engine):
        return next(scores_iter)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2,
                        score_function=score_function, score_name="val_loss")

    to_save = {'name': 42}
    for _ in range(4):
        h(None, to_save)

    expected = ['{}_{}_{}_val_loss={:.7}.pth'.format(_PREFIX, 'name', i, scores[i - 1])
                for i in [1, 3]]

    assert sorted(os.listdir(dirname)) == expected


def test_with_engine(dirname):

    def update_fn(engine, batch):
        pass

    name = 'model'
    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False,
                              n_saved=2, save_interval=1)

    engine.add_event_handler(Events.EPOCH_COMPLETED, handler, {name: 42})
    engine.run([0], max_epochs=4)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, name, i)
                for i in [3, 4]]

    assert sorted(os.listdir(dirname)) == expected

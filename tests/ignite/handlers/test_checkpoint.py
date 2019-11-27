import os
import tempfile
import shutil
import warnings

import torch
import torch.nn as nn

from ignite.engine import Engine, Events, State
from ignite.handlers import ModelCheckpoint, Checkpoint, DiskSaver

import pytest
from mock import MagicMock

_PREFIX = 'PREFIX'


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


def test_args_validation(dirname):
    existing = os.path.join(dirname, 'existing_dir')
    nonempty = os.path.join(dirname, 'nonempty')

    os.makedirs(existing)
    os.makedirs(nonempty)

    with open(os.path.join(nonempty, '{}_name_0.pth'.format(_PREFIX)), 'w'):
        pass

    with pytest.raises(ValueError, match=r"Files are already present in the directory"):
        ModelCheckpoint(nonempty, _PREFIX)

    with pytest.raises(ValueError, match=r"Argument save_interval is deprecated and should be None"):
        ModelCheckpoint(existing, _PREFIX, save_interval=42)

    with pytest.raises(ValueError, match=r"Directory path '\S+' is not found"):
        ModelCheckpoint(os.path.join(dirname, 'non_existing_dir'), _PREFIX, create_dir=False)

    with pytest.raises(ValueError, match=r"Argument save_as_state_dict is deprecated and should be True"):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, save_as_state_dict=False)


def test_simple_recovery(dirname):
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=1)

    model = DummyModel()
    to_save = {'model': model}
    h(engine, to_save)

    fname = h.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    loaded_objects = torch.load(fname)
    assert "model" in loaded_objects
    assert loaded_objects['model'] == model.state_dict()


def test_simple_recovery_from_existing_non_empty(dirname):
    previous_fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'obj', 1))
    with open(previous_fname, 'w') as f:
        f.write("test")

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=True, require_empty=False)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=1)

    model = DummyModel()
    to_save = {'model': model}
    h(engine, to_save)

    fname = h.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'model', 1)) == fname
    assert os.path.exists(fname)
    assert os.path.exists(previous_fname)
    loaded_objects = torch.load(fname)
    assert "model" in loaded_objects
    assert loaded_objects['model'] == model.state_dict()


def test_atomic(dirname):

    model = DummyModel()
    to_save_serializable = {'model': model}
    to_save_non_serializable = {'model': lambda x: x}

    def _test_existance(atomic, _to_save, expected):

        saver = DiskSaver(dirname, atomic=atomic, create_dir=False, require_empty=False)
        fname = "test.pth"
        try:
            with warnings.catch_warnings():
                # Ignore torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type
                # DummyModel. It won't be checked for correctness upon loading.
                warnings.simplefilter("ignore", category=UserWarning)
                saver(_to_save, fname)
        except Exception:
            pass
        fp = os.path.join(saver.dirname, fname)
        assert os.path.exists(fp) == expected
        if expected:
            saver.remove(fname)

    _test_existance(atomic=False, _to_save=to_save_serializable, expected=True)
    _test_existance(atomic=False, _to_save=to_save_non_serializable, expected=True)

    _test_existance(atomic=True, _to_save=to_save_serializable, expected=True)
    _test_existance(atomic=True, _to_save=to_save_non_serializable, expected=False)


def test_last_k(dirname):

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {'model': model}
    h(engine, to_save)

    for i in range(1, 9):
        engine.state.iteration = i
        h(engine, to_save)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, 'model', i) for i in [7, 8]]

    assert sorted(os.listdir(dirname)) == expected, "{} vs {}".format(sorted(os.listdir(dirname)), expected)


def test_best_k(dirname):
    scores = iter([1.2, -2., 3.1, -4.0])

    def score_function(_):
        return next(scores)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2, score_function=score_function)

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {'model': model}
    for _ in range(4):
        h(engine, to_save)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, 'model', i) for i in [1.2, 3.1]]

    assert sorted(os.listdir(dirname)) == expected


def test_best_k_with_suffix(dirname):
    scores = [0.3456789, 0.1234, 0.4567, 0.134567]
    scores_iter = iter(scores)

    def score_function(engine):
        return next(scores_iter)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2,
                        score_function=score_function, score_name="val_loss")

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {'model': model}
    for _ in range(4):
        engine.state.epoch += 1
        h(engine, to_save)

    expected = ['{}_{}_{}_val_loss={:.7}.pth'.format(_PREFIX, 'model', e, scores[e - 1]) for e in [1, 3]]

    assert sorted(os.listdir(dirname)) == expected


def test_with_engine(dirname):

    def update_fn(_1, _2):
        pass

    name = 'model'
    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2)

    model = DummyModel()
    to_save = {'model': model}
    engine.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    engine.run([0], max_epochs=4)

    expected = ['{}_{}_{}.pth'.format(_PREFIX, name, i) for i in [3, 4]]

    assert sorted(os.listdir(dirname)) == expected


def test_with_state_dict(dirname):

    def update_fn(_1, _2):
        pass

    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)

    model = DummyModel()
    to_save = {'model': model}
    engine.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    engine.run([0], max_epochs=4)

    saved_model = os.path.join(dirname, os.listdir(dirname)[0])
    load_model = torch.load(saved_model)

    assert not isinstance(load_model, DummyModel)
    assert isinstance(load_model, dict)

    model_state_dict = model.state_dict()
    loaded_model_state_dict = load_model
    for key in model_state_dict.keys():
        assert key in loaded_model_state_dict['model']

        model_value = model_state_dict[key]
        loaded_model_value = loaded_model_state_dict['model'][key]

        assert model_value.numpy() == loaded_model_value.numpy()


def test_valid_state_dict_save(dirname):
    model = DummyModel()
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    to_save = {'name': 42}
    with pytest.raises(TypeError, match=r"should have `state_dict` and `load_state_dict` methods"):
        h(engine, to_save)
    to_save = {'name': model}
    try:
        h(engine, to_save)
    except ValueError:
        pytest.fail("Unexpected ValueError")


def test_save_model_optimizer_lr_scheduler_with_state_dict(dirname):
    model = DummyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)

    def update_fn(engine, batch):
        x = torch.rand((4, 1))
        optim.zero_grad()
        y = model(x)
        loss = y.pow(2.0).sum()
        loss.backward()
        optim.step()
        lr_scheduler.step()

    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)

    engine.add_event_handler(Events.EPOCH_COMPLETED,
                             handler,
                             {
                                 "model": model,
                                 "optimizer": optim,
                                 "lr_scheduler": lr_scheduler,
                             })
    engine.run([0], max_epochs=4)

    saved_objects = sorted(os.listdir(dirname))
    # saved object is ['PREFIX_checkpoint_4.pth', ]
    saved_checkpoint = os.path.join(dirname, saved_objects[0])

    loaded_obj = torch.load(saved_checkpoint)
    for f in ["model", "optimizer", "lr_scheduler"]:
        assert f in loaded_obj
    loaded_model_state_dict = loaded_obj['model']
    loaded_optimizer_state_dict = loaded_obj['optimizer']
    loaded_lr_scheduler_state_dict = loaded_obj['lr_scheduler']

    assert isinstance(loaded_model_state_dict, dict)
    assert isinstance(loaded_optimizer_state_dict, dict)
    assert isinstance(loaded_lr_scheduler_state_dict, dict)

    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        assert key in loaded_model_state_dict
        model_value = model_state_dict[key]
        loaded_model_value = loaded_model_state_dict[key]
        assert model_value.numpy() == loaded_model_value.numpy()

    optim_state_dict = optim.state_dict()
    for key in optim_state_dict.keys():
        assert key in loaded_optimizer_state_dict
        optim_value = optim_state_dict[key]
        loaded_optim_value = loaded_optimizer_state_dict[key]
        assert optim_value == loaded_optim_value

    lr_scheduler_state_dict = lr_scheduler.state_dict()
    for key in lr_scheduler_state_dict.keys():
        assert key in loaded_lr_scheduler_state_dict
        lr_scheduler_value = lr_scheduler_state_dict[key]
        loaded_lr_scheduler_value = loaded_lr_scheduler_state_dict[key]
        assert lr_scheduler_value == loaded_lr_scheduler_value


def test_checkpoint_wrong_input():

    with pytest.raises(TypeError, match=r"Argument `to_save` should be a dictionary"):
        Checkpoint(12, lambda x: x, "prefix", )

    with pytest.raises(TypeError, match=r"Argument `to_save` should be a dictionary"):
        Checkpoint([12, ], lambda x: x, "prefix")

    with pytest.raises(ValueError, match=r"No objects to checkpoint."):
        Checkpoint({}, lambda x: x, "prefix")

    model = DummyModel()
    to_save = {'model': model}

    with pytest.raises(TypeError, match=r"Argument `save_handler` should be callable"):
        Checkpoint(to_save, 12, "prefix")

    with pytest.raises(ValueError,
                       match=r"If `score_name` is provided, then `score_function` should be also provided."):
        Checkpoint(to_save, lambda x: x, score_name="acc")


def test_checkpoint__setup_checkpoint():
    save_handler = MagicMock()

    to_save = {'model1': DummyModel(), 'model2': DummyModel()}

    checkpointer = Checkpoint(to_save, save_handler=save_handler)
    chkpt = checkpointer._setup_checkpoint()
    assert isinstance(chkpt, dict)
    for k in ['model1', 'model2']:
        assert k in chkpt
        assert chkpt[k] == to_save[k].state_dict()


def test_checkpoint_default():
    save_handler = MagicMock()
    save_handler.remove = MagicMock()

    model = DummyModel()
    to_save = {'model': model}

    checkpointer = Checkpoint(to_save, save_handler=save_handler)

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)
    checkpointer(engine)

    assert save_handler.call_count == 1
    save_handler.assert_called_with(checkpointer._setup_checkpoint(), "model_0.pth")

    engine.state.iteration += 1
    checkpointer(engine)
    assert save_handler.call_count == 2
    save_handler.assert_called_with(checkpointer._setup_checkpoint(), "model_1.pth")
    assert save_handler.remove.call_count == 1
    save_handler.remove.assert_called_with("model_0.pth")


def test_checkpoint_score_function():

    model = DummyModel()
    to_save = {'model': model}

    def _test(scores, score_name=None):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        def score_function(_):
            return next(scores)

        checkpointer = Checkpoint(to_save, save_handler=save_handler,
                                  score_function=score_function, score_name=score_name)

        engine = Engine(lambda e, b: None)
        engine.state = State(epoch=0, iteration=0)
        checkpointer(engine)

        assert save_handler.call_count == 1
        fname = "model_{}1.2.pth"
        if score_name is not None:
            fname = fname.format("0_" + score_name + "=")
        else:
            fname = fname.format("")
        save_handler.assert_called_with(checkpointer._setup_checkpoint(), fname)

        engine.state.epoch += 1
        checkpointer(engine)
        assert save_handler.call_count == 1

        engine.state.epoch += 1
        checkpointer(engine)
        assert save_handler.call_count == 2
        fname2 = "model_{}3.1.pth"
        if score_name is not None:
            fname2 = fname2.format("2_" + score_name + "=")
        else:
            fname2 = fname2.format("")
        save_handler.assert_called_with(checkpointer._setup_checkpoint(), fname2)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(fname)

    _test(iter([1.2, -2., 3.1, -4.0]))
    _test(iter([1.2, float('nan'), 3.1, -4.0]))
    _test(iter([1.2, -2., 3.1, -4.0]), score_name="acc")
    _test(iter([1.2, float('nan'), 3.1, -4.0]), score_name="acc")


def test_checkpoint_last_checkpoint():
    save_handler = MagicMock()
    save_handler.__call__ = MagicMock()
    model = DummyModel()
    to_save = {'model': model}

    checkpointer = Checkpoint(to_save, save_handler=save_handler)
    assert checkpointer.last_checkpoint is None

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    checkpointer(engine)
    assert checkpointer.last_checkpoint == "model_0.pth"


def test_checkpoint_load_objects():

    with pytest.raises(TypeError, match=r"Argument checkpoint should be a dictionary"):
        Checkpoint.load_objects({}, [])

    model = DummyModel()
    to_load = {'model': model}

    with pytest.raises(ValueError, match=r"from `to_load` is not found in the checkpoint"):
        Checkpoint.load_objects(to_load, {})

    model = DummyModel()
    to_load = {'model': model}
    model2 = DummyModel()

    chkpt = {'model': model2.state_dict()}
    Checkpoint.load_objects(to_load, chkpt)
    assert model.state_dict() == model2.state_dict()


def test_disksaver_wrong_input(dirname):

    with pytest.raises(ValueError, match=r"Directory path '\S+' is not found"):
        DiskSaver("/tmp/non-existing-folder", create_dir=False)

    previous_fname = os.path.join(dirname, '{}_{}_{}.pth'.format(_PREFIX, 'obj', 1))
    with open(previous_fname, 'w') as f:
        f.write("test")

    with pytest.raises(ValueError, match=r"Files are already present in the directory"):
        DiskSaver(dirname, require_empty=True)

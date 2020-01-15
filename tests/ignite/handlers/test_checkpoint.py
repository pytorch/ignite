import os
import warnings

import torch
import torch.nn as nn

from ignite.engine import Engine, Events, State
from ignite.handlers import ModelCheckpoint, Checkpoint, DiskSaver

import pytest
from unittest.mock import MagicMock

_PREFIX = 'PREFIX'


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


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

    with pytest.raises(TypeError, match=r"global_step_transform should be a function."):
        Checkpoint(to_save, lambda x: x, score_function=lambda e: 123, score_name="acc", global_step_transform=123)


def test_checkpoint_default():

    def _test(to_save, obj, name):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        checkpointer = Checkpoint(to_save, save_handler=save_handler)
        assert checkpointer.last_checkpoint is None

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=0, iteration=0)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        save_handler.assert_called_with(obj, "{}_0.pth".format(name))

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer(trainer)
        assert save_handler.call_count == 2
        save_handler.assert_called_with(obj, "{}_1234.pth".format(name))
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_0.pth".format(name))
        assert checkpointer.last_checkpoint == "{}_1234.pth".format(name)

    model = DummyModel()
    to_save = {'model': model}
    _test(to_save, model.state_dict(), 'model')

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {'model': model, 'optimizer': optimizer}
    _test(to_save, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint')


def test_checkpoint_with_global_step_transform():

    def _test(filename_prefix, to_save, obj, name):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        checkpointer = Checkpoint(to_save, save_handler=save_handler, filename_prefix=filename_prefix,
                                  global_step_transform=lambda e, _: e.state.epoch)

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        if len(filename_prefix) > 0:
            filename_prefix += "_"

        save_handler.assert_called_with(obj, "{}{}_1.pth".format(filename_prefix, name))

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer(trainer)
        assert save_handler.call_count == 2
        save_handler.assert_called_with(obj, "{}{}_12.pth".format(filename_prefix, name))
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}{}_1.pth".format(filename_prefix, name))
        assert checkpointer.last_checkpoint == "{}{}_12.pth".format(filename_prefix, name)

    for prefix in ["", "dummytask"]:
        model = DummyModel()
        to_save = {'model': model}
        _test(prefix, to_save, model.state_dict(), 'model')

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        to_save = {'model': model, 'optimizer': optimizer}
        _test(prefix, to_save, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint')


def test_checkpoint_with_score_function():

    def _test(to_save, obj, name):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        checkpointer = Checkpoint(to_save, save_handler=save_handler, score_function=lambda e: e.state.score)

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1, score=0.77)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        save_handler.assert_called_with(obj, "{}_0.77.pth".format(name))

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        trainer.state.score = 0.78

        checkpointer(trainer)
        assert save_handler.call_count == 2
        save_handler.assert_called_with(obj, "{}_0.78.pth".format(name))
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_0.77.pth".format(name))
        assert checkpointer.last_checkpoint == "{}_0.78.pth".format(name)

    model = DummyModel()
    to_save = {'model': model}
    _test(to_save, model.state_dict(), 'model')

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {'model': model, 'optimizer': optimizer}
    _test(to_save, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint')


def test_checkpoint_with_score_name_and_function():

    def _test(to_save, obj, name):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        checkpointer = Checkpoint(to_save, save_handler=save_handler,
                                  score_name="loss",
                                  score_function=lambda e: e.state.score)

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1, score=-0.77)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        save_handler.assert_called_with(obj, "{}_loss=-0.77.pth".format(name))

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        trainer.state.score = -0.76

        checkpointer(trainer)
        assert save_handler.call_count == 2
        save_handler.assert_called_with(obj, "{}_loss=-0.76.pth".format(name))
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_loss=-0.77.pth".format(name))
        assert checkpointer.last_checkpoint == "{}_loss=-0.76.pth".format(name)

    model = DummyModel()
    to_save = {'model': model}
    _test(to_save, model.state_dict(), 'model')

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {'model': model, 'optimizer': optimizer}
    _test(to_save, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint')


def test_checkpoint_with_score_function_and_trainer_epoch():

    def _test(to_save, obj, name):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)
        trainer.state = State(epoch=11, iteration=1)

        checkpointer = Checkpoint(to_save, save_handler=save_handler,
                                  global_step_transform=lambda _1, _2: trainer.state.epoch,
                                  score_function=lambda e: e.state.metrics['val_acc'])

        evaluator.state = State(epoch=1, iteration=1000,
                                metrics={'val_acc': 0.77})
        checkpointer(evaluator)
        assert save_handler.call_count == 1

        save_handler.assert_called_with(obj, "{}_11_0.77.pth".format(name))

        trainer.state.epoch = 12
        evaluator.state.metrics['val_acc'] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        save_handler.assert_called_with(obj, "{}_12_0.78.pth".format(name))
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_11_0.77.pth".format(name))
        assert checkpointer.last_checkpoint == "{}_12_0.78.pth".format(name)

    model = DummyModel()
    to_save = {'model': model}
    _test(to_save, model.state_dict(), 'model')


def test_checkpoint_with_score_name_and_function_and_trainer_epoch():

    def _test(to_save, obj, name):
        save_handler = MagicMock()
        save_handler.remove = MagicMock()

        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)
        trainer.state = State(epoch=11, iteration=1)

        checkpointer = Checkpoint(to_save, save_handler=save_handler,
                                  global_step_transform=lambda _1, _2: trainer.state.epoch,
                                  score_name="val_acc",
                                  score_function=lambda e: e.state.metrics['val_acc'])

        evaluator.state = State(epoch=1, iteration=1000,
                                metrics={'val_acc': 0.77})

        checkpointer(evaluator)
        assert save_handler.call_count == 1

        save_handler.assert_called_with(obj, "{}_11_val_acc=0.77.pth".format(name))

        trainer.state.epoch = 12
        evaluator.state.metrics['val_acc'] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        save_handler.assert_called_with(obj, "{}_12_val_acc=0.78.pth".format(name))
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_11_val_acc=0.77.pth".format(name))
        assert checkpointer.last_checkpoint == "{}_12_val_acc=0.78.pth".format(name)

    model = DummyModel()
    to_save = {'model': model}
    _test(to_save, model.state_dict(), 'model')


def test_model_checkpoint_args_validation(dirname):
    existing = os.path.join(dirname, 'existing_dir')
    nonempty = os.path.join(dirname, 'nonempty')

    os.makedirs(existing)
    os.makedirs(nonempty)

    with open(os.path.join(nonempty, '{}_name_0.pth'.format(_PREFIX)), 'w'):
        pass

    with pytest.raises(ValueError, match=r"with extension '.pth' or '.pth.tar' are already present "):
        ModelCheckpoint(nonempty, _PREFIX)

    with pytest.raises(ValueError, match=r"Argument save_interval is deprecated and should be None"):
        ModelCheckpoint(existing, _PREFIX, save_interval=42)

    with pytest.raises(ValueError, match=r"Directory path '\S+' is not found"):
        ModelCheckpoint(os.path.join(dirname, 'non_existing_dir'), _PREFIX, create_dir=False)

    with pytest.raises(ValueError, match=r"Argument save_as_state_dict is deprecated and should be True"):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, save_as_state_dict=False)

    with pytest.raises(ValueError, match=r"If `score_name` is provided, then `score_function` "):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, score_name='test')

    with pytest.raises(TypeError, match=r"global_step_transform should be a function"):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, global_step_transform=1234)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False)
    assert h.last_checkpoint is None
    with pytest.raises(RuntimeError, match=r"No objects to checkpoint found."):
        h(None, [])


def test_model_checkpoint_simple_recovery(dirname):
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
    assert loaded_objects == model.state_dict()


def test_model_checkpoint_simple_recovery_from_existing_non_empty(dirname):

    def _test(ext, require_empty, archived):
        previous_fname = os.path.join(dirname, '{}_{}_{}{}'.format(_PREFIX, 'obj', 1, ext))
        with open(previous_fname, 'w') as f:
            f.write("test")

        h = ModelCheckpoint(dirname, _PREFIX, create_dir=True, require_empty=require_empty, archived=archived)
        engine = Engine(lambda e, b: None)
        engine.state = State(epoch=0, iteration=1)

        model = DummyModel()
        to_save = {'model': model}
        h(engine, to_save)

        fname = h.last_checkpoint
        ext = ".pth.tar" if archived else ".pth"
        assert isinstance(fname, str)
        assert os.path.join(dirname, '{}_{}_{}{}'.format(_PREFIX, 'model', 1, ext)) == fname
        assert os.path.exists(fname)
        assert os.path.exists(previous_fname)
        loaded_objects = torch.load(fname)
        assert loaded_objects == model.state_dict()
        os.remove(fname)

    _test(".txt", require_empty=True, archived=False)
    _test(".txt", require_empty=True, archived=True)
    _test(".pth", require_empty=False, archived=False)


def test_disk_saver_atomic(dirname):

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


def test_disabled_n_saved(dirname):

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=None)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {'model': model}

    num_iters = 100
    for i in range(num_iters):
        engine.state.iteration = i
        h(engine, to_save)

    saved_files = sorted(os.listdir(dirname))
    assert len(saved_files) == num_iters, "{}".format(saved_files)

    expected = sorted(['{}_{}_{}.pth'.format(_PREFIX, 'model', i) for i in range(num_iters)])
    assert saved_files == expected, "{} vs {}".format(saved_files, expected)


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

    expected = ['{}_{}_val_loss={:.7}.pth'.format(_PREFIX, 'model', scores[e - 1]) for e in [1, 3]]

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
        assert key in loaded_model_state_dict

        model_value = model_state_dict[key]
        loaded_model_value = loaded_model_state_dict[key]

        assert model_value.numpy() == loaded_model_value.numpy()


def test_valid_state_dict_save(dirname):
    model = DummyModel()
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    to_save = {'name': 42}
    with pytest.raises(TypeError, match=r"should have `state_dict` method"):
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


def test_checkpoint_load_objects():

    with pytest.raises(TypeError, match=r"Argument checkpoint should be a dictionary"):
        Checkpoint.load_objects({}, [])

    with pytest.raises(TypeError, match=r"should have `load_state_dict` method"):
        Checkpoint.load_objects({"a": None}, {"a": None})

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

    def _test(ext):
        previous_fname = os.path.join(dirname, '{}_{}_{}{}'.format(_PREFIX, 'obj', 1, ext))
        with open(previous_fname, 'w') as f:
            f.write("test")

        with pytest.raises(ValueError, match=r"with extension '.pth' or '.pth.tar' are already present"):
            DiskSaver(dirname, require_empty=True)

    _test(".pth")
    _test(".pth.tar")

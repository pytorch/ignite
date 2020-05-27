import os
import warnings
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import ignite.distributed as idist
from ignite.engine import Engine, Events, State
from ignite.handlers import Checkpoint, DiskSaver, ModelCheckpoint
from ignite.handlers.checkpoint import BaseSaveHandler

_PREFIX = "PREFIX"


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


class DummyPretrainedModel(nn.Module):
    def __init__(self):
        super(DummyPretrainedModel, self).__init__()
        self.features = nn.Linear(4, 2, bias=False)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def test_checkpoint_wrong_input():

    with pytest.raises(TypeError, match=r"Argument `to_save` should be a dictionary"):
        Checkpoint(12, lambda x: x, "prefix")

    with pytest.raises(TypeError, match=r"Argument `to_save` should be a dictionary"):
        Checkpoint([12], lambda x: x, "prefix")

    with pytest.raises(ValueError, match=r"No objects to checkpoint."):
        Checkpoint({}, lambda x: x, "prefix")

    model = DummyModel()
    to_save = {"model": model}

    with pytest.raises(TypeError, match=r"Argument `save_handler` should be callable"):
        Checkpoint(to_save, 12, "prefix")

    with pytest.raises(
        ValueError, match=r"If `score_name` is provided, then `score_function` should be also provided."
    ):
        Checkpoint(to_save, lambda x: x, score_name="acc")

    with pytest.raises(TypeError, match=r"global_step_transform should be a function."):
        Checkpoint(to_save, lambda x: x, score_function=lambda e: 123, score_name="acc", global_step_transform=123)

    with pytest.warns(UserWarning, match=r"Argument archived is deprecated"):
        Checkpoint(to_save, lambda x: x, score_function=lambda e: 123, score_name="acc", archived=True)


def test_checkpoint_score_function_wrong_output():
    model = DummyModel()
    to_save = {"model": model}

    checkpointer = Checkpoint(to_save, lambda x: x, score_function=lambda e: {"1": 1}, score_name="acc")
    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)
    with pytest.raises(ValueError, match=r"Output of score_function should be a number"):
        checkpointer(trainer)


def test_checkpoint_default():
    def _test(to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(to_save, save_handler=save_handler)
        assert checkpointer.last_checkpoint is None

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=0, iteration=0)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": None, "priority": 0}
        save_handler.assert_called_with(obj, "{}_0.pt".format(name), metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 1234
        save_handler.assert_called_with(obj, "{}_1234.pt".format(name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_0.pt".format(name))
        assert checkpointer.last_checkpoint == "{}_1234.pt".format(name)

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_with_global_step_transform():
    def _test(filename_prefix, to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(
            to_save,
            save_handler=save_handler,
            filename_prefix=filename_prefix,
            global_step_transform=lambda e, _: e.state.epoch,
        )

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        if len(filename_prefix) > 0:
            filename_prefix += "_"

        metadata = {"basename": "{}{}".format(filename_prefix, name), "score_name": None, "priority": 1}
        save_handler.assert_called_with(obj, "{}{}_1.pt".format(filename_prefix, name), metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 1234
        save_handler.assert_called_with(obj, "{}{}_12.pt".format(filename_prefix, name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}{}_1.pt".format(filename_prefix, name))
        assert checkpointer.last_checkpoint == "{}{}_12.pt".format(filename_prefix, name)

    for prefix in ["", "dummytask"]:
        model = DummyModel()
        to_save = {"model": model}
        _test(prefix, to_save, model.state_dict(), "model")

        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        to_save = {"model": model, "optimizer": optimizer}
        _test(prefix, to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_with_score_function():
    def _test(to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(to_save, save_handler=save_handler, score_function=lambda e: e.state.score)

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1, score=0.77)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": None, "priority": 0.77}
        save_handler.assert_called_with(obj, "{}_0.7700.pt".format(name), metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        trainer.state.score = 0.78

        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, "{}_0.7800.pt".format(name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_0.7700.pt".format(name))
        assert checkpointer.last_checkpoint == "{}_0.7800.pt".format(name)

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_with_score_name_and_function():
    def _test(to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(
            to_save, save_handler=save_handler, score_name="loss", score_function=lambda e: e.state.score
        )

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1, score=-0.77)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": "loss", "priority": -0.77}
        save_handler.assert_called_with(obj, "{}_loss=-0.7700.pt".format(name), metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        trainer.state.score = -0.76

        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = -0.76
        save_handler.assert_called_with(obj, "{}_loss=-0.7600.pt".format(name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_loss=-0.7700.pt".format(name))
        assert checkpointer.last_checkpoint == "{}_loss=-0.7600.pt".format(name)

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_with_int_score():
    def _test(to_save, obj, name, score_name=None):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(
            to_save, save_handler=save_handler, score_name=score_name, score_function=lambda e: e.state.epoch
        )

        if score_name is None:
            score_name = ""
        else:
            score_name += "="

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=1, iteration=1)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": score_name[:-1] if len(score_name) > 0 else None, "priority": 1}
        save_handler.assert_called_with(obj, "{}_{}1.pt".format(name, score_name), metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234

        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 12
        save_handler.assert_called_with(obj, "{}_{}12.pt".format(name, score_name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_{}1.pt".format(name, score_name))
        assert checkpointer.last_checkpoint == "{}_{}12.pt".format(name, score_name)

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")
    _test(to_save, model.state_dict(), "model", "epoch")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint", "epoch")


def test_checkpoint_with_score_function_and_trainer_epoch():
    def _test(to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)
        trainer.state = State(epoch=11, iteration=1)

        checkpointer = Checkpoint(
            to_save,
            save_handler=save_handler,
            global_step_transform=lambda _1, _2: trainer.state.epoch,
            score_function=lambda e: e.state.metrics["val_acc"],
        )

        evaluator.state = State(epoch=1, iteration=1000, metrics={"val_acc": 0.77})
        checkpointer(evaluator)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": None, "priority": 0.77}
        save_handler.assert_called_with(obj, "{}_11_0.7700.pt".format(name), metadata)

        trainer.state.epoch = 12
        evaluator.state.metrics["val_acc"] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, "{}_12_0.7800.pt".format(name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_11_0.7700.pt".format(name))
        assert checkpointer.last_checkpoint == "{}_12_0.7800.pt".format(name)

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")


def test_checkpoint_with_score_name_and_function_and_trainer_epoch():
    def _test(to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)
        trainer.state = State(epoch=11, iteration=1)

        checkpointer = Checkpoint(
            to_save,
            save_handler=save_handler,
            global_step_transform=lambda _1, _2: trainer.state.epoch,
            score_name="val_acc",
            score_function=lambda e: e.state.metrics["val_acc"],
        )

        evaluator.state = State(epoch=1, iteration=1000, metrics={"val_acc": 0.77})

        checkpointer(evaluator)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": "val_acc", "priority": 0.77}
        save_handler.assert_called_with(obj, "{}_11_val_acc=0.7700.pt".format(name), metadata)

        trainer.state.epoch = 12
        evaluator.state.metrics["val_acc"] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, "{}_12_val_acc=0.7800.pt".format(name), metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with("{}_11_val_acc=0.7700.pt".format(name))
        assert checkpointer.last_checkpoint == "{}_12_val_acc=0.7800.pt".format(name)

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")


def test_checkpoint_last_checkpoint():
    save_handler = MagicMock(spec=BaseSaveHandler)
    to_save = {"model": DummyModel()}

    checkpointer = Checkpoint(to_save, save_handler=save_handler, n_saved=None)

    trainer = Engine(lambda e, b: None)

    for i in range(10):
        trainer.state = State(epoch=1, iteration=i)
        checkpointer(trainer)

    assert save_handler.call_count == 10
    assert checkpointer.last_checkpoint == "{}_9.pt".format("model")


def test_checkpoint_last_checkpoint_on_score():
    save_handler = MagicMock(spec=BaseSaveHandler)
    to_save = {"model": DummyModel()}

    checkpointer = Checkpoint(
        to_save,
        save_handler=save_handler,
        n_saved=None,
        score_name="val_acc",
        score_function=lambda e: e.state.metrics["val_acc"],
    )

    trainer = Engine(lambda e, b: None)

    val_acc = 0.0
    for i in range(10):
        val_acc = i * 0.1
        trainer.state = State(epoch=1, iteration=i, metrics={"val_acc": val_acc})
        checkpointer(trainer)

    assert save_handler.call_count == 10
    assert checkpointer.last_checkpoint == "{}_val_acc=0.9000.pt".format("model")


def test_checkpoint_save_handler_callable():
    def save_handler(c, f):
        assert f == "model_12.pt"

    to_save = {"model": DummyModel()}

    checkpointer = Checkpoint(to_save, save_handler=save_handler,)

    trainer = Engine(lambda e, b: None)

    trainer.state = State(epoch=1, iteration=12)
    checkpointer(trainer)


def test_model_checkpoint_args_validation(dirname):
    existing = os.path.join(dirname, "existing_dir")
    nonempty = os.path.join(dirname, "nonempty")

    os.makedirs(existing)
    os.makedirs(nonempty)

    with open(os.path.join(nonempty, "{}_name_0.pt".format(_PREFIX)), "w"):
        pass

    with pytest.raises(ValueError, match=r"with extension '.pt' are already present "):
        ModelCheckpoint(nonempty, _PREFIX)

    with pytest.raises(ValueError, match=r"Argument save_interval is deprecated and should be None"):
        ModelCheckpoint(existing, _PREFIX, save_interval=42)

    with pytest.raises(ValueError, match=r"Directory path '\S+' is not found"):
        ModelCheckpoint(os.path.join(dirname, "non_existing_dir"), _PREFIX, create_dir=False)

    with pytest.raises(ValueError, match=r"Argument save_as_state_dict is deprecated and should be True"):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, save_as_state_dict=False)

    with pytest.raises(ValueError, match=r"If `score_name` is provided, then `score_function` "):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, score_name="test")

    with pytest.raises(TypeError, match=r"global_step_transform should be a function"):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, global_step_transform=1234)

    with pytest.warns(UserWarning, match=r"Argument archived is deprecated"):
        ModelCheckpoint(existing, _PREFIX, create_dir=False, archived=True)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False)
    assert h.last_checkpoint is None
    with pytest.raises(RuntimeError, match=r"No objects to checkpoint found."):
        h(None, [])


def test_model_checkpoint_simple_recovery(dirname):
    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=1)

    model = DummyModel()
    to_save = {"model": model}
    h(engine, to_save)

    fname = h.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    loaded_objects = torch.load(fname)
    assert loaded_objects == model.state_dict()


def test_model_checkpoint_simple_recovery_from_existing_non_empty(dirname):
    def _test(ext, require_empty):
        previous_fname = os.path.join(dirname, "{}_{}_{}{}".format(_PREFIX, "obj", 1, ext))
        with open(previous_fname, "w") as f:
            f.write("test")

        h = ModelCheckpoint(dirname, _PREFIX, create_dir=True, require_empty=require_empty)
        engine = Engine(lambda e, b: None)
        engine.state = State(epoch=0, iteration=1)

        model = DummyModel()
        to_save = {"model": model}
        h(engine, to_save)

        fname = h.last_checkpoint
        ext = ".pt"
        assert isinstance(fname, str)
        assert os.path.join(dirname, "{}_{}_{}{}".format(_PREFIX, "model", 1, ext)) == fname
        assert os.path.exists(fname)
        assert os.path.exists(previous_fname)
        loaded_objects = torch.load(fname)
        assert loaded_objects == model.state_dict()
        os.remove(fname)

    _test(".txt", require_empty=True)
    _test(".pt", require_empty=False)


def test_disk_saver_atomic(dirname):

    model = DummyModel()
    to_save_serializable = {"model": model}
    to_save_non_serializable = {"model": lambda x: x}

    def _test_existance(atomic, _to_save, expected):

        saver = DiskSaver(dirname, atomic=atomic, create_dir=False, require_empty=False)
        fname = "test.pt"
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
    to_save = {"model": model}
    h(engine, to_save)

    for i in range(1, 9):
        engine.state.iteration = i
        h(engine, to_save)

    expected = ["{}_{}_{}.pt".format(_PREFIX, "model", i) for i in [7, 8]]

    assert sorted(os.listdir(dirname)) == expected, "{} vs {}".format(sorted(os.listdir(dirname)), expected)


def test_disabled_n_saved(dirname):

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=None)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {"model": model}

    num_iters = 100
    for i in range(num_iters):
        engine.state.iteration = i
        h(engine, to_save)

    saved_files = sorted(os.listdir(dirname))
    assert len(saved_files) == num_iters, "{}".format(saved_files)

    expected = sorted(["{}_{}_{}.pt".format(_PREFIX, "model", i) for i in range(num_iters)])
    assert saved_files == expected, "{} vs {}".format(saved_files, expected)


def test_best_k(dirname):
    scores = iter([1.2, -2.0, 3.1, -4.0])

    def score_function(_):
        return next(scores)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2, score_function=score_function)

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {"model": model}
    for _ in range(4):
        h(engine, to_save)

    expected = ["{}_{}_{:.4f}.pt".format(_PREFIX, "model", i) for i in [1.2, 3.1]]

    assert sorted(os.listdir(dirname)) == expected


def test_best_k_with_suffix(dirname):
    scores = [0.3456789, 0.1234, 0.4567, 0.134567]
    scores_iter = iter(scores)

    def score_function(engine):
        return next(scores_iter)

    h = ModelCheckpoint(
        dirname, _PREFIX, create_dir=False, n_saved=2, score_function=score_function, score_name="val_loss"
    )

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {"model": model}
    for _ in range(4):
        engine.state.epoch += 1
        h(engine, to_save)

    expected = ["{}_{}_val_loss={:.4}.pt".format(_PREFIX, "model", scores[e - 1]) for e in [1, 3]]

    assert sorted(os.listdir(dirname)) == expected


def test_removes_each_score_at_most_once(dirname):
    scores = [0, 1, 1, 2, 3]
    scores_iter = iter(scores)

    def score_function(_):
        return next(scores_iter)

    h = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2, score_function=score_function)

    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=0)

    model = DummyModel()
    to_save = {"model": model}
    for _ in range(len(scores)):
        h(engine, to_save)

    # If a score was removed multiple times, the code above would have raise a
    # FileNotFoundError. So this just tests the absence of such a failure
    # without futher assertions.


def test_with_engine(dirname):
    def update_fn(_1, _2):
        pass

    name = "model"
    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2)

    model = DummyModel()
    to_save = {"model": model}
    engine.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    engine.run([0], max_epochs=4)

    expected = ["{}_{}_{}.pt".format(_PREFIX, name, i) for i in [3, 4]]

    assert sorted(os.listdir(dirname)) == expected


def test_with_state_dict(dirname):
    def update_fn(_1, _2):
        pass

    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)

    model = DummyModel()
    to_save = {"model": model}
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

    to_save = {"name": 42}
    with pytest.raises(TypeError, match=r"should have `state_dict` method"):
        h(engine, to_save)
    to_save = {"name": model}
    try:
        h(engine, to_save)
    except ValueError:
        pytest.fail("Unexpected ValueError")


def _test_save_model_optimizer_lr_scheduler_with_state_dict(device, dirname):

    torch.manual_seed(23)

    model = DummyModel().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)

    def update_fn(engine, batch):
        x = torch.rand((4, 1)).to(device)
        optim.zero_grad()
        y = model(x)
        loss = y.pow(2.0).sum()
        loss.backward()
        optim.step()
        lr_scheduler.step()

    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=True, n_saved=1)

    engine.add_event_handler(
        Events.EPOCH_COMPLETED, handler, {"model": model, "optimizer": optim, "lr_scheduler": lr_scheduler}
    )
    engine.run([0], max_epochs=4)

    saved_objects = sorted(os.listdir(dirname))
    # saved object is ['PREFIX_checkpoint_4.pt', ]
    saved_checkpoint = os.path.join(dirname, saved_objects[0])

    loaded_obj = torch.load(saved_checkpoint)
    for f in ["model", "optimizer", "lr_scheduler"]:
        assert f in loaded_obj
    loaded_model_state_dict = loaded_obj["model"]
    loaded_optimizer_state_dict = loaded_obj["optimizer"]
    loaded_lr_scheduler_state_dict = loaded_obj["lr_scheduler"]

    assert isinstance(loaded_model_state_dict, dict)
    assert isinstance(loaded_optimizer_state_dict, dict)
    assert isinstance(loaded_lr_scheduler_state_dict, dict)

    # Specifically move device to CPU first
    model_state_dict = model.cpu().state_dict()
    for key in model_state_dict.keys():
        assert key in loaded_model_state_dict
        model_value = model_state_dict[key]
        loaded_model_value = loaded_model_state_dict[key]
        assert model_value.cpu().numpy() == loaded_model_value.cpu().numpy()

    optim_state_dict = optim.state_dict()
    for key in optim_state_dict.keys():
        assert key in loaded_optimizer_state_dict
        optim_value = optim_state_dict[key]
        loaded_optim_value = loaded_optimizer_state_dict[key]
        if idist.get_rank() == 0:
            assert optim_value == loaded_optim_value

    lr_scheduler_state_dict = lr_scheduler.state_dict()
    for key in lr_scheduler_state_dict.keys():
        assert key in loaded_lr_scheduler_state_dict
        lr_scheduler_value = lr_scheduler_state_dict[key]
        loaded_lr_scheduler_value = loaded_lr_scheduler_state_dict[key]
        assert lr_scheduler_value == loaded_lr_scheduler_value


def test_save_model_optimizer_lr_scheduler_with_state_dict(dirname):
    _test_save_model_optimizer_lr_scheduler_with_state_dict("cpu", dirname)


def test_checkpoint_load_objects():

    with pytest.raises(TypeError, match=r"Argument checkpoint should be a dictionary"):
        Checkpoint.load_objects({}, [])

    with pytest.raises(TypeError, match=r"should have `load_state_dict` method"):
        Checkpoint.load_objects({"a": None}, {"a": None})

    model = DummyModel()
    to_load = {"model": model, "another_model": model}

    with pytest.raises(ValueError, match=r"from `to_load` is not found in the checkpoint"):
        Checkpoint.load_objects(to_load, {})

    model = DummyModel()
    to_load = {"model": model}
    model2 = DummyModel()

    chkpt = {"model": model2.state_dict()}
    Checkpoint.load_objects(to_load, chkpt)
    assert model.state_dict() == model2.state_dict()


def test_checkpoint_load_objects_from_saved_file(dirname):
    def _get_single_obj_to_save():
        model = DummyModel()
        to_save = {"model": model}
        return to_save

    def _get_multiple_objs_to_save():
        model = DummyModel()
        optim = torch.optim.SGD(model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)
        to_save = {"model": model, "optimizer": optim, "lr_scheduler": lr_scheduler}
        return to_save

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)

    # case: multiple objects
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)
    to_save = _get_multiple_objs_to_save()
    handler(trainer, to_save)
    fname = handler.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    loaded_objects = torch.load(fname)
    Checkpoint.load_objects(to_save, loaded_objects)
    os.remove(fname)

    # case: saved multiple objects, loaded single object
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)
    to_save = _get_multiple_objs_to_save()
    handler(trainer, to_save)
    fname = handler.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    loaded_objects = torch.load(fname)
    to_load = {"model": to_save["model"]}
    Checkpoint.load_objects(to_load, loaded_objects)
    os.remove(fname)

    # case: single object
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)
    to_save = _get_single_obj_to_save()
    handler(trainer, to_save)
    fname = handler.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    loaded_objects = torch.load(fname)
    Checkpoint.load_objects(to_save, loaded_objects)


def test_load_checkpoint_with_different_num_classes(dirname):
    model = DummyPretrainedModel()
    to_save_single_object = {"model": model}

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)

    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)
    handler(trainer, to_save_single_object)

    fname = handler.last_checkpoint
    loaded_checkpoint = torch.load(fname)

    to_load_single_object = {"pretrained_features": model.features}

    with pytest.raises(RuntimeError):
        Checkpoint.load_objects(to_load_single_object, loaded_checkpoint)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        Checkpoint.load_objects(to_load_single_object, loaded_checkpoint, strict=False, blah="blah")

    loaded_weights = to_load_single_object["pretrained_features"].state_dict()["weight"]

    assert torch.all(model.state_dict()["features.weight"].eq(loaded_weights))


def test_disksaver_wrong_input(dirname):

    with pytest.raises(ValueError, match=r"Directory path '\S+' is not found"):
        DiskSaver("/tmp/non-existing-folder", create_dir=False)

    def _test(ext):
        previous_fname = os.path.join(dirname, "{}_{}_{}{}".format(_PREFIX, "obj", 1, ext))
        with open(previous_fname, "w") as f:
            f.write("test")

        with pytest.raises(ValueError, match=r"with extension '.pt' are already present"):
            DiskSaver(dirname, require_empty=True)

    _test(".pt")


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo, get_rank_zero_dirname):
    dirname = get_rank_zero_dirname("cpu")
    _test_save_model_optimizer_lr_scheduler_with_state_dict("cpu", os.path.join(dirname, "1"))


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl, get_rank_zero_dirname):
    device = idist.device()
    dirname = get_rank_zero_dirname(device)
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, os.path.join(dirname, "1"))


def _test_tpu_saves_to_cpu(device, dirname):

    h = ModelCheckpoint(dirname, _PREFIX)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=1)

    model = DummyModel().to(device)
    to_save = {"model": model}

    h(engine, to_save)

    fname = h.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    loaded_objects = torch.load(fname)
    assert loaded_objects == model.cpu().state_dict()


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla(dirname):
    assert "xla" in idist.device().type
    _test_tpu_saves_to_cpu(idist.device(), os.path.join(dirname, "1"))
    _test_save_model_optimizer_lr_scheduler_with_state_dict(idist.device(), os.path.join(dirname, "2"))


def _test_tpu_saves_to_cpu_nprocs(index, dirname):
    _test_tpu_saves_to_cpu(idist.device(), os.path.join(dirname, "1"))
    _test_save_model_optimizer_lr_scheduler_with_state_dict(idist.device(), os.path.join(dirname, "2"))


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla_nprocs(dirname, xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_tpu_saves_to_cpu_nprocs, args=(dirname,), nprocs=n)

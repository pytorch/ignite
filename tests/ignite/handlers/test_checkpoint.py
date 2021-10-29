import os
import stat
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from pkg_resources import parse_version

import ignite.distributed as idist
from ignite.engine import Engine, Events, State
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint, global_step_from_engine
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

    model = DummyModel()
    to_save = {"model": model}

    with pytest.raises(
        TypeError, match=r"Argument `save_handler` should be a string or callable or inherit from BaseSaveHandler"
    ):
        Checkpoint(to_save, 12, "prefix")

    with pytest.raises(TypeError, match=r"global_step_transform should be a function."):
        Checkpoint(to_save, lambda x: x, score_function=lambda e: 123, score_name="acc", global_step_transform=123)

    with pytest.raises(ValueError, match=r"Cannot have key 'checkpointer' if `include_self` is True"):
        Checkpoint({"checkpointer": model}, lambda x: x, include_self=True)

    class ImmutableMapping(Mapping):
        def __getitem__(self, key):
            return to_save[key]

        def __iter__(self):
            return iter(to_save)

        def __len__(self):
            return len(to_save)

    with pytest.raises(TypeError, match="If `include_self` is True, then `to_save` must be mutable"):
        Checkpoint(ImmutableMapping(), lambda x: x, include_self=True)


def test_save_handler_as_str(dirname):
    model = DummyModel()
    to_save = {"model": model}

    checkpointer = Checkpoint(to_save, save_handler=dirname)
    assert isinstance(checkpointer.save_handler, DiskSaver)


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
        save_handler.assert_called_with(obj, f"{name}_0.pt", metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 1234
        save_handler.assert_called_with(obj, f"{name}_1234.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_0.pt")
        assert checkpointer.last_checkpoint == f"{name}_1234.pt"

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_include_self_state_dict():
    def _test(to_save, obj, name):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(to_save, save_handler=save_handler, include_self=True)
        assert checkpointer.last_checkpoint is None

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=0, iteration=0)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        fname = f"{name}_0.pt"
        obj["checkpointer"] = OrderedDict([("saved", [(0, fname)])])

        metadata = {"basename": name, "score_name": None, "priority": 0}
        save_handler.assert_called_with(obj, fname, metadata)

        # Swap object, state should be maintained
        checkpointer2 = Checkpoint(to_save, save_handler=save_handler, include_self=True)
        checkpointer2.load_state_dict(checkpointer.state_dict())
        assert checkpointer2.last_checkpoint == fname

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer2(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 1234

        # This delete only happens if state was restored correctly.
        save_handler.remove.assert_called_with(f"{name}_0.pt")

        fname = f"{name}_1234.pt"
        obj["checkpointer"] = OrderedDict([("saved", [(1234, fname)])])

        save_handler.assert_called_with(obj, fname, metadata)
        assert save_handler.remove.call_count == 1
        assert checkpointer2.last_checkpoint == fname

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_with_dp():

    model = DummyModel()
    dp_model = nn.DataParallel(model)
    to_save = {"model": dp_model}

    save_handler = MagicMock(spec=BaseSaveHandler)
    checkpointer = Checkpoint(to_save, save_handler=save_handler)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)

    checkpointer(trainer)
    assert save_handler.call_count == 1
    metadata = {"basename": "model", "score_name": None, "priority": 0}
    save_handler.assert_called_with(model.state_dict(), "model_0.pt", metadata)


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
        trainer.state = State(epoch=2, iteration=1)

        checkpointer(trainer)
        assert save_handler.call_count == 1

        if len(filename_prefix) > 0:
            filename_prefix += "_"

        metadata = {"basename": f"{filename_prefix}{name}", "score_name": None, "priority": 2}
        save_handler.assert_called_with(obj, f"{filename_prefix}{name}_2.pt", metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 12
        save_handler.assert_called_with(obj, f"{filename_prefix}{name}_12.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{filename_prefix}{name}_2.pt")
        assert checkpointer.last_checkpoint == f"{filename_prefix}{name}_12.pt"

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
        save_handler.assert_called_with(obj, f"{name}_0.7700.pt", metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        trainer.state.score = 0.78

        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, f"{name}_0.7800.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_0.7700.pt")
        assert checkpointer.last_checkpoint == f"{name}_0.7800.pt"

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    to_save = {"model": model, "optimizer": optimizer}
    _test(to_save, {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint")


def test_checkpoint_with_score_name_only():
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
        )

        evaluator.state = State(epoch=1, iteration=1000, metrics={"val_acc": 0.77})

        checkpointer(evaluator)
        assert save_handler.call_count == 1

        metadata = {"basename": name, "score_name": "val_acc", "priority": 0.77}
        save_handler.assert_called_with(obj, f"{name}_11_val_acc=0.7700.pt", metadata)

        trainer.state.epoch = 12
        evaluator.state.metrics["val_acc"] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, f"{name}_12_val_acc=0.7800.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_11_val_acc=0.7700.pt")
        assert checkpointer.last_checkpoint == f"{name}_12_val_acc=0.7800.pt"

    model = DummyModel()
    to_save = {"model": model}
    _test(to_save, model.state_dict(), "model")


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
        save_handler.assert_called_with(obj, f"{name}_loss=-0.7700.pt", metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234
        trainer.state.score = -0.76

        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = -0.76
        save_handler.assert_called_with(obj, f"{name}_loss=-0.7600.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_loss=-0.7700.pt")
        assert checkpointer.last_checkpoint == f"{name}_loss=-0.7600.pt"

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
        save_handler.assert_called_with(obj, f"{name}_{score_name}1.pt", metadata)

        trainer.state.epoch = 12
        trainer.state.iteration = 1234

        checkpointer(trainer)
        assert save_handler.call_count == 2
        metadata["priority"] = 12
        save_handler.assert_called_with(obj, f"{name}_{score_name}12.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_{score_name}1.pt")
        assert checkpointer.last_checkpoint == f"{name}_{score_name}12.pt"

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
        save_handler.assert_called_with(obj, f"{name}_11_0.7700.pt", metadata)

        trainer.state.epoch = 12
        evaluator.state.metrics["val_acc"] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, f"{name}_12_0.7800.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_11_0.7700.pt")
        assert checkpointer.last_checkpoint == f"{name}_12_0.7800.pt"

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
        save_handler.assert_called_with(obj, f"{name}_11_val_acc=0.7700.pt", metadata)

        trainer.state.epoch = 12
        evaluator.state.metrics["val_acc"] = 0.78

        checkpointer(evaluator)
        assert save_handler.call_count == 2
        metadata["priority"] = 0.78
        save_handler.assert_called_with(obj, f"{name}_12_val_acc=0.7800.pt", metadata)
        assert save_handler.remove.call_count == 1
        save_handler.remove.assert_called_with(f"{name}_11_val_acc=0.7700.pt")
        assert checkpointer.last_checkpoint == f"{name}_12_val_acc=0.7800.pt"

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
    assert checkpointer.last_checkpoint == "model_9.pt"


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
    assert checkpointer.last_checkpoint == "model_val_acc=0.9000.pt"


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

    with open(os.path.join(nonempty, f"{_PREFIX}_name_0.pt"), "w"):
        pass

    with pytest.raises(ValueError, match=r"with extension '.pt' are already present "):
        ModelCheckpoint(nonempty, _PREFIX)

    with pytest.raises(ValueError, match=r"Directory path '\S+' is not found"):
        ModelCheckpoint(os.path.join(dirname, "non_existing_dir"), _PREFIX, create_dir=False)

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
        previous_fname = os.path.join(dirname, f"{_PREFIX}_obj_{1}{ext}")
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
        assert os.path.join(dirname, f"{_PREFIX}_model_{1}{ext}") == fname
        assert os.path.exists(fname)
        assert os.path.exists(previous_fname)
        loaded_objects = torch.load(fname)
        assert loaded_objects == model.state_dict()
        os.remove(fname)

    _test(".txt", require_empty=True)
    _test(".pt", require_empty=False)


def test_model_checkpoint_invalid_save_handler(dirname):
    h = ModelCheckpoint(dirname, _PREFIX)
    to_save = {"model": DummyModel()}
    # Redefine save_handler
    h.save_handler = lambda x, y: None
    h(Engine(lambda x, y: None), to_save)

    with pytest.raises(
        RuntimeError, match=rf"Unable to save checkpoint, save_handler should be DiskSaver, got {type(h.save_handler)}."
    ):
        h.last_checkpoint


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
            # related to https://github.com/pytorch/ignite/issues/1876
            mode = stat.filemode(os.stat(fp).st_mode)
            assert [mode[1], mode[4], mode[7]] == ["r", "r", "r"], mode

        if expected:
            saver.remove(fname)

    _test_existance(atomic=False, _to_save=to_save_serializable, expected=True)
    _test_existance(atomic=False, _to_save=to_save_non_serializable, expected=True)

    _test_existance(atomic=True, _to_save=to_save_serializable, expected=True)
    _test_existance(atomic=True, _to_save=to_save_non_serializable, expected=False)


@pytest.mark.skipif(
    parse_version(torch.__version__) < parse_version("1.4.0"), reason="Zipfile serialization was introduced in 1.4.0"
)
def test_disk_saver_zipfile_serialization_keyword(dirname):
    model = DummyModel()
    to_save = {"model": model}

    saver = DiskSaver(dirname, create_dir=False, _use_new_zipfile_serialization=False)
    fname = "test.pt"
    saver(to_save, fname)
    fp = os.path.join(saver.dirname, fname)
    assert os.path.exists(fp)
    saver.remove(fname)


def test_disk_saver_unknown_keyword(dirname):
    model = DummyModel()
    to_save = {"model": model}

    saver = DiskSaver(dirname, create_dir=False, unknown_keyword="")
    fname = "test.pt"
    with pytest.raises(TypeError, match=r"got an unexpected keyword argument 'unknown_keyword'"):
        saver(to_save, fname)


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

    expected = [f"{_PREFIX}_model_{i}.pt" for i in [7, 8]]

    assert sorted(os.listdir(dirname)) == expected, f"{sorted(os.listdir(dirname))} vs {expected}"


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
    assert len(saved_files) == num_iters, f"{saved_files}"

    expected = sorted([f"{_PREFIX}_model_{i}.pt" for i in range(num_iters)])
    assert saved_files == expected, f"{saved_files} vs {expected}"


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

    expected = [f"{_PREFIX}_model_{i:.4f}.pt" for i in [1.2, 3.1]]

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

    expected = [f"{_PREFIX}_model_val_loss={scores[e - 1]:.4}.pt" for e in [1, 3]]

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
    engine.run([0, 1,], max_epochs=4)

    expected = sorted([f"{_PREFIX}_{name}_{i}.pt" for i in [3 * 2, 4 * 2]])

    assert sorted(os.listdir(dirname)) == expected


def test_with_state_dict(dirname):
    def update_fn(_1, _2):
        pass

    engine = Engine(update_fn)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)

    model = DummyModel()
    to_save = {"model": model}
    engine.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    engine.run([0, 1, 2], max_epochs=4)

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


def _test_save_model_optimizer_lr_scheduler_with_state_dict(device, dirname, on_zero_rank=False):

    torch.manual_seed(23)

    model = DummyModel().to(device)

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)

    def update_fn(engine, batch):
        x = torch.rand((4, 1)).to(device)
        optim.zero_grad()
        y = model(x)
        # Below code raises: RuntimeError: torch_xla/csrc/tensor_impl.cpp:144 : XLA tensors do not have storage
        # Probably related to https://github.com/pytorch/xla/issues/2576
        # loss = y.pow(2.0).sum()
        loss = y.sum()
        print(loss.device, y.device, x.device)
        loss.backward()
        if idist.has_xla_support:
            import torch_xla.core.xla_model as xm

            xm.optimizer_step(optim, barrier=True)
        else:
            optim.step()
        lr_scheduler.step()

    engine = Engine(update_fn)

    if (not on_zero_rank) or (on_zero_rank and idist.get_rank() == 0):
        handler = ModelCheckpoint(dirname, _PREFIX, create_dir=True, n_saved=1)

        engine.add_event_handler(
            Events.EPOCH_COMPLETED, handler, {"model": model, "optimizer": optim, "lr_scheduler": lr_scheduler}
        )

    engine.run([0, 1, 2], max_epochs=4)

    idist.barrier()

    saved_objects = sorted(os.listdir(dirname))
    # saved object is ['PREFIX_checkpoint_3.pt', ]
    saved_checkpoint = os.path.join(dirname, saved_objects[0])

    if idist.has_xla_support:
        device = "cpu"

    loaded_obj = torch.load(saved_checkpoint, map_location=device)
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


def _test_save_model_optimizer_lr_scheduler_with_validation(device, dirname, on_zero_rank=False):
    torch.manual_seed(23)

    def _build_objects(acc_list):

        model = DummyModel().to(device)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)

        def update_fn(engine, batch):
            x = torch.rand((4, 1)).to(device)
            optim.zero_grad()
            y = model(x)
            loss = y.pow(2.0).sum()
            loss.backward()
            if idist.has_xla_support:
                import torch_xla.core.xla_model as xm

                xm.optimizer_step(optim, barrier=True)
            else:
                optim.step()
            lr_scheduler.step()

        trainer = Engine(update_fn)

        evaluator = Engine(lambda e, b: None)
        acc_iter = iter(acc_list)

        @evaluator.on(Events.EPOCH_COMPLETED)
        def setup_result():
            evaluator.state.metrics["accuracy"] = next(acc_iter)

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_eval():
            evaluator.run([0, 1, 2])

        def score_function(engine):
            return engine.state.metrics["accuracy"]

        save_handler = DiskSaver(dirname, create_dir=True, require_empty=False)
        early_stop = EarlyStopping(score_function=score_function, patience=2, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stop)

        checkpointer = Checkpoint(
            {
                "trainer": trainer,
                "model": model,
                "optim": optim,
                "lr_scheduler": lr_scheduler,
                "early_stop": early_stop,
            },
            save_handler,
            include_self=True,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpointer)

        return trainer, evaluator, model, optim, lr_scheduler, early_stop, checkpointer

    trainer, evaluator, model, optim, scheduler, early, checkpointer = _build_objects([0.2, 0.3, 0.2])
    trainer.run([0, 1, 2], max_epochs=3)

    saved_objects = sorted(os.listdir(dirname))
    saved_checkpoint = os.path.join(dirname, saved_objects[0])

    loaded_obj = torch.load(saved_checkpoint, map_location=device)
    for f in ["trainer", "model", "optim", "lr_scheduler", "early_stop", "checkpointer"]:
        assert f in loaded_obj

    trainer2, evaluator2, model2, optim2, scheduler2, early2, checkpointer2 = _build_objects([0.1, 0.1, 0.1])
    Checkpoint.load_objects(
        {
            "trainer": trainer2,
            "model": model2,
            "optim": optim2,
            "lr_scheduler": scheduler2,
            "early_stop": early2,
            "checkpointer": checkpointer2,
        },
        loaded_obj,
    )
    assert checkpointer2.last_checkpoint == checkpointer.last_checkpoint

    model_state_dict = model.cpu().state_dict()
    loaded_model_state_dict = model2.cpu().state_dict()
    for key in model_state_dict.keys():
        assert key in loaded_model_state_dict
        model_value = model_state_dict[key]
        loaded_model_value = loaded_model_state_dict[key]
        assert model_value.cpu().numpy() == loaded_model_value.cpu().numpy()

    optim_state_dict = optim.state_dict()
    loaded_optimizer_state_dict = optim2.state_dict()
    # "params" contains tensor IDs, which are different
    del optim_state_dict["param_groups"][0]["params"]
    del loaded_optimizer_state_dict["param_groups"][0]["params"]
    for key in optim_state_dict.keys():
        assert key in loaded_optimizer_state_dict
        optim_value = optim_state_dict[key]
        loaded_optim_value = loaded_optimizer_state_dict[key]
        if idist.get_rank() == 0:
            assert optim_value == loaded_optim_value

    def _check_state_dict(original, loaded):
        original_state_dict = original.state_dict()
        loaded_state_dict = loaded.state_dict()
        for key in original_state_dict.keys():
            assert key in loaded_state_dict
            original_value = original_state_dict[key]
            loaded_value = loaded_state_dict[key]
            assert original_value == loaded_value

    _check_state_dict(trainer, trainer2)
    _check_state_dict(scheduler, scheduler2)
    _check_state_dict(early, early2)
    _check_state_dict(checkpointer, checkpointer2)

    trainer2.run([0, 1, 2], max_epochs=6)

    # early stopping should have triggered
    assert trainer2.state.epoch == 4

    # If Checkpoint's state was restored correctly, it should continue to respect n_saved
    # and delete old checkpoints, and have the correct last_checkpoint.
    assert os.listdir(dirname) == ["checkpoint_4.pt"]
    assert checkpointer2.last_checkpoint == "checkpoint_4.pt"


def test_save_model_optimizer_lr_scheduler_with_validation(dirname):
    _test_save_model_optimizer_lr_scheduler_with_validation("cpu", dirname)


def test_checkpoint_load_objects():

    with pytest.raises(TypeError, match=r"Argument checkpoint should be a string or a dictionary"):
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

    # case: load from filepath
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=1)
    to_save = _get_multiple_objs_to_save()
    handler(trainer, to_save)
    fname = handler.last_checkpoint
    assert isinstance(fname, str)
    assert os.path.join(dirname, _PREFIX) in fname
    assert os.path.exists(fname)
    Checkpoint.load_objects(to_save, fname)
    os.remove(fname)

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
    os.remove(fname)


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
        previous_fname = os.path.join(dirname, f"{_PREFIX}_obj_{1}{ext}")
        with open(previous_fname, "w") as f:
            f.write("test")

        with pytest.raises(ValueError, match=r"with extension '.pt' are already present"):
            DiskSaver(dirname, require_empty=True)

    _test(".pt")


def _test_checkpoint_with_ddp(device):
    torch.manual_seed(0)

    model = DummyModel().to(device)
    device_ids = (
        None if "cpu" in device.type else [device,]
    )
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    to_save = {"model": ddp_model}

    save_handler = MagicMock(spec=BaseSaveHandler)
    checkpointer = Checkpoint(to_save, save_handler=save_handler)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)

    checkpointer(trainer)
    assert save_handler.call_count == 1
    metadata = {"basename": "model", "score_name": None, "priority": 0}
    save_handler.assert_called_with(model.state_dict(), "model_0.pt", metadata)


def _test_checkpoint_load_objects_ddp(device):
    model = DummyModel().to(device)
    device_ids = (
        None if "cpu" in device.type else [device,]
    )
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    # single object:
    to_load = {"model": ddp_model}
    checkpoint = ddp_model.module.state_dict()
    Checkpoint.load_objects(to_load, checkpoint)

    # multiple objects:
    to_load = {"model": ddp_model, "opt": opt}
    checkpoint = {"model": ddp_model.module.state_dict(), "opt": opt.state_dict()}
    Checkpoint.load_objects(to_load, checkpoint)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo, get_rank_zero_dirname):

    device = idist.device()
    dirname = get_rank_zero_dirname()
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, os.path.join(dirname, "1"))
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, os.path.join(dirname, "2"), on_zero_rank=True)
    _test_checkpoint_with_ddp(device)
    _test_checkpoint_load_objects_ddp(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl, get_rank_zero_dirname):

    device = idist.device()
    dirname = get_rank_zero_dirname()
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, os.path.join(dirname, "1"))
    _test_save_model_optimizer_lr_scheduler_with_state_dict("cpu", os.path.join(dirname, "2"), on_zero_rank=True)
    _test_checkpoint_with_ddp(device=device)
    _test_checkpoint_load_objects_ddp(device=device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor, get_rank_zero_dirname):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    dirname = get_rank_zero_dirname()

    gloo_hvd_executor(
        _test_save_model_optimizer_lr_scheduler_with_state_dict,
        (device, os.path.join(dirname, "1")),
        np=nproc,
        do_init=True,
    )
    gloo_hvd_executor(
        _test_save_model_optimizer_lr_scheduler_with_state_dict,
        ("cpu", os.path.join(dirname, "2"), True),
        np=nproc,
        do_init=True,
    )


def _test_tpu_saves_to_cpu(device, dirname):
    torch.manual_seed(0)

    h = ModelCheckpoint(dirname, _PREFIX)
    engine = Engine(lambda e, b: None)
    engine.state = State(epoch=0, iteration=1)

    model = DummyModel().to(device)
    to_save = {"model": model}

    h(engine, to_save)

    idist.barrier()

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
    device = idist.device()
    _test_tpu_saves_to_cpu(device, os.path.join(dirname, "1"))
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, os.path.join(dirname, "2"))

    import time

    # hack to have all proc properly sync:
    time.sleep(1)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla_nprocs(xmp_executor, dirname):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_tpu_saves_to_cpu_nprocs, args=(dirname,), nprocs=n)


def test_checkpoint_filename_pattern():
    def _test(
        to_save,
        filename_prefix="",
        score_function=None,
        score_name=None,
        global_step_transform=None,
        filename_pattern=None,
    ):
        save_handler = MagicMock(spec=BaseSaveHandler)

        checkpointer = Checkpoint(
            to_save,
            save_handler=save_handler,
            filename_prefix=filename_prefix,
            score_function=score_function,
            score_name=score_name,
            global_step_transform=global_step_transform,
            filename_pattern=filename_pattern,
        )

        trainer = Engine(lambda e, b: None)
        trainer.state = State(epoch=12, iteration=203, score=0.9999)

        checkpointer(trainer)
        return checkpointer.last_checkpoint

    model = DummyModel()
    to_save = {"model": model}

    assert _test(to_save) == "model_203.pt"
    assert _test(to_save, "best") == "best_model_203.pt"
    assert _test(to_save, score_function=lambda e: e.state.score) == "model_0.9999.pt"

    res = _test(to_save, score_function=lambda e: e.state.score, global_step_transform=lambda e, _: e.state.epoch)
    assert res == "model_12_0.9999.pt"

    assert _test(to_save, score_function=lambda e: e.state.score, score_name="acc") == "model_acc=0.9999.pt"

    res = _test(
        to_save,
        score_function=lambda e: e.state.score,
        score_name="acc",
        global_step_transform=lambda e, _: e.state.epoch,
    )
    assert res == "model_12_acc=0.9999.pt"

    assert _test(to_save, "best", score_function=lambda e: e.state.score) == "best_model_0.9999.pt"

    res = _test(
        to_save, "best", score_function=lambda e: e.state.score, global_step_transform=lambda e, _: e.state.epoch
    )
    assert res == "best_model_12_0.9999.pt"

    res = _test(to_save, "best", score_function=lambda e: e.state.score, score_name="acc")
    assert res == "best_model_acc=0.9999.pt"

    res = _test(
        to_save,
        "best",
        score_function=lambda e: e.state.score,
        score_name="acc",
        global_step_transform=lambda e, _: e.state.epoch,
    )
    assert res == "best_model_12_acc=0.9999.pt"

    pattern = "{name}.{ext}"
    assert _test(to_save, filename_pattern=pattern) == "model.pt"

    pattern = "chk-{name}--{global_step}.{ext}"
    assert _test(to_save, to_save, filename_pattern=pattern) == "chk-model--203.pt"
    pattern = "chk-{filename_prefix}--{name}--{global_step}.{ext}"
    assert _test(to_save, "best", filename_pattern=pattern) == "chk-best--model--203.pt"
    pattern = "chk-{name}--{score}.{ext}"
    assert _test(to_save, score_function=lambda e: e.state.score, filename_pattern=pattern) == "chk-model--0.9999.pt"
    pattern = "{global_step}-{name}-{score}.chk.{ext}"
    res = _test(
        to_save,
        score_function=lambda e: e.state.score,
        global_step_transform=lambda e, _: e.state.epoch,
        filename_pattern=pattern,
    )
    assert res == "12-model-0.9999.chk.pt"

    pattern = "chk-{name}--{score_name}--{score}.{ext}"
    res = _test(to_save, score_function=lambda e: e.state.score, score_name="acc", filename_pattern=pattern)
    assert res == "chk-model--acc--0.9999.pt"

    pattern = "chk-{name}-{global_step}-{score_name}-{score}.{ext}"
    res = _test(
        to_save,
        score_function=lambda e: e.state.score,
        score_name="acc",
        global_step_transform=lambda e, _: e.state.epoch,
        filename_pattern=pattern,
    )
    assert res == "chk-model-12-acc-0.9999.pt"

    pattern = "{filename_prefix}-{name}-{score}.chk"
    res = _test(to_save, "best", score_function=lambda e: e.state.score, filename_pattern=pattern)
    assert res == "best-model-0.9999.chk"

    pattern = "resnet-{filename_prefix}-{name}-{global_step}-{score}.chk"
    res = _test(
        to_save,
        "best",
        score_function=lambda e: e.state.score,
        global_step_transform=lambda e, _: e.state.epoch,
        filename_pattern=pattern,
    )
    assert res == "resnet-best-model-12-0.9999.chk"

    pattern = "{filename_prefix}-{name}-{score_name}-{score}.chk"
    res = _test(to_save, "best", score_function=lambda e: e.state.score, score_name="acc", filename_pattern=pattern)
    assert res == "best-model-acc-0.9999.chk"

    pattern = "{global_step}-{filename_prefix}-{name}-{score_name}-{score}"
    res = _test(
        to_save,
        "best",
        score_function=lambda e: e.state.score,
        score_name="acc",
        global_step_transform=lambda e, _: e.state.epoch,
        filename_pattern=pattern,
    )
    assert res == "12-best-model-acc-0.9999"

    pattern = "SAVE:{name}-{score_name}-{score}.pth"
    res = _test(
        to_save,
        "best",
        score_function=lambda e: e.state.score,
        score_name="acc",
        global_step_transform=lambda e, _: e.state.epoch,
        filename_pattern=pattern,
    )

    assert res == "SAVE:model-acc-0.9999.pth"

    pattern = "{global_step}-chk-{filename_prefix}-{name}-{score_name}-{score}.{ext}"
    assert _test(to_save, filename_pattern=pattern) == "203-chk--model-None-None.pt"

    with pytest.raises(KeyError, match=r"random_key"):
        pattern = "SAVE:{random_key}.{ext}"
        _test(to_save, filename_pattern=pattern)


def test_setup_filename_pattern():
    # default filename pattern
    assert Checkpoint.setup_filename_pattern() == "{filename_prefix}_{name}_{global_step}_{score_name}={score}.{ext}"

    assert Checkpoint.setup_filename_pattern(False) == "{name}_{global_step}_{score_name}={score}.{ext}"
    assert Checkpoint.setup_filename_pattern(False, False, False) == "{name}_{global_step}.{ext}"
    assert Checkpoint.setup_filename_pattern(False, True, False) == "{name}_{global_step}_{score}.{ext}"
    assert Checkpoint.setup_filename_pattern(False, True, False, False) == "{name}_{score}.{ext}"
    assert Checkpoint.setup_filename_pattern(False, True, True, False) == "{name}_{score_name}={score}.{ext}"

    with pytest.raises(ValueError, match=r"At least one of with_score and with_global_step should be True."):
        Checkpoint.setup_filename_pattern(False, False, False, False)

    with pytest.raises(ValueError, match=r"If with_score_name is True, with_score should be also True"):
        Checkpoint.setup_filename_pattern(True, False, True, True)


def _setup_checkpoint():
    save_handler = MagicMock(spec=BaseSaveHandler)
    model = DummyModel()
    to_save = {"model": model}

    checkpointer = Checkpoint(to_save, save_handler=save_handler, n_saved=None)
    assert checkpointer.last_checkpoint is None

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)

    checkpointer(trainer)
    trainer.state.iteration = 10
    checkpointer(trainer)
    trainer.state.iteration = 20
    checkpointer(trainer)
    assert save_handler.call_count == 3
    return checkpointer


def test_checkpoint_state_dict():
    checkpointer = _setup_checkpoint()
    sd = checkpointer.state_dict()
    assert "saved" in sd
    assert isinstance(sd["saved"], list) and len(sd["saved"]) == len(checkpointer._saved)

    for saved_item, true_item in zip(sd["saved"], checkpointer._saved):
        assert saved_item[0] == true_item.priority
        assert saved_item[1] == true_item.filename


def test_checkpoint_load_state_dict():
    true_checkpointer = _setup_checkpoint()

    save_handler = MagicMock(spec=BaseSaveHandler)
    model = DummyModel()
    to_save = {"model": model}
    checkpointer = Checkpoint(to_save, save_handler=save_handler, n_saved=None)

    sd = {"saved": [(0, "model_0.pt"), (10, "model_10.pt"), (20, "model_20.pt")]}
    checkpointer.load_state_dict(sd)
    assert checkpointer._saved == true_checkpointer._saved


def test_checkpoint_fixed_filename():
    model = DummyModel()
    to_save = {"model": model}

    def _test(n_saved):
        save_handler = MagicMock(spec=BaseSaveHandler)
        checkpointer = Checkpoint(to_save, save_handler=save_handler, n_saved=n_saved, filename_pattern="{name}.{ext}")

        trainer = Engine(lambda e, b: None)

        for i in range(10):
            trainer.state = State(epoch=i, iteration=i)
            checkpointer(trainer)
            assert save_handler.call_count == i + 1
            metadata = {"basename": "model", "score_name": None, "priority": i}
            save_handler.assert_called_with(model.state_dict(), "model.pt", metadata)

    _test(None)
    _test(1)
    _test(3)


def test_checkpoint_reset():
    model = DummyModel()
    to_save = {"model": model}

    save_handler = MagicMock(spec=BaseSaveHandler)

    checkpointer = Checkpoint(to_save, save_handler=save_handler, n_saved=2)
    assert checkpointer.last_checkpoint is None

    trainer = Engine(lambda e, b: None)

    trainer.state = State(epoch=0, iteration=123)
    checkpointer(trainer)
    trainer.state.iteration = 234
    checkpointer(trainer)

    assert save_handler.call_count == 2
    assert checkpointer.last_checkpoint == "model_234.pt"
    assert len(checkpointer._saved) == 2
    assert sorted([item.filename for item in checkpointer._saved]) == sorted(["model_123.pt", "model_234.pt"])

    checkpointer.reset()
    assert len(checkpointer._saved) == 0

    trainer.state.iteration = 124
    checkpointer(trainer)

    assert save_handler.call_count == 3
    assert checkpointer.last_checkpoint == "model_124.pt"
    assert len(checkpointer._saved) == 1
    assert sorted([item.filename for item in checkpointer._saved]) == sorted(["model_124.pt",])


def test_checkpoint_reset_with_engine(dirname):
    name = "model"
    engine = Engine(lambda e, b: None)
    handler = ModelCheckpoint(dirname, _PREFIX, create_dir=False, n_saved=2)

    model = DummyModel()
    to_save = {"model": model}
    engine.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    engine.run([0, 1,], max_epochs=10)

    expected = sorted([f"{_PREFIX}_{name}_{i}.pt" for i in [9 * 2, 10 * 2]])
    assert sorted(os.listdir(dirname)) == expected
    assert "PREFIX_model_20.pt" in handler.last_checkpoint

    handler.reset()
    engine.state.max_epochs = None
    engine.run([0, 1,], max_epochs=2)

    expected += [f"{_PREFIX}_{name}_{i}.pt" for i in [1 * 2, 2 * 2]]
    assert sorted(os.listdir(dirname)) == sorted(expected)
    assert "PREFIX_model_4.pt" in handler.last_checkpoint


def test_greater_or_equal():
    scores = iter([1, 2, 2, 2])

    def score_function(_):
        return next(scores)

    class Saver:
        def __init__(self):
            self.counter = 0

        def __call__(self, c, f, m):
            if self.counter == 0:
                assert f == "model_1.pt"
            else:
                assert f == "model_2.pt"
            self.counter += 1

    handler = Saver()

    checkpointer = Checkpoint(
        to_save={"model": DummyModel()},
        save_handler=handler,
        score_function=score_function,
        n_saved=2,
        greater_or_equal=True,
    )
    trainer = Engine(lambda e, b: None)

    for _ in range(4):
        checkpointer(trainer)
    assert handler.counter == 4


def test_get_default_score_fn():

    with pytest.raises(ValueError, match=r"Argument score_sign should be 1 or -1"):
        Checkpoint.get_default_score_fn("acc", 2.0)

    engine = Engine(lambda e, b: None)
    engine.state.metrics["acc"] = 0.9
    engine.state.metrics["loss"] = 0.123

    score_fn = Checkpoint.get_default_score_fn("acc")
    score = score_fn(engine)
    assert score == 0.9

    score_fn = Checkpoint.get_default_score_fn("loss", -1)
    score = score_fn(engine)
    assert score == -0.123

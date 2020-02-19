import copy

import matplotlib

matplotlib.use("agg")

import torch
from torch import nn
from ignite.contrib.handlers import FastaiLRFinder
from torch.optim import SGD
from ignite.engine import create_supervised_trainer, Events

import pytest


@pytest.fixture
def no_site_packages():
    import sys

    matplotlib = sys.modules["matplotlib"]
    del sys.modules["matplotlib"]
    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    sys.modules["matplotlib"] = matplotlib


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
def lr_finder():
    yield FastaiLRFinder()


@pytest.fixture()
def dummy_engine(model, optimizer):
    engine = create_supervised_trainer(model, optimizer, nn.MSELoss())
    yield engine


@pytest.fixture()
def dataloader():
    yield torch.rand(100, 2, 1)


def test_attach_incorrect_input_args(lr_finder, dummy_engine, model, optimizer):

    with pytest.raises(TypeError, match=r"Argument to_save should be a mapping"):
        with lr_finder.attach(dummy_engine, to_save=123) as f:
            pass

    with pytest.raises(TypeError, match=r"Object <class 'int'> should have `state_dict` method"):
        with lr_finder.attach(dummy_engine, to_save={1: 2}) as f:
            pass

    with pytest.raises(ValueError, match=r"Mapping to_save should contain 'optimizer' key"):
        with lr_finder.attach(dummy_engine, to_save={"model": model}) as f:
            pass

    to_save = {"model": model, "optimizer": optimizer}
    with pytest.raises(ValueError, match=r"smooth_f is outside the range \[0, 1\]"):
        with lr_finder.attach(dummy_engine, to_save=to_save, smooth_f=234) as f:
            pass

    with pytest.raises(ValueError, match=r"diverge_th should be larger than 1"):
        with lr_finder.attach(dummy_engine, to_save=to_save, diverge_th=0.0) as f:
            pass

    with pytest.raises(ValueError, match=r"if provided, num_iter should be a positive integer"):
        with lr_finder.attach(dummy_engine, to_save=to_save, num_iter=0.0) as f:
            pass


def test_attach_without_with(lr_finder, dummy_engine, model, optimizer):
    to_save = {"model": model, "optimizer": optimizer}
    _ = lr_finder.attach(dummy_engine, to_save=to_save)
    for event in dummy_engine._event_handlers:
        assert len(dummy_engine._event_handlers[event]) == 0

    with lr_finder.attach(dummy_engine, to_save=to_save) as _:
        assert any([len(dummy_engine._event_handlers[event]) != 0 for event in dummy_engine._event_handlers])
        with pytest.raises(RuntimeError):
            lr_finder.lr_suggestion()
        with pytest.raises(RuntimeError):
            lr_finder.plot()


def test_with_attach(lr_finder, model, optimizer, dummy_engine, dataloader):
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(dummy_engine, to_save=to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    assert lr_finder.get_results() is not None

    for event in dummy_engine._event_handlers:
        assert len(dummy_engine._event_handlers[event]) == 0


def test_model_optimizer_reset(lr_finder, model, optimizer, dummy_engine, dataloader):
    to_save = {"model": model, "optimizer": optimizer}

    init_optimizer = copy.deepcopy(optimizer.state_dict())
    init_model = copy.deepcopy(model.state_dict())

    with lr_finder.attach(dummy_engine, to_save=to_save, diverge_th=1e10) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    assert init_optimizer["param_groups"][0]["params"] == optimizer.param_groups[0]["params"]

    for v1, v2 in zip(init_model.values(), model.parameters()):
        assert all(v1 == v2)


def test_lr_policy(lr_finder, model, optimizer, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, model, optimizer, step_mode="linear") as trainer_with_finder:
        trainer_with_finder.run(dataloader)
    lr = lr_finder.get_results()["lr"]
    assert all([lr[i - 1] < lr[i] for i in range(1, len(lr))])

    with lr_finder.attach(dummy_engine, model, optimizer, step_mode="exp") as trainer_with_finder:
        trainer_with_finder.run(dataloader)
    lr = lr_finder.get_results()["lr"]
    assert all([lr[i - 1] < lr[i] for i in range(1, len(lr))])


def assert_output_sizes(lr_finder, dummy_engine):
    iteration = dummy_engine.state.iteration
    lr_finder_results = lr_finder.get_results()
    lr, loss = lr_finder_results["lr"], lr_finder_results["loss"]
    assert len(lr) == len(loss) == iteration


def test_num_iter_is_none(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder()
    with lr_finder.attach(dummy_engine, model, optimizer, diverge_th=np.inf) as trainer_with_finder:
        trainer_with_finder.run(dataloader)
    assert_output_sizes(lr_finder, dummy_engine)
    assert dummy_engine.state.iteration == len(dataloader)


def test_num_iter_is_enough(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder()
    with lr_finder.attach(dummy_engine, model, optimizer, num_iter=50, diverge_th=np.inf) as trainer_with_finder:
        trainer_with_finder.run(dataloader)
    assert_output_sizes(lr_finder, dummy_engine)
    # -1 because it terminates when state.iteration > num_iter
    assert dummy_engine.state.iteration - 1 == 50


def test_num_iter_is_not_enough(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder()
    with lr_finder.attach(dummy_engine, model, optimizer, num_iter=150, diverge_th=np.inf) as trainer_with_finder:
        with pytest.warns(UserWarning):
            trainer_with_finder.run(dataloader)
    assert_output_sizes(lr_finder, dummy_engine)
    assert dummy_engine.state.iteration == len(dataloader)


def test_detach_terminates(model, optimizer, dummy_engine, dataloader):
    lr_finder = FastaiLRFinder()
    with lr_finder.attach(dummy_engine, model, optimizer, end_lr=100, diverge_th=2) as trainer_with_finder:
        with pytest.warns(None) as record:
            trainer_with_finder.run(dataloader)
            assert len(record) == 0

    dummy_engine.run(dataloader, max_epochs=3)
    assert dummy_engine.state.epoch == 3


def test_lr_suggestion(lr_finder, model, optimizer, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, model, optimizer) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    assert 1e-4 <= lr_finder.lr_suggestion() <= 10


def test_bad_input(lr_finder, model, optimizer, dummy_engine, dataloader):

    with pytest.raises(ValueError):
        with lr_finder.attach(dummy_engine, model, optimizer, step_mode="bla") as trainer_with_finder:
            pass

    with pytest.raises(ValueError):
        with lr_finder.attach(dummy_engine, model, optimizer, diverge_th=0.5) as trainer_with_finder:
            pass

    with pytest.raises(ValueError):
        with lr_finder.attach(dummy_engine, model, optimizer, smooth_f=-0.5) as trainer_with_finder:
            pass

    with pytest.raises(ValueError):
        with lr_finder.attach(dummy_engine, model, optimizer, smooth_f=1.5) as trainer_with_finder:
            pass

    with pytest.raises(ValueError):
        with lr_finder.attach(dummy_engine, model, optimizer, num_iter=1.5) as trainer_with_finder:
            pass

    with pytest.raises(ValueError):
        with lr_finder.attach(dummy_engine, model, optimizer, num_iter=0) as trainer_with_finder:
            pass

    with pytest.raises(ValueError):
        st = _StateCacher(False, "not a dir")

    with pytest.raises(KeyError):
        st = _StateCacher(True)
        st.retrieve("nothing")

    with pytest.raises(RuntimeError):
        st = _StateCacher(False)
        key = "something"
        st.store(key, 5)
        path = st.cached[key]
        os.remove(path)
        st.retrieve(key)

    with lr_finder.attach(dummy_engine, model, optimizer) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    with pytest.raises(ValueError):
        lr_finder.plot(skip_start=-1)
    with pytest.raises(ValueError):
        lr_finder.plot(skip_end=-1)


def test_plot(lr_finder, model, optimizer, dummy_engine, dataloader):

    with lr_finder.attach(dummy_engine, model, optimizer) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    lr_finder.plot()
    lr_finder.plot(skip_end=0)


def test_no_matplotlib(no_site_packages):

    with pytest.raises(RuntimeError):
        FastaiLRFinder()

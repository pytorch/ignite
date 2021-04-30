import copy
from os import path

import matplotlib
import pytest
import torch
from torch import nn
from torch.optim import SGD

from ignite.contrib.handlers import FastaiLRFinder
from ignite.engine import Events, create_supervised_trainer

matplotlib.use("agg")


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
    def __init__(self, n_channels=10, out_channels=1, flatten_input=False):
        super(DummyModel, self).__init__()

        self.net = nn.Sequential(nn.Flatten() if flatten_input else nn.Identity(), nn.Linear(n_channels, out_channels))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model():
    model = DummyModel()
    yield model


@pytest.fixture
def mnist_model():
    model = DummyModel(n_channels=784, out_channels=10, flatten_input=True)
    yield model


@pytest.fixture
def optimizer(model):
    yield SGD(model.parameters(), lr=1e-4, momentum=0.0)


@pytest.fixture
def mnist_optimizer(mnist_model):
    yield SGD(mnist_model.parameters(), lr=1e-4, momentum=0.0)


@pytest.fixture
def to_save(model, optimizer):
    yield {"model": model, "optimizer": optimizer}


@pytest.fixture
def lr_finder():
    yield FastaiLRFinder()


@pytest.fixture
def dummy_engine(model, optimizer):
    engine = create_supervised_trainer(model, optimizer, nn.MSELoss())
    yield engine


@pytest.fixture
def dataloader():
    yield torch.rand(100, 2, 10)


@pytest.fixture
def mnist_dataloader():
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root="/tmp", transform=data_transform, train=True), batch_size=64, shuffle=True
    )

    yield train_loader


def test_attach_incorrect_input_args(lr_finder, dummy_engine, model, optimizer, dataloader):

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

    with pytest.raises(TypeError, match=r"if provided, num_iter should be an integer"):
        with lr_finder.attach(dummy_engine, to_save=to_save, num_iter=0.0) as f:
            pass

    with pytest.raises(ValueError, match=r"if provided, num_iter should be positive"):
        with lr_finder.attach(dummy_engine, to_save=to_save, num_iter=0) as f:
            pass

    with pytest.raises(TypeError, match=r"Object to_save\['optimizer'] should be torch optimizer"):
        with lr_finder.attach(dummy_engine, {"model": to_save["model"], "optimizer": to_save["model"]}):
            pass

    with pytest.raises(ValueError, match=r"step_mode should be 'exp' or 'linear'"):
        with lr_finder.attach(dummy_engine, to_save=to_save, step_mode="abc") as f:
            pass

    with lr_finder.attach(dummy_engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    with pytest.raises(ValueError, match=r"skip_start cannot be negative"):
        lr_finder.plot(skip_start=-1)
    with pytest.raises(ValueError, match=r"skip_end cannot be negative"):
        lr_finder.plot(skip_end=-1)


def test_attach_without_with(lr_finder, dummy_engine, to_save):
    _ = lr_finder.attach(dummy_engine, to_save=to_save)
    for event in dummy_engine._event_handlers:
        assert len(dummy_engine._event_handlers[event]) == 0

    with lr_finder.attach(dummy_engine, to_save=to_save) as _:
        assert any([len(dummy_engine._event_handlers[event]) != 0 for event in dummy_engine._event_handlers])

        with pytest.raises(
            RuntimeError, match=r"learning rate finder didn't run yet so lr_suggestion can't be returned"
        ):
            lr_finder.lr_suggestion()
        with pytest.raises(RuntimeError, match=r"learning rate finder didn't run yet so results can't be plotted"):
            lr_finder.plot()


def test_with_attach(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save=to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    assert lr_finder.get_results() is not None

    for event in dummy_engine._event_handlers:
        assert len(dummy_engine._event_handlers[event]) == 0


def test_model_optimizer_reset(lr_finder, to_save, dummy_engine, dataloader):
    optimizer = to_save["optimizer"]
    model = to_save["model"]

    init_optimizer_sd = copy.deepcopy(optimizer.state_dict())
    init_model_sd = copy.deepcopy(model.state_dict())
    init_trainer_sd = copy.deepcopy(dummy_engine.state_dict())

    with pytest.warns(UserWarning, match=r"Run completed without loss diverging"):
        with lr_finder.attach(dummy_engine, to_save=to_save, diverge_th=float("inf")) as trainer_with_finder:
            trainer_with_finder.run(dataloader)

    assert init_optimizer_sd == optimizer.state_dict()
    for tensor1, tensor2 in zip(init_model_sd.values(), model.state_dict().values()):
        assert torch.all(torch.eq(tensor1, tensor2))
    assert init_trainer_sd == dummy_engine.state_dict()


def test_lr_policy(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save=to_save, step_mode="linear") as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    lr = lr_finder.get_results()["lr"]
    assert all([lr[i - 1] < lr[i] for i in range(1, len(lr))])

    with lr_finder.attach(dummy_engine, to_save=to_save, step_mode="exp") as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    lr = lr_finder.get_results()["lr"]
    assert all([lr[i - 1] < lr[i] for i in range(1, len(lr))])


def assert_output_sizes(lr_finder, dummy_engine):
    iteration = dummy_engine.state.iteration
    lr_finder_results = lr_finder.get_results()
    lr, loss = lr_finder_results["lr"], lr_finder_results["loss"]
    assert len(lr) == len(loss) == iteration


def test_num_iter_is_none(lr_finder, to_save, dummy_engine, dataloader):

    with pytest.warns(UserWarning, match=r"Run completed without loss diverging"):
        with lr_finder.attach(dummy_engine, to_save=to_save, diverge_th=float("inf")) as trainer_with_finder:
            trainer_with_finder.run(dataloader)
            assert_output_sizes(lr_finder, dummy_engine)
            assert dummy_engine.state.iteration == len(dataloader)


def test_num_iter_is_enough(lr_finder, to_save, dummy_engine, dataloader):

    with pytest.warns(UserWarning, match=r"Run completed without loss diverging"):
        with lr_finder.attach(
            dummy_engine, to_save=to_save, num_iter=50, diverge_th=float("inf")
        ) as trainer_with_finder:
            trainer_with_finder.run(dataloader)
            assert_output_sizes(lr_finder, dummy_engine)
            # -1 because it terminates when state.iteration > num_iter
            assert dummy_engine.state.iteration - 1 == 50


def test_num_iter_is_not_enough(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save, num_iter=150, diverge_th=float("inf")) as trainer_with_finder:
        with pytest.warns(UserWarning):
            trainer_with_finder.run(dataloader)
        assert_output_sizes(lr_finder, dummy_engine)
        assert dummy_engine.state.iteration == len(dataloader)


def test_detach_terminates(lr_finder, to_save, dummy_engine, dataloader, recwarn):
    with lr_finder.attach(dummy_engine, to_save, end_lr=100, diverge_th=2) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    dummy_engine.run(dataloader, max_epochs=3)
    assert dummy_engine.state.epoch == 3
    assert len(recwarn) == 1


def test_lr_suggestion_unexpected_curve(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    lr_finder._history["loss"].insert(0, 0)
    with pytest.raises(
        RuntimeError, match=r"FastaiLRFinder got unexpected curve shape, the curve should be somehow U-shaped"
    ):
        lr_finder.lr_suggestion()


def test_lr_suggestion(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    # Avoid RuntimeError
    if torch.tensor(lr_finder._history["loss"]).argmin() < 2:
        lr_finder._history["loss"].insert(0, 10)
        lr_finder._history["loss"].insert(0, 100)
        lr_finder._history["lr"].insert(0, 10)
        lr_finder._history["lr"].insert(0, 100)

    assert 1e-4 <= lr_finder.lr_suggestion() <= 10


def test_plot(lr_finder, to_save, dummy_engine, dataloader):

    with lr_finder.attach(dummy_engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    # Avoid RuntimeError
    if torch.tensor(lr_finder._history["loss"]).argmin() < 2:
        lr_finder._history["loss"].insert(0, 10)
        lr_finder._history["loss"].insert(0, 100)
        lr_finder._history["lr"].insert(0, 10)
        lr_finder._history["lr"].insert(0, 100)

    lr_finder.plot()
    lr_finder.plot(skip_end=0)
    lr_finder.plot(skip_end=0, filepath="dummy.jpg")
    lr_finder.plot(
        skip_end=0, filepath="dummy.jpg", orientation="landscape", papertype="a4", format="png",
    )
    assert path.exists("dummy.jpg")
    lr_finder.plot(
        skip_end=0, filepath="/nonexisting/dummy.jpg", orientation="landscape", papertype="a4", format="png",
    )
    assert not path.exists("/nonexisting/dummy.jpg")


def test_no_matplotlib(no_site_packages, lr_finder):

    with pytest.raises(RuntimeError, match=r"This method requires matplotlib to be installed"):
        lr_finder.plot()


def test_mnist_lr_suggestion(lr_finder, mnist_model, mnist_optimizer, mnist_dataloader):
    criterion = nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(mnist_model, mnist_optimizer, criterion)
    to_save = {"model": mnist_model, "optimizer": mnist_optimizer}

    max_iters = 50

    with lr_finder.attach(trainer, to_save) as trainer_with_finder:

        with trainer_with_finder.add_event_handler(
            Events.ITERATION_COMPLETED(once=max_iters), lambda _: trainer_with_finder.terminate()
        ):
            trainer_with_finder.run(mnist_dataloader)

    assert 1e-4 <= lr_finder.lr_suggestion() <= 10

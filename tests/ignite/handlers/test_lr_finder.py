import copy
import os
from pathlib import Path
from unittest.mock import MagicMock

import filelock

import matplotlib
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

import ignite.distributed as idist
from ignite.engine import create_supervised_trainer, Engine, Events
from ignite.handlers import FastaiLRFinder

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


class DummyModelMulipleParamGroups(nn.Module):
    def __init__(self):
        super(DummyModelMulipleParamGroups, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@pytest.fixture
def model():
    model = DummyModel(out_channels=10)
    yield model


@pytest.fixture
def model_multiple_param_groups():
    model_multiple_param_groups = DummyModelMulipleParamGroups()
    yield model_multiple_param_groups


@pytest.fixture
def mnist_model():
    model = DummyModel(n_channels=784, out_channels=10, flatten_input=True)
    yield model


@pytest.fixture
def optimizer(model):
    yield SGD(model.parameters(), lr=1e-4, momentum=0.0)


@pytest.fixture
def optimizer_multiple_param_groups(model_multiple_param_groups):
    optimizer_multiple_param_groups = SGD(
        [
            {"params": model_multiple_param_groups.fc1.parameters(), "lr": 4e-1},
            {"params": model_multiple_param_groups.fc2.parameters(), "lr": 3e-2},
            {"params": model_multiple_param_groups.fc3.parameters(), "lr": 3e-3},
        ]
    )
    yield optimizer_multiple_param_groups


@pytest.fixture
def mnist_optimizer(mnist_model):
    yield SGD(mnist_model.parameters(), lr=1e-4, momentum=0.0)


@pytest.fixture
def to_save(model, optimizer):
    yield {"model": model, "optimizer": optimizer}


@pytest.fixture
def mnist_to_save(mnist_model, mnist_optimizer):
    yield {"model": mnist_model, "optimizer": mnist_optimizer}


@pytest.fixture
def to_save_mulitple_param_groups(model_multiple_param_groups, optimizer_multiple_param_groups):
    yield {"model": model_multiple_param_groups, "optimizer": optimizer_multiple_param_groups}


@pytest.fixture
def lr_finder():
    yield FastaiLRFinder()


@pytest.fixture
def dummy_engine(model, optimizer):
    engine = create_supervised_trainer(model, optimizer, nn.MSELoss())
    yield engine


@pytest.fixture
def dummy_engine_mnist(mnist_model, mnist_optimizer):
    mnist_engine = create_supervised_trainer(mnist_model, mnist_optimizer, nn.CrossEntropyLoss())
    yield mnist_engine


@pytest.fixture
def dummy_engine_mulitple_param_groups(model_multiple_param_groups, optimizer_multiple_param_groups):
    engine_multiple_param_groups = create_supervised_trainer(
        model_multiple_param_groups, optimizer_multiple_param_groups, nn.MSELoss()
    )
    yield engine_multiple_param_groups


@pytest.fixture
def dataloader():
    yield torch.rand(100, 2, 10)


@pytest.fixture
def dataloader_plot():
    yield torch.rand(500, 2, 10)


@pytest.fixture
def mnist_dataloader(tmp_path_factory):
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    while True:
        try:
            with filelock.FileLock(root_tmp_dir / "mnist_download.lock", timeout=0.2) as fn:
                fn.acquire()
                train_loader = DataLoader(
                    MNIST(download=True, root="/tmp", transform=data_transform, train=True),
                    batch_size=256,
                    shuffle=True,
                )
                fn.release()
                break
        except filelock._error.Timeout:
            pass

    yield train_loader


def test_attach_incorrect_input_args(lr_finder, dummy_engine, model, optimizer, dataloader):
    with pytest.raises(TypeError, match=r"Argument to_save should be a mapping"):
        with lr_finder.attach(dummy_engine, to_save=123):
            pass

    with pytest.raises(TypeError, match=r"Object <class 'int'> should have `state_dict` method"):
        with lr_finder.attach(dummy_engine, to_save={1: 2}):
            pass

    with pytest.raises(ValueError, match=r"Mapping to_save should contain 'optimizer' key"):
        with lr_finder.attach(dummy_engine, to_save={"model": model}):
            pass

    to_save = {"model": model, "optimizer": optimizer}
    with pytest.raises(ValueError, match=r"smooth_f is outside the range \[0, 1\]"):
        with lr_finder.attach(dummy_engine, to_save=to_save, smooth_f=234):
            pass

    with pytest.raises(ValueError, match=r"diverge_th should be larger than 1"):
        with lr_finder.attach(dummy_engine, to_save=to_save, diverge_th=0.0):
            pass

    with pytest.raises(TypeError, match=r"if provided, num_iter should be an integer"):
        with lr_finder.attach(dummy_engine, to_save=to_save, num_iter=0.0):
            pass

    with pytest.raises(ValueError, match=r"if provided, num_iter should be positive"):
        with lr_finder.attach(dummy_engine, to_save=to_save, num_iter=0):
            pass

    with pytest.raises(TypeError, match=r"Object to_save\['optimizer'] should be torch optimizer"):
        with lr_finder.attach(dummy_engine, {"model": to_save["model"], "optimizer": to_save["model"]}):
            pass

    with pytest.raises(ValueError, match=r"step_mode should be 'exp' or 'linear'"):
        with lr_finder.attach(dummy_engine, to_save=to_save, step_mode="abc"):
            pass

    with lr_finder.attach(dummy_engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    with pytest.raises(ValueError, match=r"skip_start cannot be negative"):
        lr_finder.plot(skip_start=-1)
    with pytest.raises(ValueError, match=r"skip_end cannot be negative"):
        lr_finder.plot(skip_end=-1)

    with pytest.raises(ValueError, match=r"Number of values of start_lr should be equal to optimizer values."):
        with lr_finder.attach(dummy_engine, to_save, start_lr=[0.1, 0.1]):
            pass
    with pytest.raises(ValueError, match=r"Number of values of end_lr should be equal to optimizer values."):
        with lr_finder.attach(dummy_engine, to_save, end_lr=[0.1, 0.1]):
            pass

    with pytest.raises(TypeError, match=r"start_lr should be a float or list of floats"):
        with lr_finder.attach(dummy_engine, to_save, start_lr=1):
            pass
    with pytest.raises(TypeError, match=r"end_lr should be a float or list of floats"):
        with lr_finder.attach(dummy_engine, to_save, end_lr=1):
            pass


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


def test_wrong_values_start_lr_and_end_lr(
    lr_finder, dummy_engine, to_save, dummy_engine_mulitple_param_groups, to_save_mulitple_param_groups
):
    with pytest.raises(ValueError, match=r"start_lr must be less than end_lr"):
        with lr_finder.attach(dummy_engine, to_save=to_save, start_lr=10.0, end_lr=1.0):
            pass

    with pytest.raises(ValueError, match=r"start_lr must be less than end_lr"):
        with lr_finder.attach(
            dummy_engine_mulitple_param_groups,
            to_save=to_save_mulitple_param_groups,
            start_lr=[1.0, 10.0, 5.0],
            end_lr=[10.0, 10.0, 10.0],
        ):
            pass


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


@pytest.mark.parametrize("step_mode", ["exp", "linear"])
def test_multiple_optimizers(
    lr_finder, dummy_engine_mulitple_param_groups, to_save_mulitple_param_groups, dataloader, step_mode
):
    start_lr = [0.1, 0.1, 0.01]
    end_lr = [1.0, 1.0, 1.0]
    with lr_finder.attach(
        dummy_engine_mulitple_param_groups,
        to_save_mulitple_param_groups,
        start_lr=start_lr,
        end_lr=end_lr,
        step_mode=step_mode,
    ) as trainer:
        trainer.run(dataloader)
    groups_lrs = lr_finder.get_results()["lr"]
    assert [all([group_lrs[i - 1] < group_lrs[i] for i in range(1, len(group_lrs))]) for group_lrs in groups_lrs]


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
        assert dummy_engine.state.iteration != len(dataloader)
        assert dummy_engine.state.iteration == 150


def test_detach_terminates(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save, end_lr=100.0, diverge_th=2) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    dummy_engine.run(dataloader, max_epochs=3)
    assert dummy_engine.state.epoch == 3


def test_different_num_iters(lr_finder, to_save, dummy_engine, dataloader):
    with pytest.warns(UserWarning, match=r"Run completed without loss diverging"):
        with lr_finder.attach(dummy_engine, to_save, num_iter=200, diverge_th=float("inf")) as trainer_with_finder:
            trainer_with_finder.run(dataloader)
            assert trainer_with_finder.state.iteration == 200  # num_iter

    with pytest.warns(UserWarning, match=r"Run completed without loss diverging"):
        with lr_finder.attach(dummy_engine, to_save, num_iter=1000, diverge_th=float("inf")) as trainer_with_finder:
            trainer_with_finder.run(dataloader)
            assert trainer_with_finder.state.iteration == 1000  # num_iter


@pytest.mark.parametrize("step_mode", ["exp", "linear"])
def test_start_lr(lr_finder, to_save, dummy_engine, dataloader, step_mode):
    with lr_finder.attach(
        dummy_engine, to_save, start_lr=0.01, end_lr=10.0, num_iter=5, step_mode=step_mode, diverge_th=1
    ) as trainer_with_finder:
        trainer_with_finder.run(dataloader)
    history = lr_finder.get_results()

    if step_mode == "exp":
        assert 0.01 < history["lr"][0] < 0.16
    else:
        assert pytest.approx(history["lr"][0]) == 0.01


def test_engine_output_type(lr_finder, dummy_engine, optimizer):
    from ignite.handlers.param_scheduler import PiecewiseLinear

    dummy_engine.state.iteration = 1
    dummy_engine.state.output = [10]
    with pytest.raises(TypeError, match=r"output of the engine should be of type float or 0d torch.Tensor"):
        lr_finder._log_lr_and_loss(dummy_engine, output_transform=lambda x: x, smooth_f=0, diverge_th=1)

    dummy_engine.state.output = (10, 5)
    with pytest.raises(TypeError, match=r"output of the engine should be of type float or 0d torch.Tensor"):
        lr_finder._log_lr_and_loss(dummy_engine, output_transform=lambda x: x, smooth_f=0, diverge_th=1)

    dummy_engine.state.output = torch.tensor([1, 2], dtype=torch.float32)
    with pytest.raises(ValueError, match=r"if output of the engine is torch.Tensor"):
        lr_finder._log_lr_and_loss(dummy_engine, output_transform=lambda x: x, smooth_f=0, diverge_th=1)

    lr_finder._lr_schedule = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=[(0, optimizer.param_groups[0]["lr"]), (100, 10)]
    )

    dummy_engine.state.output = torch.tensor(10.0, dtype=torch.float32)
    lr_finder._history = {"lr": [], "loss": []}
    lr_finder._log_lr_and_loss(dummy_engine, output_transform=lambda x: x, smooth_f=0, diverge_th=1)
    loss = lr_finder._history["loss"][-1]
    assert type(loss) is float

    dummy_engine.state.output = torch.tensor([10.0], dtype=torch.float32)
    lr_finder._history = {"lr": [], "loss": []}
    lr_finder._log_lr_and_loss(dummy_engine, output_transform=lambda x: x, smooth_f=0, diverge_th=1)
    loss = lr_finder._history["loss"][-1]
    assert type(loss) is float


def test_lr_suggestion_unexpected_curve(lr_finder, to_save, dummy_engine, dataloader):
    with lr_finder.attach(dummy_engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(dataloader)

    lr_finder._history["loss"].insert(0, 0)
    with pytest.raises(
        RuntimeError, match=r"FastaiLRFinder got unexpected curve shape, the curve should be somehow U-shaped"
    ):
        lr_finder.lr_suggestion()


def test_lr_suggestion_single_param_group(lr_finder):  # , to_save, dummy_engine, dataloader):
    import numpy as np

    noise = 0.05
    lr_finder._history["loss"] = np.linspace(-5.0, 5.0, num=100) ** 2 + noise
    lr_finder._history["lr"] = np.linspace(0.01, 10, num=100)

    # lr_finder.lr_suggestion() is supposed to return a value, but as
    # we assign loss and lr to tensors, instead of lists, it will return tensors
    suggested_lr = lr_finder.lr_suggestion()

    assert pytest.approx(suggested_lr.item()) == 0.110909089


def test_lr_suggestion_multiple_param_groups(lr_finder):
    import numpy as np

    noise = 0.06
    lr_finder._history["loss"] = np.linspace(-5.0, 5, num=50) ** 2 + noise
    # 2 param_groups
    lr_finder._history["lr"] = np.linspace(0.01, 10, num=100).reshape(50, 2)

    # lr_finder.lr_suggestion() is supposed to return a list of values,
    # but as we assign loss and lr to tensors, instead of lists, it will return tensors
    suggested_lrs = lr_finder.lr_suggestion()

    assert pytest.approx(suggested_lrs[0].item()) == 0.21181818
    assert pytest.approx(suggested_lrs[1].item()) == 0.31272727


def test_lr_suggestion_mnist(lr_finder, mnist_to_save, dummy_engine_mnist, mnist_dataloader):
    max_iters = 50

    with lr_finder.attach(dummy_engine_mnist, mnist_to_save, diverge_th=2, step_mode="linear") as trainer_with_finder:
        with trainer_with_finder.add_event_handler(
            Events.ITERATION_COMPLETED(once=max_iters), lambda _: trainer_with_finder.terminate()
        ):
            trainer_with_finder.run(mnist_dataloader)

    assert 1e-4 <= lr_finder.lr_suggestion() <= 2


def test_apply_suggested_lr_unmatched_optimizers(
    lr_finder, mnist_to_save, dummy_engine_mnist, optimizer_multiple_param_groups, mnist_dataloader
):
    with lr_finder.attach(dummy_engine_mnist, mnist_to_save) as trainer_with_finder:
        trainer_with_finder.run(mnist_dataloader)

    sug_lr = lr_finder.lr_suggestion()

    with pytest.raises(RuntimeError, match=r"The number of parameter groups does not match"):
        lr_finder.apply_suggested_lr(optimizer_multiple_param_groups)


def test_apply_suggested_lr_single_param_groups(
    lr_finder, mnist_to_save, dummy_engine_mnist, mnist_optimizer, mnist_dataloader
):
    with lr_finder.attach(dummy_engine_mnist, mnist_to_save) as trainer_with_finder:
        trainer_with_finder.run(mnist_dataloader)

    sug_lr = lr_finder.lr_suggestion()
    lr_finder.apply_suggested_lr(mnist_optimizer)

    assert mnist_optimizer.param_groups[0]["lr"] == sug_lr


def test_apply_suggested_lr_multiple_param_groups(
    lr_finder,
    to_save_mulitple_param_groups,
    dummy_engine_mulitple_param_groups,
    optimizer_multiple_param_groups,
    dataloader_plot,
):
    with lr_finder.attach(dummy_engine_mulitple_param_groups, to_save_mulitple_param_groups) as trainer_with_finder:
        trainer_with_finder.run(dataloader_plot)

    sug_lr = lr_finder.lr_suggestion()
    lr_finder.apply_suggested_lr(optimizer_multiple_param_groups)

    for i in range(len(sug_lr)):
        assert optimizer_multiple_param_groups.param_groups[i]["lr"] == sug_lr[i]


def test_no_matplotlib(no_site_packages, lr_finder):
    with pytest.raises(ModuleNotFoundError, match=r"This method requires matplotlib to be installed"):
        lr_finder.plot()


def test_plot_single_param_group(dirname, lr_finder, mnist_to_save, dummy_engine_mnist, mnist_dataloader):
    with lr_finder.attach(dummy_engine_mnist, mnist_to_save, end_lr=20.0, smooth_f=0.04) as trainer_with_finder:
        trainer_with_finder.run(mnist_dataloader)

    def _test(ax):
        assert ax is not None
        assert ax.get_xscale() == "log"
        assert ax.get_xlabel() == "Learning rate"
        assert ax.get_ylabel() == "Loss"
        filepath = Path(dirname) / "dummy.jpg"
        ax.figure.savefig(filepath)
        assert filepath.exists()
        filepath.unlink()

    lr_finder.plot()
    ax = lr_finder.plot(skip_end=0)
    _test(ax)

    # Passing axes object
    from matplotlib import pyplot as plt

    _, ax = plt.subplots()
    lr_finder.plot(skip_end=0, ax=ax)
    _test(ax)


def test_plot_multiple_param_groups(
    dirname, lr_finder, to_save_mulitple_param_groups, dummy_engine_mulitple_param_groups, dataloader_plot
):
    with lr_finder.attach(
        dummy_engine_mulitple_param_groups, to_save_mulitple_param_groups, end_lr=20.0, smooth_f=0.04
    ) as trainer_with_finder:
        trainer_with_finder.run(dataloader_plot)

    def _test(ax):
        assert ax is not None
        assert ax.get_xscale() == "log"
        assert ax.get_xlabel() == "Learning rate"
        assert ax.get_ylabel() == "Loss"
        filepath = Path(dirname) / "dummy_muliple_param_groups.jpg"
        ax.figure.savefig(filepath)
        assert filepath.exists()
        filepath.unlink()

    ax = lr_finder.plot(skip_start=0, skip_end=0)
    _test(ax)

    # Passing axes object
    from matplotlib import pyplot as plt

    _, ax = plt.subplots()
    lr_finder.plot(skip_start=0, skip_end=0, ax=ax)
    _test(ax)


def _test_distrib_log_lr_and_loss(device):
    from ignite.handlers import ParamScheduler

    lr_finder = FastaiLRFinder()
    _lr_schedule = MagicMock(spec=ParamScheduler)

    # minimal setup for lr_finder to make _log_lr_and_loss work
    rank = idist.get_rank()
    loss = 0.01 * (rank + 1)

    engine = Engine(lambda e, b: None)

    engine.state.output = loss
    engine.state.iteration = 1
    lr_finder._lr_schedule = _lr_schedule
    lr_finder._history["loss"] = []
    lr_finder._history["lr"] = []

    lr_finder._log_lr_and_loss(engine, output_transform=lambda x: x, smooth_f=0.1, diverge_th=10.0)

    expected_loss = idist.all_reduce(loss)
    assert pytest.approx(lr_finder._history["loss"][-1]) == expected_loss


def _test_distrib_integration_mnist(dirname, device):
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, Normalize, ToTensor

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root="/tmp", transform=data_transform, train=True), batch_size=256, shuffle=True
    )

    class DummyModel(nn.Module):
        def __init__(self, n_channels=10, out_channels=1, flatten_input=False):
            super(DummyModel, self).__init__()

            self.net = nn.Sequential(
                nn.Flatten() if flatten_input else nn.Identity(), nn.Linear(n_channels, out_channels)
            )

        def forward(self, x):
            return self.net(x)

    model = DummyModel(n_channels=784, out_channels=10, flatten_input=True)
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.0)
    to_save = {"model": model, "optimizer": optimizer}
    engine = create_supervised_trainer(model, optimizer, nn.CrossEntropyLoss(), device=device)
    lr_finder = FastaiLRFinder()
    with lr_finder.attach(engine, to_save) as trainer_with_finder:
        trainer_with_finder.run(train_loader)

    lr_finder.plot()

    if idist.get_rank() == 0:
        ax = lr_finder.plot(skip_end=0)
        filepath = Path(dirname) / "distrib_dummy.jpg"
        ax.figure.savefig(filepath)
        assert filepath.exists()

    sug_lr = lr_finder.lr_suggestion()
    assert 1e-3 <= sug_lr <= 1

    lr_finder.apply_suggested_lr(optimizer)
    assert optimizer.param_groups[0]["lr"] == sug_lr


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(dirname, distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_log_lr_and_loss(device)
    _test_distrib_integration_mnist(dirname, device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(dirname, distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_log_lr_and_loss(device)
    _test_distrib_integration_mnist(dirname, device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla(dirname):
    device = idist.device()
    assert "xla" in device.type
    _test_distrib_log_lr_and_loss(device)
    _test_distrib_integration_mnist(dirname, device)


def _test_distrib_log_lr_and_loss_xla_nprocs(index, dirname):
    device = idist.device()
    _test_distrib_log_lr_and_loss(device)
    _test_distrib_integration_mnist(dirname, device)

    import time

    # hack to have all proc properly sync:
    time.sleep(1)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_xla_nprocs(dirname, xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_log_lr_and_loss_xla_nprocs, args=(dirname,), nprocs=n)

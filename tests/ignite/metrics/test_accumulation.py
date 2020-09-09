import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.accumulation import Average, GeometricAverage, VariableAccumulation

torch.manual_seed(15)


def test_variable_accumulation_wrong_inputs():

    with pytest.raises(TypeError, match=r"Argument op should be a callable"):
        VariableAccumulation(1)

    with pytest.raises(TypeError, match=r"Output should be a number or torch.Tensor,"):
        mean_acc = VariableAccumulation(lambda a, x: a + x)
        mean_acc.update((1, 2))

    with pytest.raises(TypeError, match=r"Output should be a number or torch.Tensor,"):
        mean_acc = VariableAccumulation(lambda a, x: a + x)
        mean_acc.update("a")


def test_variable_accumulation_mean_variable():

    mean_var = VariableAccumulation(lambda a, x: a + x)
    y_true = torch.rand(100)

    for y in y_true:
        mean_var.update(y)

    a, n = mean_var.compute()
    assert a.item() == pytest.approx(y_true.sum().item())
    assert n == len(y_true)

    mean_var = VariableAccumulation(lambda a, x: a + x)
    y_true = torch.rand(100, 10)
    for y in y_true:
        mean_var.update(y)

    a, n = mean_var.compute()
    assert a.numpy() == pytest.approx(y_true.sum(dim=0).numpy())
    assert n == len(y_true)

    mean_var = VariableAccumulation(lambda a, x: a + x.sum(dim=0))
    # iterate by batch of 16 samples
    y_true = torch.rand(8, 16, 10)
    for y in y_true:
        mean_var.update(y)

    a, n = mean_var.compute()
    assert a.numpy() == pytest.approx(y_true.reshape(-1, 10).sum(dim=0).numpy())
    assert n == y_true.shape[0] * y_true.shape[1]


def test_average():

    with pytest.raises(NotComputableError):
        v = Average()
        v.compute()

    mean_var = Average()
    y_true = torch.rand(100) + torch.randint(0, 10, size=(100,)).float()

    for y in y_true:
        mean_var.update(y.item())

    m = mean_var.compute()
    assert m.item() == pytest.approx(y_true.mean().item())

    mean_var = Average()
    y_true = torch.rand(100, 10) + torch.randint(0, 10, size=(100, 10)).float()
    for y in y_true:
        mean_var.update(y)

    m = mean_var.compute()
    assert m.numpy() == pytest.approx(y_true.mean(dim=0).numpy())

    mean_var = Average()
    y_true = torch.rand(8, 16, 10) + torch.randint(0, 10, size=(8, 16, 10)).float()
    for y in y_true:
        mean_var.update(y)

    m = mean_var.compute()
    assert m.numpy() == pytest.approx(y_true.reshape(-1, 10).mean(dim=0).numpy())


def _geom_mean(t):
    np_t = t.numpy()
    return np.exp(np.mean(np.log(np_t), axis=0))


def test_geom_average():

    with pytest.raises(NotComputableError):
        v = GeometricAverage()
        v.compute()

    mean_var = GeometricAverage()
    y_true = torch.rand(100) + torch.randint(0, 10, size=(100,)).float()

    for y in y_true:
        mean_var.update(y.item())

    m = mean_var.compute()
    assert m.item() == pytest.approx(_geom_mean(y_true))

    mean_var = GeometricAverage()
    y_true = torch.rand(100, 10) + torch.randint(0, 10, size=(100, 10)).float()
    for y in y_true:
        mean_var.update(y)

    m = mean_var.compute()
    np.testing.assert_almost_equal(m.numpy(), _geom_mean(y_true), decimal=5)

    mean_var = GeometricAverage()
    y_true = torch.rand(8, 16, 10) + torch.randint(0, 10, size=(8, 16, 10)).float()
    for y in y_true:
        mean_var.update(y)

    m = mean_var.compute()
    np.testing.assert_almost_equal(m.numpy(), _geom_mean(y_true.reshape(-1, 10)), decimal=5)


def test_integration():
    def _test(metric_cls, true_result_fn):

        size = 100
        custom_variable = 10.0 + 5.0 * torch.rand(size, 12)

        def update_fn(engine, batch):
            return 0, custom_variable[engine.state.iteration - 1]

        engine = Engine(update_fn)

        custom_var_mean = metric_cls(output_transform=lambda output: output[1])
        custom_var_mean.attach(engine, "agg_custom_var")

        state = engine.run([0] * size)
        np.testing.assert_almost_equal(
            state.metrics["agg_custom_var"].numpy(), true_result_fn(custom_variable), decimal=5
        )

        size = 100
        custom_variable = 10.0 + 5.0 * torch.rand(size)

        def update_fn(engine, batch):
            return 0, custom_variable[engine.state.iteration - 1].item()

        engine = Engine(update_fn)

        custom_var_mean = metric_cls(output_transform=lambda output: output[1])
        custom_var_mean.attach(engine, "agg_custom_var")

        state = engine.run([0] * size)
        assert state.metrics["agg_custom_var"] == pytest.approx(true_result_fn(custom_variable))

    def _mean(y_true):
        return y_true.mean(dim=0).numpy()

    _test(Average, _mean)
    _test(GeometricAverage, _geom_mean)


def test_compute_mean_std():
    n = 8
    b = 12
    c = 3
    w = h = 64
    true_data = np.arange(0, n * b * h * w * c, dtype="float64").reshape(n * b, c, h, w) - (n * b * c * w * h * 0.75)
    mean = true_data.transpose((0, 2, 3, 1)).reshape(-1, c).mean(axis=0)
    std = true_data.transpose((0, 2, 3, 1)).reshape(-1, c).std(axis=0)

    train_loader = torch.from_numpy(true_data).reshape(n, b, c, h, w)

    def compute_mean_std(engine, batch):
        _b, _c = batch.shape[:2]
        data = batch.reshape(_b, _c, -1).to(dtype=torch.float64)
        _mean = torch.mean(data, dim=-1)
        _mean2 = torch.mean(data ** 2, dim=-1)
        return {"mean": _mean, "mean^2": _mean2}

    compute_engine = Engine(compute_mean_std)
    img_mean = Average(output_transform=lambda output: output["mean"])
    img_mean2 = Average(output_transform=lambda output: output["mean^2"])
    img_mean.attach(compute_engine, "mean")
    img_mean2.attach(compute_engine, "mean2")
    state = compute_engine.run(train_loader)
    state.metrics["std"] = torch.sqrt(state.metrics["mean2"] - state.metrics["mean"] ** 2)

    np.testing.assert_almost_equal(state.metrics["mean"].numpy(), mean, decimal=7)
    np.testing.assert_almost_equal(state.metrics["std"].numpy(), std, decimal=5)


def _test_distrib_variable_accumulation(device):
    def _test(metric_device):
        mean_var = VariableAccumulation(lambda a, x: a + x, device=metric_device)
        y_true = torch.rand(100, device=device, dtype=torch.float64)

        for y in y_true:
            mean_var.update(y)

        y_true = idist.all_reduce(y_true)
        a, n = mean_var.compute()
        assert a.item() == pytest.approx(y_true.sum().item())
        assert n == len(y_true) * idist.get_world_size()
        # check if call compute twice
        a, n = mean_var.compute()
        assert a.item() == pytest.approx(y_true.sum().item())
        assert n == len(y_true) * idist.get_world_size()

        mean_var = VariableAccumulation(lambda a, x: a + x, device=metric_device)
        y_true = torch.rand(50, 10, device=device, dtype=torch.float64)

        for y in y_true:
            mean_var.update(y)

        y_true = idist.all_reduce(y_true)
        a, n = mean_var.compute()
        assert n == len(y_true) * idist.get_world_size()
        np.testing.assert_almost_equal(a.cpu().numpy(), y_true.sum(dim=0).cpu().numpy(), decimal=4)
        a, n = mean_var.compute()
        assert n == len(y_true) * idist.get_world_size()
        np.testing.assert_almost_equal(a.cpu().numpy(), y_true.sum(dim=0).cpu().numpy(), decimal=4)

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(device)


def _test_distrib_average(device):
    def _test(metric_device):
        with pytest.raises(NotComputableError):
            v = Average(device=metric_device)
            v.compute()

        mean_var = Average(device=metric_device)
        y_true = torch.rand(100, dtype=torch.float64) + torch.randint(0, 10, size=(100,)).double()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()

        y_true = idist.all_reduce(y_true)
        assert m.item() == pytest.approx(y_true.mean().item() / idist.get_world_size())

        mean_var = Average(device=metric_device)
        y_true = torch.rand(100, 10, dtype=torch.float64) + torch.randint(0, 10, size=(100, 10)).double()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()

        y_true = idist.all_reduce(y_true)
        np.testing.assert_almost_equal(
            m.cpu().numpy(), y_true.mean(dim=0).cpu().numpy() / idist.get_world_size(), decimal=5
        )

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(device)


def _test_distrib_geom_average(device):
    def _test(metric_device):
        with pytest.raises(NotComputableError):
            v = GeometricAverage(device=metric_device)
            v.compute()

        mean_var = GeometricAverage(device=metric_device)
        y_true = torch.rand(100, dtype=torch.float64) + torch.randint(0, 10, size=(100,)).double()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()
        log_y_true = torch.log(y_true)
        log_y_true = idist.all_reduce(log_y_true)
        assert m.item() == pytest.approx(torch.exp(log_y_true.mean(dim=0) / idist.get_world_size()).item())

        mean_var = GeometricAverage(device=metric_device)
        y_true = torch.rand(100, 10, dtype=torch.float64) + torch.randint(0, 10, size=(100, 10)).double()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()
        log_y_true = torch.log(y_true)
        log_y_true = idist.all_reduce(log_y_true)
        np.testing.assert_almost_equal(
            m.cpu().numpy(), torch.exp(log_y_true.mean(dim=0) / idist.get_world_size()).cpu().numpy(), decimal=5
        )

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(device)


def _test_distrib_integration(device):
    def _test(metric_cls, true_result_fn, metric_device, tol=1e-5):

        size = 100
        custom_variable = 10.0 + 5.0 * torch.rand(size, 12, dtype=torch.float64)
        custom_variable = custom_variable.to(device)

        def update_fn(engine, batch):
            return 0, custom_variable[engine.state.iteration - 1]

        engine = Engine(update_fn)

        custom_var_mean = metric_cls(output_transform=lambda output: output[1], device=metric_device)
        custom_var_mean.attach(engine, "agg_custom_var")

        state = engine.run([0] * size)
        np.testing.assert_almost_equal(
            state.metrics["agg_custom_var"].cpu().numpy(),
            true_result_fn(custom_variable),
            decimal=int(np.log10(1.0 / tol)),
        )

        size = 100
        custom_variable = 10.0 + 5.0 * torch.rand(size, dtype=torch.float64)
        custom_variable = custom_variable.to(device)

        def update_fn(engine, batch):
            return 0, custom_variable[engine.state.iteration - 1].item()

        engine = Engine(update_fn)

        custom_var_mean = metric_cls(output_transform=lambda output: output[1], device=metric_device)
        custom_var_mean.attach(engine, "agg_custom_var")

        state = engine.run([0] * size)
        assert state.metrics["agg_custom_var"] == pytest.approx(true_result_fn(custom_variable), abs=tol)

    def _mean(y_true):
        y_true = idist.all_reduce(y_true)
        return y_true.mean(dim=0).cpu().numpy() / idist.get_world_size()

    def _geom_mean(y_true):
        log_y_true = torch.log(y_true)
        log_y_true = idist.all_reduce(log_y_true)
        np_t = log_y_true.cpu().numpy()
        return np.exp(np.mean(np_t, axis=0) / idist.get_world_size())

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(device)
    for metric_device in metric_devices:
        _test(Average, _mean, metric_device)
        _test(GeometricAverage, _geom_mean, metric_device, tol=1e-4)


def _test_distrib_accumulator_device(device):

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(device)
    for metric_device in metric_devices:

        m = VariableAccumulation(lambda a, x: x, device=metric_device)
        assert m._device == metric_device
        assert m.accumulator.device == metric_device, "{}:{} vs {}:{}".format(
            type(m.accumulator.device), m.accumulator.device, type(metric_device), metric_device
        )

        m.update(torch.tensor(1, device=device))
        assert m.accumulator.device == metric_device, "{}:{} vs {}:{}".format(
            type(m.accumulator.device), m.accumulator.device, type(metric_device), metric_device
        )


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):

    device = torch.device("cuda:{}".format(distributed_context_single_node_nccl["local_rank"]))
    _test_distrib_variable_accumulation(device)
    _test_distrib_average(device)
    _test_distrib_geom_average(device)
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distrib_variable_accumulation(device)
    _test_distrib_average(device)
    _test_distrib_geom_average(device)
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_variable_accumulation(device)
    _test_distrib_average(device)
    _test_distrib_geom_average(device)
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_variable_accumulation, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_average, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_geom_average, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device("cuda:{}".format(distributed_context_multi_node_nccl["local_rank"]))
    _test_distrib_variable_accumulation(device)
    _test_distrib_average(device)
    _test_distrib_geom_average(device)
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_variable_accumulation(device)
    _test_distrib_average(device)
    _test_distrib_geom_average(device)
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_variable_accumulation(device)
    _test_distrib_average(device)
    _test_distrib_geom_average(device)
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

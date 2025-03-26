import os
from unittest.mock import patch

import pytest
import sklearn
import torch
from sklearn.metrics import cohen_kappa_score

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import CohenKappa

torch.manual_seed(12)


@pytest.fixture()
def mock_no_sklearn():
    with patch.dict("sys.modules", {"sklearn.metrics": None}):
        yield sklearn


def test_no_sklearn(mock_no_sklearn):
    with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires scikit-learn to be installed."):
        CohenKappa()


def test_no_update():
    ck = CohenKappa()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        ck.compute()


def test_input_types():
    ck = CohenKappa()
    ck.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    ck.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        ck.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        ck.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        ck.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test_check_shape():
    ck = CohenKappa()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ck._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ck._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        ck._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


def test_cohen_kappa_wrong_weights_type():
    with pytest.raises(ValueError, match=r"Kappa Weighting type must be"):
        ck = CohenKappa(weights=7)

    with pytest.raises(ValueError, match=r"Kappa Weighting type must be"):
        ck = CohenKappa(weights="dd")


@pytest.fixture(params=range(4))
def test_data_binary(request):
    return [
        # Binary input data of shape (N,) or (N, 1)
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 16),
        (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_binary_input(n_times, weights, test_data_binary, available_device):
    y_pred, y, batch_size = test_data_binary
    ck = CohenKappa(weights=weights, device=available_device)
    assert ck._device == torch.device(available_device)

    ck.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            ck.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        ck.update((y_pred, y))

    np_y = y.numpy()
    np_y_pred = y_pred.numpy()

    res = ck.compute()
    assert isinstance(res, float)
    assert cohen_kappa_score(np_y, np_y_pred, weights=weights) == pytest.approx(res)


def test_multilabel_inputs():
    ck = CohenKappa()

    with pytest.raises(ValueError, match=r"multilabel-indicator is not supported"):
        ck.reset()
        ck.update((torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long()))
        ck.compute()

    with pytest.raises(ValueError, match=r"multilabel-indicator is not supported"):
        ck.reset()
        ck.update((torch.randint(0, 2, size=(10, 6)).long(), torch.randint(0, 2, size=(10, 6)).long()))
        ck.compute()

    with pytest.raises(ValueError, match=r"multilabel-indicator is not supported"):
        ck.reset()
        ck.update((torch.randint(0, 2, size=(10, 8)).long(), torch.randint(0, 2, size=(10, 8)).long()))
        ck.compute()


@pytest.fixture(params=range(2))
def test_data_integration_binary(request):
    return [
        # Binary input data of shape (N,) or (N, 1)
        (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 10),
        (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 10),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_integration_binary_input(n_times, weights, test_data_integration_binary, available_device):
    y_pred, y, batch_size = test_data_integration_binary

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ck_metric = CohenKappa(weights=weights, device=available_device)
    assert ck_metric._device == torch.device(available_device)

    ck_metric.attach(engine, "ck")

    np_y = y.numpy()
    np_y_pred = y_pred.numpy()

    np_ck = cohen_kappa_score(np_y, np_y_pred, weights=weights)

    data = list(range(y_pred.shape[0] // batch_size))
    ck = engine.run(data, max_epochs=1).metrics["ck"]

    assert isinstance(ck, float)
    assert np_ck == pytest.approx(ck)


def _test_distrib_binary_input(device):
    rank = idist.get_rank()

    def _test(y_pred, y, batch_size, metric_device):
        metric_device = torch.device(metric_device)
        ck = CohenKappa(device=metric_device)

        ck.reset()
        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                ck.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            ck.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        res = ck.compute()
        assert isinstance(res, float)
        assert cohen_kappa_score(np_y, np_y_pred) == pytest.approx(res)

    def get_test_cases():
        test_cases = [
            # Binary input data of shape (N,) or (N, 1)
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
            (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
            # updated batches
            (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 16),
            (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 16),
        ]
        return test_cases

    for i in range(3):
        torch.manual_seed(10 + rank + i)
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size, "cpu")
            if device.type != "xla":
                _test(y_pred, y, batch_size, idist.device())


def _test_distrib_integration_binary_input(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 16

        torch.manual_seed(12 + rank)

        # Binary input data of shape (N,) or (N, 1)
        y_true = torch.randint(0, 2, size=(n_iters * batch_size,)).to(device)
        y_preds = torch.randint(0, 2, size=(n_iters * batch_size,)).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        ck = CohenKappa(device=metric_device)
        ck.attach(engine, "ck")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_true = idist.all_gather(y_true)
        y_preds = idist.all_gather(y_preds)

        assert "ck" in engine.state.metrics

        res = engine.state.metrics["ck"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = cohen_kappa_score(y_true.cpu().numpy(), y_preds.cpu().numpy())

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_binary_input(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_binary_input(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_binary_input, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_binary_input, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_binary_input(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_binary_input(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_binary_input(device)
    _test_distrib_integration_binary_input(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_binary_input(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

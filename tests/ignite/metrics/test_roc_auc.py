import os
from unittest.mock import patch

import pytest
import sklearn
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import ROC_AUC
from ignite.metrics.epoch_metric import EpochMetricWarning

torch.manual_seed(12)


@pytest.fixture()
def mock_no_sklearn():
    with patch.dict("sys.modules", {"sklearn.metrics": None}):
        yield sklearn


def test_no_sklearn(mock_no_sklearn):
    with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires scikit-learn to be installed."):
        ROC_AUC()


def test_no_update():
    roc_auc = ROC_AUC()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        roc_auc.compute()


def test_input_types():
    roc_auc = ROC_AUC()
    roc_auc.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    roc_auc.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        roc_auc.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        roc_auc.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        roc_auc.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test_check_shape():
    roc_auc = ROC_AUC()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        roc_auc._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        roc_auc._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        roc_auc._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


@pytest.fixture(params=range(8))
def test_data_binary_and_multilabel(request):
    return [
        # Binary input data of shape (N,) or (N, 1)
        (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 1),
        (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 16),
        (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 16),
        # Binary input data of shape (N, L)
        (torch.randint(0, 2, size=(50, 4)).long(), torch.randint(0, 2, size=(50, 4)).long(), 1),
        (torch.randint(0, 2, size=(50, 7)).long(), torch.randint(0, 2, size=(50, 7)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 4)).long(), torch.randint(0, 2, size=(50, 4)).long(), 16),
        (torch.randint(0, 2, size=(50, 7)).long(), torch.randint(0, 2, size=(50, 7)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_binary_and_multilabel_inputs(n_times, available_device, test_data_binary_and_multilabel):
    y_pred, y, batch_size = test_data_binary_and_multilabel
    roc_auc = ROC_AUC(device=available_device)
    assert roc_auc._device == torch.device(available_device)
    roc_auc.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            roc_auc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        roc_auc.update((y_pred, y))

    np_y = y.numpy()
    np_y_pred = y_pred.numpy()

    res = roc_auc.compute()
    assert isinstance(res, float)
    assert roc_auc_score(np_y, np_y_pred) == pytest.approx(res)


def test_check_compute_fn():
    y_pred = torch.zeros((8, 13))
    y_pred[:, 1] = 1
    y_true = torch.zeros_like(y_pred)
    output = (y_pred, y_true)

    em = ROC_AUC(check_compute_fn=True)

    em.reset()
    with pytest.warns((UndefinedMetricWarning, EpochMetricWarning), match=r"Only one class.+present in y_true"):
        em.update(output)

    em = ROC_AUC(check_compute_fn=False)
    em.update(output)


@pytest.fixture(params=range(4))
def test_data_integration_binary_and_multilabel(request):
    return [
        # Binary input data of shape (N,) or (N, 1)
        (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 10),
        (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 10),
        # Binary input data of shape (N, L)
        (torch.randint(0, 2, size=(100, 3)).long(), torch.randint(0, 2, size=(100, 3)).long(), 10),
        (torch.randint(0, 2, size=(100, 4)).long(), torch.randint(0, 2, size=(100, 4)).long(), 10),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_integration_binary_and_multilabel_inputs(
    n_times, available_device, test_data_integration_binary_and_multilabel
):
    y_pred, y, batch_size = test_data_integration_binary_and_multilabel

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    roc_auc_metric = ROC_AUC(device=available_device)
    assert roc_auc_metric._device == torch.device(available_device)
    roc_auc_metric.attach(engine, "roc_auc")

    np_y = y.numpy()
    np_y_pred = y_pred.numpy()

    np_roc_auc = roc_auc_score(np_y, np_y_pred)

    data = list(range(y_pred.shape[0] // batch_size))
    roc_auc = engine.run(data, max_epochs=1).metrics["roc_auc"]

    assert isinstance(roc_auc, float)
    assert np_roc_auc == pytest.approx(roc_auc)


def _test_distrib_binary_and_multilabel_inputs(device):
    rank = idist.get_rank()

    def _test(y_pred, y, batch_size, metric_device):
        metric_device = torch.device(metric_device)
        roc_auc = ROC_AUC(device=metric_device)

        roc_auc.reset()
        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                roc_auc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            roc_auc.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        res = roc_auc.compute()
        assert isinstance(res, float)
        assert roc_auc_score(np_y, np_y_pred) == pytest.approx(res)

    def get_test_cases():
        test_cases = [
            # Binary input data of shape (N,) or (N, 1)
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
            (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
            # updated batches
            (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 16),
            (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 16),
            # Binary input data of shape (N, L)
            (torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long(), 1),
            (torch.randint(0, 2, size=(10, 7)).long(), torch.randint(0, 2, size=(10, 7)).long(), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 4)).long(), torch.randint(0, 2, size=(50, 4)).long(), 16),
            (torch.randint(0, 2, size=(50, 7)).long(), torch.randint(0, 2, size=(50, 7)).long(), 16),
        ]
        return test_cases

    for i in range(5):
        torch.manual_seed(12 + rank + i)
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size, "cpu")
            if device.type != "xla":
                _test(y_pred, y, batch_size, idist.device())


def _test_distrib_integration_binary_input(device):
    rank = idist.get_rank()
    n_iters = 80
    batch_size = 16
    n_classes = 2

    def _test(y_preds, y_true, n_epochs, metric_device, update_fn):
        metric_device = torch.device(metric_device)

        engine = Engine(update_fn)

        roc_auc = ROC_AUC(device=metric_device)
        roc_auc.attach(engine, "roc_auc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "roc_auc" in engine.state.metrics

        res = engine.state.metrics["roc_auc"]

        true_res = roc_auc_score(y_true.cpu().numpy(), y_preds.cpu().numpy())
        assert pytest.approx(res) == true_res

    def get_tests(is_N):
        if is_N:
            y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
            y_preds = torch.rand(n_iters * batch_size).to(device)

            def update_fn(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

        else:
            y_true = torch.randint(0, n_classes, size=(n_iters * batch_size, 10)).to(device)
            y_preds = torch.rand(n_iters * batch_size, 10).to(device)

            def update_fn(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

        return y_preds, y_true, update_fn

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for i in range(2):
            torch.manual_seed(12 + rank + i)
            # Binary input data of shape (N,)
            y_preds, y_true, update_fn = get_tests(is_N=True)
            _test(y_preds, y_true, n_epochs=1, metric_device=metric_device, update_fn=update_fn)
            _test(y_preds, y_true, n_epochs=2, metric_device=metric_device, update_fn=update_fn)
            # Binary input data of shape (N, L)
            y_preds, y_true, update_fn = get_tests(is_N=False)
            _test(y_preds, y_true, n_epochs=1, metric_device=metric_device, update_fn=update_fn)
            _test(y_preds, y_true, n_epochs=2, metric_device=metric_device, update_fn=update_fn)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_binary_and_multilabel_inputs(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_binary_and_multilabel_inputs(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_binary_and_multilabel_inputs, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_binary_input, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_binary_and_multilabel_inputs(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_binary_and_multilabel_inputs(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_binary_and_multilabel_inputs(device)
    _test_distrib_integration_binary_input(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_binary_and_multilabel_inputs(device)
    _test_distrib_integration_binary_input(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

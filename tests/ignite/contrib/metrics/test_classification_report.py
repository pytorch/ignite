import os
from unittest.mock import patch

import pytest
import sklearn
import torch
from sklearn.metrics import classification_report as sklearn_classification_report

import ignite.distributed as idist
from ignite.contrib.metrics import ClassificationReport
from ignite.engine import Engine
from ignite.exceptions import NotComputableError

torch.manual_seed(12)


@pytest.fixture()
def mock_no_sklearn():
    with patch.dict("sys.modules", {"sklearn.metrics": None}):
        yield sklearn


def test_no_sklearn(mock_no_sklearn):
    with pytest.raises(RuntimeError, match=r"This contrib module requires sklearn to be installed."):
        ClassificationReport()


def test_no_update():
    classification_report = ClassificationReport()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        classification_report.compute()


def test_input_types():
    classification_report = ClassificationReport()
    classification_report.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    classification_report.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        classification_report.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        classification_report.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        classification_report.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test_check_shape():
    classification_report = ClassificationReport()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        classification_report._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        classification_report._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        classification_report._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


@pytest.mark.parametrize("output_dict", [True, False])
def test_binary_input_N(output_dict):

    classification_report = ClassificationReport(output_dict=output_dict)

    def _test(y_true, y_pred, n_iters):
        classification_report.reset()

        np_y_true = y_true.numpy()
        np_y_pred = y_pred.numpy()

        if n_iters > 1:
            batch_size = y_true.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                classification_report.update((y_true[idx : idx + batch_size], y_pred[idx : idx + batch_size]))
        else:
            classification_report.update((y_true, y_pred))

        res = classification_report.compute()
        assert isinstance(res, dict if output_dict else str)
        assert sklearn_classification_report(np_y_true, np_y_pred, output_dict=output_dict) == res

    def get_test_cases():
        test_cases = [
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 1),
            (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
            (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 1),
            # updated batches
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 16),
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 16),
            (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 16),
            (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 16),
        ]
        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_true, y_pred, n_iters in test_cases:
            _test(y_true, y_pred, n_iters)


@pytest.mark.parametrize("output_dict", [True, False])
def test_integration_binary_input_with_output_transform(output_dict):
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        metric = ClassificationReport(output_transform=lambda x: (x[1], x[2]), output_dict=output_dict)
        metric.attach(engine, "ck")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        sklearn_report = sklearn_classification_report(np_y_pred, np_y, output_dict=output_dict)

        data = list(range(y_pred.shape[0] // batch_size))
        classification_report_res = engine.run(data, max_epochs=1).metrics["ck"]

        assert isinstance(classification_report_res, dict if output_dict else str)
        assert sklearn_report == classification_report_res

    def get_test_cases():
        test_cases = [
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 10),
            (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 10),
            (torch.randint(0, 2, size=(200,)).long(), torch.randint(0, 2, size=(200,)).long(), 10),
            (torch.randint(0, 2, size=(200, 1)).long(), torch.randint(0, 2, size=(200, 1)).long(), 10),
        ]
        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


def _test_distrib_binary_input_N(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(y_pred, y, n_iters, metric_device):

        metric_device = torch.device(metric_device)
        classification_report = ClassificationReport(device=metric_device)

        torch.manual_seed(10 + rank)

        classification_report.reset()

        if n_iters > 1:
            batch_size = y.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                classification_report.update((y[idx : idx + batch_size], y_pred[idx : idx + batch_size]))
        else:
            classification_report.update((y, y_pred))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        res = classification_report.compute()
        assert isinstance(res, str)
        assert sklearn_classification_report(np_y, np_y_pred) == res

    def get_test_cases():
        test_cases = [
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 1),
            (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
            (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 1),
            # updated batches
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 16),
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 16),
            (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 16),
            (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 16),
        ]
        return test_cases

    for _ in range(3):
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size, "cpu")
            if device.type != "xla":
                _test(y_pred, y, batch_size, idist.device())


def _test_distrib_integration_binary(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        s = 16
        n_classes = 2

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)

        def update(engine, i):
            return (
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        classification_report = ClassificationReport(device=metric_device)
        classification_report.attach(engine, "classification_report")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "classification_report" in engine.state.metrics

        res = engine.state.metrics["classification_report"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = sklearn_classification_report(y_true.cpu().numpy(), y_preds.cpu().numpy())

        assert res == true_res

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
def test_distrib_gpu(distributed_context_single_node_nccl):

    device = torch.device(f"cuda:{distributed_context_single_node_nccl['local_rank']}")
    _test_distrib_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distrib_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(
        _test_distrib_binary_input_N, (device,), np=nproc, do_init=True,
    )
    gloo_hvd_executor(
        _test_distrib_integration_binary, (device,), np=nproc, do_init=True,
    )


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):

    device = torch.device("cpu")
    _test_distrib_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):

    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():

    device = idist.device()
    _test_distrib_binary_input_N(device)
    _test_distrib_integration_binary(device)


def _test_distrib_xla_nprocs(index):

    device = idist.device()
    _test_distrib_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

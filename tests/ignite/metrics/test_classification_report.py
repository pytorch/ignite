import json
import os

import pytest
import torch
from packaging.version import Version

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics.classification_report import ClassificationReport


def _test_multiclass(metric_device, n_classes, output_dict, labels=None, distributed=False):
    if distributed:
        device = idist.device()
    else:
        device = metric_device

    classification_report = ClassificationReport(device=metric_device, output_dict=output_dict, labels=labels)
    n_iters = 80
    batch_size = 16

    y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
    y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

    def update(engine, i):
        return (
            y_preds[i * batch_size : (i + 1) * batch_size, :],
            y_true[i * batch_size : (i + 1) * batch_size],
        )

    engine = Engine(update)

    classification_report.attach(engine, "cr")

    data = list(range(n_iters))
    engine.run(data=data)

    if distributed:
        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

    assert "cr" in engine.state.metrics
    res = engine.state.metrics["cr"]
    res2 = classification_report.compute()
    assert res == res2

    assert isinstance(res, dict if output_dict else str)
    if not output_dict:
        res = json.loads(res)

    from sklearn.metrics import classification_report as sklearn_classification_report

    sklearn_result = sklearn_classification_report(
        y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy(), output_dict=True, zero_division=1
    )

    for i in range(n_classes):
        label_i = labels[i] if labels else str(i)
        assert sklearn_result[str(i)]["precision"] == pytest.approx(res[label_i]["precision"])
        assert sklearn_result[str(i)]["f1-score"] == pytest.approx(res[label_i]["f1-score"])
        assert sklearn_result[str(i)]["recall"] == pytest.approx(res[label_i]["recall"])
    assert sklearn_result["macro avg"]["precision"] == pytest.approx(res["macro avg"]["precision"])
    assert sklearn_result["macro avg"]["recall"] == pytest.approx(res["macro avg"]["recall"])
    assert sklearn_result["macro avg"]["f1-score"] == pytest.approx(res["macro avg"]["f1-score"])

    metric_state = classification_report.state_dict()
    classification_report.reset()
    classification_report.load_state_dict(metric_state)

    res2 = classification_report.compute()
    if not output_dict:
        res2 = json.loads(res2)

    for i in range(n_classes):
        label_i = labels[i] if labels else str(i)
        assert res2[label_i]["precision"] == res[label_i]["precision"]
        assert res2[label_i]["f1-score"] == res[label_i]["f1-score"]
        assert res2[label_i]["recall"] == res[label_i]["recall"]
    assert res2["macro avg"]["precision"] == res["macro avg"]["precision"]
    assert res2["macro avg"]["recall"] == res["macro avg"]["recall"]
    assert res2["macro avg"]["f1-score"] == res["macro avg"]["f1-score"]


def _test_multilabel(metric_device, n_epochs, output_dict, labels=None, distributed=False):
    if distributed:
        device = idist.device()
    else:
        device = metric_device

    classification_report = ClassificationReport(device=metric_device, output_dict=output_dict, is_multilabel=True)

    n_iters = 10
    batch_size = 16
    n_classes = 7

    y_true = torch.randint(0, 2, size=(n_iters * batch_size, n_classes, 6, 8)).to(device)
    y_preds = torch.randint(0, 2, size=(n_iters * batch_size, n_classes, 6, 8)).to(device)

    def update(engine, i):
        return (
            y_preds[i * batch_size : (i + 1) * batch_size, ...],
            y_true[i * batch_size : (i + 1) * batch_size, ...],
        )

    engine = Engine(update)

    classification_report.attach(engine, "cr")

    data = list(range(n_iters))
    engine.run(data=data, max_epochs=n_epochs)

    if distributed:
        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

    assert "cr" in engine.state.metrics
    res = engine.state.metrics["cr"]
    res2 = classification_report.compute()
    assert res == res2

    assert isinstance(res, dict if output_dict else str)
    if not output_dict:
        res = json.loads(res)

    np_y_preds = to_numpy_multilabel(y_preds)
    np_y_true = to_numpy_multilabel(y_true)

    from sklearn.metrics import classification_report as sklearn_classification_report

    sklearn_result = sklearn_classification_report(np_y_true, np_y_preds, output_dict=True, zero_division=1)

    for i in range(n_classes):
        label_i = labels[i] if labels else str(i)
        assert sklearn_result[str(i)]["precision"] == pytest.approx(res[label_i]["precision"])
        assert sklearn_result[str(i)]["f1-score"] == pytest.approx(res[label_i]["f1-score"])
        assert sklearn_result[str(i)]["recall"] == pytest.approx(res[label_i]["recall"])
    assert sklearn_result["macro avg"]["precision"] == pytest.approx(res["macro avg"]["precision"])
    assert sklearn_result["macro avg"]["recall"] == pytest.approx(res["macro avg"]["recall"])
    assert sklearn_result["macro avg"]["f1-score"] == pytest.approx(res["macro avg"]["f1-score"])


def _test_integration_multiclass(device, output_dict):
    rank = idist.get_rank()
    labels = ["label0", "label1", "label2", "label3"]

    for i in range(5):
        torch.manual_seed(12 + rank + i)
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            for n_classes in range(2, len(labels) + 1):
                for output_dict in [False, True]:
                    _test_multiclass(metric_device, n_classes, output_dict, distributed=True)
                    _test_multiclass(metric_device, n_classes, output_dict, labels=labels[:n_classes], distributed=True)


def _test_integration_multilabel(device, output_dict):
    rank = idist.get_rank()

    for i in range(3):
        torch.manual_seed(12 + rank + i)
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            for n_epochs in [1, 2]:
                for output_dict in [False, True]:
                    _test_multilabel(metric_device, n_epochs, output_dict, distributed=True)
                    _test_multilabel(
                        metric_device, n_epochs, output_dict, ["0", "1", "2", "3", "4", "5", "6"], distributed=True
                    )


@pytest.mark.parametrize("n_times", range(5))
def test_compute_multiclass(n_times, available_device):
    labels = ["label0", "label1", "label2", "label3"]
    for n_classes in range(2, len(labels) + 1):
        for output_dict in [False, True]:
            _test_multiclass(available_device, n_classes, output_dict)
            _test_multiclass(available_device, n_classes, output_dict, labels[:n_classes])


@pytest.mark.parametrize("n_times", range(5))
def test_compute_multilabel(n_times, available_device):
    for n_epochs in [1, 2]:
        for output_dict in [False, True]:
            _test_multilabel(available_device, n_epochs, output_dict)
            _test_multilabel(available_device, n_epochs, output_dict, ["0", "1", "2", "3", "4", "5", "6"])


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="Skip if < 1.7.0")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    pytest.skip("Temporarily skip failing test. See https://github.com/pytorch/ignite/pull/3301")
    # When run with 2 devices:
    #  tests/ignite/metrics/test_classification_report.py::test_distrib_nccl_gpu Fatal Python error: Aborted
    # Thread 0x00007fac95c95700 (most recent call first):
    #   <no Python frame>

    # Thread 0x00007facbb89b700 (most recent call first):
    #   <no Python frame>

    # Thread 0x00007fae637f4700 (most recent call first):
    #   File "<string>", line 534 in read
    #   File "<string>", line 567 in from_io
    #   File "<string>", line 1160 in _thread_receiver
    #   File "<string>", line 341 in run
    #   File "<string>", line 411 in _perform_spawn

    device = idist.device()
    _test_integration_multiclass(device, True)
    _test_integration_multiclass(device, False)
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="Skip if < 1.7.0")
def test_distrib_gloo_cpu_or_gpu(local_rank, distributed_context_single_node_gloo):
    device = idist.device()
    _test_integration_multiclass(device, True)
    _test_integration_multiclass(device, False)
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_integration_multiclass, (device, True), np=nproc, do_init=True)
    gloo_hvd_executor(_test_integration_multiclass, (device, False), np=nproc, do_init=True)
    gloo_hvd_executor(_test_integration_multilabel, (device, True), np=nproc, do_init=True)
    gloo_hvd_executor(_test_integration_multilabel, (device, False), np=nproc, do_init=True)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_integration_multiclass(device, True)
    _test_integration_multiclass(device, False)
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_integration_multiclass(device, True)
    _test_integration_multiclass(device, False)
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_integration_multiclass(device, True)
    _test_integration_multiclass(device, False)
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)

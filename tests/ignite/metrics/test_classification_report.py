import ast
import os

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics.classification_report import ClassificationReport


def _test_integration_binary(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):

        n_iters = 80
        s = 16
        offset = n_iters * s
        y_true = torch.randint(0, 2, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.randint(0, 2, size=(offset * idist.get_world_size(),)).to(device)
        y_true_unflat = _unflatten_binary(y_true)
        classification_report = ClassificationReport(device=metric_device, output_dict="False")

        def update(engine, i):
            return (
                y_true_unflat[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        classification_report.attach(engine, "cr")

        data = list(range(n_iters))
        engine.run(data=data)

        assert "cr" in engine.state.metrics
        res = engine.state.metrics["cr"]
        res2 = classification_report.compute()
        assert res == res2

        from sklearn.metrics import classification_report as sklearn_classification_report

        sklearn_result = sklearn_classification_report(y_preds.cpu().numpy(), y_true.cpu().numpy(), output_dict=True)

        assert pytest.approx(res["0"] == sklearn_result["0"])
        assert pytest.approx(res["1"] == sklearn_result["1"])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            _test(metric_device)


def _test_integration_binary_labels(device, labels):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):

        n_iters = 80
        s = 16
        offset = n_iters * s
        y_true = torch.randint(0, 2, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.randint(0, 2, size=(offset * idist.get_world_size(),)).to(device)
        y_true_unflat = _unflatten_binary(y_true)
        classification_report = ClassificationReport(device=metric_device, output_dict="False", labels=labels)

        def update(engine, i):
            return (
                y_true_unflat[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        classification_report.attach(engine, "cr")

        data = list(range(n_iters))
        engine.run(data=data)

        assert "cr" in engine.state.metrics
        res = engine.state.metrics["cr"]
        res2 = classification_report.compute()
        assert res == res2

        from sklearn.metrics import classification_report as sklearn_classification_report

        sklearn_result = sklearn_classification_report(y_preds.cpu().numpy(), y_true.cpu().numpy(), output_dict=True)

        assert pytest.approx(res[labels[0]] == sklearn_result["0"])
        assert pytest.approx(res[labels[1]] == sklearn_result["1"])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            _test(metric_device)


def _test_integration_multilabel(device, output_dict):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device, n_classes):

        n_iters = 80
        s = 16
        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_true_unflat = _unflatten_multilabel(y_true, n_classes)
        classification_report = ClassificationReport(device=metric_device, output_dict=output_dict)

        def update(engine, i):
            return (
                y_true_unflat[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        classification_report.attach(engine, "cr")

        data = list(range(n_iters))
        engine.run(data=data)

        assert "cr" in engine.state.metrics
        res = engine.state.metrics["cr"]
        res2 = classification_report.compute()
        assert res == res2

        assert isinstance(res, dict if output_dict else str)
        if not output_dict:
            res = ast.literal_eval(res)

        from sklearn.metrics import classification_report as sklearn_classification_report

        sklearn_result = sklearn_classification_report(y_preds.cpu().numpy(), y_true.cpu().numpy(), output_dict=True)

        for i in range(n_classes):
            assert pytest.approx(res[str(i)]["precision"] == sklearn_result[str(i)]["precision"])
            assert pytest.approx(res[str(i)]["f1-score"] == sklearn_result[str(i)]["f1-score"])
            assert pytest.approx(res[str(i)]["recall"] == sklearn_result[str(i)]["recall"])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            _test(metric_device, 3)


def _test_integration_multilabel_with_labels(device, output_dict):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device, n_classes, labels):

        n_iters = 80
        s = 16
        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_true_unflat = _unflatten_multilabel(y_true, n_classes)
        classification_report = ClassificationReport(device=metric_device, output_dict=output_dict, labels=labels)

        def update(engine, i):
            return (
                y_true_unflat[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        classification_report.attach(engine, "cr")

        data = list(range(n_iters))
        engine.run(data=data)

        assert "cr" in engine.state.metrics
        res = engine.state.metrics["cr"]
        res2 = classification_report.compute()
        assert res == res2

        assert isinstance(res, dict if output_dict else str)
        if not output_dict:
            res = ast.literal_eval(res)

        from sklearn.metrics import classification_report as sklearn_classification_report

        sklearn_result = sklearn_classification_report(y_preds.cpu().numpy(), y_true.cpu().numpy(), output_dict=True)

        for i in range(n_classes):
            assert pytest.approx(res[labels[i]]["precision"] == sklearn_result[str(i)]["precision"])
            assert pytest.approx(res[labels[i]]["f1-score"] == sklearn_result[str(i)]["f1-score"])
            assert pytest.approx(res[labels[i]]["recall"] == sklearn_result[str(i)]["recall"])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            _test(metric_device, 3, ["label0", "label1", "label2"])


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):

    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_integration_binary(device)
    _test_integration_binary_labels(device, labels=["label0", "label1"])
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)
    _test_integration_multilabel_with_labels(device, True)
    _test_integration_multilabel_with_labels(device, False)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_integration_binary(device)
    _test_integration_binary_labels(device, labels=["label0", "label1"])
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)
    _test_integration_multilabel_with_labels(device, True)
    _test_integration_multilabel_with_labels(device, False)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_integration_binary(device)
    _test_integration_binary_labels(device, labels=["label0", "label1"])
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)
    _test_integration_multilabel_with_labels(device, True)
    _test_integration_multilabel_with_labels(device, False)


def _test_distrib_xla_nprocs(index):

    device = idist.device()
    _test_integration_binary(device)
    _test_integration_binary_labels(device, labels=["label0", "label1"])
    _test_integration_multilabel(device, True)
    _test_integration_multilabel(device, False)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)


def _unflatten_multilabel(y, n_labels):
    return torch.tensor(list(map(lambda x: [1 if x == i else 0 for i in range(n_labels)], y)))


def _unflatten_binary(y):
    return _unflatten_multilabel(y, 2)

import pytest
import torch
import os

import ignite.distributed as idist
from ignite.metrics.classification_report import ClassificationReport

torch.manual_seed(12)


def test_input_types():
    classification_report = ClassificationReport()
    classification_report.reset()

    y_true, y_preds = torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long()
    classification_report.update((_unflatten_binary(y_true), y_preds))

    with pytest.raises(RuntimeError, match=r"Input data type has changed from multiclass to binary"):
        classification_report.update((y_true, y_preds))

    with pytest.raises(ValueError, match=r"y and y_pred must have compatible shapes."):
        y_true = torch.randint(0, 2, size=(100,)).long()
        classification_report.update((_unflatten_binary(y_true), y_preds))


def _test_integration_binary(device):

    from ignite.engine import Engine

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

        sklearn_result = sklearn_classification_report(y_preds, y_true, output_dict=True)

        assert pytest.approx(res["0"] == sklearn_result["0"])
        assert pytest.approx(res["1"] == sklearn_result["1"])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            _test(metric_device)


@pytest.mark.parametrize("output_dict", [True])
def test_binary_input_N(output_dict):

    classification_report = ClassificationReport(output_dict=output_dict)

    def _test(y_true, y_pred, batch_size):
        y_true_unflat = _unflatten_binary(y_true)
        classification_report.reset()

        if batch_size > 1:
            n_iters = y_true_unflat.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                classification_report.update((y_true_unflat[idx : idx + batch_size], y_pred[idx : idx + batch_size]))
        else:
            classification_report.update((y_true_unflat, y_pred))

        from sklearn.metrics import classification_report as sklearn_classification_report

        res = classification_report.compute()
        assert isinstance(res, dict if output_dict else str)

        sklearn_result = sklearn_classification_report(y_pred, y_true, output_dict=output_dict)
        if not output_dict:
            res = eval(res)
            sklearn_result = eval(sklearn_result)
        assert pytest.approx(res["0"] == sklearn_result["0"])
        assert pytest.approx(res["1"] == sklearn_result["1"])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    def get_test_cases():
        test_cases = [
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 1),
            # updated batches
            (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 16),
            (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 16),
        ]
        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_true, y_pred, batch_size in test_cases:
            _test(y_true, y_pred, batch_size)


@pytest.mark.parametrize("output_dict", [True])
def test_multilabel_input_N(output_dict):

    classification_report = ClassificationReport(output_dict=output_dict)

    def _test(y_true, y_pred, batch_size, n_classes):
        y_true_unflat = _unflatten_multilabel(y_true, n_classes)
        classification_report.reset()

        if batch_size > 1:
            n_iters = y_true_unflat.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                classification_report.update((y_true_unflat[idx : idx + batch_size], y_pred[idx : idx + batch_size]))
        else:
            classification_report.update((y_true_unflat, y_pred))

        from sklearn.metrics import classification_report as sklearn_classification_report

        res = classification_report.compute()
        assert isinstance(res, dict if output_dict else str)

        sklearn_result = sklearn_classification_report(y_pred, y_true, output_dict=output_dict)
        if not output_dict:
            res = eval(res)
            sklearn_result = eval(sklearn_result)
        for i in range(n_classes):
            assert pytest.approx(res[str(i)] == sklearn_result[str(i)])
        assert pytest.approx(res["macro avg"]["precision"] == sklearn_result["macro avg"]["precision"])
        assert pytest.approx(res["macro avg"]["recall"] == sklearn_result["macro avg"]["recall"])
        assert pytest.approx(res["macro avg"]["f1-score"] == sklearn_result["macro avg"]["f1-score"])

    def get_test_cases():
        test_cases = [
            (torch.randint(0, 3, size=(10,)).long(), torch.randint(0, 3, size=(10,)).long(), 1, 3),
            (torch.randint(0, 3, size=(100,)).long(), torch.randint(0, 3, size=(100,)).long(), 1, 3),
            # updated batches
            (torch.randint(0, 3, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 16, 3),
            (torch.randint(0, 3, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 16, 3),
        ]
        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_true, y_pred, batch_size, n_classes in test_cases:
            _test(y_true, y_pred, batch_size, n_classes)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_integration_binary(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_integration_binary(device)


def _unflatten_multilabel(y, n_labels):
    return torch.tensor(list(map(lambda x: [1 if x == i else 0 for i in range(n_labels)], y)))


def _unflatten_binary(y):
    return _unflatten_multilabel(y, 2)

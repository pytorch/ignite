import os

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import ConfusionMatrix, IoU, JaccardIndex, mIoU
from ignite.metrics.confusion_matrix import cmAccuracy, cmPrecision, cmRecall, DiceCoefficient

torch.manual_seed(12)


def test_no_update():
    cm = ConfusionMatrix(10)
    with pytest.raises(NotComputableError, match=r"Confusion matrix must have at least one example before it "):
        cm.compute()


def test_num_classes_wrong_input():
    with pytest.raises(ValueError, match="Argument num_classes needs to be > 1"):
        ConfusionMatrix(num_classes=1)


def test_multiclass_wrong_inputs():
    cm = ConfusionMatrix(10)

    with pytest.raises(
        ValueError, match=r"y_pred must have shape \(batch_size, num_classes " r"\(currently set to 10\), ...\)"
    ):
        cm.update((torch.rand(10), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y_pred does not have correct number of classes:"):
        cm.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(
        ValueError,
        match=r"y_pred must have shape \(batch_size, num_classes "
        r"\(currently set to 10\), ...\) "
        r"and y must have ",
    ):
        cm.update((torch.rand(4, 10, 12, 12), torch.randint(0, 10, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y and y_pred must have compatible shapes."):
        cm.update((torch.rand(4, 10, 12, 14), torch.randint(0, 10, size=(4, 5, 6)).long()))

    with pytest.raises(ValueError, match=r"Argument average can None or one of"):
        ConfusionMatrix(num_classes=10, average="abc")

    with pytest.raises(ValueError, match=r"Argument average should be one of 'samples', 'recall', 'precision'"):
        ConfusionMatrix.normalize(None, None)


@pytest.fixture(params=[item for item in range(10)])
def test_data(request):
    return [
        # Multiclass input data of shape (N, )
        (torch.rand(10, 4), torch.randint(0, 4, size=(10,)).long(), 4, 1),
        (torch.rand(4, 10), torch.randint(0, 10, size=(4,)).long(), 10, 1),
        (torch.rand(4, 2), torch.randint(0, 2, size=(4,)).long(), 2, 1),
        (torch.rand(100, 5), torch.randint(0, 5, size=(100,)).long(), 5, 16),
        # Multiclass input data of shape (N, L)
        (torch.rand(10, 4, 5), torch.randint(0, 4, size=(10, 5)).long(), 4, 1),
        (torch.rand(4, 10, 5), torch.randint(0, 10, size=(4, 5)).long(), 10, 1),
        (torch.rand(100, 9, 7), torch.randint(0, 9, size=(100, 7)).long(), 9, 16),
        # Multiclass input data of shape (N, H, W, ...)
        (torch.rand(4, 5, 12, 10), torch.randint(0, 5, size=(4, 12, 10)).long(), 5, 1),
        (torch.rand(4, 5, 10, 12, 8), torch.randint(0, 5, size=(4, 10, 12, 8)).long(), 5, 1),
        (torch.rand(100, 3, 8, 8), torch.randint(0, 3, size=(100, 8, 8)).long(), 3, 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_multiclass_input(n_times, test_data, available_device):
    y_pred, y, num_classes, batch_size = test_data
    cm = ConfusionMatrix(num_classes=num_classes, device=available_device)
    assert cm._device == torch.device(available_device)
    cm.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            cm.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        cm.update((y_pred, y))

    np_y_pred = y_pred.cpu().numpy().argmax(axis=1).ravel()
    np_y = y.cpu().numpy().ravel()
    assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().cpu().numpy())


def test_ignored_out_of_num_classes_indices(available_device):
    num_classes = 21
    cm = ConfusionMatrix(num_classes=num_classes, device=available_device)
    assert cm._device == torch.device(available_device)

    y_pred = torch.rand(4, num_classes, 12, 10)
    y = torch.randint(0, 255, size=(4, 12, 10)).long()
    cm.update((y_pred, y))
    np_y_pred = y_pred.cpu().numpy().argmax(axis=1).ravel()
    np_y = y.cpu().numpy().ravel()
    assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().cpu().numpy())


def get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((30, 30), dtype=np.int32)
    y_true[1:11, 1:11] = 1
    y_true[15:25, 15:25] = 2

    y_pred = np.zeros((30, 30), dtype=np.int32)
    y_pred[5:15, 1:11] = 1
    y_pred[20:30, 20:30] = 2
    return y_true, y_pred


def compute_th_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    th_y_true = torch.from_numpy(y_true).unsqueeze(0)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 720
    th_y_logits = torch.from_numpy(y_probas).unsqueeze(0)
    return th_y_true, th_y_logits


def test_multiclass_images(available_device):
    num_classes = 3
    cm = ConfusionMatrix(num_classes=num_classes, device=available_device)
    assert cm._device == torch.device(available_device)

    y_true, y_pred = get_y_true_y_pred()

    # Compute confusion matrix with sklearn
    true_res = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))

    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = cm.compute().cpu().numpy()

    assert np.all(true_res == res)

    # Another test on batch of 2 images
    num_classes = 3
    cm = ConfusionMatrix(num_classes=num_classes, device=available_device)
    assert cm._device == torch.device(available_device)

    # Create a batch of two images:
    th_y_true1 = torch.from_numpy(y_true).reshape(1, 30, 30)
    th_y_true2 = torch.from_numpy(y_true.transpose()).reshape(1, 30, 30)
    th_y_true = torch.cat([th_y_true1, th_y_true2], dim=0)

    # Create a batch of 2 logits tensors
    y_probas = np.ones((3, 30, 30)) * -10
    y_probas[0, (y_pred == 0)] = 720
    y_probas[1, (y_pred == 1)] = 720
    y_probas[2, (y_pred == 2)] = 768
    th_y_logits1 = torch.from_numpy(y_probas).reshape(1, 3, 30, 30)

    y_probas = np.ones((3, 30, 30)) * -10
    y_probas[0, (y_pred.transpose() == 0)] = 720
    y_probas[1, (y_pred.transpose() == 2)] = 720
    y_probas[2, (y_pred.transpose() == 1)] = 768
    th_y_logits2 = torch.from_numpy(y_probas).reshape(1, 3, 30, 30)

    th_y_logits = torch.cat([th_y_logits1, th_y_logits2], dim=0)

    # Update metric & compute
    output = (th_y_logits, th_y_true)
    cm.update(output)
    res = cm.compute().cpu().numpy()

    # Compute confusion matrix with sklearn
    true_res = confusion_matrix(
        th_y_true.cpu().numpy().reshape(-1), np.argmax(th_y_logits.cpu().numpy(), axis=1).reshape(-1)
    )

    assert np.all(true_res == res)


def test_iou_wrong_input():
    with pytest.raises(TypeError, match="Argument cm should be instance of ConfusionMatrix"):
        IoU(None)

    cm = ConfusionMatrix(num_classes=10)
    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given -1"):
        IoU(cm, ignore_index=-1)

    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given a"):
        IoU(cm, ignore_index="a")

    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given 10"):
        IoU(cm, ignore_index=10)

    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given 11"):
        IoU(cm, ignore_index=11)


@pytest.mark.parametrize("average", [None, "samples"])
def test_iou(average, available_device):
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        union = bin_y_true | bin_y_pred
        true_res[index] = intersection.sum() / union.sum()

    cm = ConfusionMatrix(num_classes=3, average=average, device=available_device)
    assert cm._device == torch.device(available_device)
    iou_metric = IoU(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = iou_metric.compute().cpu().numpy()

    assert np.all(res == true_res)

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3, device=available_device)
        assert cm._device == torch.device(available_device)
        iou_metric = IoU(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = iou_metric.compute().cpu().numpy()
        true_res_ = true_res[:ignore_index] + true_res[ignore_index + 1 :]
        assert np.all(res == true_res_), f"{ignore_index}: {res} vs {true_res_}"

    with pytest.raises(ValueError, match=r"ConfusionMatrix should have average attribute either"):
        cm = ConfusionMatrix(num_classes=3, average="precision")
        IoU(cm)


def test_miou(available_device):
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        union = bin_y_true | bin_y_pred
        true_res[index] = intersection.sum() / union.sum()

    true_res_ = np.mean(true_res)

    cm = ConfusionMatrix(num_classes=3, device=available_device)
    assert cm._device == torch.device(available_device)
    iou_metric = mIoU(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = iou_metric.compute().cpu().numpy()

    assert pytest.approx(res) == true_res_

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3, device=available_device)
        assert cm._device == torch.device(available_device)
        iou_metric = mIoU(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = iou_metric.compute().cpu().numpy()
        true_res_ = np.mean(true_res[:ignore_index] + true_res[ignore_index + 1 :])
        assert pytest.approx(res) == true_res_, f"{ignore_index}: {res} vs {true_res_}"


def test_cm_accuracy(available_device):
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_acc = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))

    cm = ConfusionMatrix(num_classes=3, device=available_device)
    assert cm._device == torch.device(available_device)
    acc_metric = cmAccuracy(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = acc_metric.compute().cpu().numpy()

    assert pytest.approx(res) == true_acc


def test_cm_precision(available_device):
    y_true, y_pred = np.random.randint(0, 10, size=(1000,)), np.random.randint(0, 10, size=(1000,))
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_pr = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average="macro")

    cm = ConfusionMatrix(num_classes=10, device=available_device)
    assert cm._device == torch.device(available_device)
    pr_metric = cmPrecision(cm, average=True)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = pr_metric.compute().cpu().numpy()

    assert pytest.approx(res) == true_pr

    true_pr = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average=None)
    cm = ConfusionMatrix(num_classes=10, device=available_device)
    assert cm._device == torch.device(available_device)
    pr_metric = cmPrecision(cm, average=False)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = pr_metric.compute().cpu().numpy()

    assert np.all(res == true_pr)


def test_cm_recall(available_device):
    y_true, y_pred = np.random.randint(0, 10, size=(1000,)), np.random.randint(0, 10, size=(1000,))
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_re = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average="macro")

    cm = ConfusionMatrix(num_classes=10, device=available_device)
    assert cm._device == torch.device(available_device)
    re_metric = cmRecall(cm, average=True)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = re_metric.compute().cpu().numpy()

    assert pytest.approx(res) == true_re

    true_re = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average=None)
    cm = ConfusionMatrix(num_classes=10, device=available_device)
    assert cm._device == torch.device(available_device)
    re_metric = cmRecall(cm, average=False)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = re_metric.compute().cpu().numpy()

    assert np.all(res == true_re)


def test_cm_with_average(available_device):
    num_classes = 5
    y_pred = torch.rand(40, num_classes)
    y = torch.randint(0, num_classes, size=(40,)).long()
    np_y_pred = y_pred.cpu().numpy().argmax(axis=1).ravel()
    np_y = y.cpu().numpy().ravel()

    cm = ConfusionMatrix(num_classes=num_classes, average="samples", device=available_device)
    assert cm._device == torch.device(available_device)
    cm.update((y_pred, y))
    true_res = confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) * 1.0 / len(np_y)
    res = cm.compute().cpu().numpy()
    np.testing.assert_almost_equal(true_res, res)

    cm = ConfusionMatrix(num_classes=num_classes, average="recall", device=available_device)
    assert cm._device == torch.device(available_device)
    cm.update((y_pred, y))
    true_re = recall_score(np_y, np_y_pred, average=None, labels=list(range(num_classes)))
    res = cm.compute().cpu().numpy().diagonal()
    np.testing.assert_almost_equal(true_re, res)

    res = cm.compute().cpu().numpy()
    true_res = confusion_matrix(np_y, np_y_pred, normalize="true")
    np.testing.assert_almost_equal(true_res, res)

    cm = ConfusionMatrix(num_classes=num_classes, average="precision", device=available_device)
    assert cm._device == torch.device(available_device)
    cm.update((y_pred, y))
    true_pr = precision_score(np_y, np_y_pred, average=None, labels=list(range(num_classes)))
    res = cm.compute().cpu().numpy().diagonal()
    np.testing.assert_almost_equal(true_pr, res)

    res = cm.compute().cpu().numpy()
    true_res = confusion_matrix(np_y, np_y_pred, normalize="pred")
    np.testing.assert_almost_equal(true_res, res)


def test_dice_coefficient_wrong_input():
    with pytest.raises(TypeError, match="Argument cm should be instance of ConfusionMatrix"):
        DiceCoefficient(None)

    cm = ConfusionMatrix(num_classes=10)
    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given -1"):
        DiceCoefficient(cm, ignore_index=-1)

    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given a"):
        DiceCoefficient(cm, ignore_index="a")

    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given 10"):
        DiceCoefficient(cm, ignore_index=10)

    with pytest.raises(ValueError, match=r"ignore_index should be integer and in the range of \[0, 10\), but given 11"):
        DiceCoefficient(cm, ignore_index=11)


def test_dice_coefficient(available_device):
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        # dice coefficient: 2*intersection(x, y) / (|x| + |y|)
        # union(x, y) = |x| + |y| - intersection(x, y)
        intersection = bin_y_true & bin_y_pred
        union = bin_y_true | bin_y_pred
        true_res[index] = 2.0 * intersection.sum() / (union.sum() + intersection.sum())

    cm = ConfusionMatrix(num_classes=3, device=available_device)
    assert cm._device == torch.device(available_device)
    dice_metric = DiceCoefficient(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = dice_metric.compute().cpu().numpy()
    np.testing.assert_allclose(res, true_res)

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3, device=available_device)
        assert cm._device == torch.device(available_device)
        dice_metric = DiceCoefficient(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = dice_metric.compute().cpu().numpy()
        true_res_ = true_res[:ignore_index] + true_res[ignore_index + 1 :]
        assert np.all(res == true_res_), f"{ignore_index}: {res} vs {true_res_}"


def _test_distrib_multiclass_images(device):
    def _test(metric_device):
        num_classes = 3
        cm = ConfusionMatrix(num_classes=num_classes, device=metric_device)

        y_true, y_pred = get_y_true_y_pred()

        # Compute confusion matrix with sklearn
        true_res = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))

        th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)
        th_y_true = th_y_true.to(device)
        th_y_logits = th_y_logits.to(device)

        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)

        res = cm.compute().cpu().numpy() / idist.get_world_size()

        assert np.all(true_res == res)

        # Another test on batch of 2 images
        num_classes = 3
        cm = ConfusionMatrix(num_classes=num_classes, device=metric_device)

        # Create a batch of two images:
        th_y_true1 = torch.from_numpy(y_true).reshape(1, 30, 30)
        th_y_true2 = torch.from_numpy(y_true.transpose()).reshape(1, 30, 30)
        th_y_true = torch.cat([th_y_true1, th_y_true2], dim=0)
        th_y_true = th_y_true.to(device)

        # Create a batch of 2 logits tensors
        y_probas = np.ones((3, 30, 30)) * -10
        y_probas[0, (y_pred == 0)] = 720
        y_probas[1, (y_pred == 1)] = 720
        y_probas[2, (y_pred == 2)] = 768
        th_y_logits1 = torch.from_numpy(y_probas).reshape(1, 3, 30, 30)

        y_probas = np.ones((3, 30, 30)) * -10
        y_probas[0, (y_pred.transpose() == 0)] = 720
        y_probas[1, (y_pred.transpose() == 2)] = 720
        y_probas[2, (y_pred.transpose() == 1)] = 768
        th_y_logits2 = torch.from_numpy(y_probas).reshape(1, 3, 30, 30)

        th_y_logits = torch.cat([th_y_logits1, th_y_logits2], dim=0)
        # check update if input is on another device
        th_y_logits = th_y_logits.to(device)

        # Update metric & compute
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = cm.compute().cpu().numpy()

        # Compute confusion matrix with sklearn
        th_y_true = idist.all_gather(th_y_true)
        th_y_logits = idist.all_gather(th_y_logits)

        np_y_true = th_y_true.cpu().numpy().reshape(-1)
        np_y_pred = np.argmax(th_y_logits.cpu().numpy(), axis=1).reshape(-1)
        true_res = confusion_matrix(np_y_true, np_y_pred)

        assert np.all(true_res == res)

    _test("cpu")
    if device.type != "xla":
        _test(idist.device())


def _test_distrib_accumulator_device(device):
    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        cm = ConfusionMatrix(num_classes=3, device=metric_device)
        assert cm._device == metric_device
        assert (
            cm.confusion_matrix.device == metric_device
        ), f"{type(cm.confusion_matrix.device)}:{cm._num_correct.device} vs {type(metric_device)}:{metric_device}"

        y_true, y_pred = get_y_true_y_pred()
        th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)
        cm.update((th_y_logits, th_y_true))

        assert (
            cm.confusion_matrix.device == metric_device
        ), f"{type(cm.confusion_matrix.device)}:{cm._num_correct.device} vs {type(metric_device)}:{metric_device}"


@pytest.mark.parametrize("average", [None, "samples"])
def test_jaccard_index(average, available_device):
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        union = bin_y_true | bin_y_pred
        true_res[index] = intersection.sum() / union.sum()

    cm = ConfusionMatrix(num_classes=3, average=average, device=available_device)
    assert cm._device == torch.device(available_device)
    jaccard_index = JaccardIndex(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = jaccard_index.compute().cpu().numpy()

    assert np.all(res == true_res)

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3, device=available_device)
        assert cm._device == torch.device(available_device)
        jaccard_index_metric = JaccardIndex(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = jaccard_index_metric.compute().cpu().numpy()
        true_res_ = true_res[:ignore_index] + true_res[ignore_index + 1 :]
        assert np.all(res == true_res_), f"{ignore_index}: {res} vs {true_res_}"

    with pytest.raises(ValueError, match=r"ConfusionMatrix should have average attribute either"):
        cm = ConfusionMatrix(num_classes=3, average="precision")
        JaccardIndex(cm)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_multiclass_images(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_multiclass_images(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_multiclass_images, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_multiclass_images(device)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_multiclass_images(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_multiclass_images(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_multiclass_images(device)
    _test_distrib_accumulator_device(device)

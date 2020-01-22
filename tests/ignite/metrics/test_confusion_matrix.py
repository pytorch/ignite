import os
import torch

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from ignite.exceptions import NotComputableError
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from ignite.metrics.confusion_matrix import cmAccuracy, cmPrecision, cmRecall, DiceCoefficient
import pytest


torch.manual_seed(12)


def test_no_update():
    cm = ConfusionMatrix(10)
    with pytest.raises(NotComputableError):
        cm.compute()


def test_multiclass_wrong_inputs():
    cm = ConfusionMatrix(10)

    with pytest.raises(ValueError, match=r"y_pred must have shape \(batch_size, num_categories, ...\)"):
        cm.update((torch.rand(10),
                   torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y_pred does not have correct number of categories:"):
        cm.update((torch.rand(10, 5, 4),
                   torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y_pred must have shape \(batch_size, num_categories, ...\) "
                                         r"and y must have "):
        cm.update((torch.rand(4, 10, 12, 12),
                   torch.randint(0, 10, size=(10, )).long()))

    with pytest.raises(ValueError, match=r"y and y_pred must have compatible shapes."):
        cm.update((torch.rand(4, 10, 12, 14),
                   torch.randint(0, 10, size=(4, 5, 6)).long()))

    with pytest.raises(ValueError, match=r"Argument average can None or one of"):
        ConfusionMatrix(num_classes=10, average="abc")


def test_multiclass_input_N():
    # Multiclass input data of shape (N, )
    def _test_N():
        num_classes = 4
        cm = ConfusionMatrix(num_classes=num_classes)
        y_pred = torch.rand(10, num_classes)
        y = torch.randint(0, num_classes, size=(10,)).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        num_classes = 10
        cm = ConfusionMatrix(num_classes=num_classes)
        y_pred = torch.rand(4, num_classes)
        y = torch.randint(0, num_classes, size=(4, )).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        # 2-classes
        num_classes = 2
        cm = ConfusionMatrix(num_classes=num_classes)
        y_pred = torch.rand(4, num_classes)
        y = torch.randint(0, num_classes, size=(4,)).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        # Batched Updates
        num_classes = 5
        cm = ConfusionMatrix(num_classes=num_classes)

        y_pred = torch.rand(100, num_classes)
        y = torch.randint(0, num_classes, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            cm.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test_N()


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L)
    def _test_NL():
        num_classes = 4
        cm = ConfusionMatrix(num_classes=num_classes)

        y_pred = torch.rand(10, num_classes, 5)
        y = torch.randint(0, num_classes, size=(10, 5)).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        num_classes = 10
        cm = ConfusionMatrix(num_classes=num_classes)
        y_pred = torch.rand(4, num_classes, 5)
        y = torch.randint(0, num_classes, size=(4, 5)).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        # Batched Updates
        num_classes = 9
        cm = ConfusionMatrix(num_classes=num_classes)

        y_pred = torch.rand(100, num_classes, 7)
        y = torch.randint(0, num_classes, size=(100, 7)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            cm.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test_NL()


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...)
    def _test_NHW():
        num_classes = 5
        cm = ConfusionMatrix(num_classes=num_classes)

        y_pred = torch.rand(4, num_classes, 12, 10)
        y = torch.randint(0, num_classes, size=(4, 12, 10)).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        num_classes = 5
        cm = ConfusionMatrix(num_classes=num_classes)
        y_pred = torch.rand(4, num_classes, 10, 12, 8)
        y = torch.randint(0, num_classes, size=(4, 10, 12, 8)).long()
        cm.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

        # Batched Updates
        num_classes = 3
        cm = ConfusionMatrix(num_classes=num_classes)
        y_pred = torch.rand(100, num_classes, 8, 8)
        y = torch.randint(0, num_classes, size=(100, 8, 8)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            cm.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test_NHW()


def test_ignored_out_of_num_classes_indices():
    num_classes = 21
    cm = ConfusionMatrix(num_classes=num_classes)

    y_pred = torch.rand(4, num_classes, 12, 10)
    y = torch.randint(0, 255, size=(4, 12, 10)).long()
    cm.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert np.all(confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) == cm.compute().numpy())


def get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((30, 30), dtype=np.int)
    y_true[1:11, 1:11] = 1
    y_true[15:25, 15:25] = 2

    y_pred = np.zeros((30, 30), dtype=np.int)
    y_pred[5:15, 1:11] = 1
    y_pred[20:30, 20:30] = 2
    return y_true, y_pred


def compute_th_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    th_y_true = torch.from_numpy(y_true).unsqueeze(0)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes, ) + y_true.shape) * -10
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 720
    th_y_logits = torch.from_numpy(y_probas).unsqueeze(0)
    return th_y_true, th_y_logits


def test_multiclass_images():
    num_classes = 3
    cm = ConfusionMatrix(num_classes=num_classes)

    y_true, y_pred = get_y_true_y_pred()

    # Compute confusion matrix with sklearn
    true_res = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))

    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = cm.compute().numpy()

    assert np.all(true_res == res)

    # Another test on batch of 2 images
    num_classes = 3
    cm = ConfusionMatrix(num_classes=num_classes)

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
    res = cm.compute().numpy()

    # Compute confusion matrix with sklearn
    true_res = confusion_matrix(th_y_true.numpy().reshape(-1), np.argmax(th_y_logits.numpy(), axis=1).reshape(-1))

    assert np.all(true_res == res)


def test_iou_wrong_input():

    with pytest.raises(TypeError, match="Argument cm should be instance of ConfusionMatrix"):
        IoU(None)

    cm = ConfusionMatrix(num_classes=10)
    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        IoU(cm, ignore_index=-1)

    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        IoU(cm, ignore_index="a")

    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        IoU(cm, ignore_index=10)

    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        IoU(cm, ignore_index=11)


def test_iou():

    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        union = bin_y_true | bin_y_pred
        true_res[index] = intersection.sum() / union.sum()

    cm = ConfusionMatrix(num_classes=3)
    iou_metric = IoU(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = iou_metric.compute().numpy()

    assert np.all(res == true_res)

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3)
        iou_metric = IoU(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = iou_metric.compute().numpy()
        true_res_ = true_res[:ignore_index] + true_res[ignore_index + 1:]
        assert np.all(res == true_res_), "{}: {} vs {}".format(ignore_index, res, true_res_)


def test_miou():

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

    cm = ConfusionMatrix(num_classes=3)
    iou_metric = mIoU(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = iou_metric.compute().numpy()

    assert res == true_res_

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3)
        iou_metric = mIoU(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = iou_metric.compute().numpy()
        true_res_ = np.mean(true_res[:ignore_index] + true_res[ignore_index + 1:])
        assert res == true_res_, "{}: {} vs {}".format(ignore_index, res, true_res_)


def test_cm_accuracy():

    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_acc = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))

    cm = ConfusionMatrix(num_classes=3)
    acc_metric = cmAccuracy(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = acc_metric.compute().numpy()

    assert pytest.approx(res) == true_acc


def test_cm_precision():

    y_true, y_pred = np.random.randint(0, 10, size=(1000,)), np.random.randint(0, 10, size=(1000,))
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_pr = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro')

    cm = ConfusionMatrix(num_classes=10)
    pr_metric = cmPrecision(cm, average=True)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = pr_metric.compute().numpy()

    assert pytest.approx(res) == true_pr

    true_pr = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average=None)
    cm = ConfusionMatrix(num_classes=10)
    pr_metric = cmPrecision(cm, average=False)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = pr_metric.compute().numpy()

    assert np.all(res == true_pr)


def test_cm_recall():

    y_true, y_pred = np.random.randint(0, 10, size=(1000,)), np.random.randint(0, 10, size=(1000,))
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_re = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro')

    cm = ConfusionMatrix(num_classes=10)
    re_metric = cmRecall(cm, average=True)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = re_metric.compute().numpy()

    assert pytest.approx(res) == true_re

    true_re = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average=None)
    cm = ConfusionMatrix(num_classes=10)
    re_metric = cmRecall(cm, average=False)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = re_metric.compute().numpy()

    assert np.all(res == true_re)


def test_cm_with_average():
    num_classes = 5
    y_pred = torch.rand(40, num_classes)
    y = torch.randint(0, num_classes, size=(40,)).long()
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()

    cm = ConfusionMatrix(num_classes=num_classes, average='samples')
    cm.update((y_pred, y))
    true_res = confusion_matrix(np_y, np_y_pred, labels=list(range(num_classes))) * 1.0 / len(np_y)
    res = cm.compute().numpy()
    np.testing.assert_almost_equal(true_res, res)

    cm = ConfusionMatrix(num_classes=num_classes, average='recall')
    cm.update((y_pred, y))
    true_re = recall_score(np_y, np_y_pred, average=None, labels=list(range(num_classes)))
    res = cm.compute().numpy().diagonal()
    np.testing.assert_almost_equal(true_re, res)

    cm = ConfusionMatrix(num_classes=num_classes, average='precision')
    cm.update((y_pred, y))
    true_pr = precision_score(np_y, np_y_pred, average=None, labels=list(range(num_classes)))
    res = cm.compute().numpy().diagonal()
    np.testing.assert_almost_equal(true_pr, res)


def test_dice_coefficient_wrong_input():

    with pytest.raises(TypeError, match="Argument cm should be instance of ConfusionMatrix"):
        DiceCoefficient(None)

    cm = ConfusionMatrix(num_classes=10)
    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        DiceCoefficient(cm, ignore_index=-1)

    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        DiceCoefficient(cm, ignore_index="a")

    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        DiceCoefficient(cm, ignore_index=10)

    with pytest.raises(ValueError, match="ignore_index should be non-negative integer"):
        DiceCoefficient(cm, ignore_index=11)


def test_dice_coefficient():

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

    cm = ConfusionMatrix(num_classes=3)
    dice_metric = DiceCoefficient(cm)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = dice_metric.compute().numpy()
    np.testing.assert_allclose(res, true_res)

    for ignore_index in range(3):
        cm = ConfusionMatrix(num_classes=3)
        dice_metric = DiceCoefficient(cm, ignore_index=ignore_index)
        # Update metric
        output = (th_y_logits, th_y_true)
        cm.update(output)
        res = dice_metric.compute().numpy()
        true_res_ = true_res[:ignore_index] + true_res[ignore_index + 1:]
        assert np.all(res == true_res_), "{}: {} vs {}".format(ignore_index, res, true_res_)


def _test_distrib_multiclass_images(device):

    import torch.distributed as dist

    def _gather(y):
        output = [torch.zeros_like(y) for i in range(dist.get_world_size())]
        dist.all_gather(output, y)
        y = torch.cat(output, dim=0)
        return y

    num_classes = 3
    cm = ConfusionMatrix(num_classes=num_classes, device=device)

    y_true, y_pred = get_y_true_y_pred()

    # Compute confusion matrix with sklearn
    true_res = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))

    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)
    th_y_true = th_y_true.to(device)
    th_y_logits = th_y_logits.to(device)

    # Update metric
    output = (th_y_logits, th_y_true)
    cm.update(output)

    res = cm.compute().cpu().numpy() / dist.get_world_size()

    assert np.all(true_res == res)

    # Another test on batch of 2 images
    num_classes = 3
    cm = ConfusionMatrix(num_classes=num_classes, device=device)

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
    th_y_true = _gather(th_y_true)
    th_y_logits = _gather(th_y_logits)

    np_y_true = th_y_true.cpu().numpy().reshape(-1)
    np_y_pred = np.argmax(th_y_logits.cpu().numpy(), axis=1).reshape(-1)
    true_res = confusion_matrix(np_y_true, np_y_pred)

    assert np.all(true_res == res)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = "cuda:{}".format(local_rank)
    _test_distrib_multiclass_images(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_multiclass_images(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_multiclass_images(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_distrib_multiclass_images(device)

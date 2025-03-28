import numpy as np
import pytest
import torch
from sklearn.metrics import multilabel_confusion_matrix

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix

torch.manual_seed(12)


def test_no_update():
    cm = MultiLabelConfusionMatrix(10)
    with pytest.raises(
        NotComputableError, match=r"Confusion matrix must have at least one example before it can be computed"
    ):
        cm.compute()


def test_num_classes_wrong_input():
    with pytest.raises(ValueError, match="Argument num_classes needs to be > 1"):
        MultiLabelConfusionMatrix(num_classes=1)


def test_multiclass_wrong_inputs():
    cm = MultiLabelConfusionMatrix(10)

    with pytest.raises(
        ValueError, match=r"y_pred must at least have shape \(batch_size, num_classes \(currently set to 10\), ...\)"
    ):
        cm.update((torch.rand(10), torch.randint(0, 2, size=(10, 10)).long()))

    with pytest.raises(
        ValueError, match=r"y must at least have shape \(batch_size, num_classes \(currently set to 10\), ...\)"
    ):
        cm.update((torch.rand(10, 10), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y_pred and y have different batch size: 10 vs 8"):
        cm.update((torch.rand(10, 10), torch.randint(0, 2, size=(8, 10)).long()))

    with pytest.raises(ValueError, match=r"y does not have correct number of classes: 9 vs 10"):
        cm.update((torch.rand(10, 10), torch.randint(0, 2, size=(10, 9)).long()))

    with pytest.raises(ValueError, match=r"y_pred does not have correct number of classes: 3 vs 10"):
        cm.update((torch.rand(10, 3), torch.randint(0, 2, size=(10, 10)).long()))

    with pytest.raises(ValueError, match=r"y and y_pred shapes must match."):
        cm.update((torch.rand(10, 10, 2), torch.randint(0, 2, size=(10, 10)).long()))

    with pytest.raises(
        ValueError,
        match=r"y_pred must be of any type: \(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64\)",
    ):
        cm.update((torch.rand(10, 10), torch.rand(10, 10)))

    with pytest.raises(
        ValueError, match=r"y must be of any type: \(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64\)"
    ):
        cm.update((torch.rand(10, 10).type(torch.int32), torch.rand(10, 10)))

    with pytest.raises(ValueError, match=r"y_pred must be a binary tensor"):
        y = torch.randint(0, 2, size=(10, 10)).long()
        y_pred = torch.randint(0, 2, size=(10, 10)).long()
        y_pred[0, 0] = 2
        cm.update((y_pred, y))

    with pytest.raises(ValueError, match=r"y must be a binary tensor"):
        y = torch.randint(0, 2, size=(10, 10)).long()
        y_pred = torch.randint(0, 2, size=(10, 10)).long()
        y[0, 0] = 2
        cm.update((y_pred, y))


def get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((1, 3, 30, 30), dtype=np.int64)
    y_true[0, 0, 5:17, 7:11] = 1
    y_true[0, 1, 1:11, 1:11] = 1
    y_true[0, 2, 15:25, 15:25] = 1

    y_pred = np.zeros((1, 3, 30, 30), dtype=np.int64)
    y_pred[0, 0, 0:7, 8:15] = 1
    y_pred[0, 1, 5:15, 1:11] = 1
    y_pred[0, 2, 20:30, 20:30] = 1
    return y_true, y_pred


def test_multiclass_images(available_device):
    num_classes = 3
    cm = MultiLabelConfusionMatrix(num_classes=num_classes, device=available_device)
    assert cm._device == torch.device(available_device)

    y_true, y_pred = get_y_true_y_pred()

    # Compute confusion matrix with sklearn
    sklearn_CM = multilabel_confusion_matrix(
        y_true.transpose((0, 2, 3, 1)).reshape(-1, 3), y_pred.transpose((0, 2, 3, 1)).reshape(-1, 3)
    )

    # Update metric
    output = (torch.tensor(y_pred), torch.tensor(y_true))
    cm.update(output)

    ignite_CM = cm.compute().cpu().numpy()

    assert np.all(ignite_CM == sklearn_CM)

    # Another test on batch of 2 images
    cm = MultiLabelConfusionMatrix(num_classes=num_classes, device=available_device)
    assert cm._device == torch.device(available_device)

    # Create a batch of two images:
    th_y_true1 = torch.tensor(y_true)
    th_y_true2 = torch.tensor(y_true.transpose(0, 1, 3, 2))
    th_y_true = torch.cat([th_y_true1, th_y_true2], dim=0)

    th_y_pred1 = torch.tensor(y_pred)
    th_y_pred2 = torch.tensor(y_pred.transpose(0, 1, 3, 2))
    th_y_pred = torch.cat([th_y_pred1, th_y_pred2], dim=0)

    # Update metric & compute
    output = (th_y_pred, th_y_true)
    cm.update(output)
    ignite_CM = cm.compute().cpu().numpy()

    # Compute confusion matrix with sklearn
    th_y_true = idist.all_gather(th_y_true)
    th_y_pred = idist.all_gather(th_y_pred)

    np_y_true = th_y_true.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
    np_y_pred = th_y_pred.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
    sklearn_CM = multilabel_confusion_matrix(np_y_true, np_y_pred)

    assert np.all(ignite_CM == sklearn_CM)


def _test_distrib_multiclass_images(device):
    def _test(metric_device):
        num_classes = 3
        cm = MultiLabelConfusionMatrix(num_classes=num_classes, device=metric_device)

        y_true, y_pred = get_y_true_y_pred()

        # Compute confusion matrix with sklearn
        sklearn_CM = multilabel_confusion_matrix(
            y_true.transpose((0, 2, 3, 1)).reshape(-1, 3), y_pred.transpose((0, 2, 3, 1)).reshape(-1, 3)
        )

        # Update metric
        output = (torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))
        cm.update(output)

        ignite_CM = cm.compute().cpu().numpy()

        assert np.all(ignite_CM == sklearn_CM)

        # Another test on batch of 2 images
        num_classes = 3
        cm = MultiLabelConfusionMatrix(num_classes=num_classes, device=metric_device)

        # Create a batch of two images:
        th_y_true1 = torch.tensor(y_true)
        th_y_true2 = torch.tensor(y_true.transpose(0, 1, 3, 2))
        th_y_true = torch.cat([th_y_true1, th_y_true2], dim=0)
        th_y_true = th_y_true.to(device)

        th_y_pred1 = torch.tensor(y_pred)
        th_y_pred2 = torch.tensor(y_pred.transpose(0, 1, 3, 2))
        th_y_pred = torch.cat([th_y_pred1, th_y_pred2], dim=0)
        th_y_pred = th_y_pred.to(device)

        # Update metric & compute
        output = (th_y_pred, th_y_true)
        cm.update(output)
        ignite_CM = cm.compute().cpu().numpy()

        # Compute confusion matrix with sklearn
        th_y_true = idist.all_gather(th_y_true)
        th_y_pred = idist.all_gather(th_y_pred)

        np_y_true = th_y_true.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
        np_y_pred = th_y_pred.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
        sklearn_CM = multilabel_confusion_matrix(np_y_true, np_y_pred)

        assert np.all(ignite_CM == sklearn_CM)

    _test("cpu")
    if device.type != "xla":
        _test(idist.device())


def _test_distrib_accumulator_device(device):
    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        cm = MultiLabelConfusionMatrix(num_classes=3, device=metric_device)
        assert cm._device == metric_device
        assert (
            cm.confusion_matrix.device == metric_device
        ), f"{type(cm.confusion_matrix.device)}:{cm._num_correct.device} vs {type(metric_device)}:{metric_device}"

        y_true, y_pred = get_y_true_y_pred()
        cm.update((torch.tensor(y_pred), torch.tensor(y_true)))

        assert (
            cm.confusion_matrix.device == metric_device
        ), f"{type(cm.confusion_matrix.device)}:{cm._num_correct.device} vs {type(metric_device)}:{metric_device}"


def test_simple_2D_input(available_device):
    # Tests for 2D inputs with normalized = True and False

    num_iters = 5
    num_samples = 100
    num_classes = 10
    torch.manual_seed(0)
    for _ in range(num_iters):
        target = torch.randint(0, 2, size=(num_samples, num_classes))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes))
        sklearn_CM = multilabel_confusion_matrix(target.cpu().numpy(), prediction.cpu().numpy())
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().cpu().numpy()
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=True, device=available_device)
        assert mlcm._device == torch.device(available_device)
        mlcm.update([prediction, target])
        ignite_CM_normalized = mlcm.compute().cpu().numpy()
        sklearn_CM_normalized = sklearn_CM / sklearn_CM.sum(axis=(1, 2))[:, None, None]
        assert np.allclose(sklearn_CM_normalized, ignite_CM_normalized)


def test_simple_ND_input(available_device):
    num_iters = 5
    num_samples = 100
    num_classes = 10
    torch.manual_seed(0)

    size_3d = 4
    for _ in range(num_iters):  # 3D tests
        target = torch.randint(0, 2, size=(num_samples, num_classes, size_3d))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes, size_3d))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().cpu().numpy()
        target_reshaped = target.permute(0, 2, 1).reshape(size_3d * num_samples, num_classes)
        prediction_reshaped = prediction.permute(0, 2, 1).reshape(size_3d * num_samples, num_classes)
        sklearn_CM = multilabel_confusion_matrix(target_reshaped.cpu().numpy(), prediction_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))

    size_4d = 4
    for _ in range(num_iters):  # 4D tests
        target = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().cpu().numpy()
        target_reshaped = target.permute(0, 2, 3, 1).reshape(size_3d * size_4d * num_samples, num_classes)
        prediction_reshaped = prediction.permute(0, 2, 3, 1).reshape(size_3d * size_4d * num_samples, num_classes)
        sklearn_CM = multilabel_confusion_matrix(target_reshaped.cpu().numpy(), prediction_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))

    size_5d = 4
    for _ in range(num_iters):  # 5D tests
        target = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d, size_5d))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d, size_5d))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().cpu().numpy()
        target_reshaped = target.permute(0, 2, 3, 4, 1).reshape(size_3d * size_4d * size_5d * num_samples, num_classes)
        prediction_reshaped = prediction.permute(0, 2, 3, 4, 1).reshape(
            size_3d * size_4d * size_5d * num_samples, num_classes
        )
        sklearn_CM = multilabel_confusion_matrix(target_reshaped.cpu().numpy(), prediction_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))


def test_simple_batched(available_device):
    num_iters = 5
    num_samples = 100
    num_classes = 10
    batch_size = 1
    torch.manual_seed(0)

    for _ in range(num_iters):  # 2D tests
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes))
        predictions = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes))
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])

        ignite_CM = mlcm.compute().cpu().numpy()
        targets_reshaped = targets.reshape(-1, num_classes)
        predictions_reshaped = predictions.reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.cpu().numpy(), predictions_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))

    size_3d = 4
    for _ in range(num_iters):  # 3D tests
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d))
        predictions = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d))
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])

        ignite_CM = mlcm.compute().cpu().numpy()
        targets_reshaped = targets.permute(0, 1, 3, 2).reshape(-1, num_classes)
        predictions_reshaped = predictions.permute(0, 1, 3, 2).reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.cpu().numpy(), predictions_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))

    size_4d = 4
    for _ in range(num_iters):  # 4D tests
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d))
        predictions = torch.randint(
            0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d)
        )
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])

        ignite_CM = mlcm.compute().cpu().numpy()
        targets_reshaped = targets.permute(0, 1, 3, 4, 2).reshape(-1, num_classes)
        predictions_reshaped = predictions.permute(0, 1, 3, 4, 2).reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.cpu().numpy(), predictions_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))

    size_5d = 4
    for _ in range(num_iters):  # 5D tests
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False, device=available_device)
        assert mlcm._device == torch.device(available_device)
        targets = torch.randint(
            0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d, size_5d)
        )
        predictions = torch.randint(
            0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d, size_5d)
        )
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])

        ignite_CM = mlcm.compute().cpu().numpy()
        targets_reshaped = targets.permute(0, 1, 3, 4, 5, 2).reshape(-1, num_classes)
        predictions_reshaped = predictions.permute(0, 1, 3, 4, 5, 2).reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.cpu().numpy(), predictions_reshaped.cpu().numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))


# @pytest.mark.distributed
# @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
# @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
# def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

#     device = idist.device()
#     _test_distrib_multiclass_images(device)
#     _test_distrib_accumulator_device(device)


# @pytest.mark.distributed
# @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
# def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

#     device = idist.device()
#     _test_distrib_multiclass_images(device)
#     _test_distrib_accumulator_device(device)


# @pytest.mark.distributed
# @pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
# @pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
# def test_distrib_hvd(gloo_hvd_executor):

#     device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
#     nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

#     gloo_hvd_executor(_test_distrib_multiclass_images, (device,), np=nproc, do_init=True)
#     gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)


# @pytest.mark.multinode_distributed
# @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
# @pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
# def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
#
#     device = idist.device()
#     _test_distrib_multiclass_images(device)
#     _test_distrib_accumulator_device(device)


# @pytest.mark.multinode_distributed
# @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
# @pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
# def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
#
#     device = idist.device()
#     _test_distrib_multiclass_images(device)
#     _test_distrib_accumulator_device(device)


# @pytest.mark.tpu
# @pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
# @pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
# def test_distrib_single_device_xla():
#     device = idist.device()
#     _test_distrib_multiclass_images(device)
#     _test_distrib_accumulator_device(device)


# def _test_distrib_xla_nprocs(index):
#     device = idist.device()
#     _test_distrib_multiclass_images(device)
#     _test_distrib_accumulator_device(device)


# @pytest.mark.tpu
# @pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
# @pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
# def test_distrib_xla_nprocs(xmp_executor):
#     n = int(os.environ["NUM_TPU_WORKERS"])
#     xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

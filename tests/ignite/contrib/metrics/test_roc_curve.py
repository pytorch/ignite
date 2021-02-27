import os

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_curve

import ignite.distributed as idist
from ignite.contrib.metrics.roc_auc import RocCurve
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetricWarning

torch.manual_seed(12)


def test_no_update():
    roc_curve_metric = RocCurve()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        roc_curve_metric.compute()


def test_input_types():
    roc_curve_metric = RocCurve()
    roc_curve_metric.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    roc_curve_metric.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        roc_curve_metric.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        roc_curve_metric.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        roc_curve_metric.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test_check_shape():
    roc_curve_metric = RocCurve()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        roc_curve_metric._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        roc_curve_metric._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        roc_curve_metric._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


def test_binary_input_N():
    def _test():
        roc_curve_metric = RocCurve()

        y_pred = torch.rand(10,)
        y = torch.randint(0, 2, size=(10,)).long()
        roc_curve_metric.update((y_pred, y))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        # Batched Updates
        roc_curve_metric.reset()
        y_pred = torch.rand(100,)
        y = torch.randint(0, 2, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            roc_curve_metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        roc_curve_metric.reset()
        y_pred = torch.rand(10, 1)
        y = torch.randint(0, 2, size=(10, 1)).long()
        roc_curve_metric.update((y_pred, y))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        roc_curve_metric.reset()
        y_pred = torch.rand(10, 1)
        y = torch.randint(0, 2, size=(10, 1)).long()
        roc_curve_metric.update((y_pred, y))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        # Batched Updates
        roc_curve_metric.reset()
        y_pred = torch.rand(100, 1)
        y = torch.randint(0, 2, size=(100, 1)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            roc_curve_metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

    for _ in range(10):
        _test()


def test_multilabel_inputs():
    roc_curve_metric = RocCurve()

    with pytest.raises(ValueError, match=r"multilabel-indicator format is not supported"):
        roc_curve_metric.reset()
        roc_curve_metric.update((torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long()))
        roc_curve_metric.compute()

    with pytest.raises(ValueError, match=r"multilabel-indicator format is not supported"):
        roc_curve_metric.reset()
        roc_curve_metric.update((torch.randint(0, 2, size=(10, 6)).long(), torch.randint(0, 2, size=(10, 6)).long()))
        roc_curve_metric.compute()

    with pytest.raises(ValueError, match=r"multilabel-indicator format is not supported"):
        roc_curve_metric.reset()
        roc_curve_metric.update((torch.randint(0, 2, size=(10, 8)).long(), torch.randint(0, 2, size=(10, 8)).long()))
        roc_curve_metric.compute()


def test_multiclass_inputs():
    roc_curve_metric = RocCurve()

    with pytest.raises(ValueError, match=r"Targets should be binary"):
        roc_curve_metric.update((torch.randint(0, 3, size=(10, 4)).long(), torch.randint(0, 3, size=(10, 4)).long()))

    with pytest.raises(ValueError, match=r"Targets should be binary"):
        roc_curve_metric.update((torch.randint(0, 5, size=(10, 6)).long(), torch.randint(0, 5, size=(10, 6)).long()))

    with pytest.raises(ValueError, match=r"Targets should be binary"):
        roc_curve_metric.update((torch.randint(0, 7, size=(10, 8)).long(), torch.randint(0, 7, size=(10, 8)).long()))


def test_check_compute_fn():
    y_pred = torch.zeros((8, 13))
    y_pred[:, 1] = 1
    y_true = torch.zeros_like(y_pred)
    output = (y_pred, y_true)

    em = RocCurve(check_compute_fn=True)

    em.reset()
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output)

    em = RocCurve(check_compute_fn=False)
    em.update(output)


def test_integration_binary_input_with_output_transform():
    def _test():

        y_pred = torch.rand(100)
        y = torch.randint(0, 2, size=(100,)).long()

        batch_size = 10

        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        roc_curve_metric = RocCurve(output_transform=lambda x: (x[1], x[2]))
        roc_curve_metric.attach(engine, "roc_curve")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        data = list(range(100 // batch_size))

        fpr, tpr, thresholds = engine.run(data, max_epochs=1).metrics["roc_curve"]
        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        y_pred = torch.rand(100, 1)
        y = torch.randint(0, 2, size=(100, 1)).long()

        batch_size = 10

        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        roc_curve_metric = RocCurve(output_transform=lambda x: (x[1], x[2]))
        roc_curve_metric.attach(engine, "roc_curve")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        data = list(range(100 // batch_size))

        fpr, tpr, thresholds = engine.run(data, max_epochs=1).metrics["roc_curve"]
        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

    for _ in range(10):
        _test()


def _test_distirb_binary_input_N(device):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        roc_curve_metric = RocCurve(device=metric_device)

        torch.manual_seed(10 + rank)

        y_pred = torch.rand((10,), device=device)
        y = torch.randint(0, 2, size=(10,), device=device).long()
        roc_curve_metric.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        roc_curve_metric.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.rand((100,), device=device)
        y = torch.randint(0, 2, size=(100,), device=device).long()
        roc_curve_metric.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        roc_curve_metric.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.rand((100, 1), device=device)
        y = torch.randint(0, 2, size=(100, 1), device=device).long()
        roc_curve_metric.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        # Batched Updates
        roc_curve_metric.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.rand((100,), device=device)
        y = torch.randint(0, 2, size=(100,), device=device).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            roc_curve_metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

        # Batched Updates
        roc_curve_metric.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.rand((100, 1), device=device)
        y = torch.randint(0, 2, size=(100, 1), device=device).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            roc_curve_metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(np_y, np_y_pred)
        fpr, tpr, thresholds = roc_curve_metric.compute()

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration_binary(device):

    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        s = 16
        n_classes = 2

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.rand(offset * idist.get_world_size(),).to(device)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        roc_curve_metric = RocCurve(device=metric_device)
        roc_curve_metric.attach(engine, "roc_cuve")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "roc_cuve" in engine.state.metrics

        fpr, tpr, thresholds = engine.state.metrics["roc_cuve"]
        if isinstance(fpr, torch.Tensor):
            fpr = fpr.cpu().numpy()

        if isinstance(tpr, torch.Tensor):
            tpr = tpr.cpu().numpy()

        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.cpu().numpy()

        sk_fpr, sk_tpr, sk_thresholds = roc_curve(y_true.cpu().numpy(), y_preds.cpu().numpy())

        assert np.array_equal(fpr, sk_fpr)
        assert np.array_equal(tpr, sk_tpr)
        # assert thresholds almost equal, due to numpy->torch->numpy conversion
        np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

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
    _test_distirb_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distirb_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distirb_binary_input_N, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_binary, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):

    device = torch.device("cpu")
    _test_distirb_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):

    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distirb_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():

    device = idist.device()
    _test_distirb_binary_input_N(device)
    _test_distrib_integration_binary(device)


def _test_distrib_xla_nprocs(index):

    device = idist.device()
    _test_distirb_binary_input_N(device)
    _test_distrib_integration_binary(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

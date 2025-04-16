import os

import pytest
import torch
from sklearn.metrics import fbeta_score

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics import Fbeta, Precision, Recall

torch.manual_seed(12)


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Beta should be a positive integer"):
        Fbeta(0.0)

    with pytest.raises(ValueError, match=r"Input precision metric should have average=False"):
        p = Precision(average="micro")
        Fbeta(1.0, precision=p)

    with pytest.raises(ValueError, match=r"Input recall metric should have average=False"):
        r = Recall(average="samples")
        Fbeta(1.0, recall=r)

    with pytest.raises(ValueError, match=r"If precision argument is provided, device should be None"):
        p = Precision(average=False)
        Fbeta(1.0, precision=p, device="cpu")

    with pytest.raises(ValueError, match=r"If precision argument is provided, output_transform should be None"):
        p = Precision(average=False)
        Fbeta(1.0, precision=p, output_transform=lambda x: x)

    with pytest.raises(ValueError, match=r"If recall argument is provided, device should be None"):
        r = Recall(average=False)
        Fbeta(1.0, recall=r, device="cpu")

    with pytest.raises(ValueError, match=r"If recall argument is provided, output_transform should be None"):
        r = Recall(average=False)
        Fbeta(1.0, recall=r, output_transform=lambda x: x)


def _output_transform(output):
    return output["y_pred"], output["y"]


@pytest.mark.parametrize(
    "precision_cls, recall_cls, average, output_transform",
    [
        (None, None, False, None),
        (None, None, True, None),
        (None, None, False, _output_transform),
        (None, None, True, _output_transform),
        (
            lambda device: Precision(average=False, device=device),
            lambda device: Recall(average=False, device=device),
            False,
            None,
        ),
        (
            lambda device: Precision(average=False, device=device),
            lambda device: Recall(average=False, device=device),
            True,
            None,
        ),
    ],
)
def test_integration(precision_cls, recall_cls, average, output_transform, available_device):
    if precision_cls is None:
        p = None
    else:
        p = precision_cls(available_device)
        assert p._device == torch.device(available_device)
    if recall_cls is None:
        r = None
    else:
        r = recall_cls(available_device)
        assert r._device == torch.device(available_device)

    n_iters = 10
    batch_size = 10
    n_classes = 10

    y_true = torch.arange(n_iters * batch_size, dtype=torch.long, device=available_device) % n_classes
    y_pred = 0.2 * torch.rand(n_iters * batch_size, n_classes, device=available_device)
    for i in range(n_iters * batch_size):
        if torch.rand(1) > 0.4:
            y_pred[i, y_true[i]] = 1.0
        else:
            j = torch.randint(0, n_classes, size=(1,))
            y_pred[i, j] = 0.7

    y_true_batch_values = iter(y_true.reshape(n_iters, batch_size))
    y_pred_batch_values = iter(y_pred.reshape(n_iters, batch_size, n_classes))

    def update_fn(engine, batch):
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        if output_transform is not None:
            return {"y_pred": y_pred_batch, "y": y_true_batch}
        return y_pred_batch, y_true_batch

    evaluator = Engine(update_fn)

    device = None if p is not None and r is not None else available_device
    f2 = Fbeta(beta=2.0, average=average, precision=p, recall=r, output_transform=output_transform, device=device)

    f2.attach(evaluator, "f2")

    data = list(range(n_iters))
    state = evaluator.run(data, max_epochs=1)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = torch.argmax(y_pred, dim=-1).cpu().numpy()
    f2_true = fbeta_score(y_true_np, y_pred_np, average="macro" if average else None, beta=2.0)

    assert f2_true == pytest.approx(state.metrics["f2"])


def _test_distrib_integration(device):
    rank = idist.get_rank()

    def _test(p, r, average, n_epochs, metric_device):
        n_iters = 60
        batch_size = 16
        n_classes = 7

        torch.manual_seed(12 + rank)

        y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, :],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        fbeta = Fbeta(beta=2.5, average=average, device=metric_device)
        fbeta.attach(engine, "f2.5")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "f2.5" in engine.state.metrics
        res = engine.state.metrics["f2.5"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = fbeta_score(
            y_true.cpu().numpy(),
            torch.argmax(y_preds, dim=1).cpu().numpy(),
            beta=2.5,
            average="macro" if average else None,
        )

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(None, None, average=True, n_epochs=1, metric_device=metric_device)
        _test(None, None, average=True, n_epochs=2, metric_device=metric_device)
        precision = Precision(average=False, device=metric_device)
        recall = Recall(average=False, device=metric_device)
        _test(precision, recall, average=False, n_epochs=1, metric_device=metric_device)
        _test(precision, recall, average=False, n_epochs=2, metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)


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
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)

import os

import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import TopKCategoricalAccuracy


def test_zero_div():
    acc = TopKCategoricalAccuracy(2)
    with pytest.raises(
        NotComputableError, match=r"TopKCategoricalAccuracy must have at least one example before it can be computed"
    ):
        acc.compute()


def test_compute(available_device):
    acc = TopKCategoricalAccuracy(2, device=available_device)
    assert acc._device == torch.device(available_device)

    y_pred = torch.FloatTensor([[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2]])
    y = torch.ones(2).long()
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.FloatTensor([[0.4, 0.8, 0.2, 0.6], [0.8, 0.6, 0.4, 0.2]])
    y = torch.ones(2).long()
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 1.0


def top_k_accuracy(y_true, y_pred, k=5, normalize=True):
    import numpy as np

    # Taken from
    # https://github.com/scikit-learn/scikit-learn/blob/4685cb5c50629aba4429f6701585f82fc3eee5f7/
    # sklearn/metrics/classification.py#L187
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0.0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1 :]:
            counter += 1.0
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter


def _test_distrib_integration(device):
    from ignite.engine import Engine

    def _test(n_epochs, metric_device):
        n_iters = 100
        batch_size = 16
        n_classes = 10

        y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, :],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        k = 5
        acc = TopKCategoricalAccuracy(k=k, device=metric_device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "acc" in engine.state.metrics
        res = engine.state.metrics["acc"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = top_k_accuracy(y_true.cpu().numpy(), y_preds.cpu().numpy(), k=k)

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    rank = idist.get_rank()
    for i in range(3):
        torch.manual_seed(12 + rank + i)
        for metric_device in metric_devices:
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


def _test_distrib_accumulator_device(device):
    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        acc = TopKCategoricalAccuracy(2, device=metric_device)
        assert acc._device == metric_device
        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        y_pred = torch.tensor([[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2]])
        y = torch.ones(2).long()
        acc.update((y_pred, y))

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import CosineSimilarity


def test_zero_sample():
    cos_sim = CosineSimilarity()
    with pytest.raises(
        NotComputableError, match=r"CosineSimilarity must have at least one example before it can be computed"
    ):
        cos_sim.compute()


@pytest.fixture(params=[item for item in range(4)])
def test_case(request):
    return [
        (torch.randn((100, 50)), torch.randn((100, 50)), 10 ** np.random.uniform(-8, 0), 1),
        (
            torch.normal(1.0, 2.0, size=(100, 10)),
            torch.normal(3.0, 4.0, size=(100, 10)),
            10 ** np.random.uniform(-8, 0),
            1,
        ),
        # updated batches
        (torch.rand((100, 128)), torch.rand((100, 128)), 10 ** np.random.uniform(-8, 0), 16),
        (
            torch.normal(0.0, 5.0, size=(100, 30)),
            torch.normal(5.0, 1.0, size=(100, 30)),
            10 ** np.random.uniform(-8, 0),
            16,
        ),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case):
    y_pred, y, eps, batch_size = test_case

    cos = CosineSimilarity(eps=eps)

    cos.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            cos.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        cos.update((y_pred, y))

    np_y = y.numpy()
    np_y_pred = y_pred.numpy()

    np_y_norm = np.clip(np.linalg.norm(np_y, axis=1, keepdims=True), eps, None)
    np_y_pred_norm = np.clip(np.linalg.norm(np_y_pred, axis=1, keepdims=True), eps, None)
    np_res = np.sum((np_y / np_y_norm) * (np_y_pred / np_y_pred_norm), axis=1)
    np_res = np.mean(np_res)

    assert isinstance(cos.compute(), float)
    assert pytest.approx(np_res, rel=2e-5) == cos.compute()


def _test_distrib_integration(device, tol=2e-5):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12 + rank)

    def _test(metric_device):
        n_iters = 100
        batch_size = 10
        n_dims = 100

        y_true = torch.randn((n_iters * batch_size, n_dims), dtype=torch.float).to(device)
        y_preds = torch.normal(2.0, 3.0, size=(n_iters * batch_size, n_dims), dtype=torch.float).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        m = CosineSimilarity(device=metric_device)
        m.attach(engine, "cosine_similarity")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "cosine_similarity" in engine.state.metrics
        res = engine.state.metrics["cosine_similarity"]

        y_true_np = y_true.cpu().numpy()
        y_preds_np = y_preds.cpu().numpy()
        y_true_norm = np.clip(np.linalg.norm(y_true_np, axis=1, keepdims=True), 1e-8, None)
        y_preds_norm = np.clip(np.linalg.norm(y_preds, axis=1, keepdims=True), 1e-8, None)
        true_res = np.sum((y_true_np / y_true_norm) * (y_preds_np / y_preds_norm), axis=1)
        true_res = np.mean(true_res)

        assert pytest.approx(res, rel=tol) == true_res

    _test("cpu")
    if device.type != "xla":
        _test(idist.device())


def _test_distrib_accumulator_device(device):
    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        device = torch.device(device)
        cos = CosineSimilarity(device=metric_device)

        for dev in [cos._device, cos._sum_of_cos_similarities.device]:
            assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

        y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
        y = torch.ones(2, 2, dtype=torch.float)
        cos.update((y_pred, y))

        for dev in [cos._device, cos._sum_of_cos_similarities.device]:
            assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"


def test_accumulator_detached():
    cos = CosineSimilarity()

    y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
    y = torch.ones(2, 2, dtype=torch.float)
    cos.update((y_pred, y))

    assert not cos._sum_of_cos_similarities.requires_grad


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
    _test_distrib_integration(device, tol=1e-4)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device, tol=1e-4)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)

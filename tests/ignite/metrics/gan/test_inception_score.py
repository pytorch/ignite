import os

import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.gan.inception_score import InceptionScore


def calculate_inception_score(p_yx):
    p_y = torch.unsqueeze(p_yx.mean(axis=0), 0)
    kl_d = torch.kl_div(torch.log(p_y), p_yx)

    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = torch.mean(sum_kl_d)

    is_score = torch.exp(avg_kl_d)

    return is_score


def test_inception_score(available_device):
    p_yx = torch.rand(20, 10)
    m = InceptionScore(
        num_features=10,
        feature_extractor=torch.nn.Identity(),
        device=available_device,
    )
    m.update(p_yx)
    assert pytest.approx(calculate_inception_score(p_yx)) == m.compute()

    p_yx = torch.rand(20, 3, 299, 299)
    m = InceptionScore(device=available_device)
    assert m._device == torch.device(available_device)
    m.update(p_yx)
    assert isinstance(m.compute(), float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_device_mismatch_cuda():
    p_yx = torch.rand(20, 10).to("cpu")
    m = InceptionScore(num_features=10, feature_extractor=torch.nn.Identity().to("cpu"), device="cuda")
    m.update(p_yx)
    assert pytest.approx(calculate_inception_score(p_yx)) == m.compute()


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Argument num_features must be greater to zero, got:"):
        InceptionScore(num_features=-1, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 0))

    with pytest.raises(ValueError, match=r"feature_extractor output must be a tensor of dim 2, got: 1"):
        InceptionScore(num_features=1000, feature_extractor=torch.nn.Identity()).update(torch.rand(3))

    with pytest.raises(ValueError, match=r"Batch size should be greater than one, got: 0"):
        InceptionScore(num_features=1000, feature_extractor=torch.nn.Identity()).update(torch.rand(0, 0))

    with pytest.raises(ValueError, match=r"num_features returned by feature_extractor should be 1000, got: 0"):
        InceptionScore(num_features=1000, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 0))

    with pytest.raises(
        NotComputableError, match=r"InceptionScore must have at least one example before it can be computed."
    ):
        InceptionScore(num_features=1000, feature_extractor=torch.nn.Identity()).compute()

    with pytest.raises(ValueError, match=r"Argument num_features must be provided, if feature_extractor is specified."):
        InceptionScore(feature_extractor=torch.nn.Identity())


def _test_distrib_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):
        n_iters = 60
        s = 16
        offset = n_iters * s

        n_probabilities = 10
        y = torch.rand(offset * idist.get_world_size(), n_probabilities)

        def update(_, i):
            return y[i * s + rank * offset : (i + 1) * s + rank * offset, :]

        engine = Engine(update)
        m = InceptionScore(num_features=n_probabilities, feature_extractor=torch.nn.Identity(), device=metric_device)
        m.attach(engine, "InceptionScore")

        engine.run(data=list(range(n_iters)), max_epochs=1)

        assert "InceptionScore" in engine.state.metrics

        assert pytest.approx(calculate_inception_score(y), rel=1e-5) == m.compute()

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = torch.device(f"cuda:{local_rank}")
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_integration(device)


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

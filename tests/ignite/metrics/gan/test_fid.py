import os
import re
from unittest.mock import patch

import pytest
import pytorch_fid.fid_score as pytorch_fid_score
import scipy
import torch
from numpy import cov

import ignite.distributed as idist
from ignite.metrics.gan.fid import FID, fid_score


@pytest.fixture()
def mock_no_scipy():
    with patch.dict("sys.modules", {"scipy": None}):
        yield scipy


def test_no_scipy(mock_no_scipy):
    with pytest.raises(ModuleNotFoundError, match=r"This module requires scipy to be installed."):
        FID()

    with pytest.raises(ModuleNotFoundError, match=r"fid_score requires scipy to be installed."):
        fid_score(0, 0, 0, 0)


@pytest.fixture()
def mock_no_numpy():
    with patch.dict("sys.modules", {"numpy": None}):
        yield scipy


def test_no_numpy(mock_no_numpy):
    with pytest.raises(ModuleNotFoundError, match=r"This module requires numpy to be installed."):
        FID()

    with pytest.raises(ModuleNotFoundError, match=r"fid_score requires numpy to be installed."):
        fid_score(0, 0, 0, 0)


def test_fid_function():
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)

    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)

    sigma1 = torch.tensor(sigma1, dtype=torch.float64)
    sigma2 = torch.tensor(sigma2, dtype=torch.float64)
    assert pytest.approx(fid_score(mu1, mu2, sigma1, sigma2), rel=1e-5) == pytorch_fid_score.calculate_frechet_distance(
        mu1, sigma1, mu2, sigma2
    )


def test_compute_fid_from_features(available_device):
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)

    fid_scorer = FID(
        num_features=10,
        feature_extractor=torch.nn.Identity(),
        device=available_device,
    )
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)

    tol = 5e-4 if available_device == "mps" else 1e-5
    assert (
        pytest.approx(pytorch_fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2), abs=tol)
        == fid_scorer.compute()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_device_mismatch_cuda():
    train_samples, test_samples = torch.rand(10, 10).to("cpu"), torch.rand(10, 10).to("cpu")

    fid_scorer = FID(num_features=10, feature_extractor=torch.nn.Identity().to("cpu"), device="cuda")
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)

    assert (
        pytest.approx(pytorch_fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2), rel=1e-4)
        == fid_scorer.compute()
    )


def test_compute_fid_sqrtm():
    mu1 = torch.tensor([0, 0])
    mu2 = torch.tensor([0, 0])

    sigma1 = torch.tensor([[-1, 1], [1, 1]], dtype=torch.float64)
    sigma2 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)

    with pytest.raises(ValueError, match=r"Imaginary component "):
        fid_score(mu1, mu2, sigma1, sigma2)

    sigma1 = torch.ones((2, 2), dtype=torch.float64) * torch.finfo(torch.float64).max
    sigma2 = torch.tensor([[1, 0.5], [0, 0.5]], dtype=torch.float64)

    assert torch.isinf(torch.tensor(fid_score(mu1, mu2, sigma1, sigma2)))


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Argument num_features must be greater to zero"):
        FID(num_features=-1, feature_extractor=torch.nn.Identity())

    with pytest.raises(ValueError, match=r"feature_extractor output must be a tensor of dim 2, got: 1"):
        FID(num_features=1, feature_extractor=torch.nn.Identity()).update(torch.tensor([[], []]))

    with pytest.raises(ValueError, match=r"Batch size should be greater than one, got: 0"):
        FID(num_features=1, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 0, 0))

    with pytest.raises(ValueError, match=r"num_features returned by feature_extractor should be 1, got: 0"):
        FID(num_features=1, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 2, 0))

    err_str = (
        "Number of Training Features and Testing Features should be equal (torch.Size([9, 2]) != torch.Size([5, 2]))"
    )
    with pytest.raises(ValueError, match=re.escape(err_str)):
        FID(num_features=2, feature_extractor=torch.nn.Identity()).update((torch.rand(9, 2), torch.rand(5, 2)))

    with pytest.raises(TypeError, match=r"Argument feature_extractor must be of type torch.nn.Module"):
        FID(num_features=1, feature_extractor=lambda x: x)

    with pytest.raises(ValueError, match=r"Argument num_features must be provided, if feature_extractor is specified."):
        FID(feature_extractor=torch.nn.Identity())


def test_statistics(available_device):
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)
    fid_scorer = FID(
        num_features=10,
        feature_extractor=torch.nn.Identity(),
        device=available_device,
    )
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1 = train_samples.mean(axis=0, dtype=torch.float64)
    sigma1 = torch.tensor(cov(train_samples, rowvar=False), dtype=torch.float64)
    mu2 = test_samples.mean(axis=0, dtype=torch.float64)
    sigma2 = torch.tensor(cov(test_samples, rowvar=False), dtype=torch.float64)

    fid_mu1 = fid_scorer._train_total / fid_scorer._num_examples
    fid_sigma1 = fid_scorer._get_covariance(fid_scorer._train_sigma, fid_scorer._train_total)

    fid_mu2 = fid_scorer._test_total / fid_scorer._num_examples
    fid_sigma2 = fid_scorer._get_covariance(fid_scorer._test_sigma, fid_scorer._test_total)

    assert torch.allclose(mu1, fid_mu1.cpu().to(dtype=mu1.dtype))
    assert torch.allclose(sigma1, fid_sigma1.cpu().to(dtype=sigma1.dtype), rtol=1e-04, atol=1e-04)

    assert torch.allclose(mu2, fid_mu2.cpu().to(dtype=mu2.dtype))
    assert torch.allclose(sigma2, fid_sigma2.cpu().to(dtype=mu2.dtype), rtol=1e-04, atol=1e-04)


def _test_distrib_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):
        n_iters = 60
        s = 16
        offset = n_iters * s

        n_features = 10

        y_pred = torch.rand(offset * idist.get_world_size(), n_features)
        y_true = torch.rand(offset * idist.get_world_size(), n_features)

        def update(_, i):
            return (
                y_pred[i * s + rank * offset : (i + 1) * s + rank * offset, :],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset, :],
            )

        engine = Engine(update)
        m = FID(num_features=n_features, feature_extractor=torch.nn.Identity(), device=metric_device)
        m.attach(engine, "fid")

        engine.run(data=list(range(n_iters)), max_epochs=1)

        assert "fid" in engine.state.metrics

        evaluator = pytorch_fid_score.calculate_frechet_distance
        mu1, sigma1 = y_pred.mean(axis=0).to("cpu"), cov(y_pred.to("cpu"), rowvar=False)
        mu2, sigma2 = y_true.mean(axis=0).to("cpu"), cov(y_true.to("cpu"), rowvar=False)
        assert pytest.approx(evaluator(mu1, sigma1, mu2, sigma2), rel=1e-5) == m.compute()

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

import os
from unittest.mock import patch

import pytest
import pytorch_fid.fid_score as pytorch_fid_score
import scipy
import torch
import torchvision
from numpy import cov

import ignite.distributed as idist
from ignite.metrics.gan.fid import FID, InceptionExtractor, fid_score


class DummyInceptionExtractor(InceptionExtractor):
    def __init__(self) -> None:
        try:
            from torchvision import models
        except ImportError:
            raise RuntimeError("This module requires torchvision to be installed.")
        self.model = models.inception_v3(pretrained=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()


@pytest.fixture()
def mock_no_torchvision():
    with patch.dict("sys.modules", {"torchvision": None}):
        yield torchvision


def test_no_torchvision(mock_no_torchvision):
    with pytest.raises(RuntimeError, match=r"This module requires torchvision to be installed."):
        FID()


@pytest.fixture()
def mock_no_scipy():
    with patch.dict("sys.modules", {"scipy": None}):
        yield scipy


def test_no_scipy(mock_no_scipy):
    with pytest.raises(RuntimeError, match=r"This module requires scipy to be installed."):
        FID()


def test_fid_function():
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)

    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)

    sigma1 = torch.tensor(sigma1, dtype=torch.float64)
    sigma2 = torch.tensor(sigma2, dtype=torch.float64)
    assert pytest.approx(fid_score(mu1, mu2, sigma1, sigma2)) == pytorch_fid_score.calculate_frechet_distance(
        mu1, sigma1, mu2, sigma2
    )


def test_compute_fid_from_features():
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)

    fid_scorer = FID(num_features=10, feature_extractor=lambda x: x)
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)

    assert pytest.approx(pytorch_fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)) == fid_scorer.compute()


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"num of features must be greater to zero"):
        FID(num_features=-1, feature_extractor=lambda x: x)
    with pytest.raises(ValueError, match=r"Features must be a tensor of dim 2 \(got: 1\)"):
        FID(num_features=1, feature_extractor=lambda x: x).update(torch.Tensor([[], []]))
    with pytest.raises(ValueError, match=r"Batch size should be greater than one \(got: 0\)"):
        FID(num_features=1, feature_extractor=lambda x: x).update(torch.rand(2, 0, 0))
    with pytest.raises(ValueError, match=r"Feature size should be greater than one \(got: 0\)"):
        FID(num_features=1, feature_extractor=lambda x: x).update(torch.rand(2, 2, 0))


def test_statistics():
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)
    fid_scorer = FID(num_features=10, feature_extractor=lambda x: x)
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), torch.tensor(cov(train_samples, rowvar=False))
    mu2, sigma2 = test_samples.mean(axis=0), torch.tensor(cov(test_samples, rowvar=False))

    fid_mu1 = fid_scorer._train_total / fid_scorer._num_examples
    fid_sigma1 = fid_scorer.get_covariance(fid_scorer._train_sigma, fid_scorer._train_total)

    fid_mu2 = fid_scorer._test_total / fid_scorer._num_examples
    fid_sigma2 = fid_scorer.get_covariance(fid_scorer._test_sigma, fid_scorer._test_total)

    assert torch.isclose(mu1.double(), fid_mu1).all()
    for cov1, cov2 in zip(sigma1, fid_sigma1):
        assert torch.isclose(cov1.double(), cov2, rtol=1e-04, atol=1e-04).all()
    assert torch.isclose(mu2.double(), fid_mu2).all()
    for cov1, cov2 in zip(sigma2, fid_sigma2):
        assert torch.isclose(cov1.double(), cov2, rtol=1e-04, atol=1e-04).all()


def test_inception_extractor_wrong_inputs():
    with pytest.raises(ValueError, match=r"Inputs should be a tensor of dim 4"):
        DummyInceptionExtractor()(torch.rand(2))
    with pytest.raises(ValueError, match=r"Inputs should be a tensor with 3 channels"):
        DummyInceptionExtractor()(torch.rand(2, 2, 2, 0))


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
        m = FID(num_features=n_features, feature_extractor=lambda x: x, device=metric_device)
        m.attach(engine, "fid")

        engine.run(data=list(range(n_iters)), max_epochs=1)

        assert "fid" in engine.state.metrics

        evaluator = pytorch_fid_score.calculate_frechet_distance
        mu1, sigma1 = y_pred.mean(axis=0).to("cpu"), cov(y_pred.to("cpu"), rowvar=False)
        mu2, sigma2 = y_true.mean(axis=0).to("cpu"), cov(y_true.to("cpu"), rowvar=False)
        assert pytest.approx(evaluator(mu1, sigma1, mu2, sigma2)) == m.compute()

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

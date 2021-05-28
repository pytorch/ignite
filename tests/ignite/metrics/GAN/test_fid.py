import os

import pytest
import pytorch_fid.fid_score as fid_score
import torch
from numpy import cov
from torch import nn
from torchvision import models

import ignite.distributed as idist
from ignite.metrics.GAN.fid import FID


@pytest.mark.parametrize(
    "train_samples, test_samples", [(torch.rand(10, 2048), torch.rand(10, 2048))],
)
def test_compute_fid_from_features(train_samples, test_samples):
    fid_scorer = FID(mode="features")
    fid_scorer.update([train_samples, test_samples])
    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)
    assert pytest.approx(fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)) == fid_scorer.compute()


@pytest.mark.parametrize(
    "train_samples, test_samples", [(torch.rand(10, 3, 299, 299), torch.rand(10, 3, 299, 299))],
)
def test_compute_fid_from_images(train_samples, test_samples):
    model = models.inception_v3(init_weights=False)
    model.fc = nn.Sequential()
    model.eval()
    fid_scorer = FID(model=model, mode="images")
    fid_scorer.update([train_samples, test_samples])

    train_samples = model(train_samples)[0].detach()
    test_samples = model(test_samples)[0].detach()
    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)
    assert pytest.approx(fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)) == fid_scorer.compute()


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Please enter a valid mode."):
        FID(mode="unknown").update([[], []])
    with pytest.raises(ValueError, match=r"Training Features must be passed as \(num_samples,feature_size\)."):
        FID(mode="features").update([torch.rand(1, 2, 3), []])
    with pytest.raises(ValueError, match=r"Testing Features must be passed as \(num_samples,feature_size\)."):
        FID(mode="features").update([[], torch.rand(1, 2, 3)])
    with pytest.raises(ValueError, match=r"Number of Training Features and Testing Features should be equal."):
        FID(mode="features").update([torch.rand(1, 2), torch.rand(2, 3)])
    with pytest.raises(ValueError, match=r"Training images must be passed as \(num_samples,image\)."):
        FID(mode="images").update([torch.rand(1, 2), []])
    with pytest.raises(ValueError, match=r"Testing images must be passed as \(num_samples,image\)."):
        FID(mode="images").update([[], torch.rand(1, 2)])
    with pytest.raises(ValueError, match=r"Train and Test images must be of equal dimensions."):
        FID(mode="images").update([torch.rand(1, 2, 3), torch.rand(2, 3, 4)])


@pytest.mark.parametrize(
    "train_samples, test_samples", [(torch.rand(10, 2048), torch.rand(10, 2048)),],
)
def test_statistics(train_samples, test_samples):
    fid_scorer = FID(mode="features")
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), torch.tensor(cov(train_samples, rowvar=False))
    mu2, sigma2 = test_samples.mean(axis=0), torch.tensor(cov(test_samples, rowvar=False))

    fid_mu1 = fid_scorer._train_record.mean
    fid_sigma1 = fid_scorer._train_record.get_covariance()

    fid_mu2 = fid_scorer._test_record.mean
    fid_sigma2 = fid_scorer._test_record.get_covariance()

    assert torch.isclose(mu1.double(), fid_mu1).all()
    for cov1, cov2 in zip(sigma1, fid_sigma1):
        assert torch.isclose(cov1.double(), cov2).all()
    assert torch.isclose(mu2.double(), fid_mu2).all()
    for cov1, cov2 in zip(sigma2, fid_sigma2):
        assert torch.isclose(cov1.double(), cov2).all()


def _test_distrib_integration(device):

    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)
    size = 10

    data = []
    for i in range(size):
        data += [(torch.rand(10, 2048), torch.rand(10, 2048))]

    def update(_, i):
        train, test = data[i + size * rank]
        return (train, test)

    def _test(metric_device):
        engine = Engine(update)
        m = FID(mode="features")
        m.attach(engine, "fid")

        engine.run(data=list(range(size)), max_epochs=1)

        assert "fid" in engine.state.metrics

        evaluator = fid_score.calculate_frechet_distance
        train, test = data[0]
        for train_samples, test_samples in data[1:]:
            train = torch.cat(train, train_samples)
            test = torch.cat(test, test_samples)
        mu1, sigma1 = train.mean(axis=0), cov(train, rowvar=False)
        mu2, sigma2 = test.mean(axis=0), cov(test, rowvar=False)
        assert pytest.approx(fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)) == m.compute()

    _test("cpu")

    if device.type != "xla":
        _test(idist.device())


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

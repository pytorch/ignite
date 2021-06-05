import os

import pytest
import pytorch_fid.fid_score as pytorch_fid_score
import torch
from numpy import cov

import ignite.distributed as idist
from ignite.metrics.gan.fid import FID, InceptionExtractor, fid_score


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

    fid_scorer = FID(num_features=10)
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), cov(train_samples, rowvar=False)
    mu2, sigma2 = test_samples.mean(axis=0), cov(test_samples, rowvar=False)

    assert pytest.approx(pytorch_fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)) == fid_scorer.compute()


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"num of features must be greater to zero"):
        FID(num_features=-1)
    with pytest.raises(ValueError, match=r"Features must be a tensor of dim 2 \(got: 1\)"):
        FID(num_features=1).update(torch.Tensor([[], []]))
    with pytest.raises(ValueError, match=r"Batch size should be greater than one \(got: 0\)"):
        FID(num_features=1).update(torch.rand(2, 0, 0))
    with pytest.raises(ValueError, match=r"Feature size should be greater than one \(got: 0\)"):
        FID(num_features=1).update(torch.rand(2, 2, 0))


def test_statistics():
    train_samples, test_samples = torch.rand(10, 10), torch.rand(10, 10)
    fid_scorer = FID(num_features=10)
    fid_scorer.update([train_samples[:5], test_samples[:5]])
    fid_scorer.update([train_samples[5:], test_samples[5:]])

    mu1, sigma1 = train_samples.mean(axis=0), torch.tensor(cov(train_samples, rowvar=False))
    mu2, sigma2 = test_samples.mean(axis=0), torch.tensor(cov(test_samples, rowvar=False))

    fid_mu1 = fid_scorer._train_total / fid_scorer._num_examples
    fid_sigma1 = fid_scorer._train_sigma / (fid_scorer._num_examples - 1)

    fid_mu2 = fid_scorer._test_total / fid_scorer._num_examples
    fid_sigma2 = fid_scorer._test_sigma / (fid_scorer._num_examples - 1)

    assert torch.isclose(mu1.double(), fid_mu1).all()
    for cov1, cov2 in zip(sigma1, fid_sigma1):
        assert torch.isclose(cov1.double(), cov2).all()
    assert torch.isclose(mu2.double(), fid_mu2).all()
    for cov1, cov2 in zip(sigma2, fid_sigma2):
        assert torch.isclose(cov1.double(), cov2).all()


def test_inception_extractor_wrong_inputs():
    with pytest.raises(ValueError, match=r"Images should be of size 3x299x299 \(got torch.Size\(\[2, 2, 2, 0\]\)\)"):
        InceptionExtractor()(torch.rand(2, 2, 2, 0))


def _test_distrib_integration(device):

    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(metric_device):
        size = 10
        train_data = torch.rand(size, 10, 2048).to(device)
        test_data = torch.rand(size, 10, 2048).to(device)

        def update(_, i):
            return (train_data[i], test_data[i])

        engine = Engine(update)
        m = FID(num_features=2048, device=metric_device)
        m.attach(engine, "fid")

        engine.run(data=list(range(size)), max_epochs=1)

        assert "fid" in engine.state.metrics

        evaluator = pytorch_fid_score.calculate_frechet_distance
        train, test = train_data[0], test_data[0]
        for train_samples, test_samples in zip(train_data[1:], test_data[1:]):
            train = torch.cat((train, train_samples))
            test = torch.cat((test, test_samples))
        mu1, sigma1 = train.mean(axis=0).to("cpu"), cov(train.to("cpu"), rowvar=False)
        mu2, sigma2 = test.mean(axis=0).to("cpu"), cov(test.to("cpu"), rowvar=False)
        assert pytest.approx(evaluator(mu1, sigma1, mu2, sigma2)) == m.compute()

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for _ in range(2):
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

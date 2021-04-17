import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.gan.inception_score import InceptionScore

torch.manual_seed(42)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0]

    def __len__(self):
        return len(self.dataset)


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Argument inception_model should be instance of nn.Module"):
        InceptionScore(inception_model="inceptionv3")
    with pytest.raises(NotComputableError):
        InceptionScore().compute()


def _test_distrib_integration(device):
    def _test_score(metric_device):
        from torchvision import models
        from torchvision.datasets import FakeData

        from ignite.engine import Engine

        inception_model = models.inception_v3(pretrained=True).eval().to(metric_device)
        dataset = FakeData(size=64, transform=transforms.Compose([transforms.Resize(299), transforms.ToTensor()]))
        dataset = IgnoreLabelDataset(dataset)
        dataloader = idist.auto_dataloader(dataset, batch_size=32)

        def np_compute(dataloader, splits):
            def get_pred(x):
                x = inception_model(x)
                return F.softmax(x).detach().cpu().numpy()

            preds = []
            for i, batch in enumerate(dataloader):
                preds.append(get_pred(batch))

            split_scores = np.zeros((splits,))
            preds = np.vstack(preds)
            N = preds.shape[0]
            for i in range(splits):
                part = preds[i * N // splits : (i + 1) * N // splits, :]
                kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
                kl = np.mean(np.sum(kl, axis=1))
                split_scores[i] = np.exp(kl)

            return np.mean(split_scores)

        def process_func(engine, batch):
            return batch

        inception_score = InceptionScore(device=metric_device)
        test_engine = Engine(process_func)
        inception_score.attach(test_engine, "score")
        np_is = np_compute(dataloader, 10)
        state = test_engine.run(dataloader)
        computed_is = state.metrics["score"]
        assert pytest.approx(computed_is, 0.1) == np_is

    _test_score("cpu")
    if device.type != "xla":
        _test_score(idist.device())


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

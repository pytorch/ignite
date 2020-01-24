import os
import torch

from ignite.metrics import EpochMetric

import pytest


def test_epoch_metric():

    # Wrong compute function
    with pytest.raises(TypeError):
        EpochMetric(12345)

    def compute_fn(y_preds, y_targets):
        return 0.0

    em = EpochMetric(compute_fn)

    # Wrong input dims
    with pytest.raises(ValueError):
        output = (torch.tensor(0), torch.tensor(0))
        em.update(output)

    # Wrong input dims
    with pytest.raises(ValueError):
        output = (torch.rand(4, 3), torch.rand(4, 3, 1))
        em.update(output)

    # Wrong input dims
    with pytest.raises(ValueError):
        output = (torch.rand(4, 3, 1), torch.rand(4, 3))
        em.update(output)

    # Target is not binary
    with pytest.raises(ValueError):
        output = (torch.rand(4, 3), torch.randint(0, 5, size=(4, 3)))
        em.update(output)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    assert em._predictions.device.type == 'cpu' and em._targets.device.type == 'cpu'
    assert torch.equal(em._predictions[:4, :], output1[0])
    assert torch.equal(em._predictions[4:, :], output2[0])
    assert torch.equal(em._targets[:4, :], output1[1])
    assert torch.equal(em._targets[4:, :], output2[1])
    assert em.compute() == 0.0

    # test when y and y_pred are (batch_size, 1) that are squeezed to (batch_size, )
    em.reset()
    output1 = (torch.rand(4, 1), torch.randint(0, 2, size=(4, 1), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 1), torch.randint(0, 2, size=(4, 1), dtype=torch.long))
    em.update(output2)

    assert em._predictions.device.type == 'cpu' and em._targets.device.type == 'cpu'
    assert torch.equal(em._predictions[:4], output1[0][:, 0])
    assert torch.equal(em._predictions[4:], output2[0][:, 0])
    assert torch.equal(em._targets[:4], output1[1][:, 0])
    assert torch.equal(em._targets[4:], output2[1][:, 0])
    assert em.compute() == 0.0


def test_mse_epoch_metric():

    def compute_fn(y_preds, y_targets):
        return torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2)).item()

    em = EpochMetric(compute_fn)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)
    output3 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output3)

    preds = torch.cat([output1[0], output2[0], output3[0]], dim=0)
    targets = torch.cat([output1[1], output2[1], output3[1]], dim=0)

    result = em.compute()
    assert result == compute_fn(preds, targets)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)
    output3 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output3)

    preds = torch.cat([output1[0], output2[0], output3[0]], dim=0)
    targets = torch.cat([output1[1], output2[1], output3[1]], dim=0)

    result = em.compute()
    assert result == compute_fn(preds, targets)


def test_bad_compute_fn():

    def compute_fn(y_preds, y_targets):
        # Following will raise the error:
        # The size of tensor a (3) must match the size of tensor b (4)
        # at non-singleton dimension 1
        return torch.mean(y_preds - y_targets).item()

    em = EpochMetric(compute_fn)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 4), dtype=torch.long))
    with pytest.warns(RuntimeWarning):
        em.update(output1)


def _test_warning():

    def compute_fn(y_preds, y_targets):
        return 0.0

    with pytest.warns(RuntimeWarning, match="EpochMetric class does not support distributed setting"):
        EpochMetric(compute_fn)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    _test_warning()

    # Perform some ops otherwise, next tests fail
    import torch.distributed as dist

    device = "cuda:{}".format(local_rank)

    def _gather(y):
        output = [torch.zeros_like(y) for i in range(dist.get_world_size())]
        dist.all_gather(output, y)
        y = torch.cat(output, dim=0)
        return y

    y = torch.rand(10, 12, device=device)
    y = _gather(y)
    assert isinstance(y, torch.Tensor)


@pytest.mark.distributed
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):

    _test_warning()

    # Perform some ops otherwise, next tests fail
    import torch.distributed as dist

    device = "cpu"

    def _gather(y):
        output = [torch.zeros_like(y) for i in range(dist.get_world_size())]
        dist.all_gather(output, y)
        y = torch.cat(output, dim=0)
        return y

    y = torch.rand(10, 12, device=device)
    y = _gather(y)
    assert isinstance(y, torch.Tensor)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):

    _test_warning()


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    _test_warning()

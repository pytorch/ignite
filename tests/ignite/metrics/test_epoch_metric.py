import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics import EpochMetric
from ignite.metrics.epoch_metric import EpochMetricWarning, NotComputableError


def test_epoch_metric_wrong_setup_or_input():
    # Wrong compute function
    with pytest.raises(TypeError, match=r"Argument compute_fn should be callable."):
        EpochMetric(12345)

    def compute_fn(y_preds, y_targets):
        return 0.0

    em = EpochMetric(compute_fn)

    # Wrong input dims
    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        output = (torch.tensor(0), torch.tensor(0))
        em.update(output)

    # Wrong input dims
    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        output = (torch.rand(4, 3), torch.rand(4, 3, 1))
        em.update(output)

    # Wrong input dims
    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        output = (torch.rand(4, 3, 1), torch.rand(4, 3))
        em.update(output)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        output2 = (torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3)))
        em.update(output2)

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32))
        em.update(output2)

    with pytest.raises(
        NotComputableError, match="EpochMetric must have at least one example before it can be computed"
    ):
        em = EpochMetric(compute_fn)
        em.compute()


def test_epoch_metric(available_device):
    def compute_fn(y_preds, y_targets):
        return 0.0

    em = EpochMetric(compute_fn, device=available_device)
    assert em._device == torch.device(available_device)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    assert all([t.device.type == available_device for t in em._predictions + em._targets])
    assert torch.equal(em._predictions[0].cpu(), output1[0].cpu())
    assert torch.equal(em._predictions[1].cpu(), output2[0].cpu())
    assert torch.equal(em._targets[0].cpu(), output1[1].cpu())
    assert torch.equal(em._targets[1].cpu(), output2[1].cpu())
    assert em.compute() == 0.0

    # test when y and y_pred are (batch_size, 1) that are squeezed to (batch_size, )
    em.reset()
    output1 = (torch.rand(4, 1), torch.randint(0, 2, size=(4, 1), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 1), torch.randint(0, 2, size=(4, 1), dtype=torch.long))
    em.update(output2)

    assert all([t.device.type == available_device for t in em._predictions + em._targets])
    assert torch.equal(em._predictions[0].cpu(), output1[0][:, 0].cpu())
    assert torch.equal(em._predictions[1].cpu(), output2[0][:, 0].cpu())
    assert torch.equal(em._targets[0].cpu(), output1[1][:, 0].cpu())
    assert torch.equal(em._targets[1].cpu(), output2[1][:, 0].cpu())
    assert em.compute() == 0.0


def test_mse_epoch_metric(available_device):
    def compute_fn(y_preds, y_targets):
        return torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2)).item()

    em = EpochMetric(compute_fn, device=available_device)
    assert em._device == torch.device(available_device)

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
    assert result == pytest.approx(compute_fn(preds, targets), rel=1e-6)

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
    assert result == pytest.approx(compute_fn(preds, targets), rel=1e-6)


def test_bad_compute_fn():
    def compute_fn(y_preds, y_targets):
        # Following will raise the error:
        # The size of tensor a (3) must match the size of tensor b (4)
        # at non-singleton dimension 1
        return torch.mean(y_preds - y_targets).item()

    em = EpochMetric(compute_fn)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 4), dtype=torch.long))
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output1)


def test_check_compute_fn(available_device):
    def compute_fn(y_preds, y_targets):
        raise Exception

    em = EpochMetric(compute_fn, check_compute_fn=True, device=available_device)
    assert em._device == torch.device(available_device)

    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output1)

    em = EpochMetric(compute_fn, check_compute_fn=False, device=available_device)
    assert em._device == torch.device(available_device)
    em.update(output1)


def test_distrib_integration(distributed):
    device = idist.device() if idist.device().type != "xla" else "cpu"
    rank = idist.get_rank()
    torch.manual_seed(40 + rank)

    n_iters = 3
    batch_size = 2
    n_classes = 7

    y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,), device=device)
    y_preds = torch.rand(n_iters * batch_size, n_classes, device=device)

    def update(engine, i):
        return (
            y_preds[i * batch_size : (i + 1) * batch_size, :],
            y_true[i * batch_size : (i + 1) * batch_size],
        )

    engine = Engine(update)

    def assert_data_fn(all_preds, all_targets):
        return (all_preds.argmax(dim=1) == all_targets).sum().item()

    ep_metric = EpochMetric(assert_data_fn, check_compute_fn=False, device=device)
    ep_metric.attach(engine, "epm")

    data = list(range(n_iters))

    engine.run(data=data, max_epochs=3)

    y_preds = idist.all_gather(y_preds)
    y_true = idist.all_gather(y_true)
    ep_metric_true = (y_preds.argmax(dim=1) == y_true).sum().item()

    assert engine.state.metrics["epm"] == ep_metric_true
    assert ep_metric.compute() == ep_metric_true


def test_skip_unrolling(available_device):
    def compute_fn(y_preds, y_targets):
        return 0.0

    em = EpochMetric(compute_fn, skip_unrolling=True, device=available_device)
    assert em._device == torch.device(available_device)

    em.reset()
    output1 = (torch.rand(4, 2), torch.randint(0, 2, size=(4, 2), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 2), torch.randint(0, 2, size=(4, 2), dtype=torch.long))
    em.update(output2)

    assert all([t.device.type == available_device for t in em._predictions + em._targets])
    assert torch.equal(em._predictions[0].cpu(), output1[0].cpu())
    assert torch.equal(em._predictions[1].cpu(), output2[0].cpu())
    assert torch.equal(em._targets[0].cpu(), output1[1].cpu())
    assert torch.equal(em._targets[1].cpu(), output2[1].cpu())
    assert em.compute() == 0.0

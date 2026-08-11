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


def test_epoch_metric_compute_fn_tensor_output():
    """Test EpochMetric with compute_fn returning a tensor."""

    def compute_fn(y_preds, y_targets):
        return torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2), dim=0)

    em = EpochMetric(compute_fn)
    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    result = em.compute()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)

    preds = torch.cat([output1[0], output2[0]], dim=0)
    targets = torch.cat([output1[1], output2[1]], dim=0)
    expected = compute_fn(preds, targets)
    assert torch.allclose(result, expected)


def test_epoch_metric_compute_fn_tuple_output():
    """Test EpochMetric with compute_fn returning a tuple of tensors."""

    def compute_fn(y_preds, y_targets):
        mse = torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2))
        mae = torch.mean(torch.abs(y_preds - y_targets.type_as(y_preds)))
        return (mse, mae)

    em = EpochMetric(compute_fn)
    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    result = em.compute()
    assert isinstance(result, tuple)
    assert len(result) == 2

    preds = torch.cat([output1[0], output2[0]], dim=0)
    targets = torch.cat([output1[1], output2[1]], dim=0)
    expected = compute_fn(preds, targets)
    assert torch.allclose(result[0], expected[0])
    assert torch.allclose(result[1], expected[1])


def test_epoch_metric_compute_fn_invalid_output():
    """Test EpochMetric raises TypeError for unsupported compute_fn output."""

    def compute_fn(y_preds, y_targets):
        return "invalid_output"

    em = EpochMetric(compute_fn, check_compute_fn=False)
    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    with pytest.raises(TypeError, match=r"compute_fn output type"):
        em.compute()


def test_epoch_metric_compute_fn_list_output():
    """Test EpochMetric with compute_fn returning a list of tensors."""

    def compute_fn(y_preds, y_targets):
        mse = torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2))
        mae = torch.mean(torch.abs(y_preds - y_targets.type_as(y_preds)))
        return [mse, mae]

    em = EpochMetric(compute_fn)
    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    result = em.compute()
    assert isinstance(result, list)
    assert len(result) == 2

    preds = torch.cat([output1[0], output2[0]], dim=0)
    targets = torch.cat([output1[1], output2[1]], dim=0)
    expected = compute_fn(preds, targets)
    assert torch.allclose(result[0], expected[0])
    assert torch.allclose(result[1], expected[1])


def test_epoch_metric_compute_fn_dict_output():
    """Test EpochMetric with compute_fn returning a dict of tensors."""

    def compute_fn(y_preds, y_targets):
        return {
            "mse": torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2)),
            "mae": torch.mean(torch.abs(y_preds - y_targets.type_as(y_preds))),
        }

    em = EpochMetric(compute_fn)
    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    em.update(output2)

    result = em.compute()
    assert isinstance(result, dict)
    assert "mse" in result
    assert "mae" in result

    preds = torch.cat([output1[0], output2[0]], dim=0)
    targets = torch.cat([output1[1], output2[1]], dim=0)
    expected = compute_fn(preds, targets)
    assert torch.allclose(result["mse"], expected["mse"])
    assert torch.allclose(result["mae"], expected["mae"])


def test_epoch_metric_nested_invalid_output_raises():
    """Test EpochMetric raises TypeError for container with invalid nested type."""

    def compute_fn(y_preds, y_targets):
        return [torch.tensor(1.0), "not-a-number"]

    em = EpochMetric(compute_fn, check_compute_fn=False)
    em.reset()
    em.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long)))
    em.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long)))

    with pytest.raises(TypeError, match=r"compute_fn output type .* is not supported"):
        em.compute()


def test_epoch_metric_mapping_non_str_key_raises():
    """Test EpochMetric raises TypeError for mapping with non-string keys."""

    def compute_fn(y_preds, y_targets):
        return {0: torch.tensor(1.0)}

    em = EpochMetric(compute_fn, check_compute_fn=False)
    em.reset()
    em.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long)))
    em.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long)))

    with pytest.raises(TypeError, match=r"mapping keys should be str"):
        em.compute()


def test_distrib_container_outputs(distributed):
    """Test EpochMetric with container outputs in distributed setting."""
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

    # Test tuple output
    def tuple_fn(preds, targets):
        return (torch.tensor(1.0), torch.tensor(2.0))

    ep_metric = EpochMetric(tuple_fn, check_compute_fn=False, device=device)
    ep_metric.attach(engine, "tup")

    # Test dict output
    def dict_fn(preds, targets):
        return {"a": torch.tensor(3.0), "b": torch.tensor(4.0)}

    ep_metric2 = EpochMetric(dict_fn, check_compute_fn=False, device=device)
    ep_metric2.attach(engine, "dct")

    engine.run(data=list(range(n_iters)), max_epochs=1)

    # Verify tuple
    tup = engine.state.metrics["tup"]
    assert isinstance(tup, tuple)
    assert len(tup) == 2
    assert torch.allclose(tup[0], torch.tensor(1.0))
    assert torch.allclose(tup[1], torch.tensor(2.0))

    # Verify dict
    dct = engine.state.metrics["dct"]
    assert isinstance(dct, dict)
    assert torch.allclose(dct["a"], torch.tensor(3.0))
    assert torch.allclose(dct["b"], torch.tensor(4.0))


def test_distrib_invalid_output_raises_on_all_ranks(distributed):
    """Regression test: an unsupported compute_fn output must raise TypeError on every
    rank, not just rank 0. compute_fn only runs on rank 0, so if that rank validated the
    output and raised before the broadcast collective started, rank 0 would exit while
    other ranks hung forever waiting on a broadcast rank 0 never issued.
    """
    device = idist.device() if idist.device().type != "xla" else "cpu"

    def compute_fn(y_preds, y_targets):
        return "not-a-supported-type"

    em = EpochMetric(compute_fn, check_compute_fn=False, device=device)
    em.reset()
    em.update((torch.rand(4, 3, device=device), torch.randint(0, 2, size=(4, 3), device=device, dtype=torch.long)))

    with pytest.raises(TypeError, match=r"compute_fn output type.*is not supported"):
        em.compute()


def test_distrib_mapping_non_str_key_raises_on_all_ranks(distributed):
    """Regression test: a mapping output with a non-str key must raise TypeError on every
    rank, for the same reason as test_distrib_invalid_output_raises_on_all_ranks above.
    """
    device = idist.device() if idist.device().type != "xla" else "cpu"

    def compute_fn(y_preds, y_targets):
        return {0: torch.tensor(1.0, device=device)}

    em = EpochMetric(compute_fn, check_compute_fn=False, device=device)
    em.reset()
    em.update((torch.rand(4, 3, device=device), torch.randint(0, 2, size=(4, 3), device=device, dtype=torch.long)))

    with pytest.raises(TypeError, match=r"mapping keys should be str"):
        em.compute()


def test_distrib_nested_container_outputs(distributed):
    """Test EpochMetric broadcasts nested containers (e.g. a dict holding a tuple and an
    int) correctly, not just a single level of tuple/list/dict.
    """
    device = idist.device() if idist.device().type != "xla" else "cpu"

    def compute_fn(y_preds, y_targets):
        return {"scores": (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)), "count": 3}

    em = EpochMetric(compute_fn, check_compute_fn=False, device=device)
    em.reset()
    em.update((torch.rand(4, 3, device=device), torch.randint(0, 2, size=(4, 3), device=device, dtype=torch.long)))

    result = em.compute()
    assert isinstance(result, dict)
    assert isinstance(result["scores"], tuple)
    assert torch.allclose(result["scores"][0].cpu(), torch.tensor(1.0))
    assert torch.allclose(result["scores"][1].cpu(), torch.tensor(2.0))
    assert result["count"] == 3

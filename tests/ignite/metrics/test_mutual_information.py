import numpy as np
import pytest
import torch
from scipy.special import softmax
from scipy.stats import entropy
from torch import Tensor

import ignite.distributed as idist

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import MutualInformation


def np_mutual_information(np_y_pred: np.ndarray) -> float:
    prob = softmax(np_y_pred, axis=1)
    marginal_ent = entropy(np.mean(prob, axis=0))
    conditional_ent = np.mean(entropy(prob, axis=1))
    return max(0.0, marginal_ent - conditional_ent)


def test_zero_sample():
    mi = MutualInformation()
    with pytest.raises(
        NotComputableError, match=r"MutualInformation must have at least one example before it can be computed"
    ):
        mi.compute()


def test_invalid_shape():
    mi = MutualInformation()
    y_pred = torch.randn(10).float()
    with pytest.raises(ValueError, match=r"y_pred must be in the shape of \(B, C\) or \(B, C, ...\), got"):
        mi.update((y_pred, None))


@pytest.fixture(params=list(range(4)))
def test_case(request):
    return [
        (torch.randn((100, 10)).float(), torch.randint(0, 10, size=[100]), 1),
        (torch.rand((100, 500)).float(), torch.randint(0, 500, size=[100]), 1),
        # updated batches
        (torch.normal(0.0, 5.0, size=(100, 10)).float(), torch.randint(0, 10, size=[100]), 16),
        (torch.normal(5.0, 3.0, size=(100, 200)).float(), torch.randint(0, 200, size=[100]), 16),
        # image segmentation
        (torch.randn((100, 5, 32, 32)).float(), torch.randint(0, 5, size=(100, 32, 32)), 16),
        (torch.randn((100, 5, 224, 224)).float(), torch.randint(0, 5, size=(100, 224, 224)), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case: tuple[Tensor, Tensor, int], available_device):
    mi = MutualInformation(device=available_device)
    assert mi._device == torch.device(available_device)

    y_pred, y, batch_size = test_case

    mi.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            mi.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        mi.update((y_pred, y))

    np_res = np_mutual_information(y_pred.numpy())
    res = mi.compute()

    assert isinstance(res, float)
    assert pytest.approx(np_res, rel=1e-4) == res


def test_accumulator_detached(available_device):
    mi = MutualInformation(device=available_device)
    assert mi._device == torch.device(available_device)

    y_pred = torch.tensor([[2.0, 3.0], [-2.0, -1.0]], requires_grad=True)
    y = torch.zeros(2)
    mi.update((y_pred, y))

    assert not mi._sum_of_probabilities.requires_grad


def _test_state_dict_round_trip(device, segmentation=False, updated_destination=False):
    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        device=device,
    )
    logits = logits.roll(idist.get_rank(), dims=1)
    if segmentation:
        # Two images, three classes, and two pixels per image.
        logits = logits.reshape(2, 2, 3).movedim(2, 1)

    # Save after the first batch, then restore into a fresh or used metric.
    metric = MutualInformation(device=device)
    metric.update((logits[:1], None))
    state = metric.state_dict()
    restored = MutualInformation(device=device)
    if updated_destination:
        restored.update((torch.zeros_like(logits), None))
    restored.load_state_dict(state)

    first_batch = logits[:1].movedim(1, -1).reshape(-1, 3)
    first_batch = idist.all_gather(first_batch).cpu().numpy()
    assert restored.compute() == pytest.approx(np_mutual_information(first_batch), abs=1e-6)

    # Resume with the remaining batch and compare with the complete-data result.
    restored.update((logits[1:], None))
    all_batches = logits.movedim(1, -1).reshape(-1, 3)
    all_batches = idist.all_gather(all_batches).cpu().numpy()
    assert restored.compute() == pytest.approx(np_mutual_information(all_batches), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("segmentation", [False, True])
@pytest.mark.parametrize("updated_destination", [False, True])
def test_state_dict_round_trip(available_device, segmentation, updated_destination):
    _test_state_dict_round_trip(available_device, segmentation, updated_destination)


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_state_dict_round_trip(self):
        _test_state_dict_round_trip(idist.device())

    def test_integration(self):
        tol = 1e-4
        n_iters = 100
        batch_size = 10
        n_cls = 50
        device = idist.device()
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            y_true = torch.randint(0, n_cls, size=[n_iters * batch_size], dtype=torch.long).to(device)
            y_preds = torch.normal(0.0, 3.0, size=(n_iters * batch_size, n_cls), dtype=torch.float).to(device)

            engine = Engine(
                lambda e, i: (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )
            )

            m = MutualInformation(device=metric_device)
            m.attach(engine, "mutual_information")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "mutual_information" in engine.state.metrics
            res = engine.state.metrics["mutual_information"]

            true_res = np_mutual_information(y_preds.cpu().numpy())

            assert pytest.approx(true_res, rel=tol) == res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)
        for metric_device in metric_devices:
            mi = MutualInformation(device=metric_device)

            devices = (mi._device, mi._sum_of_probabilities.device)
            for dev in devices:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            y_pred = torch.tensor([[2.0, 3.0], [-2.0, -1.0]], requires_grad=True)
            y = torch.zeros(2)
            mi.update((y_pred, y))

            devices = (mi._device, mi._sum_of_probabilities.device)
            for dev in devices:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

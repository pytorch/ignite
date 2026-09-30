import numpy as np
import pytest
import torch
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy

import ignite.distributed as idist

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import Entropy


def np_entropy(np_y_pred: np.ndarray):
    prob = softmax(np_y_pred, axis=1)
    ent = np.mean(scipy_entropy(prob, axis=1))
    return ent


def test_zero_sample():
    ent = Entropy()
    with pytest.raises(NotComputableError, match=r"Entropy must have at least one example before it can be computed"):
        ent.compute()


def test_invalid_shape():
    ent = Entropy()
    y_pred = torch.randn(10).float()
    with pytest.raises(ValueError, match=r"y_pred must be in the shape of \(B, C\) or \(B, C, ...\), got"):
        ent.update((y_pred, None))


@pytest.fixture(params=[item for item in range(4)])
def test_case(request):
    return [
        (torch.randn((100, 10)), torch.randint(0, 10, size=[100]), 1),
        (torch.rand((100, 500)), torch.randint(0, 500, size=[100]), 1),
        # updated batches
        (torch.normal(0.0, 5.0, size=(100, 10)), torch.randint(0, 10, size=[100]), 16),
        (torch.normal(5.0, 3.0, size=(100, 200)), torch.randint(0, 200, size=[100]), 16),
        # image segmentation
        (torch.randn((100, 5, 32, 32)), torch.randint(0, 5, size=(100, 32, 32)), 16),
        (torch.randn((100, 5, 224, 224)), torch.randint(0, 5, size=(100, 224, 224)), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case, available_device):
    ent = Entropy(device=available_device)
    assert ent._device == torch.device(available_device)

    y_pred, y, batch_size = test_case

    ent.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            ent.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        ent.update((y_pred, y))

    np_res = np_entropy(y_pred.cpu().numpy())

    assert isinstance(ent.compute(), float)
    assert pytest.approx(ent.compute()) == np_res


def test_accumulator_detached(available_device):
    ent = Entropy(device=available_device)
    assert ent._device == torch.device(available_device)

    y_pred = torch.tensor([[2.0, 3.0], [-2.0, -1.0]], requires_grad=True)
    y = torch.zeros(2)
    ent.update((y_pred, y))

    assert not ent._sum_of_entropies.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        tol = 1e-6
        device = idist.device()
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_iters = 100
        batch_size = 10
        n_cls = 50

        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            y_true = torch.randint(0, n_cls, size=[n_iters * batch_size], dtype=torch.long).to(device)
            y_preds = torch.normal(2.0, 3.0, size=(n_iters * batch_size, n_cls), dtype=torch.float).to(device)

            def update(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

            engine = Engine(update)

            m = Entropy(device=metric_device)
            m.attach(engine, "entropy")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "entropy" in engine.state.metrics
            res = engine.state.metrics["entropy"]

            true_res = np_entropy(y_preds.cpu().numpy())

            assert pytest.approx(res, rel=tol) == true_res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            device = torch.device(device)
            ent = Entropy(device=metric_device)

            for dev in [ent._device, ent._sum_of_entropies.device]:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            y_pred = torch.tensor([[2.0], [-2.0]])
            y = torch.zeros(2)
            ent.update((y_pred, y))

            for dev in [ent._device, ent._sum_of_entropies.device]:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

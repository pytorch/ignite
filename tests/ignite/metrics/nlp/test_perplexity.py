import pytest
import torch
import torch.nn.functional as F

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import Perplexity

torch.manual_seed(12)


def test_zero_sample():
    ppl = Perplexity()
    with pytest.raises(
        NotComputableError, match=r"Perplexity must have at least one example before it can be computed"
    ):
        ppl.compute()


def test_invalid_y_pred_shape():
    ppl = Perplexity()
    with pytest.raises(ValueError, match=r"y_pred must be at least 2-dimensional"):
        ppl.update((torch.tensor([1.0, 2.0]), torch.tensor([0])))


def test_invalid_y_shape():
    ppl = Perplexity()
    with pytest.raises(ValueError, match=r"y must be at least 1-dimensional"):
        ppl.update((torch.randn(2, 5, 3), torch.tensor(0)))


def test_invalid_ndim_difference():
    ppl = Perplexity()
    with pytest.raises(ValueError, match=r"y_pred and y must differ by exactly one dimension"):
        ppl.update((torch.randn(2, 5), torch.randn(2, 5)))


def test_invalid_batch_size():
    ppl = Perplexity()
    with pytest.raises(ValueError, match=r"y_pred and y have incompatible shapes"):
        ppl.update((torch.randn(2, 5, 3), torch.randint(0, 5, (3, 3))))


def test_invalid_seq_len():
    ppl = Perplexity()
    with pytest.raises(ValueError, match=r"y_pred and y have incompatible shapes"):
        ppl.update((torch.randn(2, 5, 3), torch.randint(0, 5, (2, 4))))


def test_reset_clears_state():
    torch.manual_seed(2)
    ppl = Perplexity()

    y_pred = torch.randn(2, 5, 3)
    y = torch.randint(0, 5, (2, 3))
    ppl.update((y_pred, y))
    ppl.reset()

    with pytest.raises(NotComputableError):
        ppl.compute()


def _reference_perplexity(y_pred, y):
    """Reference implementation: token-weighted NLL."""
    nll = F.cross_entropy(y_pred, y, reduction="sum")
    return torch.exp(nll / y.numel()).item()


@pytest.mark.parametrize("n_times", range(3))
def test_compute_matches_reference(n_times, available_device):
    ppl = Perplexity(device=available_device)
    assert ppl._device == torch.device(available_device)

    torch.manual_seed(n_times)
    y_pred = torch.randn(4, 10, 5)
    y = torch.randint(0, 10, (4, 5))

    ppl.reset()
    ppl.update((y_pred, y))

    ref = _reference_perplexity(y_pred, y)
    assert pytest.approx(ppl.compute(), abs=1e-4) == ref


@pytest.mark.parametrize("n_times", range(3))
def test_token_weighted_accumulation(n_times, available_device):
    """Token-weighted accumulation across multiple batches."""
    ppl = Perplexity(device=available_device)
    assert ppl._device == torch.device(available_device)

    torch.manual_seed(n_times)

    b1_pred = torch.randn(2, 5, 4)
    b1_y = torch.randint(0, 5, (2, 4))
    b2_pred = torch.randn(3, 5, 4)
    b2_y = torch.randint(0, 5, (3, 4))

    ppl.reset()
    ppl.update((b1_pred, b1_y))
    ppl.update((b2_pred, b2_y))

    combined_pred = torch.cat([b1_pred, b2_pred], dim=0)
    combined_y = torch.cat([b1_y, b2_y], dim=0)
    ppl_ref = _reference_perplexity(combined_pred, combined_y)

    assert pytest.approx(ppl.compute(), abs=1e-4) == ppl_ref


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_accumulator_device(self):
        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            ppl = Perplexity(device=metric_device)
            assert ppl._device == metric_device
            assert ppl._sum_of_nll.device == metric_device, f"{ppl._sum_of_nll.device} vs {metric_device}"

            y_pred = torch.randn(2, 5, 3, device=device)
            y = torch.randint(0, 5, (2, 3), device=device)
            ppl.update((y_pred, y))

            assert ppl._sum_of_nll.device == metric_device, f"{ppl._sum_of_nll.device} vs {metric_device}"

    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration(self, n_epochs):
        rank = idist.get_rank()
        torch.manual_seed(10 + rank)

        n_iters = 20
        batch_size = 4
        vocab_size = 10
        seq_len = 5

        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            y_true = torch.randint(0, vocab_size, size=(n_iters * batch_size, seq_len)).to(device)
            y_preds = torch.randn(n_iters * batch_size, vocab_size, seq_len).to(device)

            def update(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

            engine = Engine(update)
            ppl = Perplexity(device=metric_device)
            ppl.attach(engine, "ppl")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            y_true_gathered = idist.all_gather(y_true)
            y_preds_gathered = idist.all_gather(y_preds)

            assert "ppl" in engine.state.metrics
            res = engine.state.metrics["ppl"]

            ref = _reference_perplexity(y_preds_gathered, y_true_gathered)

            assert pytest.approx(res, abs=1e-4) == ref

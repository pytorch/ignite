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
    with pytest.raises(ValueError, match=r"y_pred must have exactly one more dimension than y"):
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [4, 32])
def test_uniform_logits_across_batch_sizes(dtype, batch_size):
    ppl = Perplexity()
    y_pred = torch.zeros(32, 16, 1024, dtype=dtype)
    y = torch.zeros(32, 1024, dtype=torch.long)

    for pred_batch, target_batch in zip(y_pred.split(batch_size), y.split(batch_size)):
        ppl.update((pred_batch, target_batch))

    # Uniform probabilities over 16 tokens give perplexity 16, independently of
    # batch size. The total NLL exceeds the float16 range for the larger batch.
    assert ppl.compute() == pytest.approx(16.0, rel=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_ignored_tokens_and_unequal_batches(dtype):
    ppl = Perplexity(ignore_index=-1)
    ppl.update((torch.zeros(1, 4, 2, dtype=dtype), torch.full((1, 2), -1)))
    with pytest.raises(NotComputableError):
        ppl.compute()

    # One valid token with probability 1/4.
    ppl.update((torch.zeros(1, 4, 3, dtype=dtype), torch.tensor([[0, -1, -1]])))
    assert ppl.compute() == pytest.approx(4.0)

    # Three valid tokens with probability 1/2; the other classes have negligible
    # probability. Only the four valid tokens contribute to the mean NLL.
    y_pred = torch.zeros(1, 4, 5, dtype=dtype)
    y_pred[:, 2:] = -1000
    ppl.update((y_pred, torch.tensor([[0, 1, 0, -1, -1]])))
    assert ppl.compute() == pytest.approx(2**1.25)


def test_float64_logits_keep_precision():
    ppl = Perplexity()
    # The common offset does not affect probabilities, but converting these
    # logits to float32 would erase their difference and yield perplexity 2.
    ppl.update((torch.tensor([[1e8, 1e8 + 1]], dtype=torch.float64), torch.tensor([1])))
    assert ppl.compute() == pytest.approx(1.3678794411714423)


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

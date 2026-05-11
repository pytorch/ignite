import numpy as np
import pytest
import torch

from ignite.metrics._running_stats import WelfordCovariance, WelfordVariance


# ---------------------------------------------------------------------------
# WelfordVariance
# ---------------------------------------------------------------------------


class TestWelfordVariance:
    def test_empty_accumulator(self):
        ws = WelfordVariance()
        assert ws.n_samples == 0
        assert ws.variance.item() == 0.0
        assert ws.std.item() == 0.0

    def test_update_then_compute_matches_numpy(self):
        rng = np.random.default_rng(0)
        # Use float64 throughout so we compare apples to apples; Welford
        # upcasts internally, so feeding float32 inputs and comparing against
        # float32 numpy stats would understate the helper's precision.
        data = rng.standard_normal(1000)

        ws = WelfordVariance()
        ws.update(torch.from_numpy(data))

        assert ws.n_samples == 1000
        assert ws.mean.item() == pytest.approx(float(data.mean()), abs=1e-12)
        assert ws.variance.item() == pytest.approx(float(data.var()), rel=1e-12)

    def test_multi_batch_matches_single_batch(self):
        # Use float64 so the test exercises the algorithm rather than float32
        # accumulation noise.
        rng = np.random.default_rng(1)
        data = rng.standard_normal(1000)
        data_t = torch.from_numpy(data)

        single = WelfordVariance()
        single.update(data_t)

        multi = WelfordVariance()
        for start in range(0, 1000, 37):
            multi.update(data_t[start : start + 37])

        assert multi.n_samples == single.n_samples
        assert multi.mean.item() == pytest.approx(single.mean.item(), abs=1e-12)
        assert multi.variance.item() == pytest.approx(single.variance.item(), rel=1e-12)

    def test_merge_matches_concatenated_update(self):
        rng = np.random.default_rng(2)
        a = torch.from_numpy(rng.standard_normal(400).astype(np.float64))
        b = torch.from_numpy(rng.standard_normal(600).astype(np.float64))

        merged = WelfordVariance()
        merged.update(a)
        right = WelfordVariance()
        right.update(b)
        merged.merge(right)

        baseline = WelfordVariance()
        baseline.update(torch.cat([a, b]))

        assert merged.n_samples == baseline.n_samples
        assert merged.mean.item() == pytest.approx(baseline.mean.item(), abs=1e-12)
        assert merged.variance.item() == pytest.approx(baseline.variance.item(), rel=1e-12)

    def test_merge_with_empty_accumulators(self):
        rng = np.random.default_rng(3)
        data = torch.from_numpy(rng.standard_normal(100))

        # Empty merged into populated -> unchanged.
        a = WelfordVariance()
        a.update(data)
        before_mean = a.mean.item()
        a.merge(WelfordVariance())
        assert a.mean.item() == pytest.approx(before_mean, abs=1e-12)

        # Populated merged into empty -> takes the other's state.
        b = WelfordVariance()
        b.merge(a)
        assert b.n_samples == a.n_samples
        assert b.mean.item() == pytest.approx(a.mean.item(), abs=1e-12)

    def test_numerical_stability_large_mean_float32(self):
        # The whole point of this helper: naive Σx^2 - (Σx)^2/n computed in
        # float32 catastrophically cancels at mean=1e6, returning ~0 variance
        # (or even negative). Welford fed float64 inputs stays exact.
        rng = np.random.default_rng(4)
        true_std = 1.0
        data = rng.standard_normal(10_000).astype(np.float32) * true_std + 1e6
        data_t = torch.from_numpy(data)

        # Naive formula in float32 collapses.
        sum_x_f32 = data_t.sum()
        sum_x2_f32 = (data_t * data_t).sum()
        naive_var_f32 = (sum_x2_f32 - sum_x_f32 * sum_x_f32 / len(data_t)) / len(data_t)
        # Use float64 ground truth so the assertion isn't measuring our own bug.
        true_var = float(np.var(data.astype(np.float64)))

        # The helper is dtype-agnostic; the caller is responsible for the
        # float64 upcast. Verify the upcast path recovers the true variance.
        ws = WelfordVariance()
        ws.update(data_t.to(torch.float64))
        assert ws.variance.item() == pytest.approx(true_var, rel=1e-6)

        # And the naive float32 formula must demonstrably fail on the same
        # data so the test documents what we're protecting against.
        assert abs(float(naive_var_f32) - true_var) > 0.1, (
            "naive float32 formula did NOT cancel; test is no longer exercising the failure mode it claims to."
        )

    def test_single_sample(self):
        ws = WelfordVariance()
        ws.update(torch.tensor([42.0]))
        assert ws.n_samples == 1
        assert ws.mean.item() == 42.0
        assert ws.variance.item() == 0.0

    def test_empty_batch_is_noop(self):
        ws = WelfordVariance()
        ws.update(torch.tensor([1.0, 2.0, 3.0]))
        before = (ws.n_samples, ws.mean.item(), ws.variance.item())
        ws.update(torch.tensor([]))
        after = (ws.n_samples, ws.mean.item(), ws.variance.item())
        assert before == after

    def test_fresh_instance_has_zero_state(self):
        # The dataclass starts empty; "reset" is just reconstruction. Verifies
        # that the default factories produce an empty accumulator.
        ws = WelfordVariance()
        assert ws.n_samples == 0
        assert ws.mean.item() == 0.0
        assert ws.sum_sq_dev_from_mean.item() == 0.0

    def test_state_dtype_follows_first_batch(self):
        # The helper does not handle dtype itself; it takes whatever dtype
        # the first batch arrives in and preserves it. The caller chooses.
        ws_f32 = WelfordVariance()
        ws_f32.update(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))
        assert ws_f32.mean.dtype == torch.float32
        assert ws_f32.mean.item() == pytest.approx(2.5)
        assert ws_f32.variance.item() == pytest.approx(1.25)

        ws_f64 = WelfordVariance()
        ws_f64.update(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        assert ws_f64.mean.dtype == torch.float64
        assert ws_f64.mean.item() == pytest.approx(2.5)
        assert ws_f64.variance.item() == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# WelfordCovariance
# ---------------------------------------------------------------------------


class TestWelfordCovariance:
    def test_empty_accumulator(self):
        wc = WelfordCovariance()
        assert wc.n_samples == 0
        assert wc.variance_x.item() == 0.0
        assert wc.variance_y.item() == 0.0
        assert wc.covariance.item() == 0.0

    def test_update_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(5)
        n = 1000
        x = rng.standard_normal(n)
        y = 0.7 * x + rng.standard_normal(n) * 0.3

        wc = WelfordCovariance()
        wc.update(torch.from_numpy(x), torch.from_numpy(y))

        np_var_x = float(np.var(x))
        np_var_y = float(np.var(y))
        np_cov = float(np.cov(x, y, bias=True)[0, 1])
        np_r = float(np.corrcoef(x, y)[0, 1])

        assert wc.variance_x.item() == pytest.approx(np_var_x, rel=1e-12)
        assert wc.variance_y.item() == pytest.approx(np_var_y, rel=1e-12)
        assert wc.covariance.item() == pytest.approx(np_cov, rel=1e-12)
        assert wc.correlation().item() == pytest.approx(np_r, rel=1e-10)

    def test_multi_batch_matches_single_batch(self):
        rng = np.random.default_rng(6)
        x = torch.from_numpy(rng.standard_normal(900))
        y = torch.from_numpy(rng.standard_normal(900))

        single = WelfordCovariance()
        single.update(x, y)

        multi = WelfordCovariance()
        for start in range(0, 900, 31):
            multi.update(x[start : start + 31], y[start : start + 31])

        assert multi.mean_x.item() == pytest.approx(single.mean_x.item(), abs=1e-12)
        assert multi.mean_y.item() == pytest.approx(single.mean_y.item(), abs=1e-12)
        assert multi.covariance.item() == pytest.approx(single.covariance.item(), rel=1e-12)
        assert multi.correlation().item() == pytest.approx(single.correlation().item(), rel=1e-12)

    def test_merge_matches_concatenated_update(self):
        rng = np.random.default_rng(7)
        x1 = torch.from_numpy(rng.standard_normal(300))
        y1 = torch.from_numpy(rng.standard_normal(300))
        x2 = torch.from_numpy(rng.standard_normal(500))
        y2 = torch.from_numpy(rng.standard_normal(500))

        merged = WelfordCovariance()
        merged.update(x1, y1)
        right = WelfordCovariance()
        right.update(x2, y2)
        merged.merge(right)

        baseline = WelfordCovariance()
        baseline.update(torch.cat([x1, x2]), torch.cat([y1, y2]))

        assert merged.covariance.item() == pytest.approx(baseline.covariance.item(), rel=1e-12)
        assert merged.correlation().item() == pytest.approx(baseline.correlation().item(), rel=1e-12)

    def test_numerical_stability_large_mean(self):
        # The Pearson-correlation regression case from issue #3662: mean=1e6,
        # std=1 makes the naive E[X^2] - E[X]^2 formula return garbage in
        # float32. Welford fed float64 inputs recovers the true r.
        rng = np.random.default_rng(8)
        n = 10_000
        x = rng.standard_normal(n).astype(np.float32) + 1e6
        y = (0.99 * x + rng.standard_normal(n).astype(np.float32) * 0.1).astype(np.float32)

        true_r = float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])
        # Sanity: the constructed series really is highly correlated.
        assert true_r > 0.99

        # Caller-side upcast to float64 (the helper preserves whatever it gets).
        wc = WelfordCovariance()
        wc.update(torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64))
        assert wc.correlation().item() == pytest.approx(true_r, rel=1e-4)

    def test_shape_mismatch_raises(self):
        wc = WelfordCovariance()
        with pytest.raises(ValueError, match="same shape"):
            wc.update(torch.zeros(5), torch.zeros(6))

    def test_empty_batch_is_noop(self):
        wc = WelfordCovariance()
        wc.update(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        before = (wc.n_samples, wc.covariance.item())
        wc.update(torch.tensor([]), torch.tensor([]))
        after = (wc.n_samples, wc.covariance.item())
        assert before == after

    def test_constant_variable_correlation_safe(self):
        # When one series is constant the denominator of Pearson r is zero;
        # the eps clamp keeps us from returning NaN / inf.
        wc = WelfordCovariance()
        wc.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 5.0, 5.0, 5.0]))
        r = wc.correlation().item()
        assert r == 0.0
        assert not (r != r)  # not NaN

    def test_fresh_instance_has_zero_state(self):
        # "Reset" is just reconstruction with this dataclass.
        wc = WelfordCovariance()
        assert wc.n_samples == 0
        assert wc.covariance.item() == 0.0


# ---------------------------------------------------------------------------
# Cross-class sanity: variance_x of WelfordCovariance == variance of
# WelfordVariance fed the same x. Catches drift between the two
# implementations.
# ---------------------------------------------------------------------------


def test_variance_x_matches_welford_variance():
    rng = np.random.default_rng(9)
    x = torch.from_numpy(rng.standard_normal(1000))
    y = torch.from_numpy(rng.standard_normal(1000))

    wv = WelfordVariance()
    wv.update(x)

    wc = WelfordCovariance()
    wc.update(x, y)

    assert wc.variance_x.item() == pytest.approx(wv.variance.item(), rel=1e-12)
    assert wc.mean_x.item() == pytest.approx(wv.mean.item(), abs=1e-12)

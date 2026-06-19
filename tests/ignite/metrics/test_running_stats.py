import numpy as np
import pytest
import torch
from ignite.metrics._running_stats import WelfordCovariance, WelfordVariance


def test_welford_variance_empty_accumulator():
    ws = WelfordVariance()
    assert ws.n_samples == 0
    assert ws.variance.item() == 0.0
    assert ws.std.item() == 0.0


def test_welford_variance_single_update_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1000)

    ws = WelfordVariance()
    ws.update(torch.from_numpy(data))

    assert ws.n_samples == 1000
    assert ws.mean.item() == pytest.approx(float(data.mean()), abs=1e-12)
    assert ws.variance.item() == pytest.approx(float(data.var()), rel=1e-12)


def test_welford_variance_matches_numpy_at_each_step():
    # After every individual update, the running mean/variance must equal
    # numpy's mean/variance computed on the cumulative prefix. Catches drift
    # in the incremental formula that a single end-of-stream assert would miss.
    rng = np.random.default_rng(10)
    data = rng.standard_normal(500)
    batch_size = 17

    ws = WelfordVariance()
    seen = 0
    for start in range(0, len(data), batch_size):
        chunk = data[start : start + batch_size]
        ws.update(torch.from_numpy(chunk))
        seen += len(chunk)
        prefix = data[:seen]
        assert ws.n_samples == seen
        assert ws.mean.item() == pytest.approx(float(prefix.mean()), abs=1e-12)
        assert ws.variance.item() == pytest.approx(float(prefix.var()), rel=1e-12)


def test_welford_variance_multi_batch_matches_single_batch():
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


def test_welford_variance_merge_matches_concatenated_update():
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


def test_welford_variance_merge_with_empty_accumulators():
    rng = np.random.default_rng(3)
    data = torch.from_numpy(rng.standard_normal(100))

    # Empty merged into populated: unchanged.
    a = WelfordVariance()
    a.update(data)
    before_mean = a.mean.item()
    a.merge(WelfordVariance())
    assert a.mean.item() == pytest.approx(before_mean, abs=1e-12)

    # Populated merged into empty: takes the other's state.
    b = WelfordVariance()
    b.merge(a)
    assert b.n_samples == a.n_samples
    assert b.mean.item() == pytest.approx(a.mean.item(), abs=1e-12)


def test_welford_variance_numerical_stability_large_mean_float32():
    # The whole point of this helper: naive Σx^2 - (Σx)^2 / n in float32
    # catastrophically cancels at mean=1e6, returning ~0 variance (or even
    # negative). Welford fed float64 inputs stays exact.
    rng = np.random.default_rng(4)
    true_std = 1.0
    data = rng.standard_normal(10_000).astype(np.float32) * true_std + 1e6
    data_t = torch.from_numpy(data)

    sum_x_f32 = data_t.sum()
    sum_x2_f32 = (data_t * data_t).sum()
    naive_var_f32 = (sum_x2_f32 - sum_x_f32 * sum_x_f32 / len(data_t)) / len(data_t)
    true_var = float(np.var(data.astype(np.float64)))

    ws = WelfordVariance()
    ws.update(data_t.to(torch.float64))
    assert ws.variance.item() == pytest.approx(true_var, rel=1e-6)

    # And the naive float32 formula must demonstrably fail on the same data so
    # the test documents the failure mode it claims to protect against.
    assert abs(float(naive_var_f32) - true_var) > 0.1, (
        "naive float32 formula did NOT cancel; test is no longer exercising the failure mode it claims to."
    )


def test_welford_variance_single_sample():
    ws = WelfordVariance()
    ws.update(torch.tensor([42.0]))
    assert ws.n_samples == 1
    assert ws.mean.item() == 42.0
    assert ws.variance.item() == 0.0


def test_welford_variance_empty_batch_is_noop():
    ws = WelfordVariance()
    ws.update(torch.tensor([1.0, 2.0, 3.0]))
    before = (ws.n_samples, ws.mean.item(), ws.variance.item())
    ws.update(torch.tensor([]))
    after = (ws.n_samples, ws.mean.item(), ws.variance.item())
    assert before == after


def test_welford_variance_fresh_instance_has_zero_state():
    ws = WelfordVariance()
    assert ws.n_samples == 0
    assert ws.mean.item() == 0.0
    assert ws.sum_sq_dev_from_mean.item() == 0.0


def test_welford_variance_state_dtype_follows_first_batch():
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


def test_welford_covariance_empty_accumulator():
    wc = WelfordCovariance()
    assert wc.n_samples == 0
    assert wc.variance_x.item() == 0.0
    assert wc.variance_y.item() == 0.0
    assert wc.covariance.item() == 0.0


def test_welford_covariance_single_update_matches_numpy():
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


def test_welford_covariance_matches_numpy_at_each_step():
    # Same shape of check as the univariate version: after every update,
    # running variances + covariance + correlation must match numpy on the
    # cumulative prefix.
    rng = np.random.default_rng(11)
    x = rng.standard_normal(500)
    y = 0.5 * x + rng.standard_normal(500) * 0.5
    batch_size = 19

    wc = WelfordCovariance()
    seen = 0
    for start in range(0, len(x), batch_size):
        wc.update(
            torch.from_numpy(x[start : start + batch_size]),
            torch.from_numpy(y[start : start + batch_size]),
        )
        seen += len(x[start : start + batch_size])
        px, py = x[:seen], y[:seen]
        assert wc.n_samples == seen
        assert wc.variance_x.item() == pytest.approx(float(np.var(px)), rel=1e-12)
        assert wc.variance_y.item() == pytest.approx(float(np.var(py)), rel=1e-12)
        assert wc.covariance.item() == pytest.approx(float(np.cov(px, py, bias=True)[0, 1]), rel=1e-12)
        if seen >= 2 and float(np.std(py)) > 0:
            assert wc.correlation().item() == pytest.approx(float(np.corrcoef(px, py)[0, 1]), rel=1e-10)


def test_welford_covariance_multi_batch_matches_single_batch():
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


def test_welford_covariance_merge_matches_concatenated_update():
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


def test_welford_covariance_negative_correlation_not_clamped():
    # Verifies the documented difference between variance (clamped at 0)
    # and covariance (signed). Clamping covariance would silently bias
    # negative correlations toward zero.
    rng = np.random.default_rng(12)
    x = rng.standard_normal(500)
    y = -1.0 * x + rng.standard_normal(500) * 0.1

    wc = WelfordCovariance()
    wc.update(torch.from_numpy(x), torch.from_numpy(y))

    assert wc.covariance.item() < 0
    assert wc.correlation().item() < -0.9


def test_welford_covariance_numerical_stability_large_mean():
    # Pearson regression case from issue #3662: mean=1e6, std=1 makes the
    # naive E[X^2] - E[X]^2 formula return garbage in float32. Welford fed
    # float64 inputs recovers the true r.
    rng = np.random.default_rng(8)
    n = 10_000
    x = rng.standard_normal(n).astype(np.float32) + 1e6
    y = (0.99 * x + rng.standard_normal(n).astype(np.float32) * 0.1).astype(np.float32)

    true_r = float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])
    assert true_r > 0.99

    wc = WelfordCovariance()
    wc.update(torch.from_numpy(x).to(torch.float64), torch.from_numpy(y).to(torch.float64))
    assert wc.correlation().item() == pytest.approx(true_r, rel=1e-4)


def test_welford_covariance_shape_mismatch_raises():
    wc = WelfordCovariance()
    with pytest.raises(ValueError, match="same shape"):
        wc.update(torch.zeros(5), torch.zeros(6))


def test_welford_covariance_empty_batch_is_noop():
    wc = WelfordCovariance()
    wc.update(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    before = (wc.n_samples, wc.covariance.item())
    wc.update(torch.tensor([]), torch.tensor([]))
    after = (wc.n_samples, wc.covariance.item())
    assert before == after


def test_welford_covariance_constant_variable_correlation_safe():
    # When one series is constant the denominator of Pearson r is zero;
    # the eps clamp keeps us from returning NaN / inf.
    wc = WelfordCovariance()
    wc.update(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([5.0, 5.0, 5.0, 5.0]))
    r = wc.correlation().item()
    assert r == 0.0
    assert not (r != r)  # not NaN


def test_welford_covariance_fresh_instance_has_zero_state():
    wc = WelfordCovariance()
    assert wc.n_samples == 0
    assert wc.covariance.item() == 0.0


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

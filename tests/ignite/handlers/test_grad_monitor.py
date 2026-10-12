import math
import torch
import pytest
from ignite.engine import Engine, Events
from ignite.handlers.grad_monitor import GradMonitor, _default_spike_detector

# Helpers

def _make_engine_and_monitor(k: float = 3.0, **kwargs) -> tuple:
    # Creates a fresh model, engine, and attached GradMonitor.
    model = torch.nn.Linear(10, 1)
    monitor = GradMonitor(model, k=k, **kwargs)
    engine = Engine(lambda e, b: None)
    monitor.attach(engine)
    return model, engine, monitor


def _set_grad(model: torch.nn.Module, scale: float) -> None:
    # Sets all weight gradients to a uniform value.
    model.weight.grad = torch.ones_like(model.weight) * scale


def _run_one_iteration(engine: Engine) -> None:
    # Running a single iteration without resetting monitor stats.
    engine.run(range(1), max_epochs=1)


# Unit tests for _default_spike_detector

class TestDefaultSpikeDetector:

    def test_returns_false_when_count_is_zero(self):
        assert _default_spike_detector(mean=0.0, m2=0.0, count=0, norm=999.0, k=3.0) is False

    def test_returns_false_when_count_is_one(self):
        assert _default_spike_detector(mean=1.0, m2=0.0, count=1, norm=999.0, k=3.0) is False

    def test_returns_false_when_norm_is_below_threshold(self):
        # mean=1.0, m2=0.5, count=10 => std ~ 0.235, threshold ~ 1.705
        assert _default_spike_detector(mean=1.0, m2=0.5, count=10, norm=1.5, k=3.0) is False

    def test_returns_true_when_norm_exceeds_threshold(self):
        # mean=1.0, m2=0.5, count=10 => std ~ 0.235, threshold ~ 1.705
        assert _default_spike_detector(mean=1.0, m2=0.5, count=10, norm=5.0, k=3.0) is True

    def test_returns_false_when_norm_equals_threshold_exactly(self):
        # Norm exactly at mean + k*std should NOT flag (strictly greater than).
        mean, m2, count, k = 1.0, 0.5, 10, 3.0
        std = (m2 / (count - 1)) ** 0.5
        threshold = mean + k * std
        assert _default_spike_detector(mean=mean, m2=m2, count=count, norm=threshold, k=k) is False


# Initialisation & validation

class TestGradMonitorInit:

    def test_raises_if_model_is_not_nn_module(self):
        with pytest.raises(TypeError, match="torch.nn.Module"):
            GradMonitor(model="not_a_model")

    def test_raises_if_k_is_not_numeric(self):
        model = torch.nn.Linear(10, 1)
        with pytest.raises(TypeError, match="numeric"):
            GradMonitor(model, k="3.0")

    def test_raises_if_k_is_zero(self):
        model = torch.nn.Linear(10, 1)
        with pytest.raises(ValueError, match="positive"):
            GradMonitor(model, k=0)

    def test_raises_if_k_is_negative(self):
        model = torch.nn.Linear(10, 1)
        with pytest.raises(ValueError, match="positive"):
            GradMonitor(model, k=-1.0)

    def test_default_spike_detector_assigned_when_none_provided(self):
        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model)
        assert monitor.spike_detector is _default_spike_detector

    def test_custom_spike_detector_assigned(self):
        model = torch.nn.Linear(10, 1)
        custom = lambda mean, m2, count, norm, k: False
        monitor = GradMonitor(model, spike_detector=custom)
        assert monitor.spike_detector is custom


# attach() behaviour

class TestGradMonitorAttach:

    def test_attach_returns_self_for_fluent_chaining(self):
        model, engine, monitor = _make_engine_and_monitor()
        model2 = torch.nn.Linear(10, 1)
        monitor2 = GradMonitor(model2)
        result = monitor2.attach(engine)  # second monitor, same engine is fine.
        assert result is monitor2

    def test_double_attach_raises_runtime_error(self):
        model, engine, monitor = _make_engine_and_monitor()
        with pytest.raises(RuntimeError, match="already attached"):
            monitor.attach(engine)

    def test_unhealthy_spike_initialised_to_false_before_first_iteration(self):
        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model)
        engine = Engine(lambda e, b: None)
        monitor.attach(engine)

        # Running one real iteration - Events.STARTED fires automatically.
        # It triggers _init_flag and sets unhealthy_spike to False.
        engine.run(range(1), max_epochs=1)
        assert engine.state.unhealthy_spike is False

# Gradient norm computation

class TestComputeGradNorm:

    def test_norm_is_zero_when_no_gradients(self):
        model, engine, monitor = _make_engine_and_monitor()
        # Don't set any grads — norm should be 0.
        norm = monitor._compute_grad_norm()
        assert norm == 0.0

    def test_norm_matches_manual_l2_calculation(self):
        model, engine, monitor = _make_engine_and_monitor()
        scale = 0.5
        _set_grad(model, scale)
        expected = math.sqrt(10 * scale ** 2)  # 10 weights, each = scale
        actual = monitor._compute_grad_norm()
        assert abs(actual - expected) < 1e-5

    def test_norm_is_divided_by_scaler_scale(self):
        class MockScaler:
            def get_scale(self): return 1024.0

        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model, scaler=MockScaler())
        engine = Engine(lambda e, b: None)
        monitor.attach(engine)

        scale = 1.0
        _set_grad(model, scale)
        raw_norm = math.sqrt(10 * scale ** 2)
        expected = raw_norm / 1024.0
        actual = monitor._compute_grad_norm()
        assert abs(actual - expected) < 1e-5

    def test_scaler_with_zero_scale_does_not_divide(self):
        # If scaler returns 0, division must be skipped to avoid inf/nan.
        class ZeroScaler:
            def get_scale(self): return 0.0

        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model, scaler=ZeroScaler())
        engine = Engine(lambda e, b: None)
        monitor.attach(engine)

        _set_grad(model, 1.0)
        norm = monitor._compute_grad_norm()
        assert math.isfinite(norm)


# Welford's running statistics

class TestUpdateStats:

    def test_mean_is_correct_after_several_updates(self):
        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            monitor._update_stats(v)
        assert abs(monitor.mean - 3.0) < 1e-9

    def test_variance_is_correct_after_several_updates(self):
        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model)
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for v in values:
            monitor._update_stats(v)
        # Population variance = 4.0, sample variance = 4.571...
        sample_var = monitor.m2 / (monitor.count - 1)
        assert abs(sample_var - (sum((v - 5.0) ** 2 for v in values) / 7)) < 1e-9

    def test_count_increments_correctly(self):
        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model)
        for i in range(5):
            monitor._update_stats(float(i))
        assert monitor.count == 5


# End-to-end spike detection

class TestSpikeDetection:

    # Stable gradient scales that produce std > 0.
    STABLE_SCALES = [0.08, 0.12, 0.09, 0.11, 0.10, 0.08, 0.12, 0.09, 0.11, 0.10]

    def _warm_up(self, model, engine, scales=None):
        # Run through stable gradients to build up history.
        for scale in (scales or self.STABLE_SCALES):
            _set_grad(model, scale)
            _run_one_iteration(engine)

    def test_no_spike_during_stable_training(self):
        model, engine, monitor = _make_engine_and_monitor(k=2.0)
        for scale in self.STABLE_SCALES:
            _set_grad(model, scale)
            _run_one_iteration(engine)
            assert not engine.state.unhealthy_spike

    def test_spike_detected_on_large_gradient(self):
        model, engine, monitor = _make_engine_and_monitor(k=2.0)
        self._warm_up(model, engine)

        _set_grad(model, 50.0)
        _run_one_iteration(engine)
        assert engine.state.unhealthy_spike

    def test_spike_flag_clears_after_norm_returns_to_normal(self):
        model, engine, monitor = _make_engine_and_monitor(k=2.0)
        self._warm_up(model, engine)

        # Trigger a spike.
        _set_grad(model, 50.0)
        _run_one_iteration(engine)
        assert engine.state.unhealthy_spike

        # Return to a normal gradient — flag should clear.
        _set_grad(model, 0.10)
        _run_one_iteration(engine)
        assert not engine.state.unhealthy_spike

    def test_no_spike_before_enough_history(self):
        # With fewer than 2 data points the detector must return False.
        model, engine, monitor = _make_engine_and_monitor(k=1.0)
        _set_grad(model, 999.0)
        _run_one_iteration(engine)
        assert not engine.state.unhealthy_spike

# k sensitivity

class TestKSensitivity:

    WARM_UP_SCALES = [0.08, 0.12, 0.09, 0.11, 0.10] * 2

    def _run_with_k(self, k: float, spike_scale: float) -> bool:
        model, engine, monitor = _make_engine_and_monitor(k=k)
        for scale in self.WARM_UP_SCALES:
            _set_grad(model, scale)
            _run_one_iteration(engine)
        _set_grad(model, spike_scale)
        _run_one_iteration(engine)
        return engine.state.unhealthy_spike

    def test_tight_k_catches_moderate_spike(self):
        assert self._run_with_k(k=1.0, spike_scale=0.5) is True

    def test_loose_k_ignores_moderate_spike(self):
        assert self._run_with_k(k=50.0, spike_scale=0.5) is False

# Custom spike detector

class TestCustomSpikeDetector:

    def test_custom_detector_is_called_instead_of_default(self):
        calls = []

        def my_detector(mean, m2, count, norm, k):
            calls.append(norm)
            return norm > 100.0

        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model, spike_detector=my_detector)
        engine = Engine(lambda e, b: None)
        monitor.attach(engine)

        _set_grad(model, 0.5)
        _run_one_iteration(engine)
        assert len(calls) == 1
        assert not engine.state.unhealthy_spike

        _set_grad(model, 200.0)
        _run_one_iteration(engine)
        assert engine.state.unhealthy_spike

    def test_custom_detector_absolute_threshold(self):
        model, engine, monitor = _make_engine_and_monitor(
            spike_detector=lambda mean, m2, count, norm, k: norm > 5.0
        )
        _set_grad(model, 0.1)
        _run_one_iteration(engine)
        assert not engine.state.unhealthy_spike

        _set_grad(model, 10.0)
        _run_one_iteration(engine)
        assert engine.state.unhealthy_spike
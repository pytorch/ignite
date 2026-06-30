import torch
from typing import Any, Callable, Optional
from ignite.engine import Engine, Events


def _default_spike_detector(mean: float, m2: float, count: int, norm: float, k: float) -> bool:
    """
    Default spike detection rule: flags if norm > mean + k * std.

    Args:
        mean: Running mean of gradient norms.
        m2: Running sum of squared deviations (Welford's algorithm).
        count: Number of iterations seen so far.
        norm: Current gradient norm.
        k: Standard deviation multiplier.

    Returns:
        True if the current norm is a spike, False otherwise.
    """
    if count < 2:
        return False
    std = (m2 / (count - 1)) ** 0.5
    return norm > mean + k * std + 1e-6


class GradMonitor:
    """
    Monitors the L2 gradient norm each iteration to detect training instability
    (e.g. exploding gradients, entropy collapse) using a dynamic threshold.

    The handler attaches to ``Events.ITERATION_STARTED``, meaning it reads
    gradients computed during the **previous** iteration. The spike flag is
    set on ``engine.state.unhealthy_spike`` and can be checked at the start
    of the next iteration's train step.

    .. warning::
        This handler requires that gradients are **not zeroed** at the end of
        your train step. If you call ``optimizer.zero_grad()`` at the end of
        ``train_step``, all gradients will be gone by the time this handler
        runs and the norm will always be 0. Instead, call
        ``optimizer.zero_grad()`` at the **start** of your train step, after
        checking the spike flag.

    .. warning::
        The ``unhealthy_spike`` flag on ``engine.state`` reflects gradients
        from the **previous** iteration, not the current one. Design your
        train step accordingly.

    .. warning::
        Call ``attach`` only once per engine instance. Calling it multiple
        times on the same engine will raise a ``RuntimeError`` to prevent
        doubled stat updates and corrupted running statistics.

    .. note::
        If multiple handlers are registered to ``Events.ITERATION_STARTED``,
        execution order depends on registration order. Register
        ``GradMonitor`` before any handler that reads ``unhealthy_spike``
        to ensure the flag is fresh when read.

    Args:
        model: The model whose gradient norms will be monitored.
        k: Multiplier for standard deviation when using the default spike
            detector. Ignored if ``spike_detector`` is provided. Default: 3.0.
        scaler: Optional ``torch.cuda.amp.GradScaler`` instance for AMP
            workflows. When provided, the raw gradient norm is divided by
            the current scale factor to recover the true unscaled norm.
        spike_detector: Optional callable with signature
            ``(mean, m2, count, norm, k) -> bool``. If provided, replaces
            the default ``mean + k * std`` rule entirely. Use this to
            implement custom thresholding logic.

    Example:
        .. code-block:: python

            import torch
            from ignite.engine import Engine, Events
            from ignite.handlers import GradMonitor

            model = torch.nn.Linear(10, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            def train_step(engine, batch):
                # Check the spike flag set by GradMonitor from the previous
                # iteration. On iteration 1 this will always be False
                # (not enough history yet).
                if engine.state.unhealthy_spike:
                    # Discard this batch entirely.
                    # Do NOT zero grads here so GradMonitor can still
                    # read them on the next iteration.
                    return {"loss": None, "skipped": True}

                optimizer.zero_grad()  # zero at START, not end
                x, y = batch
                loss = ((model(x) - y) ** 2).mean()
                loss.backward()
                optimizer.step()
                # Do NOT call optimizer.zero_grad() here.
                return {"loss": loss.item(), "skipped": False}

            trainer = Engine(train_step)

            # Attach GradMonitor BEFORE any handler that reads unhealthy_spike
            # to guarantee correct ordering on Events.ITERATION_STARTED.
            monitor = GradMonitor(model, k=3.0)
            monitor.attach(trainer)

            # This handler runs after GradMonitor because it is registered after.
            @trainer.on(Events.ITERATION_STARTED)
            def log_spike(engine):
                if engine.state.unhealthy_spike:
                    print(f"Spike at iteration {engine.state.iteration}!")

    Example with AMP:
        .. code-block:: python

            scaler = torch.cuda.amp.GradScaler()
            monitor = GradMonitor(model, k=3.0, scaler=scaler)
            monitor.attach(trainer)

    Example with a custom spike detector:
        .. code-block:: python

            def my_detector(mean, m2, count, norm, k):
                # Flag only if norm exceeds an absolute limit of 100.
                return norm > 100.0

            monitor = GradMonitor(model, spike_detector=my_detector)
            monitor.attach(trainer)

    .. versionadded:: 0.6.0
    """

    def __init__(
        self,
        model: torch.nn.Module,
        k: float = 3.0,
        scaler: Optional[Any] = None,
        spike_detector: Optional[Callable] = None,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"model must be a torch.nn.Module, got {type(model)}"
            )
        if not isinstance(k, (int, float)):
            raise TypeError(
                f"k must be a numeric value, got {type(k)}"
            )
        if k <= 0:
            raise ValueError(
                f"k must be a positive number, got {k}"
            )

        self.model = model
        self.k = k
        self.scaler = scaler
        self.spike_detector = spike_detector if spike_detector is not None else _default_spike_detector
        self._device: Optional[torch.device] = None
        self._attached: bool = False
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def _get_device(self) -> torch.device:
        if self._device is None:
            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        return self._device

    def _compute_grad_norm(self) -> float:
        """
        Computes the global L2 gradient norm across all model parameters.

        Handles:
        - AMP: divides by GradScaler scale if a scaler was provided.
        - DDP: sums squared norms across all distributed processes.

        Returns:
            The L2 gradient norm as a Python float.
        """
        device = self._get_device()
        total_norm_sq = torch.tensor(0.0, device=device)

        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.pow(2).sum()

        # DDP support: sum squared norms across all processes so every.
        # node sees the same global norm.
        try:
            from ignite.distributed import get_world_size, all_reduce
            if get_world_size() > 1:
                total_norm_sq = all_reduce(total_norm_sq)
        except ImportError:
            pass

        total_norm: float = torch.sqrt(total_norm_sq).item()

        # AMP support: divide by the scaler's scale factor to recover.
        # the true unscaled gradient norm.
        if self.scaler is not None:
            scale = self.scaler.get_scale()
            if scale > 0:
                total_norm /= scale

        return total_norm

    def _update_stats(self, norm: float) -> None:
        """
        Update running mean and variance using Welford's online algorithm.
        O(1) memory, numerically stable.

        Args:
            norm: The gradient norm from the current iteration.
        """
        self.count += 1
        delta = norm - self.mean
        self.mean += delta / self.count
        delta2 = norm - self.mean
        self.m2 += delta * delta2

    def __call__(self, engine: Engine) -> None:
        """
        Called at every ``Events.ITERATION_STARTED``.
        Computes the gradient norm, evaluates the spike detector, sets
        ``engine.state.unhealthy_spike``, then updates running statistics.

        Args:
            engine: The Ignite training engine.
        """
        norm = self._compute_grad_norm()

        # Evaluate spike BEFORE updating stats so the current norm is compared against history from previous iterations only.
        engine.state.unhealthy_spike = self.spike_detector(
            self.mean, self.m2, self.count, norm, self.k
        )

        self._update_stats(norm)

    def attach(self, engine: Engine) -> "GradMonitor":
        """
        Attach this handler to an engine.

        Initialises ``engine.state.unhealthy_spike = False`` at the start
        of each run so the flag is always safe to read inside the train step
        from the very first iteration.

        Raises:
            RuntimeError: If this handler has already been attached to an engine.

        Args:
            engine: The Ignite training engine.

        Returns:
            self, to allow fluent chaining:
            ``GradMonitor(model).attach(trainer)``.
        """
        if self._attached:
            raise RuntimeError(
                "GradMonitor is already attached to an engine. "
                "Create a new GradMonitor instance to attach to a different engine."
            )
        self._attached = True

        @engine.on(Events.STARTED)
        def _init_flag(e: Engine) -> None:
            e.state.unhealthy_spike = False

        if hasattr(engine, "state_dict_user_keys"):
            engine.state_dict_user_keys.append("unhealthy_spike")

        engine.add_event_handler(Events.ITERATION_STARTED, self)
        return self
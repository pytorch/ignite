from dataclasses import dataclass, field

import torch


@dataclass
class WelfordVariance:
    """Running mean and population variance via Welford's online algorithm.

    Fold batches in with :meth:`update`. Read off via :attr:`mean`,
    :attr:`variance`, :attr:`std`. Combine two accumulators with
    :meth:`merge` (Chan parallel formula).
    """

    # mean: running sample mean.
    # sum_sq_dev_from_mean: Σ (x_i - mean)^2, conventionally called M2.
    n_samples: int = 0
    mean: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_sq_dev_from_mean: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Fold ``batch`` into the running state. Empty batches are a no-op.

        Any tensor shape is accepted and treated as ``numel`` scalar samples.
        """
        if batch.numel() == 0:
            return
        batch = batch.detach()
        batch_mean = batch.mean()
        self.merge(
            WelfordVariance(
                n_samples=batch.numel(),
                mean=batch_mean,
                sum_sq_dev_from_mean=(batch - batch_mean).square().sum(),
            )
        )

    @torch.no_grad()
    def merge(self, other: "WelfordVariance") -> None:
        """Combine ``other`` into ``self`` via the Chan parallel formula.

        For two accumulators with sample counts ``n_a, n_b`` and M2 sums
        ``M2_a, M2_b``::

            M2 = M2_a + M2_b + (mean_b - mean_a)^2 * n_a * n_b / (n_a + n_b)

        The third term corrects for the spread of the two local means
        about the combined mean.
        """
        if other.n_samples == 0:
            return
        if self.n_samples == 0:
            # Copy so callers cannot mutate ``other`` and silently affect self.
            self.n_samples = other.n_samples
            self.mean = other.mean.detach().clone()
            self.sum_sq_dev_from_mean = other.sum_sq_dev_from_mean.detach().clone()
            return

        n_a, n_b = self.n_samples, other.n_samples
        n_ab = n_a + n_b
        delta = other.mean - self.mean

        self.mean = self.mean + delta * n_b / n_ab
        self.sum_sq_dev_from_mean = (
            self.sum_sq_dev_from_mean + other.sum_sq_dev_from_mean + delta * delta * n_a * n_b / n_ab
        )
        self.n_samples = n_ab

    @property
    def variance(self) -> torch.Tensor:
        """Population variance (divisor ``n``). Zero on an empty accumulator."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        # Variance is non-negative by definition; clamp guards against float
        # rounding producing a tiny negative value when all samples are equal.
        return torch.clamp(self.sum_sq_dev_from_mean / self.n_samples, min=0.0)

    @property
    def std(self) -> torch.Tensor:
        """Population standard deviation (divisor ``n``)."""
        return self.variance.sqrt()


@dataclass
class WelfordCovariance:
    """Running covariance for a pair ``(x, y)`` via Welford + Chan.

    Same online algorithm as :class:`WelfordVariance`, extended with the
    cross-product accumulator ``sum_product_of_devs = Σ (x_i - mean_x)(y_i - mean_y)``.
    Read off via :attr:`variance_x`, :attr:`variance_y`, :attr:`covariance`,
    :meth:`correlation`.
    """

    n_samples: int = 0
    mean_x: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    mean_y: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_sq_dev_x: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_sq_dev_y: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_product_of_devs: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    @torch.no_grad()
    def update(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        """Fold a paired batch into the running state. ``batch_x`` and
        ``batch_y`` must have the same shape."""
        if batch_x.shape != batch_y.shape:
            raise ValueError(
                f"batch_x and batch_y must have the same shape, got {tuple(batch_x.shape)} and {tuple(batch_y.shape)}."
            )
        if batch_x.numel() == 0:
            return

        x = batch_x.detach()
        y = batch_y.detach()
        mean_x_b = x.mean()
        mean_y_b = y.mean()
        dx = x - mean_x_b
        dy = y - mean_y_b
        self.merge(
            WelfordCovariance(
                n_samples=x.numel(),
                mean_x=mean_x_b,
                mean_y=mean_y_b,
                sum_sq_dev_x=dx.square().sum(),
                sum_sq_dev_y=dy.square().sum(),
                sum_product_of_devs=(dx * dy).sum(),
            )
        )

    @torch.no_grad()
    def merge(self, other: "WelfordCovariance") -> None:
        """Combine ``other`` into ``self``. Same correction term as
        :meth:`WelfordVariance.merge`, applied once per second moment
        (``sum_sq_dev_x``, ``sum_sq_dev_y``, ``sum_product_of_devs``)."""
        if other.n_samples == 0:
            return
        if self.n_samples == 0:
            self.n_samples = other.n_samples
            self.mean_x = other.mean_x.detach().clone()
            self.mean_y = other.mean_y.detach().clone()
            self.sum_sq_dev_x = other.sum_sq_dev_x.detach().clone()
            self.sum_sq_dev_y = other.sum_sq_dev_y.detach().clone()
            self.sum_product_of_devs = other.sum_product_of_devs.detach().clone()
            return

        n_a, n_b = self.n_samples, other.n_samples
        n_ab = n_a + n_b
        delta_x = other.mean_x - self.mean_x
        delta_y = other.mean_y - self.mean_y

        self.mean_x = self.mean_x + delta_x * n_b / n_ab
        self.mean_y = self.mean_y + delta_y * n_b / n_ab

        # Three parallel-formula combinations. Coefficient ``n_a * n_b / n_ab``
        # is inlined per term so arithmetic stays on the operand dtype/device.
        self.sum_sq_dev_x = self.sum_sq_dev_x + other.sum_sq_dev_x + delta_x * delta_x * n_a * n_b / n_ab
        self.sum_sq_dev_y = self.sum_sq_dev_y + other.sum_sq_dev_y + delta_y * delta_y * n_a * n_b / n_ab
        self.sum_product_of_devs = (
            self.sum_product_of_devs + other.sum_product_of_devs + delta_x * delta_y * n_a * n_b / n_ab
        )
        self.n_samples = n_ab

    @property
    def variance_x(self) -> torch.Tensor:
        """Population variance of ``x``."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return torch.clamp(self.sum_sq_dev_x / self.n_samples, min=0.0)

    @property
    def variance_y(self) -> torch.Tensor:
        """Population variance of ``y``."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return torch.clamp(self.sum_sq_dev_y / self.n_samples, min=0.0)

    @property
    def covariance(self) -> torch.Tensor:
        """Population covariance of ``(x, y)``."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return self.sum_product_of_devs / self.n_samples

    def correlation(self, eps: float = 1e-8) -> torch.Tensor:
        """Pearson correlation. ``eps`` floors the denominator so a
        constant-variable input returns ``0`` instead of ``NaN``."""
        denom = torch.clamp(self.variance_x.sqrt() * self.variance_y.sqrt(), min=eps)
        return self.covariance / denom

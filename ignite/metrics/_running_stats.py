"""Numerically stable running variance and covariance helpers.

Shared by metrics that need to accumulate variance / covariance from
streaming batches without falling into the catastrophic-cancellation
trap of the naive ``E[X^2] - E[X]^2`` formula. Used by
:class:`~ignite.metrics.regression.PearsonCorrelation` and
:class:`~ignite.metrics.regression.R2Score`; new metrics with the same
need should consume these helpers rather than rolling their own.

Both classes are tensor-type-agnostic dataclasses: callers supply
tensors in whatever dtype and device they want, and the helper
preserves both. For numerical stability under large means, callers
should pre-cast inputs to ``float64`` (the consumer metric classes
already do this in their own ``update`` methods).

Updates follow Welford's online algorithm. Two accumulators can be
combined into one via :meth:`merge`, which implements the Chan /
Welford parallel formula and is the basis for cross-rank distributed
reductions.

References:
    Welford, B. P. (1962). Note on a method for calculating corrected
        sums of squares and products. Technometrics 4 (3), 419 to 420.
    Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae
        and a pairwise algorithm for computing sample variances.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

from dataclasses import dataclass, field

import torch


def _zero() -> torch.Tensor:
    return torch.tensor(0.0)


@dataclass
class WelfordVariance:
    """Numerically stable running mean and variance via Welford's algorithm.

    Accumulates samples in batches via :meth:`update` and reads off the
    mean, variance, or standard deviation through the corresponding
    properties. Two accumulators can be combined with :meth:`merge`,
    which uses the Chan / Welford parallel formula and is the basis for
    distributed reductions.

    No dtype or device handling is performed inside the class: the
    state takes the dtype and device of the first batch passed to
    :meth:`update`. For numerical stability under large means, callers
    should hand in ``float64`` tensors.

    Example::

        ws = WelfordVariance()
        for batch in stream:
            ws.update(batch.to(torch.float64))
        print(ws.mean.item(), ws.variance.item(), ws.std.item())
    """

    n_samples: int = 0
    mean: torch.Tensor = field(default_factory=_zero)
    sum_sq_dev_from_mean: torch.Tensor = field(default_factory=_zero)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Fold a batch of samples into the running state.

        Empty batches are silently ignored. ``batch.mean()`` and
        ``batch.numel()`` perform a full reduction over the input, so
        any shape is accepted and treated as ``numel`` scalar samples.
        """
        if batch.numel() == 0:
            return
        batch = batch.detach()
        n_b = batch.numel()

        mean_b = batch.mean()
        m2_b = (batch - mean_b).square().sum()

        if self.n_samples == 0:
            self.mean = mean_b
            self.sum_sq_dev_from_mean = m2_b
            self.n_samples = n_b
            return

        n_a = self.n_samples
        n_ab = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * n_b / n_ab
        self.sum_sq_dev_from_mean = self.sum_sq_dev_from_mean + m2_b + delta * delta * n_a * n_b / n_ab
        self.n_samples = n_ab

    def merge(self, other: "WelfordVariance") -> None:
        """Combine ``other`` into ``self`` using the Chan / Welford parallel formula."""
        if other.n_samples == 0:
            return
        if self.n_samples == 0:
            self.n_samples = other.n_samples
            self.mean = other.mean.detach().clone()
            self.sum_sq_dev_from_mean = other.sum_sq_dev_from_mean.detach().clone()
            return
        n_a = self.n_samples
        n_b = other.n_samples
        n_ab = n_a + n_b
        delta = other.mean - self.mean
        self.mean = self.mean + delta * n_b / n_ab
        self.sum_sq_dev_from_mean = (
            self.sum_sq_dev_from_mean + other.sum_sq_dev_from_mean + delta * delta * n_a * n_b / n_ab
        )
        self.n_samples = n_ab

    @property
    def variance(self) -> torch.Tensor:
        """Population variance (divisor ``n``). Returns ``0.0`` when empty."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return torch.clamp(self.sum_sq_dev_from_mean / self.n_samples, min=0.0)

    @property
    def std(self) -> torch.Tensor:
        """Population standard deviation (divisor ``n``)."""
        return self.variance.sqrt()


@dataclass
class WelfordCovariance:
    """Numerically stable running covariance for a pair of variables (x, y).

    Exposes :attr:`variance_x`, :attr:`variance_y`, :attr:`covariance`,
    and :meth:`correlation` (Pearson) through the same Welford-style
    online update + Chan / Welford parallel merge as
    :class:`WelfordVariance`.

    Like :class:`WelfordVariance`, the class is dtype and device
    agnostic: state takes the dtype and device of the first batch
    passed to :meth:`update`.
    """

    n_samples: int = 0
    mean_x: torch.Tensor = field(default_factory=_zero)
    mean_y: torch.Tensor = field(default_factory=_zero)
    sum_sq_dev_x: torch.Tensor = field(default_factory=_zero)
    sum_sq_dev_y: torch.Tensor = field(default_factory=_zero)
    sum_product_of_devs: torch.Tensor = field(default_factory=_zero)

    @torch.no_grad()
    def update(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        """Fold a paired batch ``(x_i, y_i)`` into the running state.

        ``batch_x`` and ``batch_y`` must have the same shape; the full
        tensor is reduced as ``numel`` scalar samples.
        """
        if batch_x.shape != batch_y.shape:
            raise ValueError(
                f"batch_x and batch_y must have the same shape, got {tuple(batch_x.shape)} and {tuple(batch_y.shape)}."
            )
        if batch_x.numel() == 0:
            return

        x = batch_x.detach()
        y = batch_y.detach()
        n_b = x.numel()

        mean_x_b = x.mean()
        mean_y_b = y.mean()
        dx_b = x - mean_x_b
        dy_b = y - mean_y_b
        m2_x_b = dx_b.square().sum()
        m2_y_b = dy_b.square().sum()
        cxy_b = (dx_b * dy_b).sum()

        if self.n_samples == 0:
            self.mean_x = mean_x_b
            self.mean_y = mean_y_b
            self.sum_sq_dev_x = m2_x_b
            self.sum_sq_dev_y = m2_y_b
            self.sum_product_of_devs = cxy_b
            self.n_samples = n_b
            return

        n_a = self.n_samples
        n_ab = n_a + n_b
        cross = n_a * n_b / n_ab
        delta_x = mean_x_b - self.mean_x
        delta_y = mean_y_b - self.mean_y

        self.mean_x = self.mean_x + delta_x * n_b / n_ab
        self.mean_y = self.mean_y + delta_y * n_b / n_ab
        self.sum_sq_dev_x = self.sum_sq_dev_x + m2_x_b + delta_x * delta_x * cross
        self.sum_sq_dev_y = self.sum_sq_dev_y + m2_y_b + delta_y * delta_y * cross
        self.sum_product_of_devs = self.sum_product_of_devs + cxy_b + delta_x * delta_y * cross
        self.n_samples = n_ab

    def merge(self, other: "WelfordCovariance") -> None:
        """Combine ``other`` into ``self`` using the Chan / Welford parallel formula."""
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
        n_a = self.n_samples
        n_b = other.n_samples
        n_ab = n_a + n_b
        cross = n_a * n_b / n_ab
        delta_x = other.mean_x - self.mean_x
        delta_y = other.mean_y - self.mean_y

        self.mean_x = self.mean_x + delta_x * n_b / n_ab
        self.mean_y = self.mean_y + delta_y * n_b / n_ab
        self.sum_sq_dev_x = self.sum_sq_dev_x + other.sum_sq_dev_x + delta_x * delta_x * cross
        self.sum_sq_dev_y = self.sum_sq_dev_y + other.sum_sq_dev_y + delta_y * delta_y * cross
        self.sum_product_of_devs = self.sum_product_of_devs + other.sum_product_of_devs + delta_x * delta_y * cross
        self.n_samples = n_ab

    @property
    def variance_x(self) -> torch.Tensor:
        """Population variance of ``x`` (divisor ``n``)."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return torch.clamp(self.sum_sq_dev_x / self.n_samples, min=0.0)

    @property
    def variance_y(self) -> torch.Tensor:
        """Population variance of ``y`` (divisor ``n``)."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return torch.clamp(self.sum_sq_dev_y / self.n_samples, min=0.0)

    @property
    def covariance(self) -> torch.Tensor:
        """Population covariance of ``(x, y)`` (divisor ``n``)."""
        if self.n_samples == 0:
            return torch.tensor(0.0)
        return self.sum_product_of_devs / self.n_samples

    def correlation(self, eps: float = 1e-8) -> torch.Tensor:
        """Pearson correlation coefficient with a small clamp for safety.

        Args:
            eps: floor on the denominator to avoid division by zero when
                one of the variables is constant.
        """
        denom = torch.clamp(self.variance_x.sqrt() * self.variance_y.sqrt(), min=eps)
        return self.covariance / denom

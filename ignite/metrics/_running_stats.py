"""Numerically stable running variance and covariance via Welford's algorithm.

Shared internals for metrics that accumulate variance or covariance from
streaming batches without the catastrophic cancellation of the naive
``E[X^2] - E[X]^2`` formula. Intended consumers in follow-up PRs of
#3748: :class:`R2Score` denominator and :class:`PearsonCorrelation`
cross-product.

State is dtype/device agnostic and takes the dtype/device of the first
batch. Cast to ``float64`` caller-side when stability under large means
matters; the helper does not silently promote.

:meth:`update` and :meth:`merge` share one formula: ``update`` builds
a single-batch accumulator and calls ``merge``.

Distributed reduction
---------------------
``sync_all_reduce`` defaults to ``dist.all_reduce(SUM)``, which is not
the right operation for Welford state (the parallel formula is not a
sum of the per-rank means). The pattern is to gather each rank's
accumulator state and merge pairwise::

    import ignite.distributed as idist

    def compute(self):
        if idist.get_world_size() > 1:
            collected = idist.all_gather(self.welford)
            merged = WelfordVariance()
            for item in collected:
                merged.merge(item)
            return merged.variance
        return self.welford.variance

``idist.all_gather`` of a dataclass instance routes through
``_do_all_gather_object`` (pickle-backed, available for every backend
including NCCL via a Gloo subgroup). Negligible overhead for the
three small tensors carried by these accumulators. Consumers whose
state is significantly larger than a few KB should pack into a flat
tensor before ``all_gather`` and reconstruct on the other side, to
skip the pickle hop.

References:
    Welford, B. P. (1962). Technometrics 4(3), 419-420.
    Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae
    and a pairwise algorithm for computing sample variances.
"""

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
        self.sum_sq_dev_x = (
            self.sum_sq_dev_x + other.sum_sq_dev_x + delta_x * delta_x * n_a * n_b / n_ab
        )
        self.sum_sq_dev_y = (
            self.sum_sq_dev_y + other.sum_sq_dev_y + delta_y * delta_y * n_a * n_b / n_ab
        )
        self.sum_product_of_devs = (
            self.sum_product_of_devs
            + other.sum_product_of_devs
            + delta_x * delta_y * n_a * n_b / n_ab
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

"""Numerically stable running variance and covariance helpers.

Shared by metrics that need to accumulate variance / covariance from
streaming batches without falling into the catastrophic-cancellation
trap of the naive ``E[X^2] - E[X]^2`` formula. Used by
:class:`~ignite.metrics.regression.PearsonCorrelation` and
:class:`~ignite.metrics.regression.R2Score`; new metrics with the same
need should consume these helpers rather than rolling their own.

Both classes keep internal state in ``float64`` regardless of the input
dtype, follow Welford's online algorithm for incremental updates, and
fold accumulators together with the Chan / Welford parallel formula
(used both for batch-wise updates and for cross-rank merges in
distributed settings).

References:
    Welford, B. P. (1962). Note on a method for calculating corrected
        sums of squares and products. Technometrics 4 (3), 419 to 420.
    Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae
        and a pairwise algorithm for computing sample variances.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

from typing import Union

import torch


class WelfordVariance:
    """Numerically stable running mean and variance via Welford's algorithm.

    Accumulates samples in batches via :meth:`update` and reads off the
    mean, variance, or standard deviation through the corresponding
    properties. Two accumulators can be combined with :meth:`merge`,
    which uses the Chan / Welford parallel formula and is the basis for
    distributed reductions.

    State is kept in ``float64`` regardless of input dtype so that the
    classic ``E[X^2] - E[X]^2`` cancellation does not bite for inputs
    with large means.

    Args:
        device: device on which to keep the running-state tensors.
            Default: ``"cpu"``.

    Example:

        .. code-block:: python

            ws = WelfordVariance()
            for batch in stream:
                ws.update(batch)
            print(ws.mean.item(), ws.variance.item(), ws.std.item())
    """

    n_samples: int
    mean: torch.Tensor
    sum_sq_dev_from_mean: torch.Tensor

    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        self._device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Drop all accumulated state."""
        self.n_samples = 0
        self.mean = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.sum_sq_dev_from_mean = torch.tensor(0.0, dtype=torch.float64, device=self._device)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Fold a batch of samples into the running state.

        Empty batches are silently ignored. Inputs of any dtype are
        upcast to ``float64`` for the internal computation.
        """
        if batch.numel() == 0:
            return
        batch64 = batch.detach().to(dtype=torch.float64).flatten()
        n_b = batch64.shape[0]

        mean_b = batch64.mean().to(self._device)
        m2_b = (batch64 - batch64.mean()).square().sum().to(self._device)

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
        """Combine ``other`` into ``self`` using the Chan / Welford parallel formula.

        Used both for cross-rank reduction in distributed settings and
        for any other case where two accumulators need to be combined
        into one without re-iterating the raw data.
        """
        if other.n_samples == 0:
            return
        if self.n_samples == 0:
            self.n_samples = other.n_samples
            self.mean = other.mean.detach().clone().to(self._device)
            self.sum_sq_dev_from_mean = other.sum_sq_dev_from_mean.detach().clone().to(self._device)
            return
        n_a = self.n_samples
        n_b = other.n_samples
        n_ab = n_a + n_b
        delta = other.mean.to(self._device) - self.mean
        self.mean = self.mean + delta * n_b / n_ab
        self.sum_sq_dev_from_mean = (
            self.sum_sq_dev_from_mean + other.sum_sq_dev_from_mean.to(self._device) + delta * delta * n_a * n_b / n_ab
        )
        self.n_samples = n_ab

    @property
    def variance(self) -> torch.Tensor:
        """Population variance (divisor ``n``). Returns ``0.0`` when empty."""
        if self.n_samples == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=self._device)
        return torch.clamp(self.sum_sq_dev_from_mean / self.n_samples, min=0.0)

    @property
    def std(self) -> torch.Tensor:
        """Population standard deviation (divisor ``n``)."""
        return self.variance.sqrt()


class WelfordCovariance:
    """Numerically stable running covariance for a pair of variables (x, y).

    Exposes :attr:`variance_x`, :attr:`variance_y`, :attr:`covariance`,
    and :meth:`correlation` (Pearson) through the same Welford-style
    online update + Chan / Welford parallel merge as
    :class:`WelfordVariance`.

    Args:
        device: device on which to keep the running-state tensors.
            Default: ``"cpu"``.
    """

    n_samples: int
    mean_x: torch.Tensor
    mean_y: torch.Tensor
    sum_sq_dev_x: torch.Tensor
    sum_sq_dev_y: torch.Tensor
    sum_product_of_devs: torch.Tensor

    def __init__(self, device: Union[str, torch.device] = "cpu") -> None:
        self._device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Drop all accumulated state."""
        self.n_samples = 0
        self.mean_x = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.mean_y = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.sum_sq_dev_x = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.sum_sq_dev_y = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.sum_product_of_devs = torch.tensor(0.0, dtype=torch.float64, device=self._device)

    @torch.no_grad()
    def update(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        """Fold a paired batch ``(x_i, y_i)`` into the running state.

        ``batch_x`` and ``batch_y`` must have the same shape; both are
        flattened internally and upcast to ``float64``.
        """
        if batch_x.shape != batch_y.shape:
            raise ValueError(
                f"batch_x and batch_y must have the same shape, got {tuple(batch_x.shape)} and {tuple(batch_y.shape)}."
            )
        if batch_x.numel() == 0:
            return

        x64 = batch_x.detach().to(dtype=torch.float64).flatten()
        y64 = batch_y.detach().to(dtype=torch.float64).flatten()
        n_b = x64.shape[0]

        mean_x_b = x64.mean().to(self._device)
        mean_y_b = y64.mean().to(self._device)
        dx_b = x64 - mean_x_b
        dy_b = y64 - mean_y_b
        m2_x_b = dx_b.square().sum().to(self._device)
        m2_y_b = dy_b.square().sum().to(self._device)
        cxy_b = (dx_b * dy_b).sum().to(self._device)

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
            self.mean_x = other.mean_x.detach().clone().to(self._device)
            self.mean_y = other.mean_y.detach().clone().to(self._device)
            self.sum_sq_dev_x = other.sum_sq_dev_x.detach().clone().to(self._device)
            self.sum_sq_dev_y = other.sum_sq_dev_y.detach().clone().to(self._device)
            self.sum_product_of_devs = other.sum_product_of_devs.detach().clone().to(self._device)
            return
        n_a = self.n_samples
        n_b = other.n_samples
        n_ab = n_a + n_b
        cross = n_a * n_b / n_ab
        delta_x = other.mean_x.to(self._device) - self.mean_x
        delta_y = other.mean_y.to(self._device) - self.mean_y

        self.mean_x = self.mean_x + delta_x * n_b / n_ab
        self.mean_y = self.mean_y + delta_y * n_b / n_ab
        self.sum_sq_dev_x = self.sum_sq_dev_x + other.sum_sq_dev_x.to(self._device) + delta_x * delta_x * cross
        self.sum_sq_dev_y = self.sum_sq_dev_y + other.sum_sq_dev_y.to(self._device) + delta_y * delta_y * cross
        self.sum_product_of_devs = (
            self.sum_product_of_devs + other.sum_product_of_devs.to(self._device) + delta_x * delta_y * cross
        )
        self.n_samples = n_ab

    @property
    def variance_x(self) -> torch.Tensor:
        """Population variance of ``x`` (divisor ``n``)."""
        if self.n_samples == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=self._device)
        return torch.clamp(self.sum_sq_dev_x / self.n_samples, min=0.0)

    @property
    def variance_y(self) -> torch.Tensor:
        """Population variance of ``y`` (divisor ``n``)."""
        if self.n_samples == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=self._device)
        return torch.clamp(self.sum_sq_dev_y / self.n_samples, min=0.0)

    @property
    def covariance(self) -> torch.Tensor:
        """Population covariance of ``(x, y)`` (divisor ``n``)."""
        if self.n_samples == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=self._device)
        return self.sum_product_of_devs / self.n_samples

    def correlation(self, eps: float = 1e-8) -> torch.Tensor:
        """Pearson correlation coefficient with a small clamp for safety.

        Args:
            eps: floor on the denominator to avoid division by zero when
                one of the variables is constant.
        """
        denom = torch.clamp(self.variance_x.sqrt() * self.variance_y.sqrt(), min=eps)
        return self.covariance / denom

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

Two operations matter: :meth:`merge`, which combines two accumulators
into one via the Chan / Welford parallel formula (also the basis for
cross-rank distributed reductions), and :meth:`update`, which folds a
new batch into the running state. ``update`` is the degenerate case of
``merge`` where ``other`` is a freshly-built single-batch accumulator,
and the implementation reflects that: ``update`` builds the batch
accumulator and delegates to ``merge``. There is one formula, not two.

References:
    Welford, B. P. (1962). Note on a method for calculating corrected
        sums of squares and products. Technometrics 4 (3), 419 to 420.
    Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). Updating formulae
        and a pairwise algorithm for computing sample variances.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
"""

from dataclasses import dataclass, field

import torch


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

    # n_samples: count of samples folded in.
    # mean: running sample mean (Welford state).
    # sum_sq_dev_from_mean: Σ (x_i − mean)^2, the second central moment
    # numerator, conventionally called "M2" in the Welford literature.
    n_samples: int = 0
    mean: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_sq_dev_from_mean: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Fold a batch of samples into the running state.

        Implementation: build a single-batch accumulator from ``batch``
        and merge it into ``self``. ``update`` is the degenerate case
        of :meth:`merge` where the right-hand side has just been built
        from one batch; sharing the parallel formula keeps the two
        paths in lock-step.

        Empty batches are silently ignored. ``batch.mean()`` and
        ``batch.numel()`` perform a full reduction over the input, so
        any shape is accepted and treated as ``numel`` scalar samples.
        """
        if batch.numel() == 0:
            return
        batch = batch.detach()
        batch_mean = batch.mean()
        batch_acc = WelfordVariance(
            n_samples=batch.numel(),
            mean=batch_mean,
            sum_sq_dev_from_mean=(batch - batch_mean).square().sum(),
        )
        self.merge(batch_acc)

    def merge(self, other: "WelfordVariance") -> None:
        """Combine ``other`` into ``self`` using the Chan / Welford parallel formula.

        Used in two places: by :meth:`update` (where ``other`` is a
        freshly-built single-batch accumulator), and by callers that
        need to combine independently-accumulated state from elsewhere.
        The motivating second case is distributed training: each rank
        accumulates its own ``WelfordVariance`` over its local samples,
        then at eval time the ranks merge their accumulators rank-by-rank
        to produce the population statistic. Without :meth:`merge` that
        cross-rank reduction would have to re-iterate the raw data,
        which defeats the whole point of an online algorithm.

        Given two accumulators ``A`` and ``B`` with sample counts
        ``n_a, n_b`` and second-central-moment sums ``M2_a, M2_b``, the
        combined ``M2`` over the concatenated stream is::

            M2 = M2_a + M2_b + (mean_b - mean_a)^2 * n_a * n_b / (n_a + n_b)

        The third term is the *correction*: simply adding ``M2_a + M2_b``
        would under-count the variance whenever the two batches have
        different sample means, because each ``M2`` is measured relative
        to its own local mean. The correction folds in the spread of
        the two local means about the combined mean.
        """
        if other.n_samples == 0:
            return
        if self.n_samples == 0:
            # First-time absorb. Copy state so callers cannot mutate
            # ``other`` and silently affect ``self``.
            self.n_samples = other.n_samples
            self.mean = other.mean.detach().clone()
            self.sum_sq_dev_from_mean = other.sum_sq_dev_from_mean.detach().clone()
            return

        n_a = self.n_samples
        n_b = other.n_samples
        n_ab = n_a + n_b
        delta = other.mean - self.mean

        # Standard Welford incremental-mean update, weighted by the
        # fraction of the combined sample size that ``other`` contributes.
        self.mean = self.mean + delta * n_b / n_ab

        # Parallel-formula combined M2. The (delta * delta * ...) term
        # is the correction described in the docstring above.
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
    :class:`WelfordVariance`. The only extension over the univariate
    case is the cross-product accumulator
    :attr:`sum_product_of_devs` =  Σ (x_i - mean_x) (y_i - mean_y).

    Like :class:`WelfordVariance`, the class is dtype and device
    agnostic: state takes the dtype and device of the first batch
    passed to :meth:`update`.
    """

    # Two univariate Welford accumulators worth of state, plus the
    # cross-product term that turns them into a covariance.
    n_samples: int = 0
    mean_x: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    mean_y: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_sq_dev_x: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_sq_dev_y: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    sum_product_of_devs: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    @torch.no_grad()
    def update(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        """Fold a paired batch ``(x_i, y_i)`` into the running state.

        Same trick as :meth:`WelfordVariance.update`: build a single-batch
        accumulator from ``(batch_x, batch_y)`` and merge it. One formula,
        applied twice; see :meth:`merge` for the math.

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
        mean_x_b = x.mean()
        mean_y_b = y.mean()
        dx = x - mean_x_b
        dy = y - mean_y_b
        batch_acc = WelfordCovariance(
            n_samples=x.numel(),
            mean_x=mean_x_b,
            mean_y=mean_y_b,
            sum_sq_dev_x=dx.square().sum(),
            sum_sq_dev_y=dy.square().sum(),
            sum_product_of_devs=(dx * dy).sum(),
        )
        self.merge(batch_acc)

    def merge(self, other: "WelfordCovariance") -> None:
        """Combine ``other`` into ``self`` using the Chan / Welford parallel formula.

        Same correction term as the univariate version, applied three
        times: once for ``sum_sq_dev_x``, once for ``sum_sq_dev_y``, and
        once for ``sum_product_of_devs`` (using ``delta_x * delta_y``
        instead of ``delta * delta``). See
        :meth:`WelfordVariance.merge` for the derivation.
        """
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
        # The correction-term coefficient n_a * n_b / n_ab shows up in
        # every parallel-formula line below; compute it once.
        cross_coef = n_a * n_b / n_ab
        delta_x = other.mean_x - self.mean_x
        delta_y = other.mean_y - self.mean_y

        # Incremental means, weighted by other's share of the new total.
        self.mean_x = self.mean_x + delta_x * n_b / n_ab
        self.mean_y = self.mean_y + delta_y * n_b / n_ab

        # Three parallel-formula combinations: variance of x, variance
        # of y, and covariance of (x, y). Each is M_self + M_other plus
        # a correction for the fact that the two batches had different
        # local means.
        self.sum_sq_dev_x = self.sum_sq_dev_x + other.sum_sq_dev_x + delta_x * delta_x * cross_coef
        self.sum_sq_dev_y = self.sum_sq_dev_y + other.sum_sq_dev_y + delta_y * delta_y * cross_coef
        self.sum_product_of_devs = (
            self.sum_product_of_devs + other.sum_product_of_devs + delta_x * delta_y * cross_coef
        )
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

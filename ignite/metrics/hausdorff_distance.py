from collections.abc import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["HausdorffDistance", "HausdorffDistance95"]


def _extract_boundary_points(mask: torch.Tensor) -> torch.Tensor:
    """Extract the boundary points (foreground pixel/voxel coordinates) of a binary mask.

    A boundary point is any foreground pixel/voxel adjacent to a background one
    (or to the volume edge). For empty masks, returns an empty tensor.

    Args:
        mask: binary mask tensor of shape (H, W) or (D, H, W).

    Returns:
        Tensor of shape (N, ndim) with float coordinates of boundary points.
    """
    # pad with zeros so edge-of-volume foreground always counts as boundary
    padded = torch.nn.functional.pad(mask.float().unsqueeze(0).unsqueeze(0), [1] * (2 * mask.ndim))
    padded = padded.squeeze(0).squeeze(0)

    # neighborhood differences: a foreground voxel is a boundary if any direct neighbor is background
    # use a kernel-free approach with rolls so it works for both 2D and 3D
    fg = padded > 0
    boundary = torch.zeros_like(fg)
    for axis in range(mask.ndim):
        # shift +/- 1 along this axis
        for shift in (-1, 1):
            shifted = torch.roll(fg, shifts=shift, dims=axis)
            # boundary = foreground voxel that has a non-foreground neighbor
            boundary = boundary | (fg & (~shifted))

    # un-pad: remove the 1-cell border we added
    slices = tuple(slice(1, -1) for _ in range(mask.ndim))
    boundary = boundary[slices]

    coords = torch.nonzero(boundary, as_tuple=False).to(dtype=torch.float32)
    return coords


def _directed_hausdorff(a: torch.Tensor, b: torch.Tensor, percentile: float | None = None) -> torch.Tensor:
    """Directed Hausdorff distance from set A to set B.

    For each point in A, computes the distance to the nearest point in B,
    then returns the maximum (or `percentile`-th percentile if given).

    Args:
        a: tensor of shape (Na, ndim) with coordinates of set A.
        b: tensor of shape (Nb, ndim) with coordinates of set B.
        percentile: if given (in [0, 100]), return the percentile of the distances
            instead of the max. Used for HD95 etc.

    Returns:
        Scalar tensor.
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=a.device)

    # pairwise euclidean distances (Na, Nb) — done in chunks to bound memory
    # for typical 2D/3D masks the boundary set is small enough that a direct cdist is fine
    dists = torch.cdist(a, b, p=2)
    nearest = dists.min(dim=1).values
    if percentile is None:
        return nearest.max()
    # torch.quantile uses the same convention as numpy.percentile (q in [0, 1])
    return torch.quantile(nearest, percentile / 100.0)


def _hausdorff_distance_single(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, percentile: float | None = None
) -> torch.Tensor:
    """Symmetric Hausdorff distance between two binary masks (2D or 3D).

    Returns max(d(A, B), d(B, A)). Both empty -> 0. One empty -> inf.
    """
    a = _extract_boundary_points(pred_mask)
    b = _extract_boundary_points(gt_mask)

    if a.numel() == 0 and b.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_mask.device)
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=pred_mask.device)

    d_ab = _directed_hausdorff(a, b, percentile=percentile)
    d_ba = _directed_hausdorff(b, a, percentile=percentile)
    return torch.maximum(d_ab, d_ba)


class HausdorffDistance(Metric):
    r"""Calculates the `Hausdorff distance
    <https://en.wikipedia.org/wiki/Hausdorff_distance>`_ between two segmentation masks.

    .. math::
        HD(A, B) = \max\!\left( \max_{a \in A} \min_{b \in B} \|a - b\|,\ \max_{b \in B} \min_{a \in A} \|b - a\| \right)

    where :math:`A` and :math:`B` are the sets of boundary points of the predicted
    and ground truth masks, respectively.

    Supports binary and multi-class 2D/3D segmentation masks. For multi-class inputs
    the per-class HD is averaged across classes (with optional ``ignore_index``).
    Across batches, per-sample HD is averaged.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - For binary inputs, both tensors should be of shape ``(B, H, W)`` or ``(B, D, H, W)``.
    - For multi-class inputs, ``y_pred`` should be of shape ``(B, C, H, W)`` / ``(B, C, D, H, W)``
      with class logits or one-hot, and ``y`` should be of shape ``(B, H, W)`` / ``(B, D, H, W)``
      with integer class indices, or the same shape as ``y_pred`` (one-hot).

    Args:
        num_classes: Number of classes. ``1`` (default) for binary, ``> 1`` for multi-class.
        ignore_index: Class index to ignore (multi-class only). Default ``None``.
        percentile: If set, returns the given percentile of distances instead of max.
            ``95`` corresponds to the standard ``HD95`` used in medical imaging.
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into
            the form expected by the metric.
        device: specifies which device updates are accumulated on.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method.

    Examples:
        Binary 2D segmentation:

        .. code-block:: python

            metric = HausdorffDistance()
            y_pred = torch.zeros(1, 4, 4)
            y_pred[0, 0, 0] = 1
            y = torch.zeros(1, 4, 4)
            y[0, 3, 3] = 1
            metric.update((y_pred, y))
            metric.compute()  # ~4.2426

    .. versionadded:: 0.6.0
    """

    _state_dict_all_req_keys = ("_sum_of_distances", "_num_examples")

    def __init__(
        self,
        num_classes: int = 1,
        ignore_index: int | None = None,
        percentile: float | None = None,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if num_classes < 1:
            raise ValueError(f"Argument num_classes must be >= 1. Got {num_classes}.")
        if percentile is not None and not (0.0 <= percentile <= 100.0):
            raise ValueError(f"Argument percentile must be in [0, 100]. Got {percentile}.")
        if ignore_index is not None and num_classes <= 1:
            raise ValueError("Argument ignore_index can only be used with num_classes > 1.")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.percentile = percentile

        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_distances = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    def _to_class_masks(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize inputs to integer class-label tensors of shape (B, *spatial)."""
        if self.num_classes == 1:
            # binary: accept (B, *) or (B, 1, *)
            if y_pred.ndim == y.ndim + 1 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)
            if y.ndim == y_pred.ndim + 1 and y.shape[1] == 1:
                y = y.squeeze(1)
            # threshold for floating-point predictions
            if y_pred.is_floating_point():
                y_pred = (y_pred > 0.5).long()
            else:
                y_pred = y_pred.long()
            y = y.long()
        else:
            # multi-class: y_pred either (B, C, *) -> argmax, or already integer (B, *)
            if y_pred.ndim == y.ndim + 1:
                if y_pred.shape[1] != self.num_classes:
                    raise ValueError(
                        f"Expected y_pred channel dim to be {self.num_classes}, got shape {tuple(y_pred.shape)}."
                    )
                y_pred = y_pred.argmax(dim=1)
            else:
                y_pred = y_pred.long()
            # y could also be one-hot
            if y.ndim == y_pred.ndim + 1 and y.shape[1] == self.num_classes:
                y = y.argmax(dim=1)
            else:
                y = y.long()
        return y_pred, y

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        y_pred, y = self._to_class_masks(y_pred, y)

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape after normalization. "
                f"Got y_pred: {tuple(y_pred.shape)} and y: {tuple(y.shape)}."
            )
        if y_pred.ndim not in (3, 4):
            raise ValueError(
                f"Expected inputs of rank 3 (B, H, W) or 4 (B, D, H, W). "
                f"Got rank {y_pred.ndim} with shape {tuple(y_pred.shape)}."
            )

        batch_size = y_pred.shape[0]

        if self.num_classes == 1:
            class_iter: list[int] = [1]
        else:
            class_iter = [c for c in range(self.num_classes) if c != self.ignore_index]

        for i in range(batch_size):
            sample_dists = []
            for c in class_iter:
                pred_mask = (y_pred[i] == c) if self.num_classes > 1 else (y_pred[i] > 0)
                gt_mask = (y[i] == c) if self.num_classes > 1 else (y[i] > 0)
                d = _hausdorff_distance_single(pred_mask, gt_mask, percentile=self.percentile)
                sample_dists.append(d)

            stacked = torch.stack(sample_dists)
            # average across classes for this sample, ignoring inf classes (one mask empty)
            finite = stacked[torch.isfinite(stacked)]
            if finite.numel() == 0:
                # all classes were "one side empty" for this sample -> use raw mean of stack
                # (which will be inf); skip this sample to avoid polluting the running sum
                continue
            sample_value = finite.mean()
            self._sum_of_distances += sample_value.to(dtype=torch.float64, device=self._device)
            self._num_examples += 1

    @sync_all_reduce("_sum_of_distances", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError(
                "HausdorffDistance must have at least one example before it can be computed."
            )
        return (self._sum_of_distances / self._num_examples).item()


class HausdorffDistance95(HausdorffDistance):
    r"""95th-percentile Hausdorff distance (``HD95``), the standard variant used in
    medical-image segmentation benchmarks. Equivalent to
    :class:`HausdorffDistance` with ``percentile=95``.

    .. versionadded:: 0.6.0
    """

    def __init__(
        self,
        num_classes: int = 1,
        ignore_index: int | None = None,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            percentile=95.0,
            output_transform=output_transform,
            device=device,
            skip_unrolling=skip_unrolling,
        )

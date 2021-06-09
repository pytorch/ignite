from typing import Callable, Sequence, Union

import numpy as np
import torch

from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

# import torch.distributed as idist


__all__ = ["FID", "InceptionExtractor"]


def fid_score(
    mu1: torch.Tensor, mu2: torch.Tensor, sigma1: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6
) -> float:
    mu1, mu2 = mu1.cpu(), mu2.cpu()
    sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()

    diff = mu1 - mu2
    import scipy

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.mm(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class InceptionExtractor:
    def __init__(self) -> None:
        try:
            from torchvision import models  # noqa: F401
        except ImportError:
            raise RuntimeError("This contrib module requires torchvision to be installed.")
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    @torch.no_grad()
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[1] < 3 or data.shape[2] < 299 or data.shape[3] < 299:
            raise ValueError(f"Images should be of size greater than 3x299x299 (got {data.shape})")
        return self.model(data).detach()


class FID(Metric):
    r"""Calculates Frechet Inception Distance.

    .. math::
       \text{FID} = \text{|mu1} - \text{mu2|} + \text{Trace(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))}

    where :math:`mu1` and :math:`sigma1` refer to the mean and covariance of the train data and
    :math:`mu2` and :math:`sigma2` refer to the mean and covariance of the test data.

    More details can be found in `Heusel et al. 2002`__.

    __ https://arxiv.org/pdf/1706.08500.pdf

    In addition, a faster and online computation approach can be found in `Chen et al. 2014`__

    __ https://arxiv.org/pdf/2009.14075.pdf

    Remark:

        This implementation is inspired by pytorch_fid package which can be found `here`__.

    __ https://github.com/mseitzer/pytorch-fid

    Args:
        num_features: specifies number of features the evaluation samples should have.
        feature_extractor: A Callable Object for extracting features from input data.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

    .. code-block:: python

        from ignite.metric.gan import FID
        import torch

        y_pred, y = torch.rand(10,2048), torch.rand(10,2048)

        m = FID(num_features=2048)
        m.update((y_pred,y))
        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        num_features: int,
        feature_extractor: Callable = InceptionExtractor(),
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise RuntimeError("This contrib module requires scipy to be installed.")

        if num_features <= 0:
            raise ValueError(f"num of features must be greater to zero (got: {num_features})")
        self._num_features = num_features
        self._feature_extractor = feature_extractor
        self._eps = 1e-6
        super(FID, self).__init__(output_transform=output_transform, device=device)

    @staticmethod
    def _online_update(features: torch.Tensor, total: torch.Tensor, sigma: torch.Tensor) -> None:
        total += features
        sigma += torch.outer(features, features)

    def get_covariance(self, sigma: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates covariance from mean and sum of products of variables
        """
        sub_matrix = torch.outer(total, total)
        sub_matrix = sub_matrix / self._num_examples
        return (sigma - sub_matrix) / (self._num_examples - 1)

    @staticmethod
    def _check_feature_input(train: torch.Tensor, test: torch.Tensor) -> None:
        for feature in [train, test]:
            if feature.dim() != 2:
                raise ValueError(f"Features must be a tensor of dim 2 (got: {feature.dim()})")
            if feature.shape[0] == 0:
                raise ValueError(f"Batch size should be greater than one (got: {feature.shape[0]})")
            if feature.shape[1] == 0:
                raise ValueError(f"Feature size should be greater than one (got: {feature.shape[1]})")
        if train.shape[0] != test.shape[0] or train.shape[1] != test.shape[1]:
            raise ValueError(
                f"Number of Training Features and Testing Features should be equal ({train.shape} != {test.shape})"
            )

    @reinit__is_reduced
    def reset(self) -> None:
        self._train_sigma = torch.zeros((self._num_features, self._num_features), dtype=torch.float64).to(self._device)
        self._train_total = torch.zeros(self._num_features, dtype=torch.float64).to(self._device)
        self._test_sigma = torch.zeros((self._num_features, self._num_features), dtype=torch.float64).to(self._device)
        self._test_total = torch.zeros(self._num_features, dtype=torch.float64).to(self._device)
        self._num_examples = 0
        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:

        # Extract the features from the outputs
        train_features = self._feature_extractor(output[0].detach()).to(self._device)
        test_features = self._feature_extractor(output[1].detach()).to(self._device)

        # Check the feature shapess
        self._check_feature_input(train_features, test_features)

        # Updates the mean and covariance for the train features
        for i, features in enumerate(train_features, start=self._num_examples + 1):
            self._online_update(features, self._train_total, self._train_sigma)

        # Updates the mean and covariance for the test features
        for i, features in enumerate(test_features, start=self._num_examples + 1):
            self._online_update(features, self._test_total, self._test_sigma)

        self._num_examples += train_features.shape[0]

    @sync_all_reduce("_num_examples", "_train_total", "_test_total", "_train_sigma", "_test_sigma")
    def compute(self) -> float:
        return fid_score(
            mu1=self._train_total / self._num_examples,
            mu2=self._test_total / self._num_examples,
            sigma1=self.get_covariance(self._train_sigma, self._train_total),
            sigma2=self.get_covariance(self._test_sigma, self._test_total),
            eps=self._eps,
        )

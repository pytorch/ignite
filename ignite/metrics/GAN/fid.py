from typing import Sequence, Union
import numpy as np
import torch
from scipy.linalg import sqrtm

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["FID", "InceptionExtractor"]


class InceptionExtractor:

    def __init__(self):
        from torchvision import models
        self.model = models.inception_v3(init_weights=True)
        self.model.fc = torch.nn.Sequential()
        self.model.eval()

    def __call__(self, data):
        return self.model(data).detach()

class Record:
    r"""Contains mean and covariance records for train and test data.

    """

    def __init__(self, device="cpu"):
        self.covariance_matrix = None
        self.mean = None
        self.num_examples = 0
        self.device = torch.device(device)

    def reset(self, num_features):
        self.covariance_matrix = torch.zeros((num_features, num_features), dtype=torch.float64).to(self.device)
        self.mean = torch.zeros(num_features, dtype=torch.float64).to(self.device)
        self.num_examples = 0

    def get_covariance(self):
        return self.covariance_matrix / (self.num_examples - 1)


class FID(Metric):
    r"""Calculates Frechet Inception Distance.

    .. math::
       \text{FID} = \mod{\text{mu1} - \text{mu2}} + \text{Trace(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))}

    where :math:`mu1` and :math:`sigma1` refer to the mean and covariance of the train data and
    :math:`mu2` and :math:`sigma2` refer to the mean and covariance of the test data.

    More details can be found in `Heusel et al. 2002`__.

    __ https://arxiv.org/pdf/1706.08500.pdf

    In addition, a faster and online computation approach can be found in `Chen et al. 2014`__

    __ https://arxiv.org/pdf/2009.14075.pdf

    Remark:

        This implementation is inspired by pytorch_fid package.

    Args:
        eps: small value added during matric division if denominator is 0.
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
        from ignite.metric.GAN import FID
        import torch

        train, test = torch.rand(10,2048), torch.rand(10,2048)

        m = FID()
        m.update((train,test))
        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        feature_extractor=lambda x: x,
        eps=10 ** -6,
        output_transform=lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu")
    ):
        self._feature_extractor = feature_extractor
        self._train_record = Record(device=device)
        self._test_record = Record(device=device)
        self.eps = eps
        super(FID, self).__init__(output_transform=output_transform, device=device)

    def calculate_fid(self):
        mu1 = self._train_record.mean
        mu2 = self._test_record.mean

        sigma1 = self._train_record.get_covariance()
        sigma2 = self._test_record.get_covariance()

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = sqrtm(sigma1.mm(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = f"fid calculation produces singular product. Adding {self.eps} to diagonal of cov estimates"
            print(msg)
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    @staticmethod
    def _calculate_statistics(feature_list, record):

        feature_length = len(feature_list)
        if type(record.covariance_matrix) != torch.Tensor:
            record.reset(feature_length)
        record.num_examples += 1

        # Calculating difference between new sample and old and new means
        mean_difference = feature_list - record.mean
        record.mean += mean_difference / record.num_examples
        new_mean_difference = feature_list - record.mean

        # Outer product to obtain pairwise covariance between each features
        record.covariance_matrix += torch.outer(mean_difference, new_mean_difference)

    def _update_from_features(self, train_data, test_data):
        # Updates mean and covariance for train data
        for features in train_data:
            self._calculate_statistics(features, self._train_record)

        # Updates mean and covariance for test data
        for features in test_data:
            self._calculate_statistics(features, self._test_record)

    @staticmethod
    def _check_feature_input(train: torch.Tensor, test: torch.Tensor) -> None:
        for feature in [train, test]:
            if feature.dim() != 2:
                raise ValueError(f"Features must be a tensor of dim 2 (got: {feature.dim()})")
            if feature.shape[0] == 0:
                raise ValueError(f"Batch size should be greater than one (got: {feature.shape[0]})")
            if feature.shape[1] == 0:
                raise ValueError(f"Feature size should be greater than one (got: {feature.shape[1]})")
        if train.shape[0] != test.shape[0] != 0 or train.shape[1] != test.shape[1]:
            raise ValueError(f"Number of Training Features and Testing Features should be equal "
                             f"({train.shape} != {test.shape})")

    @reinit__is_reduced
    def reset(self):
        del self._train_record
        del self._test_record
        self._train_record = Record(device=self._device)
        self._test_record = Record(device=self._device)
        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        train_data = self._feature_extractor(output[0].detach())
        test_data = self._feature_extractor(output[1].detach())
        self._check_feature_input(train_data, test_data)
        self._update_from_features(train_data, test_data)

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._train_record.num_examples != self._test_record.num_examples:
            raise NotComputableError("Number of Train and Test samples provided so far do not match.")
        return self.calculate_fid()

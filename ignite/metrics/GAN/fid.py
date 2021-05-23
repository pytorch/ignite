import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.GAN.utils import Record, files
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

torch.set_printoptions(precision=8)


class FID(Metric):
    def __init__(self, model=None, eps=10 ** -6, output_transform=lambda x: x, device="cpu"):
        self._num_examples = 0
        self._train_record = Record(device=device)
        self._test_record = Record(device=device)
        self._active_record = None
        self._model = model
        self.device = torch.device(device)
        self.eps = eps
        super(FID, self).__init__(output_transform=output_transform, device=device)

    def calculate_fid(self):
        """
        Compute the Frechet Inception Distance based on given pair of mean and covariances.

        Returns:
            Float representing the Frechet Inception Distance.

        .. versionadded:: 0.5.0
        """
        mu1 = self._train_record.mean
        mu2 = self._test_record.mean

        sigma1 = self._train_record.get_covariance()
        sigma2 = self._test_record.get_covariance()

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = sqrtm(sigma1.mm(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % self.eps
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

    def _calculate_statistics(self, feature_list):
        feature_length = len(feature_list)
        if type(self._active_record.covariance_matrix) != torch.Tensor:
            self._active_record.reset(feature_length)
        self._active_record._num_examples += 1
        mean_difference = feature_list - self._active_record.mean
        self._active_record.mean += mean_difference / self._active_record._num_examples
        new_mean_difference = feature_list - self._active_record.mean
        self._active_record.covariance_matrix += torch.outer(mean_difference, new_mean_difference)

    def _get_features(self, act):
        return self._model.pred(act)

    def _update_from_data(self, data):
        features = self._get_features(data)
        for feature in features:
            self._calculate_statistics(feature)

    def _update_from_images(self, train_images, test_images):
        self._active_record = self._train_record
        self._update_from_data(train_images)
        self._active_record = self._test_record
        self._update_from_data(test_images)

    def _update_from_file(self, path):
        image = torch.tensor(Image.open(str(path))).to(self.device)
        features = self._get_features(image)
        self._calculate_statistics(features)

    def _update_from_paths(self, train_paths, test_paths):
        self._active_record = self._train_record
        for path in train_paths:
            self._update_from_file(path)
        self._active_record = self._test_record
        for path in test_paths:
            self._update_from_file(path)

    def _update_from_folder(self, train_folder, test_folder):
        self._active_record = self._train_record
        for file in files(train_folder):
            self._update_from_file(file)
        self._active_record = self._test_record
        for file in files(test_folder):
            self._update_from_file(file)

    def _update_from_features(self, train_data, test_data):
        self._active_record = self._train_record
        for features in train_data:
            self._calculate_statistics(features)
        self._active_record = self._test_record
        for features in test_data:
            self._calculate_statistics(features)

    @reinit__is_reduced
    def reset(self):
        self._num_examples = 0
        del self._train_record
        del self._test_record
        self._train_record = Record()
        self._test_record = Record()
        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output):
        train_data, test_data, mode = output[0], output[1], output[2]
        if mode == "features":
            self._update_from_features(
                torch.tensor(train_data).to(self.device), torch.tensor(test_data).to(self.device)
            )
        if mode == "images":
            self._update_from_images(torch.tensor(train_data).to(self.device), torch.tensor(test_data).to(self.device))
        if mode == "file_path":
            self._update_from_paths(train_data, test_data)
        if mode == "folder":
            self._update_from_folder()

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._active_record._num_examples < 2:
            raise NotComputableError("FID must have at least two example before it can be computed.")
        if self._train_record._num_examples != self._test_record._num_examples:
            raise NotComputableError("Number of Train and Test samples provided so far do not match.")
        return self.calculate_fid()

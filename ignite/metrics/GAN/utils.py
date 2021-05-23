import os

import torch


class Record:
    def __init__(self, device="cpu"):
        self.covariance_matrix = None
        self.mean = None
        self._num_examples = 0
        self.device = torch.device(device)

    def reset(self, num_features):
        self.covariance_matrix = torch.zeros((num_features, num_features)).to(self.device)
        self.mean = torch.zeros(num_features).to(self.device)
        self._num_examples = 0

    def get_covariance(self):
        return self.covariance_matrix / (self._num_examples - 1)


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.lower().endswith((".png", ".jpg", ".jpeg")):
            yield os.path.join(path, file)

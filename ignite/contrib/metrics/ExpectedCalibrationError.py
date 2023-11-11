import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

class ExpectedCalibrationError(Metric):
    def __init__(self, num_bins=10, device=None):
        super(ExpectedCalibrationError, self).__init__()
        self.num_bins = num_bins
        self.device = device
        self.reset()

    def reset(self):
        self.confidences = torch.tensor([], device=self.device)
        self.corrects = torch.tensor([], device=self.device)

    def update(self, output):
        y_pred, y = output

        assert y_pred.dim() == 2 and y_pred.shape[1] == 2, "This metric is for binary classification."

        softmax_probs = torch.softmax(y_pred, dim=1)
        max_probs, predicted_class = torch.max(softmax_probs, dim=1)

        self.confidences = torch.cat((self.confidences, max_probs))
        self.corrects = torch.cat((self.corrects, predicted_class == y))

    def compute(self):
        if self.confidences.numel() == 0:
            raise NotComputableError("ExpectedCalibrationError must have at least one example before it can be computed.")

        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=self.device)

        bin_indices = torch.searchsorted(bin_edges, self.confidences)

        ece = 0.0
        bin_sizes = torch.zeros(self.num_bins, device=self.device)
        bin_accuracies = torch.zeros(self.num_bins, device=self.device)

        for i in range(self.num_bins):
            mask = bin_indices == i
            bin_confidences = self.confidences[mask]
            bin_corrects = self.corrects[mask]

            accuracy = torch.mean(bin_corrects)

            avg_confidence = torch.mean(bin_confidences)

            bin_size = bin_confidences.numel()
            ece += (bin_size / len(self.confidences)) * abs(accuracy - avg_confidence)
            bin_sizes[i] = bin_size
            bin_accuracies[i] = accuracy

        return ece

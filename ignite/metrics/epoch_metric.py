import torch

from ignite.metrics.metric import Metric


class EpochMetric(Metric):
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
    datatype should be `float32`. Target datatype should be `long`.

    Args:
        compute_fn (callable): a callable 

    - `update` must receive output of the form `(y_pred, y)`. If target shape is `(batch_size, n_classes)`
    and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`
    """

    def __init__(self, compute_fn, output_transform=lambda x: x):
        assert callable(compute_fn), "Argument compute_fn should be callable"
        super(EpochMetric, self).__init__(output_transform=output_transform)

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.long)

    def update(self, output):
        y_pred, y = output

        assert 1 <= y_pred.ndimension() <= 2, "Predictions should be of shape (batch_size, n_classes)"
        assert 1 <= y.ndimension() <= 2, "Targets should be of shape (batch_size, n_classes)"

        if y.ndimension() == 2:
            assert torch.equal(y ** 2, y), 'Targets should be binary (0 or 1)'

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

    def compute(self):
        pass

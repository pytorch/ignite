from typing import Callable, List, Optional, Union

import torch

from ignite.metrics import EpochMetric


class ClassificationReport(EpochMetric):
    """Build a text report showing the main classification metrics. applying
    `sklearn.metrics.classification_report https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.classification_report.html`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        labels: optional list of label indices to include in the report
        check_compute_fn: Default False. If True, `classification-report
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification-report.html>`_
            is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.
        output_dict: Default False. If True, the result is a dict, otherwise is a str
        device: optional device specification for internal storage.

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            return y_pred, y

        classification_report = ClassificationReport(activated_output_transform)

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        labels: Optional[List[str]] = None,
        output_dict: bool = False,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        try:
            from sklearn.metrics import classification_report  # noqa: F401
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        self.classification_report_compute = self.get_classification_report_fn()

        self.labels = labels
        self.output_dict = output_dict

        super(ClassificationReport, self).__init__(
            self.classification_report_compute,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
        )

    def get_classification_report_fn(self) -> Callable[[torch.Tensor, torch.Tensor], Union[str, dict]]:
        """Return a function computing classification report from scikit-learn."""
        from sklearn.metrics import classification_report

        def wrapper(y_targets: torch.Tensor, y_preds: torch.Tensor) -> Union[str, dict]:
            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return classification_report(y_true, y_pred, labels=self.labels, output_dict=self.output_dict)

        return wrapper

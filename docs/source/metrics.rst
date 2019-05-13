ignite.metrics
==============

Metrics provide a way to compute various quantities of interest in an online
fashion without having to store the entire output history of a model.

In practice a user needs to attach the metric instance to an engine. The metric
value is then computed using the output of the engine's `process_function`:

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y

        engine = Engine(process_function)
        metric = Accuracy()
        metric.attach(engine, "accuracy")

If the engine's output is not in the format `y_pred, y`, the user can
use the `output_transform` argument to transform it:

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return {'y_pred': y_pred, 'y_true': y, ...}

        engine = Engine(process_function)

        def output_transform(output):
            # `output` variable is returned by above `process_function`
            y_pred = output['y_pred']
            y = output['y_true']
            return y_pred, y  # output format is according to `Accuracy` docs

        metric = Accuracy(output_transform=output_transform)
        metric.attach(engine, "accuracy")

Metrics could be combined together to form new metrics. This could be done through arithmetics, such
as ``metric1 + metric2``, use PyTorch operators, such as ``(metric1 + metric2).pow(2).mean()``,
or use a lambda function, such as ``MetricsLambda(lambda a, b: torch.mean(a + b), metric1, metric2)``.

For example:

    .. code-block:: python

        precision = Precision(average=False)
        recall = Recall(average=False)
        F1 = (precision * recall * 2 / (precision + recall)).mean()

    .. note::  This example computes the mean of F1 across classes. To combine
        precision and recall to get F1 or other F metrics, we have to be careful
        that `average=False`, i.e. to use the unaveraged precision and recall,
        otherwise we will not be computing F-beta metrics.

Metrics also support indexing operation (if metric's result is a vector/matrix/tensor). For example, this can be useful to compute mean metric (e.g. precision, recall or IoU) ignoring the background:

    .. code-block:: python

        cm = ConfusionMatrix(num_classes=10)
        iou_metric = IoU(cm)
        iou_no_bg_metric = iou_metric[:9]  # We assume that the background index is 9
        mean_iou_no_bg_metric = iou_no_bg_metric.mean()
        # mean_iou_no_bg_metric.compute() -> tensor(0.12345)


.. currentmodule:: ignite.metrics

.. autoclass:: Accuracy

.. autoclass:: Loss

.. autoclass:: MeanAbsoluteError

.. autoclass:: MeanPairwiseDistance

.. autoclass:: MeanSquaredError

.. autoclass:: Metric
    :members:

.. autoclass:: Precision

.. autoclass:: Recall

.. autoclass:: RootMeanSquaredError

.. autoclass:: TopKCategoricalAccuracy

.. autoclass:: EpochMetric

.. autoclass:: RunningAverage

.. autoclass:: MetricsLambda

.. autoclass:: ConfusionMatrix

.. autofunction:: IoU

.. autofunction:: mIoU

.. autoclass:: VariableAccumulation

.. autoclass:: Average

.. autoclass:: GeometricAverage

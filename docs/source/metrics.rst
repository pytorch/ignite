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

Metrics could be combined together to form a new metric through arithmetics,
for example:

    .. code-block:: python

        precision = Precision(average=False)
        recall = Recall(average=False)
        F1 = precision * recall * 2 / (precision + recall)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    .. note::  This example computes the mean of F1 across classes. To combine 
        precision and recall to get F1 or other F metrics, we have to be careful 
        that `average=False`, i.e. to use the unaveraged precision and recall, 
        otherwise we will not be computing F-beta metrics.


.. currentmodule:: ignite.metrics

.. autoclass:: Accuracy

.. autoclass:: BinaryAccuracy

.. autoclass:: CategoricalAccuracy

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

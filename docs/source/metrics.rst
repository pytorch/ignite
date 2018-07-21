ignite.metrics
==============

Metrics provide a way to compute various quantities of interest in an online
fashion without having to store the entire output history of a model.

In practice user needs to attach metric instance to engine. Metric value is
computed using engine's output:

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y

        engine = Engine(process_function)
        metric = CategoricalAccuracy()
        metric.attach(engine, "accuracy")

If engine's output is not in the format `y_pred, y`, user can use `output_transform`:

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return {'y_pred': y_pred, 'y_true': y, ...}

        engine = Engine(process_function)

        def output_transform(output):
            # `output` variable is returned by above `process_function`
            y_pred = output['y_pred']
            y = output['y_true']
            return y_pred, y  # output format is according to `CategoricalAccuracy` docs

        metric = CategoricalAccuracy(output_transform=output_transform)
        metric.attach(engine, "accuracy")


.. currentmodule:: ignite.metrics

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

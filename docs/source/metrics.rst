ignite.metrics
==============

Metrics provide a way to compute various quantities of interest in an online
fashion without having to store the entire output history of a model.

In practice a user needs to attach the metric instance to an engine. The metric
value is then computed using the output of the engine's ``process_function``:

.. code-block:: python

    def process_function(engine, batch):
        # ...
        return y_pred, y

    engine = Engine(process_function)
    metric = Accuracy()
    metric.attach(engine, "accuracy")

If the engine's output is not in the format ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``, the user can
use the ``output_transform`` argument to transform it:

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


.. Note ::

    Most of implemented metrics are adapted to distributed computations and reduce their internal states across supported
    devices before computing metric value. This can be helpful to run the evaluation on multiple nodes/GPU instances/TPUs
    with a distributed data sampler. Following code snippet shows in detail how to use metrics:

    .. code-block:: python

        device = "cuda:{}".format(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank, ],
                                                          output_device=local_rank)
        test_sampler = DistributedSampler(test_dataset)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()}, device=device)

.. Note ::

   Metrics cannot be serialized using `pickle` module because the implementation is based on lambda functions.
   Therefore, use the third party library `dill` to overcome the limitation of `pickle`.

Metric arithmetics
------------------

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
    that ``average=False``, i.e. to use the unaveraged precision and recall,
    otherwise we will not be computing F-beta metrics.

Metrics also support indexing operation (if metric's result is a vector/matrix/tensor). For example, this can be useful to compute mean metric (e.g. precision, recall or IoU) ignoring the background:

.. code-block:: python

    cm = ConfusionMatrix(num_classes=10)
    iou_metric = IoU(cm)
    iou_no_bg_metric = iou_metric[:9]  # We assume that the background index is 9
    mean_iou_no_bg_metric = iou_no_bg_metric.mean()
    # mean_iou_no_bg_metric.compute() -> tensor(0.12345)

How to create a custom metric
-----------------------------

To create a custom metric one needs to create a new class inheriting from :class:`~ignite.metrics.Metric` and override
three methods :

- :meth:`~ignite.metrics.Metric.reset()` : resets internal variables and accumulators
- :meth:`~ignite.metrics.Metric.update()` : updates internal variables and accumulators with provided batch output ``(y_pred, y)``
- :meth:`~ignite.metrics.Metric.compute()` : computes custom metric and return the result

For example, we would like to implement for illustration purposes a multi-class accuracy metric with some
specific condition (e.g. ignore user-defined classes):

.. code-block:: python

    from ignite.metrics import Metric
    from ignite.exceptions import NotComputableError

    # These decorators helps with distributed settings
    from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


    class CustomAccuracy(Metric):

        def __init__(self, ignored_class, output_transform=lambda x: x):
            self.ignored_class = ignored_class
            self._num_correct = None
            self._num_examples = None
            super(CustomAccuracy, self).__init__(output_transform=output_transform)

        @reinit__is_reduced
        def reset(self):
            self._num_correct = 0
            self._num_examples = 0
            super(CustomAccuracy, self).reset()

        @reinit__is_reduced
        def update(self, output):
            y_pred, y = output

            indices = torch.argmax(y_pred, dim=1)

            mask = (y != self.ignored_class)
            mask &= (indices != self.ignored_class)
            y = y[mask]
            indices = indices[mask]
            correct = torch.eq(indices, y).view(-1)

            self._num_correct += torch.sum(correct).item()
            self._num_examples += correct.shape[0]

        @sync_all_reduce("_num_examples", "_num_correct")
        def compute(self):
            if self._num_examples == 0:
                raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
            return self._num_correct / self._num_examples


We imported necessary classes as :class:`~ignite.metrics.Metric`, :class:`~ignite.exceptions.NotComputableError` and
decorators to adapt the metric for distributed setting. In ``reset`` method, we reset internal variables ``_num_correct``
and ``_num_examples`` which are used to compute the custom metric. In ``updated`` method we define how to update
the internal variables. And finally in ``compute`` method, we compute metric value.

We can check this implementation in a simple case:

.. code-block:: python

    import torch
    torch.manual_seed(8)

    m = CustomAccuracy(ignored_class=3)

    batch_size = 4
    num_classes = 5

    y_pred = torch.rand(batch_size, num_classes)
    y = torch.randint(0, num_classes, size=(batch_size, ))

    m.update((y_pred, y))
    res = m.compute()

    print(y, torch.argmax(y_pred, dim=1))
    # Out: tensor([2, 2, 2, 3]) tensor([2, 1, 0, 0])

    print(m._num_correct, m._num_examples, res)
    # Out: 1 3 0.3333333333333333

Metrics and its usages
----------------------

By default, `Metrics` are epoch-wise, it means

- :meth:`~ignite.metrics.Metric.reset()` is triggered every ``EPOCH_STARTED`` (See :class:`~ignite.engine.events.Events`).
- :meth:`~ignite.metrics.Metric.update()` is triggered every ``ITERATION_COMPLETED``.
- :meth:`~ignite.metrics.Metric.compute()` is triggered every ``EPOCH_COMPLETED``.

Usages can be user defined by creating a class inheriting for :class:`~ignite.metrics.MetricUsage`. See the list below of usages.

Complete list of usages
```````````````````````

    - :class:`~ignite.metrics.MetricUsage`
    - :class:`~ignite.metrics.EpochWise`
    - :class:`~ignite.metrics.BatchWise`
    - :class:`~ignite.metrics.BatchFiltered`

Metrics and distributed computations
------------------------------------

In the above example, ``CustomAccuracy`` has ``reset``, ``update``, ``compute`` methods
decorated with ``reinit__is_reduced``, ``sync_all_reduce``. The purpose of these features is to adapt metrics in distributed
computations on supported backend and devices (:doc:`distributed`). More precisely, in the above example we
added ``@sync_all_reduce("_num_examples", "_num_correct")`` over ``compute`` method. This means that when ``compute`` method
is called, metric's interal variables ``self._num_examples`` and ``self._num_correct`` are summed up over all participating
devices. Therefore, once collected, these internal variables can be used to compute the final metric value.

Complete list of metrics
------------------------

    - :class:`~ignite.metrics.Accuracy`
    - :class:`~ignite.metrics.Average`
    - :class:`~ignite.metrics.ConfusionMatrix`
    - :meth:`~ignite.metrics.DiceCoefficient`
    - :class:`~ignite.metrics.EpochMetric`
    - :meth:`~ignite.metrics.Fbeta`
    - :class:`~ignite.metrics.GeometricAverage`
    - :meth:`~ignite.metrics.IoU`
    - :meth:`~ignite.metrics.mIoU`
    - :class:`~ignite.metrics.Loss`
    - :class:`~ignite.metrics.MeanAbsoluteError`
    - :class:`~ignite.metrics.MeanPairwiseDistance`
    - :class:`~ignite.metrics.MeanSquaredError`
    - :class:`~ignite.metrics.Metric`
    - :class:`~ignite.metrics.MetricsLambda`
    - :class:`~ignite.metrics.Precision`
    - :class:`~ignite.metrics.Recall`
    - :class:`~ignite.metrics.RootMeanSquaredError`
    - :class:`~ignite.metrics.RunningAverage`
    - :class:`~ignite.metrics.SSIM`
    - :class:`~ignite.metrics.TopKCategoricalAccuracy`
    - :class:`~ignite.metrics.VariableAccumulation`

.. currentmodule:: ignite.metrics

.. autoclass:: Accuracy

.. autoclass:: Average

.. autoclass:: ConfusionMatrix

.. autofunction:: DiceCoefficient

.. autoclass:: EpochMetric

.. autofunction:: Fbeta

.. autoclass:: GeometricAverage

.. autofunction:: IoU

.. autofunction:: mIoU

.. autoclass:: Loss

.. autoclass:: MeanAbsoluteError

.. autoclass:: MeanPairwiseDistance

.. autoclass:: MeanSquaredError

.. autoclass:: Metric
    :members:

.. autoclass:: MetricsLambda

.. autoclass:: Precision

.. autoclass:: Recall

.. autoclass:: RootMeanSquaredError

.. autoclass:: RunningAverage

.. autoclass:: SSIM

.. autoclass:: TopKCategoricalAccuracy

.. autoclass:: VariableAccumulation

.. autoclass:: MetricUsage

.. autoclass:: EpochWise

.. autoclass:: BatchWise

.. autoclass:: BatchFiltered


.. currentmodule:: ignite.metrics.metric

.. autofunction:: sync_all_reduce

.. autofunction:: reinit__is_reduced

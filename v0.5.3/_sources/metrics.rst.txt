ignite.metrics
==============

Metrics provide a way to compute various quantities of interest in an online
fashion without having to store the entire output history of a model.

.. _attach-engine:

Attach Engine API
------------------

The metrics as stated above are computed in a online fashion, which means that the metric instance accumulates some internal counters on
each iteration and metric value is computed once the epoch is ended. Internal counters are reset after every epoch. In practice, this is done with the
help of three methods: :meth:`~ignite.metrics.metric.Metric.reset()`, :meth:`~ignite.metrics.metric.Metric.update()` and :meth:`~ignite.metrics.metric.Metric.compute()`.

Therefore, a user needs to attach the metric instance to the engine so that the above three methods can be triggered on execution of certain :class:`~ignite.engine.events.Events`.
The :meth:`~ignite.metrics.metric.Metric.reset()` method is triggered on ``EPOCH_STARTED`` event and it is responsible to reset the metric to its initial state. The :meth:`~ignite.metrics.metric.Metric.update()` method is triggered
on ``ITERATION_COMPLETED`` event as it updates the state of the metric using the passed batch output. And :meth:`~ignite.metrics.metric.Metric.compute()` is triggered on ``EPOCH_COMPLETED``
event. It computes the metric based on its accumulated states. The metric value is computed using the output of the engine's ``process_function``:

.. code-block:: python

    from ignite.engine import Engine
    from ignite.metrics import Accuracy

    def process_function(engine, batch):
        # ...
        return y_pred, y

    engine = Engine(process_function)
    metric = Accuracy()
    metric.attach(engine, "accuracy")
    # ...
    state = engine.run(data)
    print(f"Accuracy: {state.metrics['accuracy']}")


If the engine's output is not in the format ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``, the user can
use the ``output_transform`` argument to transform it:

.. code-block:: python

    from ignite.engine import Engine
    from ignite.metrics import Accuracy

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
    # ...
    state = engine.run(data)
    print(f"Accuracy: {state.metrics['accuracy']}")


.. warning::

    Please, be careful when using ``lambda`` functions to setup multiple ``output_transform`` for multiple metrics

    .. code-block:: python

        # Wrong
        # metrics_group = [Accuracy(output_transform=lambda output: output[name]) for name in names]
        # As lambda can not store `name` and all `output_transform` will use the last `name`

        # A correct way. For example, using functools.partial
        from functools import partial

        def ot_func(output, name):
            return output[name]

        metrics_group = [Accuracy(output_transform=partial(ot_func, name=name)) for name in names]

    For more details, see `here <https://discuss.pytorch.org/t/evaluate-multiple-models-with-one-evaluator-results-weird-metrics/96695>`_

.. Note ::

    Most of implemented metrics are adapted to distributed computations and reduce their internal states across supported
    devices before computing metric value. This can be helpful to run the evaluation on multiple nodes/GPU instances/TPUs
    with a distributed data sampler. Following code snippet shows in detail how to use metrics:

    .. code-block:: python

        device = f"cuda:{local_rank}"
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


Reset, Update, Compute API
--------------------------

User can also call directly the following methods on the metric:

- :meth:`~ignite.metrics.metric.Metric.reset()` : resets internal variables and accumulators
- :meth:`~ignite.metrics.metric.Metric.update()` : updates internal variables and accumulators with provided batch output ``(y_pred, y)``
- :meth:`~ignite.metrics.metric.Metric.compute()` : computes custom metric and return the result

This API gives a more fine-grained/custom usage on how to compute a metric. For example:

.. code-block:: python

    from ignite.metrics import Precision

    # Define the metric
    precision = Precision()

    # Start accumulation:
    for x, y in data:
        y_pred = model(x)
        precision.update((y_pred, y))

    # Compute the result
    print("Precision: ", precision.compute())

    # Reset metric
    precision.reset()

    # Start new accumulation:
    for x, y in data:
        y_pred = model(x)
        precision.update((y_pred, y))

    # Compute new result
    print("Precision: ", precision.compute())


Metric arithmetics
------------------

Metrics could be combined together to form new metrics. This could be done through arithmetics, such
as ``metric1 + metric2``, use PyTorch operators, such as ``(metric1 + metric2).pow(2).mean()``,
or use a lambda function, such as ``MetricsLambda(lambda a, b: torch.mean(a + b), metric1, metric2)``.

For example:

.. code-block:: python

    from ignite.metrics import Precision, Recall

    precision = Precision(average=False)
    recall = Recall(average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()

.. note::  This example computes the mean of F1 across classes. To combine
    precision and recall to get F1 or other F metrics, we have to be careful
    that ``average=False``, i.e. to use the unaveraged precision and recall,
    otherwise we will not be computing F-beta metrics.

Metrics also support indexing operation (if metric's result is a vector/matrix/tensor). For example, this can be useful to compute mean metric (e.g. precision, recall or IoU) ignoring the background:

.. code-block:: python

    from ignite.metrics import ConfusionMatrix

    cm = ConfusionMatrix(num_classes=10)
    iou_metric = IoU(cm)
    iou_no_bg_metric = iou_metric[:9]  # We assume that the background index is 9
    mean_iou_no_bg_metric = iou_no_bg_metric.mean()
    # mean_iou_no_bg_metric.compute() -> tensor(0.12345)

How to create a custom metric
-----------------------------

To create a custom metric one needs to create a new class inheriting from :class:`~ignite.metrics.metric.Metric` and override
three methods :

- :meth:`~ignite.metrics.metric.Metric.reset()` : resets internal variables and accumulators
- :meth:`~ignite.metrics.metric.Metric.update()` : updates internal variables and accumulators with provided batch output ``(y_pred, y)``
- :meth:`~ignite.metrics.metric.Metric.compute()` : computes custom metric and return the result

For example, we would like to implement for illustration purposes a multi-class accuracy metric with some
specific condition (e.g. ignore user-defined classes):

.. code-block:: python

    from ignite.metrics import Metric
    from ignite.exceptions import NotComputableError

    # These decorators helps with distributed settings
    from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


    class CustomAccuracy(Metric):

        def __init__(self, ignored_class, output_transform=lambda x: x, device="cpu"):
            self.ignored_class = ignored_class
            self._num_correct = None
            self._num_examples = None
            super(CustomAccuracy, self).__init__(output_transform=output_transform, device=device)

        @reinit__is_reduced
        def reset(self):
            self._num_correct = torch.tensor(0, device=self._device)
            self._num_examples = 0
            super(CustomAccuracy, self).reset()

        @reinit__is_reduced
        def update(self, output):
            y_pred, y = output[0].detach(), output[1].detach()

            indices = torch.argmax(y_pred, dim=1)

            mask = (y != self.ignored_class)
            mask &= (indices != self.ignored_class)
            y = y[mask]
            indices = indices[mask]
            correct = torch.eq(indices, y).view(-1)

            self._num_correct += torch.sum(correct).to(self._device)
            self._num_examples += correct.shape[0]

        @sync_all_reduce("_num_examples", "_num_correct:SUM")
        def compute(self):
            if self._num_examples == 0:
                raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
            return self._num_correct.item() / self._num_examples


We imported necessary classes as :class:`~ignite.metrics.metric.Metric`, :class:`~ignite.exceptions.NotComputableError` and
decorators to adapt the metric for distributed setting. In ``reset`` method, we reset internal variables ``_num_correct``
and ``_num_examples`` which are used to compute the custom metric. In ``updated`` method we define how to update
the internal variables. And finally in ``compute`` method, we compute metric value.

Notice that ``_num_correct`` is a tensor, since in ``update`` we accumulate tensor values. ``_num_examples`` is a python
scalar since we accumulate normal integers. For differentiable metrics, you must detach the accumulated values before
adding them to the internal variables.

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

- :meth:`~ignite.metrics.metric.Metric.reset()` is triggered every ``EPOCH_STARTED`` (See :class:`~ignite.engine.events.Events`).
- :meth:`~ignite.metrics.metric.Metric.update()` is triggered every ``ITERATION_COMPLETED``.
- :meth:`~ignite.metrics.metric.Metric.compute()` is triggered every ``EPOCH_COMPLETED``.

Usages can be user defined by creating a class inheriting from :class:`~ignite.metrics.metric.MetricUsage`. See the list below of usages.

Complete list of usages
~~~~~~~~~~~~~~~~~~~~~~~

    - :class:`~ignite.metrics.metric.MetricUsage`
    - :class:`~ignite.metrics.metric.EpochWise`
    - :class:`~ignite.metrics.metric.RunningEpochWise`
    - :class:`~ignite.metrics.metric.BatchWise`
    - :class:`~ignite.metrics.metric.RunningBatchWise`
    - :class:`~ignite.metrics.metric.SingleEpochRunningBatchWise`
    - :class:`~ignite.metrics.metric.BatchFiltered`

Metrics and distributed computations
------------------------------------

In the above example, ``CustomAccuracy`` has ``reset``, ``update``, ``compute`` methods decorated
with :meth:`~ignite.metrics.metric.reinit__is_reduced`, :meth:`~ignite.metrics.metric.sync_all_reduce`. The purpose of these features is to adapt metrics in distributed
computations on supported backend and devices (see :doc:`distributed` for more details). More precisely, in the above
example we added ``@sync_all_reduce("_num_examples", "_num_correct:SUM")`` over ``compute`` method. This means that when ``compute``
method is called, metric's interal variables ``self._num_examples`` and ``self._num_correct:SUM`` are summed up over all participating
devices. We specify the reduction operation ``self._num_correct:SUM`` or we keep the default ``self._num_examples`` as the default is ``SUM``.
We currently support four reduction operations (SUM, MAX, MIN, PRODUCT).
Therefore, once collected, these internal variables can be used to compute the final metric value.

Complete list of metrics
------------------------

.. currentmodule:: ignite.metrics

.. autosummary::
    :nosignatures:
    :toctree: generated

    Average
    GeometricAverage
    VariableAccumulation
    Accuracy
    confusion_matrix.ConfusionMatrix
    ClassificationReport
    DiceCoefficient
    JaccardIndex
    IoU
    mIoU
    EpochMetric
    Fbeta
    Frequency
    Loss
    MeanAbsoluteError
    MeanAveragePrecision
    MeanPairwiseDistance
    MeanSquaredError
    metric.Metric
    metric_group.MetricGroup
    metrics_lambda.MetricsLambda
    MultiLabelConfusionMatrix
    MutualInformation
    ObjectDetectionAvgPrecisionRecall
    CommonObjectDetectionMetrics
    vision.object_detection_average_precision_recall.coco_tensor_list_to_dict_list
    precision.Precision
    PSNR
    recall.Recall
    RootMeanSquaredError
    RunningAverage
    SSIM
    TopKCategoricalAccuracy
    Bleu
    Rouge
    RougeL
    RougeN
    InceptionScore
    FID
    CosineSimilarity
    Entropy
    KLDivergence
    JSDivergence
    MaximumMeanDiscrepancy
    HSIC
    AveragePrecision
    CohenKappa
    GpuInfo
    PrecisionRecallCurve
    RocCurve
    ROC_AUC
    regression.CanberraMetric
    regression.FractionalAbsoluteError
    regression.FractionalBias
    regression.GeometricMeanAbsoluteError
    regression.GeometricMeanRelativeAbsoluteError
    regression.ManhattanDistance
    regression.MaximumAbsoluteError
    regression.MeanAbsoluteRelativeError
    regression.MeanError
    regression.MeanNormalizedBias
    regression.MedianAbsoluteError
    regression.MedianAbsolutePercentageError
    regression.MedianRelativeAbsoluteError
    regression.PearsonCorrelation
    regression.SpearmanRankCorrelation
    regression.KendallRankCorrelation
    regression.R2Score
    regression.WaveHedgesDistance
    clustering.SilhouetteScore
    clustering.DaviesBouldinScore
    clustering.CalinskiHarabaszScore


.. note::

    Module ignite.metrics.regression provides implementations of metrics useful
    for regression tasks. Definitions of metrics are based on
    `Botchkarev 2018`_, page 30 "Appendix 2. Metrics mathematical definitions".


Helpers for customizing metrics
-------------------------------

MetricUsage
~~~~~~~~~~~
.. autoclass:: ignite.metrics.metric.MetricUsage

EpochWise
~~~~~~~~~
.. autoclass:: ignite.metrics.metric.EpochWise

RunningEpochWise
~~~~~~~~~~~~~~~~
.. autoclass:: ignite.metrics.metric.RunningEpochWise

BatchWise
~~~~~~~~~
.. autoclass:: ignite.metrics.metric.BatchWise

RunningBatchWise
~~~~~~~~~~~~~~~~
.. autoclass:: ignite.metrics.metric.RunningBatchWise

SingleEpochRunningBatchWise
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ignite.metrics.metric.SingleEpochRunningBatchWise

BatchFiltered
~~~~~~~~~~~~~
.. autoclass:: ignite.metrics.metric.BatchFiltered

.. currentmodule:: ignite.metrics.metric

reinit__is_reduced
~~~~~~~~~~~~~~~~~~
.. autofunction:: reinit__is_reduced

sync_all_reduce
~~~~~~~~~~~~~~~
.. autofunction:: sync_all_reduce

.. _`Botchkarev 2018`:
        https://arxiv.org/abs/1809.03006

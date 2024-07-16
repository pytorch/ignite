from typing import Callable, Union

import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers.timing import Timer
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class Frequency(Metric):
    """Provides metrics for the number of examples processed per second.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. code-block:: python

            # Compute number of tokens processed
            wps_metric = Frequency(output_transform=lambda x: x['ntokens'])
            wps_metric.attach(trainer, name='wps')
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will look like
            # Epoch [2/10]: [12/24]  50%|█████      , wps=400 [00:17<1:23]


        To compute examples processed per second every 50th iteration:

        .. code-block:: python

            # Compute number of tokens processed
            wps_metric = Frequency(output_transform=lambda x: x['ntokens'])
            wps_metric.attach(trainer, name='wps', event_name=Events.ITERATION_COMPLETED(every=50))
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will look like
            # Epoch [2/10]: [50/100]  50%|█████      , wps=400 [00:17<00:35]

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        super(Frequency, self).__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._timer = Timer()
        self._acc = 0
        self._n = 0
        self._elapsed = 0.0
        super(Frequency, self).reset()  # type: ignore

    @reinit__is_reduced
    def update(self, output: int) -> None:
        self._acc += output
        self._n = self._acc
        self._elapsed = self._timer.value()

    @sync_all_reduce("_n", "_elapsed")
    def compute(self) -> float:
        time_divisor = 1.0

        if idist.get_world_size() > 1:
            time_divisor *= idist.get_world_size()

        # Returns the average processed objects per second across all workers
        return self._n / self._elapsed * time_divisor

    def completed(self, engine: Engine, name: str) -> None:
        engine.state.metrics[name] = int(self.compute())

    # TODO: see issue https://github.com/pytorch/ignite/issues/1405
    def attach(  # type: ignore
        self, engine: Engine, name: str, event_name: Events = Events.ITERATION_COMPLETED
    ) -> None:
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(event_name, self.completed, name)

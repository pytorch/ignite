import numbers
from typing import Callable, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["VariableAccumulation", "GeometricAverage", "Average"]


class VariableAccumulation(Metric):
    """Single variable accumulator helper to compute (arithmetic, geometric, harmonic) average of a single variable.

    - ``update`` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.

    Note:

        The class stores input into two public variables: `accumulator` and `num_examples`.
        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

    Args:
        op: a callable to update accumulator. Method's signature is `(accumulator, output)`.
            For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    """

    required_output_keys = None
    _state_dict_all_req_keys = ("accumulator", "num_examples")

    def __init__(
        self,
        op: Callable,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if not callable(op):
            raise TypeError(f"Argument op should be a callable, but given {type(op)}")

        self._op = op

        super(VariableAccumulation, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.accumulator = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self.num_examples = 0

    def _check_output_type(self, output: Union[float, torch.Tensor]) -> None:
        if not isinstance(output, (numbers.Number, torch.Tensor)):
            raise TypeError(f"Output should be a number or torch.Tensor, but given {type(output)}")

    @reinit__is_reduced
    def update(self, output: Union[float, torch.Tensor]) -> None:
        self._check_output_type(output)

        if isinstance(output, torch.Tensor):
            output = output.detach()
            if not (output.device == self._device and output.dtype == self.accumulator.dtype):
                output = output.to(self.accumulator)

        self.accumulator = self._op(self.accumulator, output)

        if isinstance(output, torch.Tensor):
            self.num_examples += output.shape[0] if len(output.shape) > 1 else 1
        else:
            self.num_examples += 1

    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> Tuple[torch.Tensor, int]:
        return self.accumulator, self.num_examples


class Average(VariableAccumulation):
    """Helper class to compute arithmetic average of a single variable.

    - ``update`` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.

    Note:

        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is an ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

        For input `x` being an ND `torch.Tensor` with N > 1, the first dimension is seen as the number of samples and
        is summed up and added to the accumulator: `accumulator += x.sum(dim=0)`

        ``output_tranform`` can be added to the metric to transform the output into the form expected by the metric.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = Average()
            metric.attach(default_evaluator, 'avg')
            # Case 1. input is er
            data = torch.tensor([0, 1, 2, 3, 4])
            state = default_evaluator.run(data)
            print(state.metrics['avg'])

        .. testoutput::

            2.0

        .. testcode::

            metric = Average()
            metric.attach(default_evaluator, 'avg')
            # Case 2. input is a 1D torch.Tensor
            data = torch.tensor([
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]
            ])
            state = default_evaluator.run(data)
            print(state.metrics['avg'])

        .. testoutput::

            tensor([1.5000, 1.5000, 1.5000], dtype=torch.float64)

        .. testcode::

            metric = Average()
            metric.attach(default_evaluator, 'avg')
            # Case 3. input is a ND torch.Tensor
            data = [
                torch.tensor([[0, 0, 0], [1, 1, 1]]),
                torch.tensor([[2, 2, 2], [3, 3, 3]])
            ]
            state = default_evaluator.run(data)
            print(state.metrics['avg'])

        .. testoutput::

            tensor([1.5000, 1.5000, 1.5000], dtype=torch.float64)
    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ):
        def _mean_op(a: Union[float, torch.Tensor], x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
            if isinstance(x, torch.Tensor) and x.ndim > 1:
                x = x.sum(dim=0)
            return a + x

        super(Average, self).__init__(op=_mean_op, output_transform=output_transform, device=device)

    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self.num_examples < 1:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )

        return self.accumulator / self.num_examples


class GeometricAverage(VariableAccumulation):
    """Helper class to compute geometric average of a single variable.

    - ``update`` must receive output of the form `x`.
    - `x` can be a positive number or a positive `torch.Tensor`, such that ``torch.log(x)`` is not `nan`.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Note:

        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

        For input `x` being an ND `torch.Tensor` with N > 1, the first dimension is seen as the number of samples and
        is aggregated and added to the accumulator: `accumulator *= prod(x, dim=0)`

        ``output_tranform`` can be added to the metric to transform the output into the form expected by the metric.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = GeometricAverage()
            metric.attach(default_evaluator, 'avg')
            # Case 1. input is er
            data = torch.tensor([1, 2, 3])
            state = default_evaluator.run(data)
            print(state.metrics['avg'])

        .. testoutput::

            1.8171...

        .. testcode::

            metric = GeometricAverage()
            metric.attach(default_evaluator, 'avg')
            # Case 2. input is a 1D torch.Tensor
            data = torch.tensor([
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
            ])
            state = default_evaluator.run(data)
            print(state.metrics['avg'])

        .. testoutput::

            tensor([2.2134, 2.2134, 2.2134], dtype=torch.float64)

        .. testcode::

            metric = GeometricAverage()
            metric.attach(default_evaluator, 'avg')
            # Case 3. input is a ND torch.Tensor
            data = [
                torch.tensor([[1, 1, 1], [2, 2, 2]]),
                torch.tensor([[3, 3, 3], [4, 4, 4]])
            ]
            state = default_evaluator.run(data)
            print(state.metrics['avg'])

        .. testoutput::

            tensor([2.2134, 2.2134, 2.2134], dtype=torch.float64)
    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ):
        def _geom_op(a: torch.Tensor, x: Union[float, torch.Tensor]) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            x = torch.log(x)
            if x.ndim > 1:
                x = x.sum(dim=0)
            return a + x

        super(GeometricAverage, self).__init__(op=_geom_op, output_transform=output_transform, device=device)

    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self.num_examples < 1:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )

        tensor = torch.exp(self.accumulator / self.num_examples)

        if tensor.numel() == 1:
            return tensor.item()

        return tensor

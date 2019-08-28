import numbers
import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class VariableAccumulation(Metric):
    """Single variable accumulator helper to compute (arithmetic, geometric, harmonic) average of a single variable.

    - `update` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.

    Note:

        The class stores input into two public variables: `accumulator` and `num_examples`.
        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

    Args:
        op (callable): a callable to update accumulator. Method's signature is `(accumulator, output)`.
            For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """

    def __init__(self, op, output_transform=lambda x: x):
        if not callable(op):
            raise TypeError("Argument op should be a callable, but given {}".format(type(op)))
        self.accumulator = None
        self.num_examples = None
        self._op = op
        super(VariableAccumulation, self).__init__(output_transform=output_transform)

    def reset(self):
        self.accumulator = torch.tensor(0.0, dtype=torch.float64)
        self.num_examples = torch.tensor(0.0, dtype=torch.float64)
        super(VariableAccumulation, self).reset()

    def _check_output_type(self, output):
        if not (isinstance(output, numbers.Number) or isinstance(output, torch.Tensor)):
            raise TypeError("Output should be a number or torch.Tensor, but given {}".format(type(output)))

    def update(self, output):
        self._check_output_type(output)

        self.accumulator = self._op(self.accumulator, output)
        if hasattr(output, 'shape'):
            self.num_examples += output.shape[0] if len(output.shape) > 1 else 1
        else:
            self.num_examples += 1

    def compute(self):
        return [self.accumulator, self.num_examples]


class Average(VariableAccumulation):
    """Helper class to compute arithmetic average of a single variable.

    - `update` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.

    Note:

        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

    Examples:

    .. code-block:: python

        evaluator = ...

        custom_var_mean = Average(output_transform=lambda output: output['custom_var'])
        custom_var_mean.attach(evaluator, 'mean_custom_var')

        state = evaluator.run(dataset)
        # state.metrics['mean_custom_var'] -> average of output['custom_var']

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """
    def __init__(self, output_transform=lambda x: x):

        def _mean_op(a, x):
            return a + x

        super(Average, self).__init__(op=_mean_op, output_transform=output_transform)

    def compute(self):
        if self.num_examples < 1:
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed.".format(self.__class__.__name__))

        return self.accumulator / self.num_examples


class GeometricAverage(VariableAccumulation):
    """Helper class to compute geometric average of a single variable.

    - `update` must receive output of the form `x`.
    - `x` can be a number or `torch.Tensor`.

    Note:

        Number of samples is updated following the rule:

        - `+1` if input is a number
        - `+1` if input is a 1D `torch.Tensor`
        - `+batch_size` if input is a ND `torch.Tensor`. Batch size is the first dimension (`shape[0]`).

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """
    def __init__(self, output_transform=lambda x: x):

        def _geom_op(a, x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return a + torch.log(x)

        super(GeometricAverage, self).__init__(op=_geom_op, output_transform=output_transform)

    def compute(self):
        if self.num_examples < 1:
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed.".format(self.__class__.__name__))

        return torch.exp(self.accumulator / self.num_examples)

import torch
from ignite.engine import Engine, Events
from tensorboardX import SummaryWriter
from types import MethodType


class TensorBoardX(object):

    def __init__(self, model=None,
                 input_shape=None,
                 use_output=False,
                 use_metrics=False,
                 train_evaluator=None,
                 validation_evaluator=None,
                 state_keys=None,
                 log_dir=None,
                 write_graph=False,
                 write_grads=False,
                 histogram_freq=0
                 ):
        self.model = model
        self.use_output = use_output
        self.use_metrics = use_metrics
        self.state_keys = state_keys

        self.train_evaluator = train_evaluator
        self.validation_evaluator = validation_evaluator

        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.histogram_freq = histogram_freq

        self.epoch_scalars = {}
        self.iteration_scalars = {}

        self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.write_graph:
            x_shape = [1, *input_shape]
            x = torch.zeros(*x_shape)
            self.writer.add_graph(self.model, x)

    def attach(self, engine):

        # Use user created state attributes such as 'reward' for RL
        if self.state_keys is not None:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_custom_state)

        # Use engine.state.output
        if self.use_output:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_engine_output)

        engine.add_event_handler(Events.EPOCH_COMPLETED, self._on_epoch_end)

        self._attach_child_function_to_epoch(engine)

    def _on_epoch_end(self, engine):
        # Train Evaluator
        if self.train_evaluator is not None:
            self._log_evaluator(engine, self.train_evaluator, mode='training')

        # Validation Evaluator
        if self.validation_evaluator is not None:
            self._log_evaluator(engine, self.validation_evaluator, mode='validation')

        if self.model is not None and self.histogram_freq:
            if engine.state.epoch % self.histogram_freq==0:
                self._log_histograms(engine, self.model)

    def _log_engine_output(self, engine):
        if engine.state.output is None:
            raise ValueError('If use_output is True, engine.state.output cannot be None.')
        else:
            if isinstance(engine.state.output, dict):
                for key, value in engine.state.output.items():
                    self.writer.add_scalar('trainer/' + key, value, engine.state.iteration)
            elif isinstance(engine.state.output, (int, float)):
                self.writer.add_scalar('trainer/output', engine.state.output, engine.state.iteration)
            else:
                raise ValueError(
                    'Preferred format for engine.state.output is a single scalar or dictionary of scalars.')

    def _log_custom_state(self, engine):
        for key in self.state_keys:
            if hasattr(engine.state, key):
                self.writer.add_scalar('trainer/'+key, getattr(engine.state, key), engine.state.iteration)
            else:
                raise ValueError('engine.state does not have attribute %s.' % key)

    def _log_histograms(self, engine, model, add_name=False):
        for name, param in model.named_parameters():
            if add_name:
                name = model.__class__.__name__ + '/' + name

            if param.requires_grad:
                self.writer.add_histogram(name, param.cpu().data.numpy().flatten(), engine.state.epoch)
                if self.write_grads:
                    self.writer.add_histogram(name + '_grad', param.grad.cpu().data.numpy().flatten(),
                                              engine.state.epoch)

    def _attach_child_function_to_epoch(self, engine):
        functions = [getattr(self, x) for x in dir(self) if
                     (not x.startswith('_') and type(getattr(self, x)) == MethodType)]

        for function in functions:
            engine.add_event_handler(Events.EPOCH_COMPLETED, function)

    def _log_evaluator(self, engine, evaluator, mode='training'):
        scalars = {}
        if (not isinstance((evaluator), Engine) and evaluator.state.metrics is not None):
            raise ValueError('evaluator must be an instance of ignite.Engine and have metrics attached to it.')
        else:
            for key, value in evaluator.state.metrics.items():
                self.writer.add_scalar(mode +'/' + key, value, engine.state.epoch)

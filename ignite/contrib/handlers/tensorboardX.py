try:
    from tensorboardX import SummaryWriter
except:
    raise RuntimeError("This contrib module requires tensorboardX to be installed.")

from ignite.engine import Engine, Events
import torch
from torch import nn as nn


class TensorBoardX(object):
    """
    TensorBoard handler to log metrics related to training, validation and model.

    Examples:

        Create a TensorBoard summary that visualizes metrics related to training,
        validation, and model parameters, by simply attaching the handler to your engine.

        ..code-block:: python

            tbLogger = TensorBoardX()
            tbLogger.attach(engine=trainer,
                            mode='iteration',
                            model=model,
                            input_shape=[1, 28, 28],
                            write_graph=True,
                            validation_evaluator=evaluator,
                            use_metrics=True,
                            histogram_freq=1,
                            write_grads=True)
    Note:
        When adding tensorboard event handler to an engine, it is recommended that the evaluator
        is run before the code block above. If custom functions are needed, use tbLogger.writer.add_*.
    """

    def __init__(self, log_dir=None):
        self.writer = SummaryWriter(log_dir=log_dir)

    def _start(self, engine, model, input_shape, write_graph):
        if write_graph:
            x = torch.zeros([1] + input_shape)
            x = x.cuda() if next(model.parameters()).is_cuda else x
            self.writer.add_graph(model, x)
            del x

    def _close(self, engine):
        self.writer.close()

    def _update_iteration(self, engine, mode, use_metrics, state_keys):
        if use_metrics and mode == 'iteration':
            for key, value in engine.state.metrics.items():
                self.writer.add_scalar(tag='trainer/' + key,
                                       scalar_value=value,
                                       global_step=engine.state.iteration)
        if state_keys:
            for key in state_keys:
                state = getattr(engine.state, key)
                if isinstance(state, (int, float)):
                    self.writer.add_scalar(tag='trainer/' + key,
                                           scalar_value=state,
                                           global_step=engine.state.iteration)

    def _update_epoch(self, engine, mode, train_evaluator, validation_evaluator,
                      use_metrics, model, histogram_freq, write_grads):
        if use_metrics and mode == 'epoch':
            for key, value in engine.state.metrics.items():
                self.writer.add_scalar(tag='trainer/' + key,
                                       scalar_value=value,
                                       global_step=engine.state.epoch)
        if use_metrics:
            if train_evaluator:
                for key, value in train_evaluator.state.metrics.items():
                    self.writer.add_scalar(tag='training/' + key,
                                           scalar_value=value,
                                           global_step=engine.state.epoch)

            if validation_evaluator:
                for key, value in validation_evaluator.state.metrics.items():
                    self.writer.add_scalar(tag='validation/' + key,
                                           scalar_value=value,
                                           global_step=engine.state.epoch)

        if engine.state.epoch % histogram_freq == 0:
            for name, param in model.named_parameters():

                if param.requires_grad:
                    self.writer.add_histogram(tag=name,
                                              values=param.cpu().data.numpy().flatten(),
                                              global_step=engine.state.epoch)
                    if write_grads:
                        self.writer.add_histogram(tag=name + '_grad',
                                                  values=param.grad.cpu().data.numpy().flatten(),
                                                  global_step=engine.state.epoch)

    def attach(self,
               engine,
               mode='iteration',
               model=None,
               input_shape=None,
               write_graph=False,
               train_evaluator=None,
               validation_evaluator=None,
               use_metrics=False,
               state_keys=None,
               histogram_freq=0,
               write_grads=False
               ):
        """
        Attaches the TensorBoard event handler to an engine object.

        Args:
            engine (Engine): engine object.
            mode (str): (Optional) iteration or epoch.
            model (nn.Module): (Optional) model to train.
            input_shape (list): (Optional) shape of input to model.
            write_graph (bool): (Optional) True, if model graph should be written.
            train_evaluator (Engine): (Optional) ignite engine that has been run on the training set.
            validation_evaluator (Engine): (Optional) ignite engine that has been run on the validation set.
            use_metrics (bool): (Optional) True, if metrics for engine and evaluator should be visualized.
            state_keys (list): (Optional) list of string, state attributes to be visualized.
            histogram_freq (int): (Optional) frequency histograms should be plotted.
            write_grads (bool): (Optional) True, if gradients to model parameters should be visualized.
        """

        if mode not in ['iteration', 'epoch']:
            raise ValueError("mode should be iteration or epoch, got {} instead.").format(mode)

        if not isinstance(model, nn.Module):
            raise TypeError("model should be of type nn.Module, got {} instead.".format(type(model)))

        if not isinstance(input_shape, list):
            raise TypeError("input_shape should be a list of integers, got {} instead.").format(type(input_shape))

        if not isinstance(write_graph, bool):
            raise TypeError("write_graph should be a boolean, got {} instead.").format(type(write_graph))

        if not isinstance(train_evaluator, Engine):
            raise TypeError("train_evaluator should be an ignite.engine, got {} instead.").format(type(train_evaluator))

        if not isinstance(validation_evaluator, Engine):
            raise TypeError("validation_evaluator should be an "
                            "ignite.engine, got {} instead.").format(type(validation_evaluator))

        if not isinstance(use_metrics, bool):
            raise TypeError("write_graph should be a boolean, got {} instead.").format(type(use_metrics))

        if not isinstance(state_keys, list):
            raise TypeError("state_keys should be a list, got {} instead.").format(type(state_keys))

        if not isinstance(histogram_freq, int):
            raise TypeError("histogram_freq should be an int, got {} instead.").format(type(histogram_freq))

        if not isinstance(write_grads, bool):
            raise TypeError("write_grads should be a boolean, got {} instead.").format(type(write_grads))

        engine.add_event_handler(Events.STARTED, self._start, model, input_shape, write_graph)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update_iteration, mode, use_metrics, state_keys)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._update_epoch, mode, train_evaluator,
                                 validation_evaluator, use_metrics, model, histogram_freq, write_grads)
        engine.add_event_handler(Events.COMPLETED, self._close)

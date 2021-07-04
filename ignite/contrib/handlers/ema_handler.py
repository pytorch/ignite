import warnings
from copy import deepcopy
from typing import Any, Optional, OrderedDict, Union

import torch
import torch.nn as nn
from torch import Tensor

from ignite.engine import Engine, Events

__all__ = ["EMAHandler"]


class EMAHandler:
    r"""Exponential moving average (EMA) handler can be used to compute a EMA smoothed version of model weights.
    The EMA weights are updated as follows:

    .. math:: \theta_{\text{EMA}, t+1} = (1 - \lambda) \cdot \theta_{\text{EMA}, t} + \lambda \cdot \theta_{t}

    where :math:`\theta_{\text{EMA}, t}` and :math:`\theta_{t}` are the EMA weights and model weights at
    :math:`t`-th iteration, respectively; :math:`\lambda` is the update momentum.

    Args:
          model: the model for which EMA weights will be computed. If ``model`` is ``DataParallel`` or
              ``DistributedDataParallel``, the EMA smoothing will be applied to ``model.module`` .
          momentum: the update momentum, should be float in range :math:`\left(0, 1 \right)`.
          momentum_warmup: the initial update momentum, the value should be smaller than ``momentum``. Momentum will
              increase from this value to ``momentum`` linearly.
          interval: update interval (in iterations).
          warmup_iters: iterations of warmup.
          device: device to store the EMA weights.

    - The handler allows for linearly warmup the momentum in the beginning when training process is not stable.
    - The handler supports saving the EMA backup to checkpoint via ``state_dict``, and loading from the checkpoint via
      ``load_state_dict``.
    - In the end of each train epoch, the handler will swap weights with the model, so that the EMA weights
      will be moved to the model and used for validation. In the beginning of each epoch, the handler will againswap
      weights with the model, so that the EMA weights are moved back to the EMA handlers, and model weights are moved
      back to the model.


    Note:
          It is recommended to initialize and use an EMA handler in following order:

          1. Initialize ``model`` (``nn.Module`` or ``DistributedDataParallel``) and ``ema_handler`` (``EMAHandler``).
          2. Build ``trainer`` (``ignite.engine.Engine``).
          3. Resume from checkpoint for ``model`` and ``ema_handler``.
          4. Attach ``ema_handler`` to ``trainer``.
          5. Attach other handlers to ``trainer``.

    Note:
          When the trainer completes running, the EMA weights are moved to the model. If both the handler's
          ``state_dict`` and model's ``state_dict`` are saved by ``Checkpoint`` handler, users can retrieve the EMA
          weights as follows:

          .. code-block:: python

              to_load = {"model" model, "ema_handler", ema_handler}
              ckpt = torch.load("checkpoint.pt")
              Checkpoint.load_objects(to_load, ckpt)

              # EMA weights are actually in the model's state_dict
              smoothed_weights = model.state_dict()

          Consider resuming training from a saved checkpoint: although the EMA weights are in the model when loaded
          from a checkpoint, there is no need to manually swap the ``sate_dict`` of handler and model. The Handler
          will automatically perform this step when the training resumes.

    Note:
          Users need to register the EMA handler before the validation handler, so that swapping weights will be
          performed before validation begins. It is recommended to register the EMA handler as the first handler for
          the event ``Events.EPOCH_COMPLETED``.

    Examples:
          .. code-block:: python

              device = torch.device("cuda:0")
              model = nn.Linear(2, 1).to(device)
              # update the ema every 5 iterations
              ema_handler = EMAHandler(
                model, momentum=0.0002, momentum_warmup=0.0001, interval=5, warmup_iters=10000, device=device)
              trainer = Engine(step_fn)
              to_load = {"model": model, "ema_handler", ema_handler, "trainer", trainer}
              if resume_from is not None:
                  Checkpoint.load_objects(to_load, checkpoint=resume_from)

              # add other handlers
              to_save = to_load
              ckpt_handler = Checkpoint(to_save, DiskSaver(...), ...)
              trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_handler)

              trainer.run(...)


    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.0002,
        momentum_warmup: float = 0.0001,
        interval: int = 1,
        warmup_iters: int = 100,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        assert 0 < momentum < 1
        assert 0 < momentum_warmup < 1
        assert momentum_warmup <= momentum
        assert isinstance(interval, int) and interval > 0
        assert isinstance(warmup_iters, int) and warmup_iters > 0
        assert isinstance(model, nn.Module)
        self.momentum = momentum
        self.interval = interval
        self.momentum_warmup = momentum_warmup
        self.warmup_iters = warmup_iters
        self.device = device

        self.ema = deepcopy(self._unwrap_model(model))
        self.ema.eval()
        if self.device is not None:
            self.ema.to(device=device)

    def state_dict(self, **kwargs: Optional[Any]) -> OrderedDict[str, Tensor]:
        """Return ``state_dict`` of the EMA copy.

        Args:
            kwargs: arguments of ``nn.Module.state_dict``.

        """
        return self.ema.state_dict(**kwargs)

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True) -> None:
        """Load the ``sate_dict`` to the EMA copy.

        Args:
            state_dict: dict contains the weights.
            strict: whether to strictly enforce that the keys match.

        """
        self.ema.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        else:
            raise ValueError("Invalid model class")

    def _get_momentum(self, curr_iter: int):
        """Get current momentum, `curr_iter` should be 1-based. When `curr_iter = 1`, `momentum =
        self.momentum_warmup`; when `curr_iter >= self.warmup_iters`, `momentum = self.momentum`"""
        assert curr_iter >= 1
        denominator = max(1, self.warmup_iters - 1)
        momentum = self.momentum_warmup + (self.momentum - self.momentum_warmup) * (curr_iter - 1) / denominator
        return min(self.momentum, momentum)

    @torch.no_grad()
    def _swap_params(self, engine: Engine, model: nn.Module) -> None:

        model = self._unwrap_model(model)
        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_v = model_v.to(device=self.device)
            tmp = model_v.data.clone()
            model_v.data.copy_(ema_v.data)
            ema_v.data.copy_(tmp)

    @torch.no_grad()
    def _update(self, engine: Engine, model: nn.Module) -> None:

        model = self._unwrap_model(model)
        curr_iter = engine.state.iteration
        momentum = self._get_momentum(curr_iter)

        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_v = model_v.to(device=self.device)
            ema_v.mul_(1 - momentum).add_(model_v.data, alpha=momentum)

    def attach(self, engine: Engine, model: nn.Module) -> None:
        """Attach the handler to engine.

        Note:
              If model is ``DataParallel`` or ``DistributedDataParallel``, the EMA smoothing will be applied to
              ``model.module`` .

        Args:
            engine: Trainer to which the handler will be attached.
            model: Model to which the EMA smoothing will be applied.

        """
        # swap the parameters at the end of each epoch. If using Events.EPOCH_COMPLETED here, the handler of swap
        # parameters might be called after validation handler.
        # TODO Find a better hack to ensure that handler of swapping parameter is called before validation.
        num_handlers_epoch_completed = engine._event_handlers[Events.EPOCH_COMPLETED]
        if len(num_handlers_epoch_completed) > 0:
            warnings.warn(
                f"engine already has {num_handlers_epoch_completed} handlers at Events.EPOCH_COMPLETED, "
                f"please manually check if the EMA handler is attached to the engine before other "
                f"handlers"
            )
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._swap_params, model)
        engine.add_event_handler(Events.EPOCH_STARTED, self._swap_params, model)
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self._update, model)

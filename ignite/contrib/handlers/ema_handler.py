from copy import deepcopy
from typing import Any, Dict, Optional, Union

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


    Note:
          It is recommended to initialize and use an EMA handler in following order:

          1. Initialize ``model`` (``nn.Module`` or ``DistributedDataParallel``) and ``ema_handler`` (``EMAHandler``).
          2. Build ``trainer`` (``ignite.engine.Engine``).
          3. Resume from checkpoint for ``model`` and ``ema_handler``.
          4. Attach ``ema_handler`` to ``trainer``.

    Examples:
          .. code-block:: python

              device = torch.device("cuda:0")
              model = nn.Linear(2, 1).to(device)
              # update the ema every 5 iterations
              ema_handler = EMAHandler(
                  model, momentum=0.0002, momentum_warmup=0.0001, interval=5, warmup_iters=10000, device=device)
              trainer = Engine(train_step_fn)
              to_load = {"model": model, "ema_handler", ema_handler, "trainer", trainer}
              if resume_from is not None:
                  Checkpoint.load_objects(to_load, checkpoint=resume_from)
              ema_handler.attach(trainer)

              # add other handlers
              to_save = to_load
              ckpt_handler = Checkpoint(to_save, DiskSaver(...), ...)
              trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_handler)

              # use ema model for validation
              val_step_fn = get_val_step_fn(ema_handler.get_ema_model())
              evaluator = Engine(val_step_fn)

              @trainer.on(Events.EPOCH_COMPLETED)
              def run_validation(engine):
                  engine.run(val_data_loader)

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
        if not 0 < momentum < 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0 < momentum_warmup < 1:
            raise ValueError(f"Invalid momentum_warmup: {momentum_warmup}")
        if not momentum_warmup <= momentum:
            raise ValueError(
                f"momentum_warmup should be less than or equal to momentum, but got "
                f"momentum_warmup: {momentum_warmup} and momentum: {momentum}"
            )
        if not (isinstance(interval, int) and interval > 0):
            raise ValueError(f"Invalid interval: {interval}")
        if not isinstance(warmup_iters, int) and warmup_iters > 0:
            raise ValueError(f"Invalid warmup_iters: {warmup_iters}")
        if not isinstance(model, nn.Module):
            raise ValueError(
                f"model should be an instance of nn.Module or its subclasses, but got"
                f"model: {model.__class__.__name__}"
            )
        self.momentum = momentum
        self.interval = interval
        self.momentum_warmup = momentum_warmup
        self.warmup_iters = warmup_iters
        self.device = device

        self.unwrapped_model = self._unwrap_model(model)
        self.ema = deepcopy(self.unwrapped_model)
        self.ema.eval()
        if self.device is not None:
            self.ema.to(device=device)  # type: ignore

    def get_ema_model(self) -> nn.Module:
        """Get the model with EMA weights.

        Note:
              This function returns an instance of ``nn.Module`` rather than ``DistributedDataParallel`` or
              ``DataParallel``.

        Returns:

        """
        return self.ema

    def state_dict(self, **kwargs: Any) -> Dict[str, Tensor]:
        """Return ``state_dict`` of the EMA model. It is a shortcut for ``get_ema_model().state_dict()``.

        Args:
            kwargs: arguments of ``nn.Module.state_dict``.

        """
        return self.ema.state_dict(**kwargs)

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> None:
        """Load the ``sate_dict`` to the EMA model. It is a shortcut for ``get_ema_model().load_state_dict()``.

        Args:
            state_dict: dict contains the weights.
            strict: whether to strictly enforce that the keys match.

        """
        self.ema.load_state_dict(state_dict, strict=strict)  # type: ignore

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        else:
            return model

    def _get_momentum(self, curr_iter: int) -> float:
        """Get current momentum, `curr_iter` should be 1-based. When `curr_iter = 1`, `momentum =
        self.momentum_warmup`; when `curr_iter >= self.warmup_iters`, `momentum = self.momentum`"""
        assert curr_iter >= 1
        denominator = max(1, self.warmup_iters - 1)
        momentum = self.momentum_warmup + (self.momentum - self.momentum_warmup) * (curr_iter - 1) / denominator
        return min(self.momentum, momentum)

    @torch.no_grad()
    def _update(self, engine: Engine) -> None:
        curr_iter = engine.state.iteration
        momentum = self._get_momentum(curr_iter)

        for ema_v, model_v in zip(self.ema.state_dict().values(), self.unwrapped_model.state_dict().values()):
            if self.device is not None:
                model_v = model_v.to(device=self.device)
            ema_v.mul_(1 - momentum).add_(model_v.data, alpha=momentum)

    def attach(self, engine: Engine) -> None:
        """Attach the handler to engine.

        Args:
            engine: Trainer to which the handler will be attached.

        """
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self._update)

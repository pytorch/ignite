from copy import deepcopy
from typing import Optional, Union

import torch.nn as nn

from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList

__all__ = ["EMAHandler"]


class EMAHandler:
    r"""Exponential moving average (EMA) handler can be used to compute a smoothed version of model.
    The EMA model is updated as follows:

    .. math:: \theta_{\text{EMA}, t+1} = (1 - \lambda) \cdot \theta_{\text{EMA}, t} + \lambda \cdot \theta_{t}

    where :math:`\theta_{\text{EMA}, t}` and :math:`\theta_{t}` are the EMA weights and online model weights at
    :math:`t`-th iteration, respectively; :math:`\lambda` is the update momentum. The handler allows for linearly
    warming up the momentum in the beginning when training process is not stable. Current momentum can be retrieved
    from ``Engine.state.ema_momentum``.

    Args:
          model: the online model for which an EMA model will be computed. If ``model`` is ``DataParallel`` or
              ``DistributedDataParallel``, the EMA smoothing will be applied to ``model.module`` .
          momentum: the update momentum after warmup phase, should be float in range :math:`\left(0, 1 \right)`.
          momentum_warmup: the initial update momentum during warmup phase, the value should be smaller than
              ``momentum``. Momentum will linearly increase from this value to ``momentum`` in ``warmup_iters``
              iterations. If ``None``, no warmup will be performed.
          warmup_iters: iterations of warmup. If ``None``, no warmup will be performed.

    Attributes:
          ema_model: the exponential moving averaged model.
          model: the online model that is tracked by EMAHandler. It is ``model.module`` if ``model`` in
              the initialization method is an instance of ``DistributedDataParallel``.
          momentum: the update momentum after warmup phase.
          momentum_warmup: the initial update momentum.
          warmup_iters: number of warmup iterations.

    Note:
          The EMA model is already in ``eval`` mode. If model in the arguments is an ``nn.Module`` or
          ``DistributedDataParallel``, the EMA model is an ``nn.Module`` and it is on the same device as the online
          model. If the model is an ``nn.DataParallel``, then the EMA model is an ``nn.DataParallel``.


    Note:
          It is recommended to initialize and use an EMA handler in following steps:

          1. Initialize ``model`` (``nn.Module`` or ``DistributedDataParallel``) and ``ema_handler`` (``EMAHandler``).
          2. Build ``trainer`` (``ignite.engine.Engine``).
          3. Resume from checkpoint for ``model`` and ``ema_handler.ema_model``.
          4. Attach ``ema_handler`` to ``trainer``.

    Examples:
          .. code-block:: python

              device = torch.device("cuda:0")
              model = nn.Linear(2, 1).to(device)
              # update the ema every 5 iterations
              ema_handler = EMAHandler(
                  model, momentum=0.0002, momentum_warmup=0.0001, warmup_iters=10000)
              # get the ema model, which is an instance of nn.Module
              ema_model = ema_handler.ema_model
              trainer = Engine(train_step_fn)
              to_load = {"model": model, "ema_model", ema_model, "trainer", trainer}
              if resume_from is not None:
                  Checkpoint.load_objects(to_load, checkpoint=resume_from)

              # update the EMA model every 5 iterations
              ema_handler.attach(trainer, name="ema_momentum", event=Events.ITERATION_COMPLETED(every=5))

              # add other handlers
              to_save = to_load
              ckpt_handler = Checkpoint(to_save, DiskSaver(...), ...)
              trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_handler)

              # current momentum can be retrieved from engine.state,
              # the attribute name is the `name` parameter used in the attach function
              @trainer.on(Events.ITERATION_COMPLETED):
              def print_ema_momentum(engine):
                  print(f"current momentum: {engine.state.ema_momentum}"

              # use ema model for validation
              val_step_fn = get_val_step_fn(ema_model)
              evaluator = Engine(val_step_fn)

              @trainer.on(Events.EPOCH_COMPLETED)
              def run_validation(engine):
                  engine.run(val_data_loader)

              trainer.run(...)

          The following example shows how to attach two handlers to the same trainer:

          .. code-block:: python

              generator = build_generator(...)
              discriminator = build_discriminator(...)

              gen_handler = EMAHandler(generator)
              disc_handler = EMAHandler(discriminator)

              step_fn = get_step_fn(...)
              engine = Engine(step_fn)

              # update EMA model of generator every 1 iteration
              gen_handler.attach(engine, "gen_ema_momentum", event=Events.ITERATION_COMPLETED)
              # update EMA model of discriminator every 2 iteration
              disc_handler.attach(engine, "dis_ema_momentum", event=Events.ITERATION_COMPLETED(every=2))

              @engine.on(Events.ITERATION_COMPLETED)
              def print_ema_momentum(engine):
                  print(f"current momentum for generator: {engine.state.gen_ema_momentum}")
                  print(f"current momentum for discriminator: {engine.state.disc_ema_momentum}")

              engine.run(...)

    .. versionadded:: 0.4.6

    """

    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.0002,
        momentum_warmup: Optional[float] = None,
        warmup_iters: Optional[int] = None,
    ) -> None:
        if momentum_warmup is not None and not 0 < momentum_warmup < 1:
            raise ValueError(f"Invalid momentum_warmup: {momentum_warmup}")
        if not 0 < momentum < 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if momentum_warmup is not None and not momentum_warmup <= momentum:
            raise ValueError(
                f"momentum_warmup should be less than or equal to momentum, but got "
                f"momentum_warmup: {momentum_warmup} and momentum: {momentum}"
            )
        if warmup_iters is not None and not (isinstance(warmup_iters, int) and warmup_iters > 0):
            raise ValueError(f"Invalid warmup_iters: {warmup_iters}")
        if not isinstance(model, nn.Module):
            raise ValueError(
                f"model should be an instance of nn.Module or its subclasses, but got"
                f"model: {model.__class__.__name__}"
            )
        self.momentum_warmup = momentum_warmup
        self.momentum = momentum
        self.warmup_iters = warmup_iters

        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        self.model = model

        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model.eval()

    def _get_momentum(self, curr_iter: int) -> float:
        """Get current momentum, `curr_iter` should be 1-based. When `curr_iter = 1`, `momentum =
        self.momentum_warmup`; when `curr_iter >= self.warmup_iters`, `momentum = self.momentum`"""

        # TODO: use ignite's parameter scheduling, see also GitHub issue #2090
        if curr_iter < 1:
            raise ValueError(f"curr_iter should be at least 1, but got {curr_iter}.")

        # no warmup
        if self.momentum_warmup is None or self.warmup_iters is None:
            return self.momentum

        denominator = max(1, self.warmup_iters - 1)
        momentum = self.momentum_warmup + (self.momentum - self.momentum_warmup) * (curr_iter - 1) / denominator
        return min(self.momentum, momentum)

    def _update_ema_model(self, engine: Engine, name: str) -> None:
        """Update weights of ema model"""
        momentum = getattr(engine.state, name)
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.mul_(1.0 - momentum).add_(model_p.data, alpha=momentum)
        # assign the buffers
        for ema_b, model_b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.data = model_b.data

    def _update_ema_momentum(self, engine: Engine, name: str) -> None:
        """Update momentum in engine.state"""
        curr_iter = engine.state.iteration
        momentum = self._get_momentum(curr_iter)
        setattr(engine.state, name, momentum)

    def attach(
        self,
        engine: Engine,
        name: str = "ema_momentum",
        event: Union[str, Events, CallableEventWithFilter, EventsList] = Events.ITERATION_COMPLETED,
    ) -> None:
        """Attach the handler to engine. After the handler is attached, the ``Engine.state`` will add an new attribute
        with name ``name``. Then, current momentum can be retrieved by from ``Engine.state`` when the engine runs.

        Args:
            engine: trainer to which the handler will be attached.
            name: attribute name for retrieving EMA momentum from ``Engine.state``. It should be a unique name since a
                trainer can have multiple EMA handlers.
            event: event when the EMA momentum and EMA model are updated.

        """
        if hasattr(engine.state, name):
            raise ValueError(
                f"Attribute: '{name}' is already in Engine.state. Thus it might be "
                f"overridden by other EMA handlers. Please select another name."
            )

        setattr(engine.state, name, 0.0)

        # first update momentum, then update ema model
        engine.add_event_handler(event, self._update_ema_momentum, name)
        engine.add_event_handler(event, self._update_ema_model, name)

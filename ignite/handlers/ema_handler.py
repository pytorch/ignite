import warnings
from copy import deepcopy
from typing import Optional, Union

import torch.nn as nn

from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from ignite.handlers.param_scheduler import BaseParamScheduler
from ignite.handlers.state_param_scheduler import LambdaStateScheduler

__all__ = ["EMAHandler"]


class EMAWarmUp:
    def __init__(self, momentum_warmup: float, warmup_iters: int, momentum: float) -> None:
        self.momentum_warmup = momentum_warmup
        self.warmup_iters = warmup_iters
        self.momentum = momentum

    def __call__(self, event_index: int) -> float:
        denominator = max(1, self.warmup_iters - 1)
        curr_momentum = self.momentum_warmup + (self.momentum - self.momentum_warmup) * (event_index - 1) / denominator
        if self.momentum >= self.momentum_warmup:
            return min(self.momentum, curr_momentum)
        else:
            return max(self.momentum, curr_momentum)


class EMAHandler:
    r"""Exponential moving average (EMA) handler can be used to compute a smoothed version of model.
    The EMA model is updated as follows:

    .. math:: \theta_{\text{EMA}, t+1} = (1 - \lambda) \cdot \theta_{\text{EMA}, t} + \lambda \cdot \theta_{t}

    where :math:`\theta_{\text{EMA}, t}` and :math:`\theta_{t}` are the EMA weights and online model weights at
    :math:`t`-th iteration, respectively; :math:`\lambda` is the update momentum. Current momentum can be retrieved
    from ``Engine.state.ema_momentum``.

    Args:
          model: the online model for which an EMA model will be computed. If ``model`` is ``DataParallel`` or
              ``DistributedDataParallel``, the EMA smoothing will be applied to ``model.module`` .
          momentum: the update momentum after warmup phase, should be float in range :math:`\left(0, 1 \right)`.
          momentum_warmup: the initial update momentum during warmup phase.
          warmup_iters: iterations of warmup.
          handle_buffers: how to handle model buffers during training. There are three options: 1. "copy" means
              copying the buffers of the online model; 2. "update" means applying EMA to the buffers of the online
              model; 3. "ema_train" means set the EMA model to ``train`` mode and skip copying or updating the buffers.

    Attributes:
          ema_model: the exponential moving averaged model.
          model: the online model that is tracked by EMAHandler. It is ``model.module`` if ``model`` in
              the initialization method is an instance of ``DistributedDataParallel``.
          momentum: the update momentum.
          handle_buffers: how to handle model buffers during training.

    Note:
          The EMA model is already in ``eval`` mode if ``handle_buffers`` is "copy" or "update". If model in the
          arguments is an ``nn.Module`` or ``DistributedDataParallel``, the EMA model is an ``nn.Module`` and it is on
          the same device as the online model. If the model is an ``nn.DataParallel``, then the EMA model is an
          ``nn.DataParallel``.


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
              ema_handler = EMAHandler(model, momentum=0.0002)
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

          The following example shows how to perform warm-up to the EMA momentum:

          .. code-block:: python

              device = torch.device("cuda:0")
              model = nn.Linear(2, 1).to(device)
              # linearly change the EMA momentum from 0.2 to 0.002 in the first 100 iterations,
              # then keep a constant EMA momentum of 0.002 afterwards
              ema_handler = EMAHandler(model, momentum=0.002, momentum_warmup=0.2, warmup_iters=100)
              engine = Engine(step_fn)
              ema_handler.attach(engine, name="ema_momentum")
              engine.run(...)

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
        handle_buffers: str = "copy",
    ) -> None:
        if not 0 < momentum < 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        self.momentum = momentum
        self._momentum_lambda_obj: Optional[EMAWarmUp] = None
        if momentum_warmup is not None and warmup_iters is not None:
            self.momentum_scheduler: Optional[BaseParamScheduler] = None
            self._momentum_lambda_obj = EMAWarmUp(momentum_warmup, warmup_iters, momentum)

        if not isinstance(model, nn.Module):
            raise ValueError(
                f"model should be an instance of nn.Module or its subclasses, but got"
                f"model: {model.__class__.__name__}"
            )

        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        self.model = model

        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()

        if handle_buffers not in ("copy", "update", "ema_train"):
            raise ValueError(
                f"handle_buffers can only be one of 'copy', 'update', 'ema_train', " f"but got {handle_buffers}"
            )

        self.handle_buffers = handle_buffers
        if self.handle_buffers == "ema_train":
            self.ema_model.train()
        else:
            self.ema_model.eval()

    def _update_ema_model(self, engine: Engine, name: str) -> None:
        """Update weights of ema model"""
        momentum = getattr(engine.state, name)
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.mul_(1.0 - momentum).add_(model_p.data, alpha=momentum)

        if self.handle_buffers == "update":
            for ema_b, model_b in zip(self.ema_model.buffers(), self.model.buffers()):
                try:
                    ema_b.mul_(1.0 - momentum).add_(model_b.data, alpha=momentum)
                except RuntimeError:
                    # Handle the case where ema_b is torch.int64, torch.int32 etc.,
                    # where a runtime error will be thrown when performing the in-place operations with floats.
                    # In this case, just copy the data
                    ema_b.data = model_b.data
        elif self.handle_buffers == "copy":
            # assign the buffers
            for ema_b, model_b in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_b.data = model_b.data
        else:
            pass

    def attach(
        self,
        engine: Engine,
        name: str = "ema_momentum",
        warn_if_exists: bool = True,
        event: Union[str, Events, CallableEventWithFilter, EventsList] = Events.ITERATION_COMPLETED,
    ) -> None:
        """Attach the handler to engine. After the handler is attached, the ``Engine.state`` will add an new attribute
        with name ``name`` if the attribute does not exist. Then, the current momentum can be retrieved from
        ``Engine.state`` when the engine runs.


        Note:
            There are two cases where a momentum with name ``name`` already exists: 1. the engine has loaded its
            state dict after resuming. In this case, there is no need to initialize the momentum again, and users
            can set ``warn_if_exists`` to False to suppress the warning message; 2. another handler has created
            a state attribute with the same name. In this case, users should choose another name for the ema momentum.


        Args:
            engine: trainer to which the handler will be attached.
            name: attribute name for retrieving EMA momentum from ``Engine.state``. It should be a unique name since a
                trainer can have multiple EMA handlers.
            warn_if_exists: if True, a warning will be thrown if the momentum with name ``name`` already exists.
            event: event when the EMA momentum and EMA model are updated.

        """
        if hasattr(engine.state, name):
            if warn_if_exists:
                warnings.warn(
                    f"Attribute '{name}' already exists in Engine.state. It might because 1. the engine has loaded its "
                    f"state dict or 2. {name} is already created by other handlers. Turn off this warning by setting"
                    f"warn_if_exists to False.",
                    category=UserWarning,
                )
        else:
            setattr(engine.state, name, self.momentum)

        if self._momentum_lambda_obj is not None:
            self.momentum_scheduler = LambdaStateScheduler(self._momentum_lambda_obj, param_name="ema_momentum")

            # first update the momentum and then update the EMA model
            self.momentum_scheduler.attach(engine, event)
        engine.add_event_handler(event, self._update_ema_model, name)

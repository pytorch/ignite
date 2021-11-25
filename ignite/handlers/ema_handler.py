import warnings
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
    :math:`t`-th iteration, respectively; :math:`\lambda` is the update momentum. Current momentum can be retrieved
    from ``Engine.state.ema_momentum``.

    Args:
          model: the online model for which an EMA model will be computed. If ``model`` is ``DataParallel`` or
              ``DistributedDataParallel``, the EMA smoothing will be applied to ``model.module`` .
          momentum: the update momentum after warmup phase, should be float in range :math:`\left(0, 1 \right)`.
          momentum_warmup: the initial update momentum during warmup phase. This argument is not used.
          warmup_iters: iterations of warmup. This argument is not used.

    Attributes:
          ema_model: the exponential moving averaged model.
          model: the online model that is tracked by EMAHandler. It is ``model.module`` if ``model`` in
              the initialization method is an instance of ``DistributedDataParallel``.
          momentum: the update momentum.

    Note:
          The EMA model is already in ``eval`` mode. If model in the arguments is an ``nn.Module`` or
          ``DistributedDataParallel``, the EMA model is an ``nn.Module`` and it is on the same device as the online
          model. If the model is an ``nn.DataParallel``, then the EMA model is an ``nn.DataParallel``.

    .. warning::
        The arguments `momentum_warmup` and `warmup_iters` will be deprecated in the future. In the current version,
        these two argument have no effect on the momentum. I.e., the momentum will be constant during the training.
        If users need to change the momentum, please use a ``StateParamScheduler``.


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
        if not 0 < momentum < 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if momentum_warmup is not None:
            warnings.warn(
                "Argument 'momentum_warmup' will be deprecated in the future. In the current version,"
                "it has no effect on the ema momentum. Please use the ParamStateScheduler to schedule"
                "the ema momentum."
            )
        if warmup_iters is not None:
            warnings.warn(
                "Argument 'warmup_iters' will be deprecated in the future. In the current version, "
                "it has no effect on the ema momentum. Please use the ParamStateScheduler to schedule"
                "the ema momentum."
            )
        if not isinstance(model, nn.Module):
            raise ValueError(
                f"model should be an instance of nn.Module or its subclasses, but got"
                f"model: {model.__class__.__name__}"
            )
        # TODO: in the next version, rename momentum to init_momentum, which is more rigorous
        self.momentum = momentum

        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        self.model = model

        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model.eval()

    def _update_ema_model(self, engine: Engine, name: str) -> None:
        """Update weights of ema model"""
        momentum = getattr(engine.state, name)
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.mul_(1.0 - momentum).add_(model_p.data, alpha=momentum)
        # assign the buffers
        for ema_b, model_b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.data = model_b.data

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
                    f"Attribute '{name}' already exists in Engine.state. Turn off this warning by setting"
                    f"warn_if_exists to False.",
                    category=UserWarning,
                )
        else:
            setattr(engine.state, name, self.momentum)
        engine.add_event_handler(event, self._update_ema_model, name)

import functools
import logging
import math
import time
import warnings
import weakref
from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

from torch.utils.data import DataLoader

from ignite.base import Serializable
from ignite.engine.events import CallableEventWithFilter, EventEnum, Events, EventsList, RemovableEventHandle, State
from ignite.engine.utils import _check_signature, _to_hours_mins_secs

__all__ = ["Engine"]


class Engine(Serializable):
    """Runs a given ``process_function`` over each batch of a dataset, emitting events as it goes.

    Args:
        process_function: A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.

    Attributes:
        state: object that is used to pass internal and user-defined state between event handlers.
            It is created with the engine and its attributes (e.g. ``state.iteration``, ``state.epoch`` etc) are reset
            on every :meth:`~ignite.engine.engine.Engine.run`.
        last_event_name: last event name triggered by the engine.

    Note:
        :class:`~ignite.engine.engine.Engine` implementation has changed in v0.4.10 with "interrupt/resume" feature.
        Engine may behave differently on certain corner cases compared to the one from v0.4.9 and before.
        In such case, you can set ``Engine.interrupt_resume_enabled = False`` to restore previous behaviour.

    Examples:
        Create a basic trainer

        .. code-block:: python

            model = ...
            model = model.cuda()
            optimized = ...
            criterion = ...

            def train_step(engine, batch):
                model.train()
                inputs, targets = batch[0].cuda(), batch[1].cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                return loss.item()

            trainer = Engine(update_model)

            @trainer.on(Events.ITERATION_COMPLETED(every=100))
            def log_training(engine):
                batch_loss = engine.state.output
                lr = optimizer.param_groups[0]['lr']
                e = engine.state.epoch
                n = engine.state.max_epochs
                i = engine.state.iteration
                print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}, lr: {lr}")

            trainer.run(data_loader, max_epochs=5)

            > Epoch 1/5 : 100 - batch loss: 0.10874069479016124, lr: 0.01
            > ...
            > Epoch 2/5 : 1700 - batch loss: 0.4217900575859437, lr: 0.01

        Create a basic evaluator to compute metrics

        .. code-block:: python

            from ignite.metrics import Accuracy

            def predict_on_batch(engine, batch)
                model.eval()
                with torch.no_grad():
                    x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
                    y_pred = model(x)

                return y_pred, y

            evaluator = Engine(predict_on_batch)
            Accuracy().attach(evaluator, "val_acc")
            evaluator.run(val_dataloader)

        Compute image mean/std on training dataset

        .. code-block:: python

            from ignite.metrics import Average

            def compute_mean_std(engine, batch):
                b, c, *_ = batch['image'].shape
                data = batch['image'].reshape(b, c, -1).to(dtype=torch.float64)
                mean = torch.mean(data, dim=-1).sum(dim=0)
                mean2 = torch.mean(data ** 2, dim=-1).sum(dim=0)
                return {"mean": mean, "mean^2": mean2}

            compute_engine = Engine(compute_mean_std)
            img_mean = Average(output_transform=lambda output: output['mean'])
            img_mean.attach(compute_engine, 'mean')
            img_mean2 = Average(output_transform=lambda output: output['mean^2'])
            img_mean2.attach(compute_engine, 'mean2')
            state = compute_engine.run(train_loader)
            state.metrics['std'] = torch.sqrt(state.metrics['mean2'] - state.metrics['mean'] ** 2)
            mean = state.metrics['mean'].tolist()
            std = state.metrics['std'].tolist()

        Resume engine's run from a state. User can load a `state_dict` and run engine starting from loaded state :

        .. code-block:: python

            # Restore from an epoch
            state_dict = {"epoch": 3, "max_epochs": 100, "epoch_length": len(data_loader)}
            # or an iteration
            # state_dict = {"iteration": 500, "max_epochs": 100, "epoch_length": len(data_loader)}

            trainer = Engine(...)
            trainer.load_state_dict(state_dict)
            trainer.run(data)

    """

    _state_dict_all_req_keys = ("epoch_length", "max_epochs")
    _state_dict_one_of_opt_keys = ("iteration", "epoch")

    # Flag to disable engine._internal_run as generator feature for BC
    interrupt_resume_enabled = True

    def __init__(self, process_function: Callable[["Engine", Any], Any]):
        self._event_handlers: Dict[Any, List] = defaultdict(list)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._process_function = process_function
        self.last_event_name: Optional[Events] = None
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.should_interrupt = False
        self.state = State()
        self._state_dict_user_keys: List[str] = []
        self._allowed_events: List[EventEnum] = []

        self._dataloader_iter: Optional[Iterator[Any]] = None
        self._init_iter: Optional[int] = None

        self.register_events(*Events)

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run.")

        _check_signature(process_function, "process_function", self, None)

        # generator provided by self._internal_run_as_gen
        self._internal_run_generator: Optional[Generator] = None

    def register_events(
        self, *event_names: Union[List[str], List[EventEnum]], event_to_attr: Optional[dict] = None
    ) -> None:
        """Add events that can be fired.

        Registering an event will let the user trigger these events at any point.
        This opens the door to make the :meth:`~ignite.engine.engine.Engine.run` loop even more
        configurable.

        By default, the events from :class:`~ignite.engine.events.Events` are registered.

        Args:
            event_names: Defines the name of the event being supported. New events can be a str
                or an object derived from :class:`~ignite.engine.events.EventEnum`. See example below.
            event_to_attr: A dictionary to map an event to a state attribute.

        Examples:
            .. code-block:: python

                from ignite.engine import Engine, Events, EventEnum

                class CustomEvents(EventEnum):
                    FOO_EVENT = "foo_event"
                    BAR_EVENT = "bar_event"

                def process_function(e, batch):
                    # ...
                    trainer.fire_event("bwd_event")
                    loss.backward()
                    # ...
                    trainer.fire_event("opt_event")
                    optimizer.step()

                trainer = Engine(process_function)
                trainer.register_events(*CustomEvents)
                trainer.register_events("bwd_event", "opt_event")

                @trainer.on(Events.EPOCH_COMPLETED)
                def trigger_custom_event():
                    if required(...):
                        trainer.fire_event(CustomEvents.FOO_EVENT)
                    else:
                        trainer.fire_event(CustomEvents.BAR_EVENT)

                @trainer.on(CustomEvents.FOO_EVENT)
                def do_foo_op():
                    # ...

                @trainer.on(CustomEvents.BAR_EVENT)
                def do_bar_op():
                    # ...

            Example with State Attribute:

            .. code-block:: python

                from enum import Enum
                from ignite.engine import Engine, EventEnum

                class TBPTT_Events(EventEnum):
                    TIME_ITERATION_STARTED = "time_iteration_started"
                    TIME_ITERATION_COMPLETED = "time_iteration_completed"

                TBPTT_event_to_attr = {
                    TBPTT_Events.TIME_ITERATION_STARTED: 'time_iteration',
                    TBPTT_Events.TIME_ITERATION_COMPLETED: 'time_iteration'
                }

                engine = Engine(process_function)
                engine.register_events(*TBPTT_Events, event_to_attr=TBPTT_event_to_attr)
                engine.run(data)
                # engine.state contains an attribute time_iteration, which can be accessed
                # using engine.state.time_iteration
        """
        if not (event_to_attr is None or isinstance(event_to_attr, dict)):
            raise ValueError(f"Expected event_to_attr to be dictionary. Got {type(event_to_attr)}.")

        for index, e in enumerate(event_names):
            if not isinstance(e, (str, EventEnum)):
                raise TypeError(f"Value at {index} of event_names should be a str or EventEnum, but given {e}")
            self._allowed_events.append(e)
            if event_to_attr and e in event_to_attr:
                State.event_to_attr[e] = event_to_attr[e]
        # we need to update state attributes associated with new custom events
        self.state._update_attrs()

    def _handler_wrapper(self, handler: Callable, event_name: Any, event_filter: Callable) -> Callable:
        # signature of the following wrapper will be inspected during registering to check if engine is necessary
        # we have to build a wrapper with relevant signature : solution is functools.wraps
        @functools.wraps(handler)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            event = self.state.get_event_attrib_value(event_name)
            if event_filter(self, event):
                return handler(*args, **kwargs)

        # setup input handler as parent to make has_event_handler work
        setattr(wrapper, "_parent", weakref.ref(handler))
        return wrapper

    def _assert_allowed_event(self, event_name: Any) -> None:
        if event_name not in self._allowed_events:
            self.logger.error(f"attempt to add event handler to an invalid event {event_name}")
            raise ValueError(f"Event {event_name} is not a valid event for this {self.__class__.__name__}.")

    def add_event_handler(self, event_name: Any, handler: Callable, *args: Any, **kwargs: Any) -> RemovableEventHandle:
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event or a list of events to attach the handler. Valid events are
                from :class:`~ignite.engine.events.Events` or any ``event_name`` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            handler: the callable event handler that should be invoked. No restrictions on its signature.
                The first argument can be optionally `engine`, the :class:`~ignite.engine.engine.Engine` object,
                handler is bound to.
            args: optional args to be passed to ``handler``.
            kwargs: optional keyword args to be passed to ``handler``.

        Returns:
            :class:`~ignite.engine.events.RemovableEventHandle`, which can be used to remove the handler.

        Note:
            Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
            passed here, for example during :attr:`~ignite.engine.events.Events.EXCEPTION_RAISED`.

        Examples:
            .. code-block:: python

                engine = Engine(process_function)

                def print_epoch(engine):
                    print(f"Epoch: {engine.state.epoch}")

                engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

                events_list = Events.EPOCH_COMPLETED | Events.COMPLETED

                def execute_something():
                    # do some thing not related to engine
                    pass

                engine.add_event_handler(events_list, execute_something)

        Note:
            Since v0.3.0, Events become more flexible and allow to pass an event filter to the Engine.
            See :class:`~ignite.engine.events.Events` for more details.

        """
        if isinstance(event_name, EventsList):
            for e in event_name:
                self.add_event_handler(e, handler, *args, **kwargs)
            return RemovableEventHandle(event_name, handler, self)
        if isinstance(event_name, CallableEventWithFilter) and event_name.filter is not None:
            event_filter = event_name.filter
            handler = self._handler_wrapper(handler, event_name, event_filter)

        self._assert_allowed_event(event_name)

        event_args: Tuple[Any, ...] = ()
        if event_name == Events.EXCEPTION_RAISED:
            event_args += (Exception(),)
        elif event_name == Events.TERMINATE_SINGLE_EPOCH:
            event_args += (0,)

        try:
            _check_signature(handler, "handler", self, *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, (self,) + args, kwargs))
        except ValueError:
            _check_signature(handler, "handler", *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, args, kwargs))
        self.logger.debug(f"Added handler for event {event_name}")

        return RemovableEventHandle(event_name, handler, self)

    def has_event_handler(self, handler: Callable, event_name: Optional[Any] = None) -> bool:
        """Check if the specified event has the specified handler.

        Args:
            handler: the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events: Union[List[Any], Dict[Any, List]] = [event_name]
        else:
            events = self._event_handlers
        for e in events:
            for h, _, _ in self._event_handlers[e]:
                if self._compare_handlers(handler, h):
                    return True
        return False

    @staticmethod
    def _compare_handlers(user_handler: Callable, registered_handler: Callable) -> bool:
        if hasattr(registered_handler, "_parent"):
            registered_handler = registered_handler._parent()
        return registered_handler == user_handler

    def remove_event_handler(self, handler: Callable, event_name: Any) -> None:
        """Remove event handler `handler` from registered handlers of the engine

        Args:
            handler: the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError(f"Input event name '{event_name}' does not exist")

        new_event_handlers = [
            (h, args, kwargs)
            for h, args, kwargs in self._event_handlers[event_name]
            if not self._compare_handlers(handler, h)
        ]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError(f"Input handler '{handler}' is not found among registered event handlers")
        self._event_handlers[event_name] = new_event_handlers

    def on(self, event_name: Any, *args: Any, **kwargs: Any) -> Callable:
        """Decorator shortcut for :meth:`~ignite.engine.engine.Engine.add_event_handler`.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.events.Events`
                or any ``event_name`` added by :meth:`~ignite.engine.engine.Engine.register_events`.
            args: optional args to be passed to `handler`.
            kwargs: optional keyword args to be passed to `handler`.

        Examples:
            .. code-block:: python

                engine = Engine(process_function)

                @engine.on(Events.EPOCH_COMPLETED)
                def print_epoch():
                    print(f"Epoch: {engine.state.epoch}")

                @engine.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
                def execute_something():
                    # do some thing not related to engine
                    pass
        """

        def decorator(f: Callable) -> Callable:
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

    def _fire_event(self, event_name: Any, *event_args: Any, **event_kwargs: Any) -> None:
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        arguments updates arguments passed using :meth:`~ignite.engine.engine.Engine.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        self.logger.debug(f"{self.state.epoch} | {self.state.iteration}, Firing handlers for event {event_name}")
        self.last_event_name = event_name
        for func, args, kwargs in self._event_handlers[event_name]:
            kwargs.update(event_kwargs)
            first, others = ((args[0],), args[1:]) if (args and args[0] == self) else ((), args)
            func(*first, *(event_args + others), **kwargs)

    def fire_event(self, event_name: Any) -> None:
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. This is the method used in :meth:`~ignite.engine.engine.Engine.run` to call the
        core events found in :class:`~ignite.engine.events.Events`.

        Custom events can be fired if they have been registered before with
        :meth:`~ignite.engine.engine.Engine.register_events`. The engine `state` attribute should be used
        to exchange "dynamic" data among `process_function` and handlers.

        This method is called automatically for core events. If no custom
        events are used in the engine, there is no need for the user to call
        the method.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.

        """
        self._assert_allowed_event(event_name)
        return self._fire_event(event_name)

    def interrupt(self) -> None:
        """Sends interrupt signal to the engine, so that it interrupts the run after
        the current iteration. The run can be resumed by calling
        :meth:`~ignite.engine.engine.Engine.run`. Data iteration will continue from the interrupted state.

        Examples:
            .. testcode::

                from ignite.engine import Engine, Events

                data = range(10)
                max_epochs = 3

                def check_input_data(e, b):
                    print(f"Epoch {engine.state.epoch}, Iter {engine.state.iteration} | data={b}")
                    i = (e.state.iteration - 1) % len(data)
                    assert b == data[i]

                engine = Engine(check_input_data)

                @engine.on(Events.ITERATION_COMPLETED(every=11))
                def call_interrupt():
                    engine.interrupt()

                print("Start engine run with interruptions:")
                state = engine.run(data, max_epochs=max_epochs)
                print("1 Engine run is interrupted at ", state.epoch, state.iteration)
                state = engine.run(data, max_epochs=max_epochs)
                print("2 Engine run is interrupted at ", state.epoch, state.iteration)
                state = engine.run(data, max_epochs=max_epochs)
                print("3 Engine ended the run at ", state.epoch, state.iteration)

            .. dropdown:: Output

                .. testoutput::

                    Start engine run with interruptions:
                    Epoch 1, Iter 1 | data=0
                    Epoch 1, Iter 2 | data=1
                    Epoch 1, Iter 3 | data=2
                    Epoch 1, Iter 4 | data=3
                    Epoch 1, Iter 5 | data=4
                    Epoch 1, Iter 6 | data=5
                    Epoch 1, Iter 7 | data=6
                    Epoch 1, Iter 8 | data=7
                    Epoch 1, Iter 9 | data=8
                    Epoch 1, Iter 10 | data=9
                    Epoch 2, Iter 11 | data=0
                    1 Engine run is interrupted at  2 11
                    Epoch 2, Iter 12 | data=1
                    Epoch 2, Iter 13 | data=2
                    Epoch 2, Iter 14 | data=3
                    Epoch 2, Iter 15 | data=4
                    Epoch 2, Iter 16 | data=5
                    Epoch 2, Iter 17 | data=6
                    Epoch 2, Iter 18 | data=7
                    Epoch 2, Iter 19 | data=8
                    Epoch 2, Iter 20 | data=9
                    Epoch 3, Iter 21 | data=0
                    Epoch 3, Iter 22 | data=1
                    2 Engine run is interrupted at  3 22
                    Epoch 3, Iter 23 | data=2
                    Epoch 3, Iter 24 | data=3
                    Epoch 3, Iter 25 | data=4
                    Epoch 3, Iter 26 | data=5
                    Epoch 3, Iter 27 | data=6
                    Epoch 3, Iter 28 | data=7
                    Epoch 3, Iter 29 | data=8
                    Epoch 3, Iter 30 | data=9
                    3 Engine ended the run at  3 30


        .. versionadded:: 0.4.10
        """
        if not self.interrupt_resume_enabled:
            raise RuntimeError(
                "Engine 'interrupt/resume' feature is disabled. "
                "Please, set Engine.interrupt_resume_enabled=True to enable it"
            )

        self.logger.info("interrupt signaled. Engine will interrupt the run after current iteration is finished.")
        self.should_interrupt = True

    def terminate(self) -> None:
        """Sends terminate signal to the engine, so that it terminates completely the run. The run is
        terminated after the event on which ``terminate`` method was called. The following events are triggered:

        - ...
        - Terminating event
        - :attr:`~ignite.engine.events.Events.TERMINATE`
        - :attr:`~ignite.engine.events.Events.COMPLETED`


        Examples:
            .. testcode::

                from ignite.engine import Engine, Events

                def func(engine, batch):
                    print(engine.state.epoch, engine.state.iteration, " | ", batch)

                max_epochs = 4
                data = range(10)
                engine = Engine(func)

                @engine.on(Events.ITERATION_COMPLETED(once=14))
                def terminate():
                    print(f"-> terminate at iteration: {engine.state.iteration}")
                    engine.terminate()

                print("Start engine run:")
                state = engine.run(data, max_epochs=max_epochs)
                print("1 Engine run is terminated at ", state.epoch, state.iteration)
                state = engine.run(data, max_epochs=max_epochs)
                print("2 Engine ended the run at ", state.epoch, state.iteration)

            .. dropdown:: Output

                .. testoutput::

                    Start engine run:
                    1 1  |  0
                    1 2  |  1
                    1 3  |  2
                    1 4  |  3
                    1 5  |  4
                    1 6  |  5
                    1 7  |  6
                    1 8  |  7
                    1 9  |  8
                    1 10  |  9
                    2 11  |  0
                    2 12  |  1
                    2 13  |  2
                    2 14  |  3
                    -> terminate at iteration: 14
                    1 Engine run is terminated at  2 14
                    3 15  |  0
                    3 16  |  1
                    3 17  |  2
                    3 18  |  3
                    3 19  |  4
                    3 20  |  5
                    3 21  |  6
                    3 22  |  7
                    3 23  |  8
                    3 24  |  9
                    4 25  |  0
                    4 26  |  1
                    4 27  |  2
                    4 28  |  3
                    4 29  |  4
                    4 30  |  5
                    4 31  |  6
                    4 32  |  7
                    4 33  |  8
                    4 34  |  9
                    2 Engine ended the run at  4 34

        .. versionchanged:: 0.4.10
            Behaviour changed, for details see https://github.com/pytorch/ignite/issues/2669

        """
        self.logger.info("Terminate signaled. Engine will stop after current iteration is finished.")
        self.should_terminate = True

    def terminate_epoch(self) -> None:
        """Sends terminate signal to the engine, so that it terminates the current epoch. The run
        continues from the next epoch. The following events are triggered:

        - ...
        - Event on which ``terminate_epoch`` method is called
        - :attr:`~ignite.engine.events.Events.TERMINATE_SINGLE_EPOCH`
        - :attr:`~ignite.engine.events.Events.EPOCH_COMPLETED`
        - :attr:`~ignite.engine.events.Events.EPOCH_STARTED`
        - ...
        """
        self.logger.info(
            "Terminate current epoch is signaled. "
            "Current epoch iteration will stop after current iteration is finished."
        )
        self.should_terminate_single_epoch = True

    def _handle_exception(self, e: BaseException) -> None:
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    @property
    def state_dict_user_keys(self) -> List:
        return self._state_dict_user_keys

    def state_dict(self) -> OrderedDict:
        """Returns a dictionary containing engine's state: "epoch_length", "max_epochs" and "iteration" and
        other state values defined by `engine.state_dict_user_keys`

        .. code-block:: python

            engine = Engine(...)
            engine.state_dict_user_keys.append("alpha")
            engine.state_dict_user_keys.append("beta")
            ...

            @engine.on(Events.STARTED)
            def init_user_value(_):
                 engine.state.alpha = 0.1
                 engine.state.beta = 1.0

            @engine.on(Events.COMPLETED)
            def save_engine(_):
                state_dict = engine.state_dict()
                assert "alpha" in state_dict and "beta" in state_dict
                torch.save(state_dict, "/tmp/engine.pt")

        Returns:
            OrderedDict:
                a dictionary containing engine's state

        """
        keys: Tuple[str, ...] = self._state_dict_all_req_keys + (self._state_dict_one_of_opt_keys[0],)
        keys += tuple(self._state_dict_user_keys)
        return OrderedDict([(k, getattr(self.state, k)) for k in keys])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Setups engine from `state_dict`.

        State dictionary should contain keys: `iteration` or `epoch`, `max_epochs` and `epoch_length`.
        If `engine.state_dict_user_keys` contains keys, they should be also present in the state dictionary.
        Iteration and epoch values are 0-based: the first iteration or epoch is zero.

        This method does not remove any custom attributes added by user.

        Args:
            state_dict: a dict with parameters

        .. code-block:: python

            # Restore from the 4rd epoch
            state_dict = {"epoch": 3, "max_epochs": 100, "epoch_length": len(data_loader)}
            # or 500th iteration
            # state_dict = {"iteration": 499, "max_epochs": 100, "epoch_length": len(data_loader)}

            trainer = Engine(...)
            trainer.load_state_dict(state_dict)
            trainer.run(data)

        """
        super(Engine, self).load_state_dict(state_dict)

        for k in self._state_dict_user_keys:
            if k not in state_dict:
                raise ValueError(
                    f"Required user state attribute '{k}' is absent in provided state_dict '{state_dict.keys()}'"
                )
        self.state.max_epochs = state_dict["max_epochs"]
        self.state.epoch_length = state_dict["epoch_length"]
        for k in self._state_dict_user_keys:
            setattr(self.state, k, state_dict[k])

        if "iteration" in state_dict:
            self.state.iteration = state_dict["iteration"]
            self.state.epoch = 0
            if self.state.epoch_length is not None:
                self.state.epoch = self.state.iteration // self.state.epoch_length
        elif "epoch" in state_dict:
            self.state.epoch = state_dict["epoch"]
            if self.state.epoch_length is None:
                raise ValueError(
                    "If epoch is provided in the state dict, epoch_length should not be None. "
                    f"Input state_dict: {state_dict}"
                )
            self.state.iteration = self.state.epoch_length * self.state.epoch

    @staticmethod
    def _is_done(state: State) -> bool:
        is_done_iters = state.max_iters is not None and state.iteration >= state.max_iters
        is_done_count = (
            state.epoch_length is not None
            and state.max_epochs is not None
            and state.iteration >= state.epoch_length * state.max_epochs
        )
        is_done_epochs = state.max_epochs is not None and state.epoch >= state.max_epochs
        return is_done_iters or is_done_count or is_done_epochs

    def set_data(self, data: Union[Iterable, DataLoader]) -> None:
        """Method to set data. After calling the method the next batch passed to `processing_function` is
        from newly provided data. Please, note that epoch length is not modified.

        Args:
            data: Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).

        Examples:
            User can switch data provider during the training:

            .. code-block:: python

                data1 = ...
                data2 = ...

                switch_iteration = 5000

                def train_step(e, batch):
                    # when iteration <= switch_iteration
                    # batch is from data1
                    # when iteration > switch_iteration
                    # batch is from data2
                    ...

                trainer = Engine(train_step)

                @trainer.on(Events.ITERATION_COMPLETED(once=switch_iteration))
                def switch_dataloader():
                    trainer.set_data(data2)

                trainer.run(data1, max_epochs=100)

        """
        self.state.dataloader = data
        self._dataloader_iter = iter(self.state.dataloader)

    def run(
        self,
        data: Optional[Iterable] = None,
        max_epochs: Optional[int] = None,
        max_iters: Optional[int] = None,
        epoch_length: Optional[int] = None,
    ) -> State:
        """Runs the ``process_function`` over the passed data.

        Engine has a state and the following logic is applied in this function:

        - At the first call, new state is defined by `max_epochs`, `max_iters`, `epoch_length`, if provided.
          A timer for total and per-epoch time is initialized when Events.STARTED is handled.
        - If state is already defined such that there are iterations to run until `max_epochs` and no input arguments
          provided, state is kept and used in the function.
        - If state is defined and engine is "done" (no iterations to run until `max_epochs`), a new state is defined.
        - If state is defined, engine is NOT "done", then input arguments if provided override defined state.

        Args:
            data: Collection of batches allowing repeated iteration (e.g., list or `DataLoader`). If not provided, then
                ``epoch_length`` is required and ``batch`` argument of ``process_function`` will be ``None``.
            max_epochs: Max epochs to run for (default: None).
                If a new state should be created (first run or run again from ended engine), it's default value is 1.
                If run is resuming from a state, provided `max_epochs` will be taken into account and should be larger
                than `engine.state.max_epochs`.
            epoch_length: Number of iterations to count as one epoch. By default, it can be set as
                `len(data)`. If `data` is an iterator and `epoch_length` is not set, then it will be automatically
                determined as the iteration on which data iterator raises `StopIteration`.
                This argument should not change if run is resuming from a state.
            max_iters: Number of iterations to run for.
                `max_iters` and `max_epochs` are mutually exclusive; only one of the two arguments should be provided.

        Returns:
            State: output state.

        Note:
            User can dynamically preprocess input batch at :attr:`~ignite.engine.events.Events.ITERATION_STARTED` and
            store output batch in `engine.state.batch`. Latter is passed as usually to `process_function` as argument:

            .. code-block:: python

                trainer = ...

                @trainer.on(Events.ITERATION_STARTED)
                def switch_batch(engine):
                    engine.state.batch = preprocess_batch(engine.state.batch)

            Restart the training from the beginning. User can reset `max_epochs = None`:

            .. code-block:: python

                # ...
                trainer.run(train_loader, max_epochs=5)

                # Reset model weights etc. and restart the training
                trainer.state.max_epochs = None
                trainer.run(train_loader, max_epochs=2)

        """
        if data is not None and not isinstance(data, Iterable):
            raise TypeError("Argument data should be iterable")

        if self.state.max_epochs is not None:
            # Check and apply overridden parameters
            if max_epochs is not None:
                if max_epochs < self.state.epoch:
                    raise ValueError(
                        "Argument max_epochs should be greater than or equal to the start "
                        f"epoch defined in the state: {max_epochs} vs {self.state.epoch}. "
                        "Please, set engine.state.max_epochs = None "
                        "before calling engine.run() in order to restart the training from the beginning."
                    )
                self.state.max_epochs = max_epochs
            if epoch_length is not None:
                if epoch_length != self.state.epoch_length:
                    raise ValueError(
                        "Argument epoch_length should be same as in the state, "
                        f"but given {epoch_length} vs {self.state.epoch_length}"
                    )

        if self.state.max_epochs is None or (self._is_done(self.state) and self._internal_run_generator is None):
            # Create new state
            if epoch_length is None:
                if data is None:
                    raise ValueError("epoch_length should be provided if data is None")

                epoch_length = self._get_data_length(data)
                if epoch_length is not None and epoch_length < 1:
                    raise ValueError("Input data has zero size. Please provide non-empty data")

            if max_iters is None:
                if max_epochs is None:
                    max_epochs = 1
            else:
                if max_epochs is not None:
                    raise ValueError(
                        "Arguments max_iters and max_epochs are mutually exclusive."
                        "Please provide only max_epochs or max_iters."
                    )
                if epoch_length is not None:
                    max_epochs = math.ceil(max_iters / epoch_length)

            self.state.iteration = 0
            self.state.epoch = 0
            self.state.max_epochs = max_epochs
            self.state.max_iters = max_iters
            self.state.epoch_length = epoch_length
            # Reset generator if previously used
            self._internal_run_generator = None
            self.logger.info(f"Engine run starting with max_epochs={max_epochs}.")
        else:
            self.logger.info(
                f"Engine run resuming from iteration {self.state.iteration}, "
                f"epoch {self.state.epoch} until {self.state.max_epochs} epochs"
            )
            if self.state.epoch_length is None and data is None:
                raise ValueError("epoch_length should be provided if data is None")

            if self.should_terminate:
                # If engine was terminated and now is resuming from terminated state
                # we need to initialize iter_counter as 0
                self._init_iter = 0

        if self._dataloader_iter is None:
            self.state.dataloader = data

        if self.interrupt_resume_enabled:
            return self._internal_run()
        else:
            return self._internal_run_legacy()

    @staticmethod
    def _init_timers(state: State) -> None:
        state.times[Events.EPOCH_COMPLETED.name] = 0.0
        state.times[Events.COMPLETED.name] = 0.0

    def _get_data_length(self, data: Iterable) -> Optional[int]:
        try:
            if hasattr(data, "__len__"):
                return len(data)  # type: ignore[arg-type]
        except TypeError:
            # _InfiniteConstantSampler can raise a TypeError on DataLoader length of a IterableDataset
            pass
        return None

    def _setup_dataloader_iter(self) -> None:
        if self.state.dataloader is None:
            if self.state.epoch_length is None:
                raise RuntimeError(
                    "Internal error, self.state.epoch_length is None. "
                    "Please, file an issue if you encounter this error."
                )
            self._dataloader_iter = _get_none_data_iter(self.state.epoch_length)
        else:
            self._dataloader_iter = iter(self.state.dataloader)

    def _setup_engine(self) -> None:
        self._setup_dataloader_iter()

        if self._init_iter is None:
            iteration = self.state.iteration
            # Below we define initial counter value for _run_once_on_dataset to measure a single epoch
            if self.state.epoch_length is not None:
                iteration %= self.state.epoch_length
            self._init_iter = iteration

    def _internal_run(self) -> State:
        if self._internal_run_generator is None:
            self._internal_run_generator = self._internal_run_as_gen()
        try:
            return next(self._internal_run_generator)
        except StopIteration as out:
            self._internal_run_generator = None
            return out.value

    def _internal_run_as_gen(self) -> Generator:
        self.should_terminate = self.should_terminate_single_epoch = self.should_interrupt = False
        self._init_timers(self.state)
        try:
            try:
                start_time = time.time()
                self._fire_event(Events.STARTED)
                yield from self._maybe_terminate_or_interrupt()

                while not self._is_done(self.state) and not self.should_terminate:
                    self.state.epoch += 1
                    handlers_start_time = time.time()
                    self._fire_event(Events.EPOCH_STARTED)
                    epoch_time_taken = time.time() - handlers_start_time
                    yield from self._maybe_terminate_or_interrupt()

                    if self._dataloader_iter is None:
                        self._setup_engine()

                    epoch_time_taken += yield from self._run_once_on_dataset_as_gen()

                    # time is available for handlers but must be updated after fire
                    self.state.times[Events.EPOCH_COMPLETED.name] = epoch_time_taken

                    handlers_start_time = time.time()
                    self._fire_event(Events.EPOCH_COMPLETED)
                    epoch_time_taken += time.time() - handlers_start_time
                    # update time wrt handlers
                    self.state.times[Events.EPOCH_COMPLETED.name] = epoch_time_taken
                    yield from self._maybe_terminate_or_interrupt()

                    hours, mins, secs = _to_hours_mins_secs(epoch_time_taken)
                    self.logger.info(
                        f"Epoch[{self.state.epoch}] Complete. Time taken: {hours:02d}:{mins:02d}:{secs:06.3f}"
                    )

            except _EngineTerminateException:
                self._fire_event(Events.TERMINATE)

            time_taken = time.time() - start_time
            # time is available for handlers but must be updated after fire
            self.state.times[Events.COMPLETED.name] = time_taken
            handlers_start_time = time.time()
            self._fire_event(Events.COMPLETED)
            time_taken += time.time() - handlers_start_time
            # update time wrt handlers
            self.state.times[Events.COMPLETED.name] = time_taken
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self.logger.info(f"Engine run complete. Time taken: {hours:02d}:{mins:02d}:{secs:06.3f}")

        except BaseException as e:
            self._dataloader_iter = None
            self.logger.error(f"Engine run is terminating due to exception: {e}")
            self._handle_exception(e)

        self._dataloader_iter = None
        return self.state

    def _maybe_terminate_or_interrupt(self) -> Generator:
        if self.should_terminate:
            raise _EngineTerminateException()

        if self.should_terminate_single_epoch:
            raise _EngineTerminateSingleEpochException()

        if self.should_interrupt:
            self._fire_event(Events.INTERRUPT)
            self.should_interrupt = False
            yield self.state

    def _run_once_on_dataset_as_gen(self) -> Generator[State, None, float]:
        start_time = time.time()

        # We need to setup iter_counter > 0 if we resume from an iteration
        iter_counter = 0 if self._init_iter is None else self._init_iter
        self._init_iter = None
        should_exit = False
        try:
            if self._dataloader_iter is None:
                raise RuntimeError(
                    "Internal error, self._dataloader_iter is None. "
                    "Please, file an issue if you encounter this error."
                )

            while True:
                self.state.batch = self.state.output = None
                try:
                    # Avoid Events.GET_BATCH_STARTED triggered twice when data iter is restarted
                    if self.last_event_name != Events.DATALOADER_STOP_ITERATION:
                        self._fire_event(Events.GET_BATCH_STARTED)
                        yield from self._maybe_terminate_or_interrupt()

                    self.state.batch = next(self._dataloader_iter)
                    self._fire_event(Events.GET_BATCH_COMPLETED)
                    yield from self._maybe_terminate_or_interrupt()

                    iter_counter += 1
                    should_exit = False
                except StopIteration:
                    # Define self.state.epoch_length if it is not yet set
                    if self.state.epoch_length is None:
                        # Define epoch length and stop the epoch
                        self.state.epoch_length = iter_counter
                        if self.state.max_iters is not None:
                            self.state.max_epochs = math.ceil(self.state.max_iters / self.state.epoch_length)
                        break

                    # Should exit while loop if we can not iterate
                    if should_exit:
                        if not self._is_done(self.state):
                            total_iters = (
                                self.state.epoch_length * self.state.max_epochs
                                if self.state.max_epochs is not None
                                else self.state.max_iters
                            )

                            warnings.warn(
                                "Data iterator can not provide data anymore but required total number of "
                                "iterations to run is not reached. "
                                f"Current iteration: {self.state.iteration} vs Total iterations to run : {total_iters}"
                            )
                        break

                    self._fire_event(Events.DATALOADER_STOP_ITERATION)
                    yield from self._maybe_terminate_or_interrupt()

                    self._setup_dataloader_iter()
                    should_exit = True

                    continue

                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                yield from self._maybe_terminate_or_interrupt()

                self.state.output = self._process_function(self, self.state.batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                yield from self._maybe_terminate_or_interrupt()

                if self.state.epoch_length is not None and iter_counter == self.state.epoch_length:
                    break

                if self.state.max_iters is not None and self.state.iteration == self.state.max_iters:
                    self.should_terminate = True
                    raise _EngineTerminateException()

        except _EngineTerminateSingleEpochException:
            self._fire_event(Events.TERMINATE_SINGLE_EPOCH, iter_counter=iter_counter)
            self.should_terminate_single_epoch = False
            self._setup_dataloader_iter()

        except _EngineTerminateException as e:
            # we need to reraise this exception such that it is not handled
            # as a general exception by the code below
            raise e

        except Exception as e:
            self.logger.error(f"Current run is terminating due to exception: {e}")
            self._handle_exception(e)

        return time.time() - start_time

    def _maybe_terminate_legacy(self) -> None:
        if self.should_terminate:
            raise _EngineTerminateException()

        if self.should_terminate_single_epoch:
            raise _EngineTerminateSingleEpochException()

    def _internal_run_legacy(self) -> State:
        # internal_run without generator for BC
        self.should_terminate = self.should_terminate_single_epoch = self.should_interrupt = False
        self._init_timers(self.state)
        try:
            try:
                start_time = time.time()
                self._fire_event(Events.STARTED)
                self._maybe_terminate_legacy()

                while not self._is_done(self.state) and not self.should_terminate:
                    self.state.epoch += 1
                    handlers_start_time = time.time()
                    self._fire_event(Events.EPOCH_STARTED)
                    epoch_time_taken = time.time() - handlers_start_time
                    self._maybe_terminate_legacy()

                    if self._dataloader_iter is None:
                        self._setup_engine()

                    epoch_time_taken += self._run_once_on_dataset_legacy()

                    # time is available for handlers but must be updated after fire
                    self.state.times[Events.EPOCH_COMPLETED.name] = epoch_time_taken

                    handlers_start_time = time.time()
                    self._fire_event(Events.EPOCH_COMPLETED)
                    epoch_time_taken += time.time() - handlers_start_time
                    # update time wrt handlers
                    self.state.times[Events.EPOCH_COMPLETED.name] = epoch_time_taken
                    self._maybe_terminate_legacy()

                    hours, mins, secs = _to_hours_mins_secs(epoch_time_taken)
                    self.logger.info(
                        f"Epoch[{self.state.epoch}] Complete. Time taken: {hours:02d}:{mins:02d}:{secs:06.3f}"
                    )

            except _EngineTerminateException:
                self._fire_event(Events.TERMINATE)

            time_taken = time.time() - start_time
            # time is available for handlers but must be updated after fire
            self.state.times[Events.COMPLETED.name] = time_taken
            handlers_start_time = time.time()
            self._fire_event(Events.COMPLETED)
            time_taken += time.time() - handlers_start_time
            # update time wrt handlers
            self.state.times[Events.COMPLETED.name] = time_taken
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self.logger.info(f"Engine run complete. Time taken: {hours:02d}:{mins:02d}:{secs:06.3f}")

        except BaseException as e:
            self._dataloader_iter = None
            self.logger.error(f"Engine run is terminating due to exception: {e}")
            self._handle_exception(e)

        self._dataloader_iter = None
        return self.state

    def _run_once_on_dataset_legacy(self) -> float:
        start_time = time.time()

        # We need to setup iter_counter > 0 if we resume from an iteration
        iter_counter = 0 if self._init_iter is None else self._init_iter
        self._init_iter = None
        should_exit = False
        try:
            if self._dataloader_iter is None:
                raise RuntimeError(
                    "Internal error, self._dataloader_iter is None. "
                    "Please, file an issue if you encounter this error."
                )

            while True:
                self.state.batch = self.state.output = None
                try:
                    # Avoid Events.GET_BATCH_STARTED triggered twice when data iter is restarted
                    if self.last_event_name != Events.DATALOADER_STOP_ITERATION:
                        self._fire_event(Events.GET_BATCH_STARTED)
                        self._maybe_terminate_legacy()

                    self.state.batch = next(self._dataloader_iter)
                    self._fire_event(Events.GET_BATCH_COMPLETED)
                    self._maybe_terminate_legacy()

                    iter_counter += 1
                    should_exit = False
                except StopIteration:
                    # Define self.state.epoch_length if it is not yet set
                    if self.state.epoch_length is None:
                        # Define epoch length and stop the epoch
                        self.state.epoch_length = iter_counter
                        if self.state.max_iters is not None:
                            self.state.max_epochs = math.ceil(self.state.max_iters / self.state.epoch_length)
                        break

                    # Should exit while loop if we can not iterate
                    if should_exit:
                        if not self._is_done(self.state):
                            total_iters = (
                                self.state.epoch_length * self.state.max_epochs
                                if self.state.max_epochs is not None
                                else self.state.max_iters
                            )

                            warnings.warn(
                                "Data iterator can not provide data anymore but required total number of "
                                "iterations to run is not reached. "
                                f"Current iteration: {self.state.iteration} vs Total iterations to run : {total_iters}"
                            )
                        break

                    self._fire_event(Events.DATALOADER_STOP_ITERATION)
                    self._maybe_terminate_legacy()

                    self._setup_dataloader_iter()
                    should_exit = True

                    continue

                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self._maybe_terminate_legacy()

                self.state.output = self._process_function(self, self.state.batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                self._maybe_terminate_legacy()

                if self.state.epoch_length is not None and iter_counter == self.state.epoch_length:
                    break

                if self.state.max_iters is not None and self.state.iteration == self.state.max_iters:
                    self.should_terminate = True
                    raise _EngineTerminateException()

        except _EngineTerminateSingleEpochException:
            self._fire_event(Events.TERMINATE_SINGLE_EPOCH, iter_counter=iter_counter)
            self.should_terminate_single_epoch = False
            self._setup_dataloader_iter()

        except _EngineTerminateException as e:
            # we need to reraise this exception such that it is not handled
            # as a general exception by the code below
            raise e

        except Exception as e:
            self.logger.error(f"Current run is terminating due to exception: {e}")
            self._handle_exception(e)

        return time.time() - start_time


def _get_none_data_iter(size: int) -> Iterator:
    # Sized iterator for data as None
    for _ in range(size):
        yield None


class _EngineTerminateSingleEpochException(Exception):
    """
    Exception associated with Terminate Single Epoch event
    """

    pass


class _EngineTerminateException(Exception):
    """
    Exception associated with Terminate event
    """

    pass

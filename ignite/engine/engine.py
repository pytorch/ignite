import functools
import logging
import time
import warnings
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from typing import Any, Callable, Iterable, List, Optional

from ignite._utils import _to_hours_mins_secs
from ignite.base import Serializable
from ignite.engine.events import CallableEventWithFilter, Events, EventsList, RemovableEventHandle, State
from ignite.engine.utils import _check_signature

__all__ = ["Engine"]


class Engine(Serializable):
    """Runs a given ``process_function`` over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.

    Attributes:
        state (State): object that is used to pass internal and user-defined state between event handlers.
            It is created with the engine and its attributes (e.g. ``state.iteration``, ``state.epoch`` etc) are reset
            on every :meth:`~ignite.engine.engine.Engine.run`.
        last_event_name (Events): last event name triggered by the engine.

    Examples:

        Create a basic trainer

        .. code-block:: python

            def update_model(engine, batch):
                inputs, targets = batch
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
                print("Epoch {}/{} : {} - batch loss: {}, lr: {}".format(e, n, i, batch_loss, lr))

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

    def __init__(self, process_function: Callable):
        self._event_handlers = defaultdict(list)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._process_function = process_function
        self.last_event_name = None
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = State()
        self._state_dict_user_keys = []
        self._allowed_events = []

        self._dataloader_iter = None
        self._init_iter = []

        self.register_events(*Events)

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run.")

        _check_signature(process_function, "process_function", self, None)

    def register_events(self, *event_names: Any, event_to_attr: Optional[dict] = None) -> None:
        """Add events that can be fired.

        Registering an event will let the user fire these events at any point.
        This opens the door to make the :meth:`~ignite.engine.engine.Engine.run` loop even more
        configurable.

        By default, the events from :class:`~ignite.engine.events.Events` are registered.

        Args:
            *event_names: An object (ideally a string or int) to define the name of the event being supported.
            event_to_attr (dict, optional): A dictionary to map an event to a state attribute.

        Example usage:

        .. code-block:: python

            from ignite.engine import Engine, EventEnum

            class CustomEvents(EventEnum):
                FOO_EVENT = "foo_event"
                BAR_EVENT = "bar_event"

            engine = Engine(process_function)
            engine.register_events(*CustomEvents)


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
            # engine.state contains an attribute time_iteration, which can be accessed using engine.state.time_iteration
        """
        if not (event_to_attr is None or isinstance(event_to_attr, dict)):
            raise ValueError("Expected event_to_attr to be dictionary. Got {}.".format(type(event_to_attr)))

        for e in event_names:
            self._allowed_events.append(e)
            if event_to_attr and e in event_to_attr:
                State.event_to_attr[e] = event_to_attr[e]
        # we need to update state attributes associated with new custom events
        self.state._update_attrs()

    def _handler_wrapper(self, handler: Callable, event_name: Any, event_filter: Callable) -> Callable:
        # signature of the following wrapper will be inspected during registering to check if engine is necessary
        # we have to build a wrapper with relevant signature : solution is functools.wraps
        @functools.wraps(handler)
        def wrapper(*args, **kwargs) -> Any:
            event = self.state.get_event_attrib_value(event_name)
            if event_filter(self, event):
                return handler(*args, **kwargs)

        # setup input handler as parent to make has_event_handler work
        wrapper._parent = weakref.ref(handler)
        return wrapper

    def add_event_handler(self, event_name: Any, handler: Callable, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event or a list of events to attach the handler. Valid events are
                from :class:`~ignite.engine.events.Events` or any ``event_name`` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            handler (callable): the callable event handler that should be invoked. No restrictions on its signature.
                The first argument can be optionally `engine`, the :class:`~ignite.engine.engine.Engine` object,
                handler is bound to.
            *args: optional args to be passed to ``handler``.
            **kwargs: optional keyword args to be passed to ``handler``.

        Note:
            Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
            passed here, for example during :attr:`~ignite.engine.events.Events.EXCEPTION_RAISED`.

        Returns:
            :class:`~ignite.engine.RemovableEventHandle`, which can be used to remove the handler.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print("Epoch: {}".format(engine.state.epoch))

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
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            event_filter = event_name.filter
            handler = self._handler_wrapper(handler, event_name, event_filter)

        if event_name not in self._allowed_events:
            self.logger.error("attempt to add event handler to an invalid event %s.", event_name)
            raise ValueError("Event {} is not a valid event for this Engine.".format(event_name))

        event_args = (Exception(),) if event_name == Events.EXCEPTION_RAISED else ()
        try:
            _check_signature(handler, "handler", self, *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, (self,) + args, kwargs))
        except ValueError:
            _check_signature(handler, "handler", *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, args, kwargs))
        self.logger.debug("added handler for event %s.", event_name)

        return RemovableEventHandle(event_name, handler, self)

    @staticmethod
    def _assert_non_filtered_event(event_name: Any):
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            raise TypeError(
                "Argument event_name should not be a filtered event, " "please use event without any event filtering"
            )

    def has_event_handler(self, handler: Callable, event_name: Optional[Any] = None):
        """Check if the specified event has the specified handler.

        Args:
            handler (callable): the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events = [event_name]
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

    def remove_event_handler(self, handler: Callable, event_name: Any):
        """Remove event handler `handler` from registered handlers of the engine

        Args:
            handler (callable): the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError("Input event name '{}' does not exist".format(event_name))

        new_event_handlers = [
            (h, args, kwargs)
            for h, args, kwargs in self._event_handlers[event_name]
            if not self._compare_handlers(handler, h)
        ]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError("Input handler '{}' is not found among registered event handlers".format(handler))
        self._event_handlers[event_name] = new_event_handlers

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.events.Events`
                or any ``event_name`` added by :meth:`~ignite.engine.engine.Engine.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            @engine.on(Events.EPOCH_COMPLETED)
            def print_epoch():
                print("Epoch: {}".format(engine.state.epoch))

            @engine.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
            def execute_something():
                # do some thing not related to engine
                pass
        """

        def decorator(f: Callable) -> Callable:
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

    def _fire_event(self, event_name: Any, *event_args, **event_kwargs) -> None:
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
        if event_name in self._allowed_events:
            self.logger.debug("firing handlers for event %s ", event_name)
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
        return self._fire_event(event_name)

    def terminate(self) -> None:
        """Sends terminate signal to the engine, so that it terminates completely the run after the current iteration.
        """
        self.logger.info("Terminate signaled. Engine will stop after current iteration is finished.")
        self.should_terminate = True

    def terminate_epoch(self) -> None:
        """Sends terminate signal to the engine, so that it terminates the current epoch after the current iteration.
        """
        self.logger.info(
            "Terminate current epoch is signaled. "
            "Current epoch iteration will stop after current iteration is finished."
        )
        self.should_terminate_single_epoch = True

    def _handle_exception(self, e: Exception) -> None:
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    @property
    def state_dict_user_keys(self) -> List:
        return self._state_dict_user_keys

    def state_dict(self) -> OrderedDict:
        """Returns a dictionary containing engine's state: "seed", "epoch_length", "max_epochs" and "iteration" and
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
        keys = self._state_dict_all_req_keys + (self._state_dict_one_of_opt_keys[0],)
        keys += tuple(self._state_dict_user_keys)
        return OrderedDict([(k, getattr(self.state, k)) for k in keys])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Setups engine from `state_dict`.

        State dictionary should contain keys: `iteration` or `epoch` and `max_epochs`, `epoch_length` and
        `seed`. If `engine.state_dict_user_keys` contains keys, they should be also present in the state dictionary.
        Iteration and epoch values are 0-based: the first iteration or epoch is zero.

        This method does not remove any custom attributs added by user.

        Args:
            state_dict (Mapping): a dict with parameters

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
                    "Required user state attribute '{}' is absent in provided state_dict '{}'".format(
                        k, state_dict.keys()
                    )
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
                    "Input state_dict: {}".format(state_dict)
                )
            self.state.iteration = self.state.epoch_length * self.state.epoch

    @staticmethod
    def _is_done(state: State) -> bool:
        return state.iteration == state.epoch_length * state.max_epochs

    def set_data(self, data):
        """Method to set data. After calling the method the next batch passed to `processing_function` is
        from newly provided data. Please, note that epoch length is not modified.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).

        Example usage:
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
        data: Iterable,
        max_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> State:
        """Runs the `process_function` over the passed data.

        Engine has a state and the following logic is applied in this function:

        - At the first call, new state is defined by `max_epochs`, `epoch_length`, `seed` if provided. A timer for
            total and per-epoch time is initialized when Events.STARTED is handled.
        - If state is already defined such that there are iterations to run until `max_epochs` and no input arguments
            provided, state is kept and used in the function.
        - If state is defined and engine is "done" (no iterations to run until `max_epochs`), a new state is defined.
        - If state is defined, engine is NOT "done", then input arguments if provided override defined state.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).
            max_epochs (int, optional): Max epochs to run for (default: None).
                If a new state should be created (first run or run again from ended engine), it's default value is 1.
                If run is resuming from a state, provided `max_epochs` will be taken into account and should be larger
                than `engine.state.max_epochs`.
            epoch_length (int, optional): Number of iterations to count as one epoch. By default, it can be set as
                `len(data)`. If `data` is an iterator and `epoch_length` is not set, then it will be automatically
                determined as the iteration on which data iterator raises `StopIteration`.
                This argument should not change if run is resuming from a state.
            seed (int, optional): Deprecated argument. Please, use `torch.manual_seed` or
                :meth:`~ignite.utils.manual_seed`.

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

        """
        if seed is not None:
            warnings.warn(
                "Argument seed is deprecated. It will be removed in 0.5.0. "
                "Please, use torch.manual_seed or ignite.utils.manual_seed"
            )

        if self.state.max_epochs is not None:
            # Check and apply overridden parameters
            if max_epochs is not None:
                if max_epochs < self.state.epoch:
                    raise ValueError(
                        "Argument max_epochs should be larger than the start epoch "
                        "defined in the state: {} vs {}".format(max_epochs, self.state.epoch)
                    )
                self.state.max_epochs = max_epochs
            if epoch_length is not None:
                if epoch_length != self.state.epoch_length:
                    raise ValueError(
                        "Argument epoch_length should be same as in the state, given {} vs {}".format(
                            epoch_length, self.state.epoch_length
                        )
                    )

        if self.state.max_epochs is None or self._is_done(self.state):
            # Create new state
            if max_epochs is None:
                max_epochs = 1
            if epoch_length is None:
                epoch_length = self._get_data_length(data)
                if epoch_length is not None and epoch_length < 1:
                    raise ValueError("Input data has zero size. Please provide non-empty data")

            self.state.iteration = 0
            self.state.epoch = 0
            self.state.max_epochs = max_epochs
            self.state.epoch_length = epoch_length
            self.logger.info("Engine run starting with max_epochs={}.".format(max_epochs))
        else:
            self.logger.info(
                "Engine run resuming from iteration {}, epoch {} until {} epochs".format(
                    self.state.iteration, self.state.epoch, self.state.max_epochs
                )
            )

        self.state.dataloader = data
        return self._internal_run()

    @staticmethod
    def _init_timers(state: State):
        state.times[Events.EPOCH_COMPLETED.name] = 0.0
        state.times[Events.COMPLETED.name] = 0.0

    def _get_data_length(self, data):
        data_length = None
        try:
            if hasattr(data, "__len__"):
                data_length = len(data)
        except TypeError:
            # _InfiniteConstantSampler can raise a TypeError on DataLoader length of a IterableDataset
            pass
        return data_length

    def _setup_engine(self) -> None:
        iteration = self.state.iteration
        self._dataloader_iter = iter(self.state.dataloader)

        # Below we define initial counter value for _run_once_on_dataset to measure a single epoch
        if self.state.epoch_length is not None:
            iteration %= self.state.epoch_length
        self._init_iter.append(iteration)

    def _internal_run(self) -> State:
        self.should_terminate = self.should_terminate_single_epoch = False
        self._init_timers(self.state)
        try:
            start_time = time.time()
            self._fire_event(Events.STARTED)
            while self.state.epoch < self.state.max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(Events.EPOCH_STARTED)

                if self._dataloader_iter is None:
                    self._setup_engine()

                time_taken = self._run_once_on_dataset()
                # time is available for handlers but must be update after fire
                self.state.times[Events.EPOCH_COMPLETED.name] = time_taken
                handlers_start_time = time.time()
                if self.should_terminate:
                    self._fire_event(Events.TERMINATE)
                else:
                    self._fire_event(Events.EPOCH_COMPLETED)
                time_taken += time.time() - handlers_start_time
                # update time wrt handlers
                self.state.times[Events.EPOCH_COMPLETED.name] = time_taken
                hours, mins, secs = _to_hours_mins_secs(time_taken)
                self.logger.info(
                    "Epoch[%s] Complete. Time taken: %02d:%02d:%02d" % (self.state.epoch, hours, mins, secs)
                )
                if self.should_terminate:
                    break

            time_taken = time.time() - start_time
            # time is available for handlers but must be update after fire
            self.state.times[Events.COMPLETED.name] = time_taken
            handlers_start_time = time.time()
            self._fire_event(Events.COMPLETED)
            time_taken += time.time() - handlers_start_time
            # update time wrt handlers
            self.state.times[Events.COMPLETED.name] = time_taken
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self.logger.info("Engine run complete. Time taken: %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._dataloader_iter = None
            self.logger.error("Engine run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        self._dataloader_iter = None
        return self.state

    def _run_once_on_dataset(self) -> float:
        start_time = time.time()

        # We need to setup iter_counter > 0 if we resume from an iteration
        iter_counter = self._init_iter.pop() if len(self._init_iter) > 0 else 0
        should_exit = False
        try:
            while True:
                try:
                    # Avoid Events.GET_BATCH_STARTED triggered twice when data iter is restarted
                    if self.last_event_name != Events.DATALOADER_STOP_ITERATION:
                        self._fire_event(Events.GET_BATCH_STARTED)
                    self.state.batch = next(self._dataloader_iter)
                    self._fire_event(Events.GET_BATCH_COMPLETED)
                    iter_counter += 1
                    should_exit = False
                except StopIteration:
                    # Define self.state.epoch_length if it is not yet set
                    if self.state.epoch_length is None:
                        # Define epoch length and stop the epoch
                        self.state.epoch_length = iter_counter
                        break

                    # Should exit while loop if we can not iterate
                    if should_exit:
                        if not self._is_done(self.state):
                            warnings.warn(
                                "Data iterator can not provide data anymore but required total number of "
                                "iterations to run is not reached. "
                                "Current iteration: {} vs Total iterations to run : {}".format(
                                    self.state.iteration, self.state.epoch_length * self.state.max_epochs
                                )
                            )
                        break

                    self._fire_event(Events.DATALOADER_STOP_ITERATION)
                    self.set_data(self.state.dataloader)

                    should_exit = True

                    continue

                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, self.state.batch)
                self._fire_event(Events.ITERATION_COMPLETED)

                # TODO: remove refs on batch to avoid high mem consumption ? -> need verification
                # self.state.batch = None

                if self.should_terminate or self.should_terminate_single_epoch:
                    self._fire_event(Events.TERMINATE_SINGLE_EPOCH, iter_counter=iter_counter)
                    self.should_terminate_single_epoch = False
                    self.set_data(self.state.dataloader)
                    break

                if self.state.epoch_length is not None and iter_counter == self.state.epoch_length:
                    break

        except Exception as e:
            self.logger.error("Current run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        return time.time() - start_time

import logging
import time
from collections import defaultdict, OrderedDict
from collections.abc import Mapping
import weakref
import random
import warnings
from typing import Union, Optional, Callable, Iterable, Iterator, Any, Tuple

import torch

from ignite.engine.events import Events, State, CallableEventWithFilter, RemovableEventHandle, EventsList
from ignite.engine.utils import ReproducibleBatchSampler, _update_dataloader, _check_signature
from ignite._utils import _to_hours_mins_secs

__all__ = ["Engine"]


class Engine:
    """Runs a given `process_function` over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.

    Attributes:
        state (State): object that is used to pass internal and user-defined state between event handlers.
            It is created and reset on every :meth:`~ignite.engine.Engine.run`.
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
            state_dict = {"seed": 0, "epoch": 3, "max_epochs": 100, "epoch_length": len(data_loader)}
            # or an iteration
            # state_dict = {"seed": 0, "iteration": 500, "max_epochs": 100, "epoch_length": len(data_loader)}

            trainer = Engine(...)
            trainer.load_state_dict(state_dict)
            trainer.run(data)

    """

    _state_dict_all_req_keys = ("seed", "epoch_length", "max_epochs")
    _state_dict_one_of_opt_keys = ("iteration", "epoch")

    def __init__(self, process_function: Callable):
        self._event_handlers = defaultdict(list)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._process_function = process_function
        self.last_event_name = None
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = None
        self._allowed_events = []

        self._dataloader_iter = None
        self._init_iter = []

        self.register_events(*Events)

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run.")

        _check_signature(self, process_function, "process_function", None)

    def register_events(self, *event_names: Any, event_to_attr: Optional[dict] = None) -> None:
        """Add events that can be fired.

        Registering an event will let the user fire these events at any point.
        This opens the door to make the :meth:`~ignite.engine.Engine.run` loop even more
        configurable.

        By default, the events from :class:`~ignite.engine.Events` are registered.

        Args:
            *event_names: An object (ideally a string or int) to define the
                name of the event being supported.
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

    @staticmethod
    def _handler_wrapper(handler: Callable, event_name: Any, event_filter: Callable) -> Callable:
        def wrapper(engine: Engine, *args, **kwargs) -> Any:
            event = engine.state.get_event_attrib_value(event_name)
            if event_filter(engine, event):
                return handler(engine, *args, **kwargs)

        # setup input handler as parent to make has_event_handler work
        wrapper._parent = weakref.ref(handler)
        return wrapper

    def add_event_handler(self, event_name: Any, handler: Callable, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event or a list of events to attach the handler. Valid events are
                from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.
            handler (callable): the callable event handler that should be invoked
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        Note:
            The handler function's first argument will be `self`, the :class:`~ignite.engine.Engine` object it
            was bound to.

            Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
            passed here, for example during :attr:`~ignite.engine.Events.EXCEPTION_RAISED`.

        Returns:
            :class:`~ignite.engine.RemovableEventHandle`, which can be used to remove the handler.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print("Epoch: {}".format(engine.state.epoch))

            engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

            events_list = Events.EPOCH_COMPLETED | Events.COMPLETED

            def execute_validation(engine):
                # do some validations

            engine.add_event_handler(events_list, execute_validation)

        Note:
            Since v0.3.0, Events become more flexible and allow to pass an event filter to the Engine.
            See :class:`~ignite.engine.Events` for more details.

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
            handler = Engine._handler_wrapper(handler, event_name, event_filter)

        if event_name not in self._allowed_events:
            self.logger.error("attempt to add event handler to an invalid event %s.", event_name)
            raise ValueError("Event {} is not a valid event for this Engine.".format(event_name))

        event_args = (Exception(),) if event_name == Events.EXCEPTION_RAISED else ()
        _check_signature(self, handler, "handler", *(event_args + args), **kwargs)

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
            self._assert_non_filtered_event(event_name)

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
        self._assert_non_filtered_event(event_name)
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
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events` or
                any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

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
        arguments updates arguments passed using :meth:`~ignite.engine.Engine.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        if event_name in self._allowed_events:
            self.logger.debug("firing handlers for event %s ", event_name)
            self.last_event_name = event_name
            for func, args, kwargs in self._event_handlers[event_name]:
                kwargs.update(event_kwargs)
                func(self, *(event_args + args), **kwargs)

    def fire_event(self, event_name: Any) -> None:
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. This is the method used in :meth:`~ignite.engine.Engine.run` to call the
        core events found in :class:`~ignite.engine.Events`.

        Custom events can be fired if they have been registered before with
        :meth:`~ignite.engine.Engine.register_events`. The engine `state` attribute should be used
        to exchange "dynamic" data among `process_function` and handlers.

        This method is called automatically for core events. If no custom
        events are used in the engine, there is no need for the user to call
        the method.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.

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

    def _run_once_on_dataset(self) -> Tuple[int, int, int]:
        start_time = time.time()

        # We need to setup iter_counter > 0 if we resume from an iteration
        iter_counter = self._init_iter.pop() if len(self._init_iter) > 0 else 0
        should_exit = False
        try:
            while True:
                try:
                    self._fire_event(Events.GET_BATCH_STARTED)
                    batch = next(self._dataloader_iter)
                    self._fire_event(Events.GET_BATCH_COMPLETED)
                    iter_counter += 1
                    should_exit = False
                except StopIteration:
                    if self._dataloader_len is None:
                        if iter_counter > 0:
                            self._dataloader_len = iter_counter
                        else:
                            # this can happen when data is finite iterator and epoch_length is equal to its size
                            self._dataloader_len = self.state.iteration

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

                    # set seed on restart of data iterator
                    self.setup_seed()
                    self._dataloader_iter = iter(self.state.dataloader)

                    should_exit = True

                    continue

                self.state.batch = batch
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, self.state.batch)
                self._fire_event(Events.ITERATION_COMPLETED)

                # TODO: remove refs on batch to avoid high mem consumption ? -> need verification
                # self.state.batch = batch = None

                if self.should_terminate or self.should_terminate_single_epoch:
                    self.should_terminate_single_epoch = False
                    self._manual_seed(self.state.seed, self.state.iteration // iter_counter)
                    self._dataloader_iter = iter(self.state.dataloader)
                    break

                if iter_counter == self.state.epoch_length:
                    break

        except BaseException as e:
            self.logger.error("Current run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)

        return hours, mins, secs

    def _handle_exception(self, e: Exception) -> None:
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    def state_dict(self) -> OrderedDict:
        """Returns a dictionary containing engine's state: "seed", "epoch_length", "max_epochs" and "iteration"

        Returns:
            dict:
                a dictionary containing engine's state

        """
        if self.state is None:
            return OrderedDict()
        keys = self._state_dict_all_req_keys + (self._state_dict_one_of_opt_keys[0],)
        return OrderedDict([(k, getattr(self.state, k)) for k in keys])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Setups engine from `state_dict`.

        State dictionary should contain keys: `iteration` or `epoch` and `max_epochs`, `epoch_length` and
        `seed`. Iteration and epoch values are 0-based: the first iteration or epoch is zero.

        Args:
            state_dict (Mapping): a dict with parameters


        .. code-block:: python

            # Restore from an epoch
            state_dict = {"seed": 0, "epoch": 3, "max_epochs": 100, "epoch_length": len(data_loader)}
            # or an iteration
            # state_dict = {"seed": 0, "iteration": 500, "max_epochs": 100, "epoch_length": len(data_loader)}

            trainer = Engine(...)
            trainer.load_state_dict(state_dict)
            trainer.run(data)

        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("Argument state_dict should be a dictionary, but given {}".format(type(state_dict)))

        for k in self._state_dict_all_req_keys:
            if k not in state_dict:
                raise ValueError(
                    "Required state attribute '{}' is absent in provided state_dict '{}'".format(k, state_dict.keys())
                )

        opts = [k in state_dict for k in self._state_dict_one_of_opt_keys]
        if (not any(opts)) or (all(opts)):
            raise ValueError("state_dict should contain only one of '{}' keys".format(self._state_dict_one_of_opt_keys))

        self.state = State(
            seed=state_dict["seed"],
            max_epochs=state_dict["max_epochs"],
            epoch_length=state_dict["epoch_length"],
            metrics={},
        )

        if "iteration" in state_dict:
            self.state.iteration = state_dict["iteration"]
            self.state.epoch = self.state.iteration // self.state.epoch_length
        elif "epoch" in state_dict:
            self.state.epoch = state_dict["epoch"]
            self.state.iteration = self.state.epoch_length * self.state.epoch

    @staticmethod
    def _is_done(state: State) -> bool:
        return state.iteration == state.epoch_length * state.max_epochs

    def run(
        self,
        data: Iterable,
        max_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> State:
        """Runs the `process_function` over the passed data.

        Engine has a state and the following logic is applied in this function:

        - At the first call, new state is defined by `max_epochs`, `epoch_length`, `seed` if provided.
        - If state is already defined such that there are iterations to run until `max_epochs` and no input arguments
            provided, state is kept and used in the function.
        - If state is defined and engine is "done" (no iterations to run until `max_epochs`), a new state is defined.
        - If state is defined, engine is NOT "done", then input arguments if provided override defined state.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).
            max_epochs (int, optional): Max epochs to run for (default: None).
                If a new state should be created (first run or run again from ended engine), it's default value is 1.
                This argument should be `None` if run is resuming from a state.
            epoch_length (int, optional): Number of iterations to count as one epoch. By default, it can be set as
                `len(data)`. If `data` is an iterator and `epoch_length` is not set, an error is raised.
                This argument should be `None` if run is resuming from a state.
            seed (int, optional): Seed to use for dataflow consistency, by default it
                will respect the global random state. This argument should be `None` if run is resuming from a state.

        Returns:
            State: output state.

        Note:
            User can dynamically preprocess input batch at :attr:`~ignite.engine.Events.ITERATION_STARTED` and store
            output batch in `engine.state.batch`. Latter is passed as usually to `process_function` as argument:

            .. code-block:: python

                trainer = ...

                @trainer.on(Events.ITERATION_STARTED)
                def switch_batch(engine):
                    engine.state.batch = preprocess_batch(engine.state.batch)

        Note:
            In order to perform a reproducible run, if input `data` is `torch.utils.data.DataLoader`, its batch sampler
            is replaced by a batch sampler (:class:`~ignite.engine.engine.ReproducibleBatchSampler`) such that random
            sampling indices are reproducible by prefetching them before data iteration.

        """

        if self.state is None or self._is_done(self.state):
            # Create new state
            if max_epochs is None:
                max_epochs = 1
            if seed is None:
                seed = torch.randint(0, int(1e9), (1,)).item()
            if epoch_length is None:
                if hasattr(data, "__len__"):
                    epoch_length = len(data)
                    if epoch_length < 1:
                        raise ValueError("Input data has zero size. Please provide non-empty data")
                else:
                    raise ValueError("Argument `epoch_length` should be defined if `data` is an iterator")
            self.state = State(seed=seed, iteration=0, epoch=0, max_epochs=max_epochs, epoch_length=epoch_length)
            self.logger.info("Engine run starting with max_epochs={}.".format(max_epochs))
        else:
            # Keep actual state and override it if input args provided
            if max_epochs is not None:
                self.state.max_epochs = max_epochs
            if seed is not None:
                self.state.seed = seed
            if epoch_length is not None:
                self.state.epoch_length = epoch_length
            self.logger.info(
                "Engine run resuming from iteration {}, epoch {} until {} epochs".format(
                    self.state.iteration, self.state.epoch, self.state.max_epochs
                )
            )

        self.state.dataloader = data
        return self._internal_run()

    def _setup_engine(self) -> None:

        try:
            self._dataloader_len = len(self.state.dataloader) if hasattr(self.state.dataloader, "__len__") else None
        except TypeError:
            # _InfiniteConstantSampler can raise a TypeError on DataLoader length of a IterableDataset
            self._dataloader_len = None

        # setup seed here, as iter(data) can start prefetching
        self.setup_seed()

        # if input data is torch dataloader we replace batch sampler by a batch sampler
        # such that its random sampling indices are reproducible by prefetching them before data iteration
        if isinstance(self.state.dataloader, torch.utils.data.DataLoader):
            _dataloader_kind = self.state.dataloader._dataset_kind
            if _dataloader_kind == torch.utils.data.dataloader._DatasetKind.Map:
                if (self._dataloader_len is not None) and hasattr(self.state.dataloader.sampler, "epoch"):
                    if self._dataloader_len != self.state.epoch_length:
                        warnings.warn(
                            "When defined engine's epoch length is different of input dataloader length, "
                            "distributed sampler indices can not be setup in a reproducible manner"
                        )

                batch_sampler = self.state.dataloader.batch_sampler
                if not isinstance(batch_sampler, ReproducibleBatchSampler):
                    self.state.dataloader = _update_dataloader(
                        self.state.dataloader, ReproducibleBatchSampler(batch_sampler)
                    )

        iteration = self.state.iteration
        self._dataloader_iter = self._from_iteration(self.state.dataloader, iteration)

        # Below we define initial counter value for _run_once_on_dataset to measure a single epoch
        if self.state.epoch_length is not None:
            iteration %= self.state.epoch_length
        self._init_iter.append(iteration)

    @staticmethod
    def _from_iteration(data: Union[Iterable, torch.utils.data.DataLoader], iteration: int) -> Iterator:
        if isinstance(data, torch.utils.data.DataLoader):
            try:
                # following is unsafe for IterableDatasets
                iteration %= len(data.batch_sampler)
                if iteration > 0:
                    # batch sampler is ReproducibleBatchSampler
                    data.batch_sampler.start_iteration = iteration
            except TypeError:
                # Probably we can do nothing with DataLoader built upon IterableDatasets
                pass
            data_iter = iter(data)
        else:
            if hasattr(data, "__len__"):
                iteration %= len(data)
            data_iter = iter(data)
            counter = 0
            while counter < iteration:
                try:
                    next(data_iter)
                    counter += 1
                except StopIteration:
                    data_iter = iter(data)

        return data_iter

    @staticmethod
    def _manual_seed(seed: int, epoch: int) -> None:
        random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        try:
            import numpy as np

            np.random.seed(seed + epoch)
        except ImportError:
            pass

    def setup_seed(self) -> None:
        # seed value should be related to input data iterator length -> iteration at data iterator restart
        # - seed can not be epoch because during a single epoch we can have multiple `_dataloader_len`
        # - seed can not be iteration because when resuming from iteration we need to set the seed from the start of the
        #   dataloader and then rewind to required iteration
        le = self._dataloader_len if self._dataloader_len is not None else 1
        self._manual_seed(self.state.seed, self.state.iteration // le)

    def _internal_run(self) -> State:
        self.should_terminate = self.should_terminate_single_epoch = False
        try:
            start_time = time.time()
            self._fire_event(Events.STARTED)
            while self.state.epoch < self.state.max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(Events.EPOCH_STARTED)

                if self._dataloader_iter is None:
                    self._setup_engine()

                hours, mins, secs = self._run_once_on_dataset()

                self.logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED)

            self._fire_event(Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self.logger.info("Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._dataloader_iter = self._dataloader_len = None
            self.logger.error("Engine run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        self._dataloader_iter = self._dataloader_len = None
        return self.state

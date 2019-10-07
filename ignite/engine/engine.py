import inspect
import logging
import sys
import time
from collections import defaultdict
from enum import Enum
import weakref
import numbers
import random

import torch


from ignite._utils import _to_hours_mins_secs

IS_PYTHON2 = sys.version_info[0] < 3


class Events(Enum):
    """Events that are fired by the :class:`~ignite.engine.Engine` during execution."""
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State(object):
    """An object that is used to pass internal and user-defined state between event handlers. By default, state
    contains the following attributes:

    .. code-block:: python

        state.iteration         # 1-based, the first iteration is 1
        state.epoch             # 1-based, the first epoch is 1
        state.seed              # seed to set at each epoch
        state.dataloader        # data passed to engine
        state.epoch_length      # optional length of an epoch
        state.batch             # batch passed to `process_function`
        state.output            # output of `process_function` after a single iteration
        state.metrics           # dictionary with defined metrics if any

    """

    event_to_attr = {
        Events.ITERATION_STARTED: "iteration",
        Events.ITERATION_COMPLETED: "iteration",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.COMPLETED: "epoch"
    }

    def __init__(self, **kwargs):
        self.output = None
        self.batch = None
        self.iteration = 0
        self.epoch = 0
        self.epoch_length = None

        for k, v in kwargs.items():
            setattr(self, k, v)

        for value in self.event_to_attr.values():
            if not hasattr(self, value):
                setattr(self, value, 0)

    def get_event_attrib_value(self, event_name):
        if event_name not in State.event_to_attr:
            raise RuntimeError("Unknown event name '{}'".format(event_name))
        return getattr(self, State.event_to_attr[event_name])

    def __repr__(self):
        s = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str)):
                value = type(value)
            s += "\t{}: {}\n".format(attr, value)
        return s


class RemovableEventHandle(object):
    """A weakref handle to remove a registered event.

    A handle that may be used to remove a registered event handler via the
    remove method, with-statement, or context manager protocol. Returned from
    :meth:`~ignite.engine.Engine.add_event_handler`.


    Args:
        event_name: Registered event name.
        handler: Registered event handler, stored as weakref.
        engine: Target engine, stored as weakref.

    Example usage:

    .. code-block:: python

        engine = Engine()

        def print_epoch(engine):
            print("Epoch: {}".format(engine.state.epoch))

        with engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch):
            # print_epoch handler registered for a single run
            engine.run(data)

        # print_epoch handler is now unregistered
    """

    def __init__(self, event_name, handler, engine):
        self.event_name = event_name
        self.handler = weakref.ref(handler)
        self.engine = weakref.ref(engine)

    def remove(self):
        """Remove handler from engine."""
        handler = self.handler()
        engine = self.engine()

        if handler is None or engine is None:
            return

        if engine.has_event_handler(handler, self.event_name):
            engine.remove_event_handler(handler, self.event_name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()


class Engine(object):
    """Runs a given `process_function` over each batch of a dataset, emitting events as it goes.

    Args:
        process_function (callable): A function receiving a handle to the engine and the current batch
            in each iteration, and returns data to be stored in the engine's state.

    Example usage:

    .. code-block:: python

        def train_and_store_loss(engine, batch):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            return loss.item()

        engine = Engine(train_and_store_loss)
        engine.run(data_loader)

        # Loss value is now stored in `engine.state.output`.

    """
    def __init__(self, process_function):
        self._event_handlers = defaultdict(list)
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self._process_function = process_function
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = None
        self._allowed_events = []

        self._dataloader_iter = None
        self._init_iter = []

        self.register_events(*Events)

        if self._process_function is None:
            raise ValueError("Engine must be given a processing function in order to run.")

        self._check_signature(process_function, 'process_function', None)

    def register_events(self, *event_names, **kwargs):
        """Add events that can be fired.

        Registering an event will let the user fire these events at any point.
        This opens the door to make the :meth:`~ignite.engine.Engine.run` loop even more
        configurable.

        By default, the events from :class:`~ignite.engine.Events` are registered.

        Args:
            *event_names: An object (ideally a string or int) to define the
                name of the event being supported.
            event_to_attr (dict): A dictionary to map an event to a state attribute.

        Example usage:

        .. code-block:: python

            from enum import Enum

            class Custom_Events(Enum):
                FOO_EVENT = "foo_event"
                BAR_EVENT = "bar_event"

            engine = Engine(process_function)
            engine.register_events(*Custom_Events)


        Example with State Attribute:

        .. code-block:: python

            from enum import Enum

            class TBPTT_Events(Enum):
                TIME_ITERATION_STARTED = "time_iteration_started"
                TIME_ITERATION_COMPLETED = "time_iteration_completed"

            TBPTT_event_to_attr = {TBPTT_Events.TIME_ITERATION_STARTED: 'time_iteration',
                                   TBPTT_Events.TIME_ITERATION_COMPLETED: 'time_iteration'}

            engine = Engine(process_function)
            engine.register_events(*TBPTT_Events, event_to_attr=TBPTT_event_to_attr)
            engine.run(data)
            # engine.state contains an attribute time_iteration, which can be accessed using engine.state.time_iteration
        """
        event_to_attr = kwargs.get('event_to_attr', None)
        if event_to_attr:
            if not isinstance(event_to_attr, dict):
                raise ValueError('Expected event_to_attr to be dictionary. Got {}.'.format(type(event_to_attr)))

        for name in event_names:
            self._allowed_events.append(name)
            if event_to_attr:
                State.event_to_attr[name] = event_to_attr[name]

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events`
                or any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            handler (callable): the callable event handler that should be invoked
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        Note:
              The handler function's first argument will be `self`, the :class:`~ignite.engine.Engine` object it
              was bound to.

              Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
              passed here, for example during :attr:`~ignite.engine.Events.EXCEPTION_RAISED`.

        Returns:
            :class:`~ignite.engine.RemovableEventHandler`, which can be used to remove the handler.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print("Epoch: {}".format(engine.state.epoch))

            engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

        """
        if event_name not in self._allowed_events:
            self._logger.error("attempt to add event handler to an invalid event %s.", event_name)
            raise ValueError("Event {} is not a valid event for this Engine.".format(event_name))

        event_args = (Exception(), ) if event_name == Events.EXCEPTION_RAISED else ()
        self._check_signature(handler, 'handler', *(event_args + args), **kwargs)

        self._event_handlers[event_name].append((handler, args, kwargs))
        self._logger.debug("added handler for event %s.", event_name)

        return RemovableEventHandle(event_name, handler, self)

    def has_event_handler(self, handler, event_name=None):
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
                if h == handler:
                    return True
        return False

    def remove_event_handler(self, handler, event_name):
        """Remove event handler `handler` from registered handlers of the engine

        Args:
            handler (callable): the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError("Input event name '{}' does not exist".format(event_name))

        new_event_handlers = [(h, args, kwargs) for h, args, kwargs in self._event_handlers[event_name]
                              if h != handler]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError("Input handler '{}' is not found among registered event handlers".format(handler))
        self._event_handlers[event_name] = new_event_handlers

    def _check_signature(self, fn, fn_description, *args, **kwargs):
        exception_msg = None

        if IS_PYTHON2:
            try:
                callable_ = fn if hasattr(fn, '__name__') else fn.__call__
                inspect.getcallargs(callable_, self, *args, **kwargs)
            except TypeError as exc:
                spec = inspect.getargspec(callable_)
                fn_params = list(spec.args)
                exception_msg = str(exc)
        else:
            signature = inspect.signature(fn)
            try:
                signature.bind(self, *args, **kwargs)
            except TypeError as exc:
                fn_params = list(signature.parameters)
                exception_msg = str(exc)

        if exception_msg:
            passed_params = [self] + list(args) + list(kwargs)
            raise ValueError("Error adding {} '{}': "
                             "takes parameters {} but will be called with {} "
                             "({}).".format(fn, fn_description, fn_params, passed_params, exception_msg))

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.Events` or
                any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.

        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _fire_event(self, event_name, *event_args, **event_kwargs):
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        aguments updates arguments passed using :meth:`~ignite.engine.Engine.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        if event_name in self._allowed_events:
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                kwargs.update(event_kwargs)
                func(self, *(event_args + args), **kwargs)

    def fire_event(self, event_name):
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

    def terminate(self):
        """Sends terminate signal to the engine, so that it terminates completely the run after the current iteration.
        """
        self._logger.info("Terminate signaled. Engine will stop after current iteration is finished.")
        self.should_terminate = True

    def terminate_epoch(self):
        """Sends terminate signal to the engine, so that it terminates the current epoch after the current iteration.
        """
        self._logger.info("Terminate current epoch is signaled. "
                          "Current epoch iteration will stop after current iteration is finished.")
        self.should_terminate_single_epoch = True

    def _run_once_on_dataset(self):
        start_time = time.time()

        has_epoch_length = self.state.epoch_length is not None
        # We need to setup iter_counter > 0 if we resume from an iteration
        iter_counter = self._init_iter.pop() if len(self._init_iter) > 0 else 0
        should_exit = False
        try:
            while True:
                try:
                    batch = next(self._dataloader_iter)
                    iter_counter += 1
                    should_exit = False
                except StopIteration:
                    self._dataloader_iter = iter(self.state.dataloader)
                    # Define self.state.epoch_length if it is not yet set
                    if self.state.epoch_length is None:
                        self.state.epoch_length = self.state.iteration

                    # Should exit while loop if we can not iterate
                    if should_exit:
                        break

                    should_exit = True

                    continue

                self.state.batch = batch
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)
                self.state.output = self._process_function(self, self.state.batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                if self.should_terminate or self.should_terminate_single_epoch:
                    self.should_terminate_single_epoch = False
                    self._dataloader_iter = iter(self.state.dataloader)
                    break

                if has_epoch_length and iter_counter == self.state.epoch_length:
                    break

        except BaseException as e:
            self._logger.error("Current run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)

        return hours, mins, secs

    def _handle_exception(self, e):
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    def state_dict(self):
        if self.state is None:
            return {}
        return {
            "iteration": self.state.iteration,
            "epoch": self.state.epoch,
            "epoch_length": self.state.epoch_length,
            "max_epochs": self.state.max_epochs,
            "seed": self.state.seed,
        }

    def run(self, data, max_epochs=1, epoch_length=None, seed=12):
        """Runs the `process_function` over the passed data.

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).
            max_epochs (int, optional): Max epochs to run for (default: 1).
            epoch_length (int, optional): Number of iterations to count as one epoch. By default, at the first place
                epoch_length is tried to be set as `len(data)`. If previous command is failed, engine measures
                automatically the length by iterating over the data until `StopIteration` is raised.
            seed (int, optional): Seed to setup at each epoch for reproducible runs.

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

        """
        if epoch_length is None and hasattr(data, "__len__"):
            epoch_length = len(data)
            if epoch_length < 1:
                raise ValueError("Input data has zero size. Please provide non-empty data")

        self.state = State(iteration=0,
                           epoch=0,
                           dataloader=data,
                           max_epochs=max_epochs,
                           metrics={},
                           epoch_length=epoch_length,
                           seed=seed)
        # setup seed here, as iter(data) can start prefetching
        self._manual_seed(self.state.seed, self.state.epoch)
        self._dataloader_iter = iter(data)
        self._logger.info("Engine run starting with max_epochs={}.".format(max_epochs))
        return self._internal_run()

    def resume(self, data, state_dict, strict=True):
        """Resume engine run from a checkpoint

        Args:
            data (Iterable): Collection of batches allowing repeated iteration (e.g., list or `DataLoader`).
            state_dict (Mapping): A dictionary with keys: iteration, epoch, max_epochs, epoch_length, seed.
                See notes below for more details.
            strict (bool, optional): Flag to indicate whether user needs a strict data resuming. If
                True, data is iterated without running `process_function` or executing handlers until `state.iteration`.
                This phase can take time depending on the data. If type of data is `torch.utils.DataLoader`,
                its batch sampler is used to iterate over data and accelerate resuming. If False, data is used from
                the beginning.

        Returns:
            State: output state.

        Note:
            State dictionary should contain keys: `iteration` or `epoch` and `max_epochs`, optionally `epoch_length`,
            `seed`.
            Iteration, epoch values are 0-based: the first iteration or epoch is zero.


            ~~If specified, the value of `iteration` determines starting
            iteration, e.g. if `iteration=5` then engine skips 4 previous iterations and starts from 5th
            (1-based) iteration. Similarly, if specified, the value of `epoch` should determine the epoch to
            start engine from. Epoch value is also 1-based.~~

        """
        # TODO: bad API and we need to decide what is self.state and how it is managed by Engine
        # - resuming state contains only serializable info
        # - self.state is a mix of serializable info + other stuff
        # - we need to provide Engine.state_dict() = serializable info <--> it is not self.state ?
        # - we would like to keep Engine "stateless" for each run ?

        # In any case, we have requirements:
        # 1) object to pass between handlers with running info, minimal data and other custom stuff
        #    -> We need to record somewhere running info: iteration, epoch, max_epochs, epoch_length
        #    -> Engine.state with a scope of Engine.run and for ease of using Engine.state is kept until next Engine.run
        # 2) Engine attributes should_terminate, should_terminate_single_epoch should not be at Engine.state ?

        self._setup_state(state_dict, data, strict)
        self._logger.info("Engine run resuming from epoch {} and iteration {} with max_epochs={}"
                          .format(self.state.epoch, self.state.iteration, self.state.max_epochs))
        return self._internal_run()

    def _setup_state(self, state_dict, data, strict):

        for req_field in ["max_epochs", "seed"]:
            if req_field not in state_dict:
                raise ValueError("state_dict should contain '{}' key".format(req_field))

        opt_fields = ("iteration", "epoch")
        opts = [opt_field in state_dict for opt_field in opt_fields]
        if not any(opts):
            raise ValueError("state_dict should contain one of '{}' keys".format(opt_fields))

        if all(opts):
            raise ValueError("state_dict should contain only on of '{}' keys".format(opt_fields))

        self.state = State(seed=state_dict['seed'],
                           max_epochs=state_dict['max_epochs'],
                           epoch_length=state_dict.get('epoch_length', None),
                           metrics={})
        if self.state.epoch_length is None and hasattr(data, "__len__"):
            self.state.epoch_length = len(data)
            if self.state.epoch_length < 1:
                raise ValueError("Input data has zero size. Please provide non-empty data")

        if "iteration" in state_dict:
            self.state.iteration = state_dict['iteration']
            if self.state.epoch_length is None:
                self.state.epoch = 0
            else:
                self.state.epoch = self.state.iteration // self.state.epoch_length
        elif "epoch" in state_dict:
            # As we would like to start from `state.epoch` epoch, we need to subtract 1
            if self.state.epoch_length is None:
                raise ValueError("When start epoch is specified, state.epoch={}, "
                                 "resuming state should have epoch_length value, but state.epoch_length={}"
                                 .format(self.state.epoch, self.state.epoch_length))
            self.state.epoch = state_dict['epoch']
            self.state.iteration = self.state.epoch_length * self.state.epoch

        self.state.dataloader = data

        iteration = self.state.iteration
        if self.state.epoch_length is not None:
            iteration %= self.state.epoch_length

        # setup seed here, as iter(data) can start prefetching
        self._manual_seed(self.state.seed, self.state.epoch)
        if not strict:
            self._dataloader_iter = iter(data)
        else:
            try:
                self._dataloader_iter = self._from_iteration(data, iteration)
            except StopIteration:
                raise ValueError("Specified resume iteration {} is larger than the size of the input data"
                                 .format(iteration))

        self._init_iter.append(iteration)

    @staticmethod
    def _from_iteration(data, iteration):
        if not isinstance(data, torch.utils.data.DataLoader):
            data_iter = iter(data)
            counter = 0
            while counter < iteration:
                next(data_iter)
                counter += 1
        else:
            if iteration % len(data) > 0:
                # patch batch sampler
                if not isinstance(data.batch_sampler, _RewindableProxyBatchSampler):
                    data._original_batch_sampler = data.batch_sampler
                    data.batch_sampler = _RewindableProxyBatchSampler(data.batch_sampler)
                counter = 0
                while counter < iteration:
                    next(data.batch_sampler.batch_sampler_iter)
                    counter += 1

            data_iter = iter(data)

        return data_iter

    @staticmethod
    def _manual_seed(seed, epoch):
        random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        try:
            import numpy as np
            np.random.seed(seed + epoch)
        except ImportError:
            pass

    def _internal_run(self):
        self.should_terminate = self.should_terminate_single_epoch = False
        try:
            start_time = time.time()
            self._fire_event(Events.STARTED)
            while self.state.epoch < self.state.max_epochs and not self.should_terminate:
                self.state.epoch += 1
                self._fire_event(Events.EPOCH_STARTED)

                hours, mins, secs = self._run_once_on_dataset()

                self._logger.info("Epoch[%s] Complete. Time taken: %02d:%02d:%02d", self.state.epoch, hours, mins, secs)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED)
                # We set manual seed in the end of the loop as the first time it is called in run()/resume()
                self._manual_seed(self.state.seed, self.state.epoch)

            self._fire_event(Events.COMPLETED)
            time_taken = time.time() - start_time
            hours, mins, secs = _to_hours_mins_secs(time_taken)
            self._logger.info("Engine run complete. Time taken %02d:%02d:%02d" % (hours, mins, secs))

        except BaseException as e:
            self._logger.error("Engine run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

        if hasattr(self.state.dataloader, "_original_batch_sampler"):
            self.state.dataloader.batch_sampler = self.state.dataloader._original_batch_sampler
            del self.state.dataloader._original_batch_sampler

        return self.state


class _RewindableProxyBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Rewindable proxy batch sampler

    Args:
        batch_sampler: batch sampler same as used with torch.utils.data.DataLoader
    """
    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        self.batch_sampler_iter = iter(self.batch_sampler)

    def __iter__(self):
        for batch in self.batch_sampler_iter:
            yield batch
        self.batch_sampler_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

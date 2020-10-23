from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Union

import torch

from ignite.engine import Engine, Events
from ignite.handlers import Timer


class BasicTimeProfiler:
    """
    BasicTimeProfiler can be used to profile the handlers,
    events, data loading and data processing times.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers import BasicTimeProfiler

        trainer = Engine(train_updater)

        # Create an object of the profiler and attach an engine to it
        profiler = BasicTimeProfiler()
        profiler.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_intermediate_results():
            profiler.print_results(profiler.get_results())

        trainer.run(dataloader, max_epochs=3)

        profiler.write_results('path_to_dir/time_profiling.csv')

    """

    events_to_ignore = [
        Events.EXCEPTION_RAISED,
        Events.TERMINATE,
        Events.TERMINATE_SINGLE_EPOCH,
        Events.DATALOADER_STOP_ITERATION,
    ]

    def __init__(self):
        self._dataflow_timer = Timer()
        self._processing_timer = Timer()
        self._event_handlers_timer = Timer()

        self.dataflow_times = None
        self.processing_times = None
        self.event_handlers_times = None

        self._events = [
            Events.EPOCH_STARTED,
            Events.EPOCH_COMPLETED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
            Events.COMPLETED,
        ]
        self._fmethods = [
            self._as_first_epoch_started,
            self._as_first_epoch_completed,
            self._as_first_iter_started,
            self._as_first_iter_completed,
            self._as_first_get_batch_started,
            self._as_first_get_batch_completed,
            self._as_first_completed,
        ]
        self._lmethods = [
            self._as_last_epoch_started,
            self._as_last_epoch_completed,
            self._as_last_iter_started,
            self._as_last_iter_completed,
            self._as_last_get_batch_started,
            self._as_last_get_batch_completed,
            self._as_last_completed,
        ]

    def _reset(self, num_epochs: int, total_num_iters: int):
        self.dataflow_times = torch.zeros(total_num_iters)
        self.processing_times = torch.zeros(total_num_iters)
        self.event_handlers_times = {
            Events.STARTED: torch.zeros(1),
            Events.COMPLETED: torch.zeros(1),
            Events.EPOCH_STARTED: torch.zeros(num_epochs),
            Events.EPOCH_COMPLETED: torch.zeros(num_epochs),
            Events.ITERATION_STARTED: torch.zeros(total_num_iters),
            Events.ITERATION_COMPLETED: torch.zeros(total_num_iters),
            Events.GET_BATCH_COMPLETED: torch.zeros(total_num_iters),
            Events.GET_BATCH_STARTED: torch.zeros(total_num_iters),
        }

    def _as_first_started(self, engine: Engine):
        if hasattr(engine.state.dataloader, "__len__"):
            num_iters_per_epoch = len(engine.state.dataloader)
        else:
            num_iters_per_epoch = engine.state.epoch_length

        self.max_epochs = engine.state.max_epochs
        self.total_num_iters = self.max_epochs * num_iters_per_epoch
        self._reset(self.max_epochs, self.total_num_iters)

        self.event_handlers_names = {
            e: [
                h.__qualname__ if hasattr(h, "__qualname__") else h.__class__.__name__
                for (h, _, _) in engine._event_handlers[e]
                if "BasicTimeProfiler." not in repr(h)  # avoid adding internal handlers into output
            ]
            for e in Events
            if e not in self.events_to_ignore
        }

        # Setup all other handlers:
        engine._event_handlers[Events.STARTED].append((self._as_last_started, (engine,), {}))

        for e, m in zip(self._events, self._fmethods):
            engine._event_handlers[e].insert(0, (m, (engine,), {}))

        for e, m in zip(self._events, self._lmethods):
            engine._event_handlers[e].append((m, (engine,), {}))

        # Let's go
        self._event_handlers_timer.reset()

    def _as_last_started(self, engine: Engine):
        self.event_handlers_times[Events.STARTED][0] = self._event_handlers_timer.value()

    def _as_first_epoch_started(self, engine: Engine):
        self._event_handlers_timer.reset()

    def _as_last_epoch_started(self, engine: Engine):
        t = self._event_handlers_timer.value()
        e = engine.state.epoch - 1
        self.event_handlers_times[Events.EPOCH_STARTED][e] = t

    def _as_first_get_batch_started(self, engine: Engine):
        self._event_handlers_timer.reset()
        self._dataflow_timer.reset()

    def _as_last_get_batch_started(self, engine: Engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.GET_BATCH_STARTED][i] = t

    def _as_first_get_batch_completed(self, engine: Engine):
        self._event_handlers_timer.reset()

    def _as_last_get_batch_completed(self, engine: Engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.GET_BATCH_COMPLETED][i] = t

        d = self._dataflow_timer.value()
        self.dataflow_times[i] = d

        self._dataflow_timer.reset()

    def _as_first_iter_started(self, engine: Engine):
        self._event_handlers_timer.reset()

    def _as_last_iter_started(self, engine: Engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.ITERATION_STARTED][i] = t

        self._processing_timer.reset()

    def _as_first_iter_completed(self, engine: Engine):
        t = self._processing_timer.value()
        i = engine.state.iteration - 1
        self.processing_times[i] = t

        self._event_handlers_timer.reset()

    def _as_last_iter_completed(self, engine: Engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.ITERATION_COMPLETED][i] = t

    def _as_first_epoch_completed(self, engine: Engine):
        self._event_handlers_timer.reset()

    def _as_last_epoch_completed(self, engine: Engine):
        t = self._event_handlers_timer.value()
        e = engine.state.epoch - 1
        self.event_handlers_times[Events.EPOCH_COMPLETED][e] = t

    def _as_first_completed(self, engine: Engine):
        self._event_handlers_timer.reset()

    def _as_last_completed(self, engine: Engine):
        self.event_handlers_times[Events.COMPLETED][0] = self._event_handlers_timer.value()

        # Remove added handlers:
        engine.remove_event_handler(self._as_last_started, Events.STARTED)

        for e, m in zip(self._events, self._fmethods):
            engine.remove_event_handler(m, e)

        for e, m in zip(self._events, self._lmethods):
            engine.remove_event_handler(m, e)

    def attach(self, engine: Engine):
        if not isinstance(engine, Engine):
            raise TypeError("Argument engine should be ignite.engine.Engine, " "but given {}".format(type(engine)))

        if not engine.has_event_handler(self._as_first_started):
            engine._event_handlers[Events.STARTED].insert(0, (self._as_first_started, (engine,), {}))

    @staticmethod
    def _compute_basic_stats(data: Sequence):
        # compute on non-zero data:
        data = data[data > 0]
        out = [("total", torch.sum(data).item() if len(data) > 0 else "not yet triggered")]
        if len(data) > 1:
            out += [
                ("min/index", (torch.min(data).item(), torch.argmin(data).item())),
                ("max/index", (torch.max(data).item(), torch.argmax(data).item())),
                ("mean", torch.mean(data).item()),
                ("std", torch.std(data).item()),
            ]
        return OrderedDict(out)

    def get_results(self):
        """
        Method to fetch the aggregated profiler results after the engine is run

        .. code-block:: python

            results = profiler.get_results()

        """
        total_eh_time = sum([(self.event_handlers_times[e]).sum() for e in Events if e not in self.events_to_ignore])

        return OrderedDict(
            [
                ("processing_stats", self._compute_basic_stats(self.processing_times)),
                ("dataflow_stats", self._compute_basic_stats(self.dataflow_times)),
                (
                    "event_handlers_stats",
                    dict(
                        [
                            (str(e.name).replace(".", "_"), self._compute_basic_stats(self.event_handlers_times[e]))
                            for e in Events
                            if e not in self.events_to_ignore
                        ]
                        + [("total_time", total_eh_time)]
                    ),
                ),
                (
                    "event_handlers_names",
                    {str(e.name).replace(".", "_") + "_names": v for e, v in self.event_handlers_names.items()},
                ),
            ]
        )

    def write_results(self, output_path: str):
        """
        Method to store the unaggregated profiling results to a csv file

        .. code-block:: python

            profiler.write_results('path_to_dir/awesome_filename.csv')

        Example output:

        .. code-block:: text

            -----------------------------------------------------------------
            epoch iteration processing_stats dataflow_stats Event_STARTED ...
            1.0     1.0        0.00003         0.252387        0.125676
            1.0     2.0        0.00029         0.252342        0.125123

        """
        try:
            import pandas as pd
        except ImportError:
            print("Need pandas to write results as files")
            return

        iters_per_epoch = self.total_num_iters // self.max_epochs

        epochs = torch.arange(self.max_epochs, dtype=torch.float32).repeat_interleave(iters_per_epoch) + 1
        iterations = torch.arange(self.total_num_iters, dtype=torch.float32) + 1
        processing_stats = self.processing_times
        dataflow_stats = self.dataflow_times

        event_started = self.event_handlers_times[Events.STARTED].repeat_interleave(self.total_num_iters)
        event_completed = self.event_handlers_times[Events.COMPLETED].repeat_interleave(self.total_num_iters)
        event_epoch_started = self.event_handlers_times[Events.EPOCH_STARTED].repeat_interleave(iters_per_epoch)
        event_epoch_completed = self.event_handlers_times[Events.EPOCH_COMPLETED].repeat_interleave(iters_per_epoch)

        event_iter_started = self.event_handlers_times[Events.ITERATION_STARTED]
        event_iter_completed = self.event_handlers_times[Events.ITERATION_COMPLETED]
        event_batch_started = self.event_handlers_times[Events.GET_BATCH_STARTED]
        event_batch_completed = self.event_handlers_times[Events.GET_BATCH_COMPLETED]

        results_dump = torch.stack(
            [
                epochs,
                iterations,
                processing_stats,
                dataflow_stats,
                event_started,
                event_completed,
                event_epoch_started,
                event_epoch_completed,
                event_iter_started,
                event_iter_completed,
                event_batch_started,
                event_batch_completed,
            ],
            dim=1,
        ).numpy()

        results_df = pd.DataFrame(
            data=results_dump,
            columns=[
                "epoch",
                "iteration",
                "processing_stats",
                "dataflow_stats",
                "Event_STARTED",
                "Event_COMPLETED",
                "Event_EPOCH_STARTED",
                "Event_EPOCH_COMPLETED",
                "Event_ITERATION_STARTED",
                "Event_ITERATION_COMPLETED",
                "Event_GET_BATCH_STARTED",
                "Event_GET_BATCH_COMPLETED",
            ],
        )
        results_df.to_csv(output_path, index=False)

    @staticmethod
    def print_results(results: Union[Dict, Iterable]):
        """
        Method to print the aggregated results from the profiler

        .. code-block:: python

            profiler.print_results(results)

        Example output:

        .. code-block:: text

             ----------------------------------------------------
            | Time profiling stats (in seconds):                 |
             ----------------------------------------------------
            total  |  min/index  |  max/index  |  mean  |  std

            Processing function:
            157.46292 | 0.01452/1501 | 0.26905/0 | 0.07730 | 0.01258

            Dataflow:
            6.11384 | 0.00008/1935 | 0.28461/1551 | 0.00300 | 0.02693

            Event handlers:
            2.82721

            - Events.STARTED: []
            0.00000

            - Events.EPOCH_STARTED: []
            0.00006 | 0.00000/0 | 0.00000/17 | 0.00000 | 0.00000

            - Events.ITERATION_STARTED: ['PiecewiseLinear']
            0.03482 | 0.00001/188 | 0.00018/679 | 0.00002 | 0.00001

            - Events.ITERATION_COMPLETED: ['TerminateOnNan']
            0.20037 | 0.00006/866 | 0.00089/1943 | 0.00010 | 0.00003

            - Events.EPOCH_COMPLETED: ['empty_cuda_cache', 'training.<locals>.log_elapsed_time', ]
            2.57860 | 0.11529/0 | 0.14977/13 | 0.12893 | 0.00790

            - Events.COMPLETED: []
            not yet triggered

        """

        def to_str(v: Union[str, tuple]):
            if isinstance(v, str):
                return v
            elif isinstance(v, tuple):
                return "{:.5f}/{}".format(v[0], v[1])
            return "{:.5f}".format(v)

        def odict_to_str(d: Mapping):
            out = " | ".join([to_str(v) for v in d.values()])
            return out

        others = {
            k: odict_to_str(v) if isinstance(v, OrderedDict) else v for k, v in results["event_handlers_stats"].items()
        }

        others.update(results["event_handlers_names"])

        output_message = """
 ----------------------------------------------------
| Time profiling stats (in seconds):                 |
 ----------------------------------------------------
total  |  min/index  |  max/index  |  mean  |  std

Processing function:
{processing_stats}

Dataflow:
{dataflow_stats}

Event handlers:
{total_time:.5f}

- Events.STARTED: {STARTED_names}
{STARTED}

- Events.EPOCH_STARTED: {EPOCH_STARTED_names}
{EPOCH_STARTED}

- Events.ITERATION_STARTED: {ITERATION_STARTED_names}
{ITERATION_STARTED}

- Events.ITERATION_COMPLETED: {ITERATION_COMPLETED_names}
{ITERATION_COMPLETED}

- Events.EPOCH_COMPLETED: {EPOCH_COMPLETED_names}
{EPOCH_COMPLETED}

- Events.COMPLETED: {COMPLETED_names}
{COMPLETED}
""".format(
            processing_stats=odict_to_str(results["processing_stats"]),
            dataflow_stats=odict_to_str(results["dataflow_stats"]),
            **others,
        )
        print(output_message)
        return output_message


class HandlersTimeProfiler:
    """
    HandlersTimeProfiler can be used to profile the handlers,
    data loading and data processing times.

    Examples:

    .. code-block:: python

        from ignite.contrib.handlers import HandlersTimeProfiler

        trainer = Engine(train_updater)

        # Create an object of the profiler and attach an engine to it
        profiler = HandlersTimeProfiler()
        profiler.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_intermediate_results():
            profiler.print_results(profiler.get_results())

        trainer.run(dataloader, max_epochs=3)

        profiler.write_results('path_to_dir/time_profiling.csv')

    """

    def __init__(self):
        self._dataflow_timer = Timer()
        self._processing_timer = Timer()
        self._event_handlers_timer = Timer()

        self.dataflow_times = None
        self.processing_times = None
        self.event_handlers_times = None

        # Events to profile
        self._events = [
            Events.STARTED,
            Events.EPOCH_STARTED,
            Events.EPOCH_COMPLETED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
            Events.COMPLETED,
        ]

        # Events which are triggered only once
        self._global_events = [Events.STARTED, Events.COMPLETED]
        # Events which are triggered on each EPOCH
        self._epoch_events = [Events.EPOCH_STARTED, Events.EPOCH_COMPLETED]
        # Events which are triggered on each ITERATION
        self._iteration_events = [
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
        ]

    @staticmethod
    def _get_callable_name(handler: Callable):
        # get name of the callable handler
        return handler.__qualname__ if hasattr(handler, "__qualname__") else handler.__class__.__name__

    def _timeit_global_events(self, handler: Callable, event: Events):
        # wrapper for profiling handlers attached to global events
        self._event_handlers_timer.reset()
        func, args, kwargs = handler
        func(*args, **kwargs)
        t = self._event_handlers_timer.value()
        hname = self._get_callable_name(func)
        self.event_handlers_times[event][hname][0] = t

    def _timeit_epoch_events(self, handler: Callable, engine: Engine, event: Events, *args, **kwargs):
        # wrapper for profiling handlers attached to epoch events
        self._event_handlers_timer.reset()
        func, args, kwargs = handler
        func(*args, **kwargs)
        e = engine.state.epoch - 1
        t = self._event_handlers_timer.value()
        hname = hname = self._get_callable_name(func)
        self.event_handlers_times[event][hname][e] = t

    def _timeit_iteration_events(self, handler: Callable, engine: Engine, event: Events, *args, **kwargs):
        # wrapper for profiling handlers attached to iteration events
        self._event_handlers_timer.reset()
        func, args, kwargs = handler
        func(*args, **kwargs)
        i = engine.state.iteration - 1
        t = self._event_handlers_timer.value()
        hname = self._get_callable_name(func)
        self.event_handlers_times[event][hname][i] = t

    def _timeit_processing(self, engine: Engine):
        # handler used for profiling processing times
        t = self._processing_timer.value()
        i = engine.state.iteration - 1
        self.processing_times[i] = t

    def _timeit_dataflow(self, engine: Engine):
        # handler used for profiling dataflow times
        t = self._dataflow_timer.value()
        i = engine.state.iteration - 1
        self.dataflow_times[i] = t

    def _reset(self, num_epochs: int, total_num_iters: int):
        # reset the variable used for profiling
        self.dataflow_times = torch.zeros(total_num_iters)
        self.processing_times = torch.zeros(total_num_iters)

        self.event_handlers_times = {
            e: {h: torch.zeros(1) for h in self.event_handlers_names[e]} for e in self._global_events
        }

        self.event_handlers_times.update(
            {e: {h: torch.zeros(num_epochs) for h in self.event_handlers_names[e]} for e in self._epoch_events}
        )

        self.event_handlers_times.update(
            {e: {h: torch.zeros(total_num_iters) for h in self.event_handlers_names[e]} for e in self._iteration_events}
        )

    @staticmethod
    def _is_internal_handler(handler: Callable):
        # checks whether the handler is internal
        return "HandlersTimeProfiler." in repr(handler)

    def _detach_profiler_handlers(self, engine):
        # reverts handlers to original handlers
        engine._event_handlers = self._orig_event_handlers

    def _as_first_started(self, engine):
        if hasattr(engine.state.dataloader, "__len__"):
            num_iters_per_epoch = len(engine.state.dataloader)
        else:
            num_iters_per_epoch = engine.state.epoch_length

        self.max_epochs = engine.state.max_epochs
        self.total_num_iters = self.max_epochs * num_iters_per_epoch

        self.event_handlers_names = {
            e: [
                h.__qualname__ if hasattr(h, "__qualname__") else h.__class__.__name__
                for (h, _, _) in engine._event_handlers[e]
                if not self._is_internal_handler(h)
            ]
            for e in self._events
        }

        self._reset(self.max_epochs, self.total_num_iters)

        self._orig_event_handlers = engine._event_handlers.copy()

        for e in self._global_events:
            for i, h in enumerate(self._orig_event_handlers[e]):
                if self._is_internal_handler(h):
                    pass
                engine._event_handlers[e][i] = (partial(self._timeit_global_events, h), (e,), {})

        for e in self._epoch_events:
            for i, h in enumerate(self._orig_event_handlers[e]):
                if self._is_internal_handler(h):
                    pass
                engine._event_handlers[e][i] = (partial(self._timeit_epoch_events, h), (engine, e,), {})

        for e in self._iteration_events:
            for i, h in enumerate(self._orig_event_handlers[e]):
                if self._is_internal_handler(h):
                    pass
                engine._event_handlers[e][i] = (partial(self._timeit_iteration_events, h), (engine, e,), {})

        # processing timer
        engine._event_handlers[Events.ITERATION_STARTED].append((self._processing_timer.reset, (), {}))
        engine._event_handlers[Events.ITERATION_COMPLETED].insert(0, (self._timeit_processing, (engine,), {}))

        # dataflow timer
        engine._event_handlers[Events.GET_BATCH_STARTED].append((self._dataflow_timer.reset, (), {}))
        engine._event_handlers[Events.GET_BATCH_COMPLETED].insert(0, (self._timeit_dataflow, (engine,), {}))

        # revert back the wrapped handlers with original handlers at the end
        engine._event_handlers[Events.COMPLETED].append((self._detach_profiler_handlers, (engine,), {}))

    def attach(self, engine: Engine):
        if not isinstance(engine, Engine):
            raise TypeError("Argument engine should be ignite.engine.Engine, " "but given {}".format(type(engine)))

        if not engine.has_event_handler(self._as_first_started):
            engine._event_handlers[Events.STARTED].insert(0, (self._as_first_started, (engine,), {}))

    def get_results(self):
        """
        Method to fetch the aggregated profiler results after the engine is run

        .. code-block:: python

            results = profiler.get_results()

        """
        total_eh_time = round(
            float(
                sum(
                    [
                        sum([(self.event_handlers_times[e][h]).sum() for h in self.event_handlers_times[e]])
                        for e in self._events
                    ]
                )
            ),
            5,
        )

        def compute_basic_stats(data: Sequence):
            # compute on non-zero data:
            data = data[data > 0]
            total = round(torch.sum(data).item(), 5) if len(data) > 0 else "not triggered"
            min_index = ("None", "None")
            max_index = ("None", "None")
            mean = "None"
            std = "None"
            if len(data) > 1:
                min_index = (round(torch.min(data).item(), 5), torch.argmin(data).item())
                max_index = (round(torch.max(data).item(), 5), torch.argmax(data).item())
                mean = round(torch.mean(data).item(), 5)
                std = round(torch.std(data).item(), 5)
            return [total, min_index, max_index, mean, std]

        event_handler_stats = [
            [h, e.name, *compute_basic_stats(self.event_handlers_times[e][h])]
            for e in self._events
            for h in self.event_handlers_times[e]
        ]
        event_handler_stats.append(["Total", "", total_eh_time, "", "", "", ""])
        event_handler_stats.append(["Processing", "None", *compute_basic_stats(self.processing_times)])
        event_handler_stats.append(["Dataflow", "None", *compute_basic_stats(self.dataflow_times)])

        return event_handler_stats

    def write_results(self, output_path: str):
        """
        Method to store the unaggregated profiling results to a csv file

        .. code-block:: python

            profiler.write_results('path_to_dir/awesome_filename.csv')

        Example output:

        .. code-block:: text

            -----------------------------------------------------------------
            epoch iteration processing_stats dataflow_stats training.<locals>.log_elapsed_time (EPOCH_COMPLETED) ...
            1.0     1.0        0.00003         0.252387                          0.125676
            1.0     2.0        0.00029         0.252342                          0.125123

        """
        try:
            import pandas as pd
        except ImportError:
            print("Need pandas to write results as files")
            return

        iters_per_epoch = self.total_num_iters // self.max_epochs

        epochs = torch.arange(self.max_epochs, dtype=torch.float32).repeat_interleave(iters_per_epoch) + 1
        iterations = torch.arange(self.total_num_iters, dtype=torch.float32) + 1
        processing_stats = self.processing_times
        dataflow_stats = self.dataflow_times

        cols = [epochs, iterations, processing_stats, dataflow_stats]
        headers = ["epoch", "iteration", "processing_stats", "dataflow_stats"]

        # global event cols
        for e in self._global_events:
            for h in self.event_handlers_names[e]:
                headers.append("{} ({})".format(h, e.name))
                cols.append(self.event_handlers_times[e][h].repeat_interleave(self.total_num_iters))

        # epoch event cols
        for e in self._epoch_events:
            for h in self.event_handlers_names[e]:
                headers.append("{} ({})".format(h, e.name))
                cols.append(self.event_handlers_times[e][h].repeat_interleave(iters_per_epoch))

        # iteration event cols
        for e in self._iteration_events:
            for h in self.event_handlers_names[e]:
                headers.append("{} ({})".format(h, e.name))
                cols.append(self.event_handlers_times[e][h])

        results_dump = torch.stack(cols, dim=1,).numpy()

        results_df = pd.DataFrame(data=results_dump, columns=headers,)
        results_df.to_csv(output_path, index=False)

    @staticmethod
    def print_results(results: List[List[Union[str, float]]]):
        # adopted implementation of torch.autograd.profiler.build_table
        handler_column_width = max([len(item[0]) for item in results]) + 4
        event_column_width = max([len(item[1]) for item in results]) + 4

        DEFAULT_COLUMN_WIDTH = 14

        headers = [
            "Handler",
            "Event Name",
            "Total(s)",
            "Min(s)/IDX",
            "Max(s)/IDX",
            "Mean(s)",
            "Std(s)",
        ]

        # Have to use a list because nonlocal is Py3 only...
        SPACING_SIZE = 2
        row_format_lst = [""]
        header_sep_lst = [""]
        line_length_lst = [-SPACING_SIZE]

        def add_column(padding: int, text_dir: str = ">"):
            row_format_lst[0] += "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
            header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
            line_length_lst[0] += padding + SPACING_SIZE

        add_column(handler_column_width, text_dir="<")
        add_column(event_column_width, text_dir="<")
        for _ in headers[2:]:
            add_column(DEFAULT_COLUMN_WIDTH)

        row_format = row_format_lst[0]
        header_sep = header_sep_lst[0]

        result = []

        def append(s: str):
            result.append(s)
            result.append("\n")

        result.append("\n")
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)
        # TODO: add proper printing for total, processing and dataflow stats
        for row in results[:-3]:
            # format min/idx and max/idx
            row[3] = "{}/{}".format(*row[3])
            row[4] = "{}/{}".format(*row[4])

            append(row_format.format(*row))

        append(header_sep)
        # print total handlers time row
        append(row_format.format(*results[-3]))
        append(header_sep)

        summary_format = "{} took total {}s [min/index: {}, max/index: {}, mean: {}s, std: {}s]"
        for row in results[-2:]:
            row[3] = "{}s/{}".format(*row[3])
            row[4] = "{}s/{}".format(*row[4])
            del row[1]
            append(summary_format.format(*row))
        print("".join(result))

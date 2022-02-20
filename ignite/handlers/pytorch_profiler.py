# coding: utf-8
import os
import socket
from datetime import datetime
from typing import Any, Callable, Union

import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events


class PyTorchProfiler:
    """PyTorch Profiler for performance debugging.

    The PyTorch profiler is a tool that collects both GPU hardware and PyTorch related
    information, correlates them, performs automatic detection of bottlenecks in the model,
    and generates recommendations on how to resolve these bottlenecks.

    Args:
        cuda_activity: If true, records GPU activity in addition to CPU activity,
        on_trace_ready: Function that takes a reference to the profiler as an input
        and is called by the profiler each time the new trace is ready,
        Accepts custom function definition, as well as `tensorboard`, `flame_graph` and `chrome` as handlers.
        record_shapes: whether to record shapes of the inputs (necessary if you want to group profiler output by shapes)
        profile_memory: whether to report amount of memory consumed by model's Tensors
        with_stack: whether to record source information for the operations (necessary for flamegraph),
        with_flops: whether to use formula to estimate the FLOPS of specific ops (matrix multiplication and 2D conv),
        with_modules: whether to record module hierarchy (including function names) corresponding
        to the callstack of the op. e.g. If module A's forward call's module B's forward which
        contains an aten::add op, then aten::add's module hierarchy is A.B
        output_path: Directory where file should be placed,
        file_name: name of output file generated,
        skip_first: Scheduling parameter, the profiler first skips the first `skip_first` number of steps
        wait: Scheduling parameter, the profiler waits for `wait` number of steps
        warmup: Scheduling Parameter, the profile warms up for `warmup` number of steps
        active: Scheduling Parameter, the profiler does active profiling for the `active` number of steps
        repeat: Scheduling Parameter, number of cycles, 0 means that cycles will continue until profiling is finished.

    Examples:
        .. code-block:: python

            from ignite.handlers import PyTorchProfiler

            trainer = ...
            model = ...
            optimizer = ...

            pt_profiler = PyTorchProfiler(on_trace_ready="tensorboard", output_path="logs/train")
            pt_profiler.attach(trainer)

            # Get profiler results of time
            pt_profiler.print_results()

            # Save profiler result to text file
            pt_profiler.write_results()

            Both these methods can also be used as the on_trace_ready function which gets called after trace is ready.
            pt_profiler = PyTorchProfiler(on_trace_ready=profiler.write_to_file(10), output_path="logs/train")

            #The on_trace_handler accepts 3 strings `tensorboard`, `chrome` and `flamegraph`
            #Tensorboard
            pt_profiler = PyTorchProfiler(on_trace_ready="tensorboard", output_path="./logs/train")

            #To view this file enusre you have the PyTorch Profiler Tensorboard Plugin
            pip install torch_tb_profiler

            #Then launch tensorboard
            tensorboard --logdir=./logs

            #Chrome
            #Profiling results can be outputted as a .json trace file which can be viewed in the Chrome Trace viewer
            pt_profiler = PyTorchProfiler(on_trace_ready="chrome", output_path="./logs/train")

            #Open `chrome://tracing` on chrome and upload this file

            #Flamegraph
            Execution times can be visualised as a flamegraph
            pt_profiler = PyTorchProfiler(on_trace_ready="flamegraph",
                                        output_path="./logs/train", file_name = "fg", with_stack=True)

            # To view as an interactive SVG
            # git clone https://github.com/brendangregg/FlameGraph
            # cd FlameGraph
            # ./flamegraph.pl --title "CPU time" --countname "us." ./logs/train/fg_cpu_flamegraph.txt > perf_viz.svg

            #Custom Trace Handlers can also be used
            def trace_handler(p):
                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
                print(output)
                p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
            pt_profiler = PyTorchProfiler(on_trace_ready=trace_handler, output_path="logs/train")

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        cuda_activity: bool = False,
        on_trace_ready: Union[Callable[..., Any], str] = "tensorboard",
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        output_path: str = None,
        file_name: str = None,
        skip_first: int = 0,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
    ) -> None:

        self._activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda_activity and torch.cuda.is_available():
            self._activities.append(torch.profiler.ProfilerActivity.GPU)

        self._output_path = output_path
        self._file_name = file_name

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self._file_name:
            self._file_name = f"{idist.backend()}_{now}_{socket.gethostname()}_{str(os.getpid())}"

        self._with_stack = with_stack

        self._schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first
        )

        if on_trace_ready == "tensorboard":
            self._trace_handler = torch.profiler.tensorboard_trace_handler(self._output_path)

        elif on_trace_ready == "chrome":

            def chrome_trace(prof) -> None:
                prof.export_chrome_trace(os.path.join(self._output_path, self._file_name + "_chrome_trace.json"))

            self._trace_handler = chrome_trace

        elif on_trace_ready == "flamegraph":
            if not with_stack:
                raise ValueError("The flag with_stack must be true in order to use flamegraph")

            def flamegraph_trace(prof) -> None:
                prof.export_stacks(
                    os.path.join(self._output_path, self._file_name + "_cpu_flamegraph.txt"), "self_cpu_time_total"
                )
                if cuda_activity:
                    prof.export_stacks(
                        os.path.join(self._output_path, self._file_name + "_gpu_flamegraph.json"),
                        "self_cuda_time_total",
                    )

            self._trace_handler = flamegraph_trace
        else:
            if not isinstance(on_trace_ready, Callable):
                raise ValueError(
                    "Trace Handler should be a callable or one of"
                    f"[`tensorboard`, `chrome`, `flamegraph`]. Found: {on_trace_ready}"
                )
            self._trace_handler = on_trace_ready

        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_flops = with_flops
        self._with_modules = with_modules

        self._SORT_KEYS = {
            "cpu_time",
            "cuda_time",
            "cpu_time_total",
            "cuda_time_total",
            "cpu_memory_usage",
            "cuda_memory_usage",
            "self_cpu_memory_usage",
            "self_cuda_memory_usage",
            "count",
        }

    def _profiler_create(self):
        self._profiler = torch.profiler.profile(
            activities=self._activities,
            schedule=self._schedule,
            on_trace_ready=self._trace_handler,
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_stack=self._with_stack,
            with_flops=self._with_flops,
        )

    def _profiler_enter(self):
        self._profiler.__enter__()

    def _exit_profiler(self):
        self._profiler.__exit__(None, None, None)

    def _profiler_step(self):
        self._profiler.step()

    def attach(
        self,
        engine: Engine,
    ) -> None:
        """Attach the profiler to the engine.

        Args:
            engine: engine object.
        """
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        engine.add_event_handler(Events.EPOCH_STARTED, self._profiler_create)
        engine.add_event_handler(Events.EPOCH_STARTED, self._profiler_enter)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._profiler_step)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._exit_profiler)

    def get_results(
        self, n: int = -1, sort_key: str = "self_cuda_memory_usage", top_level_events_only=False, group_by_shapes=False
    ):
        if sort_key not in self._SORT_KEYS:
            raise ValueError(
                f" The sort_key {sort_key} is not accepted. Please choose a sort key from {self._SORT_KEYS}"
            )

        if group_by_shapes and self._record_shapes is False:
            raise ValueError(
                "Running with group_by_input_shape=True requires running the profiler with record_shapes=True"
            )

        return self._profiler.key_averages(group_by_input_shape=group_by_shapes).table(
            sort_by=sort_key, row_limit=n, top_level_events_only=top_level_events_only
        )

    def write_results(self, n: int = -1, sort_key: str = "self_cuda_memory_usage", top_level_events_only=False):

        with open(os.path.join(self._output_path, self._file_name + ".txt"), "w") as f:
            f.write(self.get_results(n, sort_key, top_level_events_only))

    def print_results(self, n: int = -1, sort_key: str = "self_cuda_memory_usage", top_level_events_only=False):
        print(self.get_results(n, sort_key, top_level_events_only))

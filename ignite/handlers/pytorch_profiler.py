# coding: utf-8
import datetime
import os
from typing import Any, Callable, Union

import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events


class PyTorchProfiler:
    """PyTorch Profiler for performance debugging.

    The PyTorch profiler is a tool that collects both GPU hardware and PyTorch related
    information, correlates them, performs automatic detection of bottlenecks in the model,
    and generates recommendations on how to resolve these bottlenecks.

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

            # Save profiler result to CSV file (requires pandas)
            pt_profiler.write_results()

            Both these methods can also be used as the on_trace_ready function which gets called after trace is ready.

            pt_profiler = PyTorchProfiler(on_trace_ready=profiler.write_to_file(10), output_path="logs/train")

    .. versionadded:: 0.4.8
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
        wait: int = 2,
        warmup: int = 2,
        active: int = 6,
        repeat: int = 1,
    ) -> None:

        self.activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda_activity and torch.cuda.is_available():
            self.activities.append(torch.profiler.ProfilerActivity.GPU)

        self.output_path = output_path

        self.schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

        self.trace_handler = (
            torch.profiler.tensorboard_trace_handler(self.output_path)
            if on_trace_ready == "tensorboard"
            else on_trace_ready
        )

        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules

        self.SORT_KEYS = {
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
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=self.trace_handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )
        self._profiler.__enter__()

    def _exit_profiler(self):
        self._profiler.__exit__()

    def _profiler_step(self):
        self.profiler.step()

    def attach(self, engine: Engine,) -> None:
        """Attach the profiler to the engine.

        Args:
            engine: engine object.
        """

        engine._event_handlers[Events.EPOCH_STARTED].append((self._profiler_create, {}, {}))
        engine._event_handlers[Events.GET_BATCH_COMPLETED].append((self._profiler_step, {}, {}))
        engine._event_handlers[Events.EPOCH_COMPLETED].append((self._profile_create.__exit__(), {}, {}))

    def get_results(self, n: int = -1, sort_key: str = "self_cuda_memory_usage", top_level_events_only=False):
        if sort_key not in self.SORT_KEYS:
            raise ValueError(
                f" The sort_key {sort_key} is not accepted. Please choose a sort key from {self.SORT_KEYS}"
            )

        return self.profiler.key_averages().table(
            sort_by=sort_key, row_limit=n, top_level_events_only=top_level_events_only
        )

    def write_results(self, n: int = -1, sort_key: str = "self_cuda_memory_usage", top_level_events_only=False):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("Need pandas to write results as files")

        results_df = pd.DataFrame(self.get_results(n, sort_key, top_level_events_only))

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{idist.backend()}_{now}.csv"

        results_df.to_csv(os.path.join(self.output_path, file_name), index=False)

    def print_results(self, n: int = -1, sort_key: str = "self_cuda_memory_usage", top_level_events_only=False):
        print(self.get_results(n, sort_key, top_level_events_only))

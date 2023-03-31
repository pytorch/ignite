import os

import pytest
import torch

from ignite.engine import Engine
from ignite.handlers import PyTorchProfiler


def clean_string(s):
    return s.lstrip().rstrip()


def update_fn(engine, batch):
    x = torch.randn((1, 8), requires_grad=True)
    y = torch.randn((8, 1), requires_grad=True)
    z = torch.matmul(x, y)
    z.backward()


def get_engine():
    dummy_trainer = Engine(update_fn)
    return dummy_trainer


def output_string_to_dict(output_string):
    output_string = output_string.split("\n")

    # Removing the formatting and headers
    output_string = output_string[3:-3]

    output_string_split = dict()

    for _output_string in output_string:
        split_string = _output_string.split("    ")
        split_string = [clean_string(i) for i in split_string if i != ""]
        # Using name and shape as key to distinguish between same operation with different shapes
        output_string_split[split_string[0] + split_string[-1]] = split_string[1:]

    return output_string_split


def check_profiler_output(data, sort_key="cpu_time", wait=1, warmup=1, active=3, repeat=1):
    # Returns output of PyTorch Profiler directly (Without using Ignite handler) for comparison

    from torch.profiler import ProfilerActivity, profile, schedule

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        record_shapes=True,
    ) as prof:
        for d in data:
            x = torch.randn((1, 8), requires_grad=True)
            y = torch.randn((8, 1), requires_grad=True)
            z = torch.matmul(x, y)
            z.backward()
            prof.step()
    return prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key)


def get_both_profiler_outputs(data_len, path, epoch, wait=1, warmup=1, active=3, repeat=1):
    data = [i for i in range(data_len)]
    trainer = get_engine()
    pt_profiler = PyTorchProfiler(
        on_trace_ready="tensorboard",
        output_path=path,
        record_shapes=True,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
        with_stack=True,
    )
    pt_profiler.attach(trainer)
    trainer.run(data, max_epochs=epoch)
    output_string = pt_profiler.get_results(sort_key="cpu_time", group_by_shapes=True)

    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            ref_output = check_profiler_output(data, "cpu_time", wait=wait, warmup=warmup, active=active, repeat=repeat)
    else:
        ref_output = check_profiler_output(data, "cpu_time", wait=wait, warmup=warmup, active=active, repeat=repeat)
    return ref_output, output_string


def test_profilers_wrong_inputs():
    pt_profiler = PyTorchProfiler()

    with pytest.raises(TypeError, match=r"Argument engine should be ignite.engine.Engine"):
        pt_profiler.attach(None)

    with pytest.raises(ValueError, match=r" The sort_key cpu_times is not accepted. Please choose a sort key from"):
        pt_profiler.get_results(sort_key="cpu_times")

    with pytest.raises(
        ValueError,
        match=r"Running with group_by_input_shape=True requires running the profiler with record_shapes=True",
    ):
        pt_profiler.get_results(group_by_shapes=True)

    with pytest.raises(ValueError, match=r"The flag with_stack must be true in order to use flamegraph"):
        pt_profiler = PyTorchProfiler(on_trace_ready="flamegraph", with_stack=False)

    with pytest.raises(ValueError, match=r"Trace Handler should be a callable or one of"):
        pt_profiler = PyTorchProfiler(on_trace_ready=10, with_stack=False)


@pytest.mark.parametrize("data_len", [1, 6, 10])
@pytest.mark.parametrize("epoch", [1, 2, 10])
def test_get_results(epoch, data_len, tmp_path):
    ref_output, output_string = get_both_profiler_outputs(data_len, tmp_path, epoch)
    print(output_string, ref_output)
    output_dict = output_string_to_dict(output_string)
    ref_output_dict = output_string_to_dict(ref_output)

    for _key in output_dict.keys():
        # Checks number of calls are same in both profilers
        assert output_dict[_key][5] == ref_output_dict[_key][5]
        # Checks shapes
        assert output_dict[_key][6] == ref_output_dict[_key][6]

    # Check number of elements recorded
    assert len(output_dict) == len(ref_output_dict)


@pytest.mark.parametrize("wait,warmup,active,repeat", [(99, 2, 1, 1), (2, 99, 1, 1), (99, 2, 1, 2)])
@pytest.mark.parametrize("epoch", [1, 2, 10])
def test_none_output(epoch, tmp_path, wait, warmup, active, repeat):
    trainer = get_engine()
    pt_profiler = PyTorchProfiler(
        on_trace_ready="tensorboard", output_path=tmp_path, wait=wait, warmup=warmup, active=active, repeat=repeat
    )
    pt_profiler.attach(trainer)
    trainer.run(range(100), max_epochs=epoch)
    assert pt_profiler.get_results() == ""


@pytest.mark.parametrize("wait,warmup,active,repeat", [(1, 1, 2, 1), (6, 2, 92, 2), (99, 1, 10, 10)])
@pytest.mark.parametrize("epoch", [1, 2, 10])
def test_schedule(epoch, tmp_path, wait, warmup, active, repeat):
    ref_output, output_string = get_both_profiler_outputs(100, tmp_path, epoch, wait, warmup, active, repeat)

    output_dict = output_string_to_dict(output_string)
    ref_output_dict = output_string_to_dict(ref_output)
    print(output_string, ref_output)

    for _key in output_dict.keys():
        assert output_dict[_key][5] == ref_output_dict[_key][5], print(_key)
        assert output_dict[_key][6] == ref_output_dict[_key][6]

    # Check number of elements recorded
    assert len(output_dict) == len(ref_output_dict)


@pytest.mark.parametrize("epoch", [1, 5, 100])
def test_multiple_epochs_files(epoch, tmp_path):
    # Number of files should be same as epochs
    trainer = get_engine()
    pt_profiler = PyTorchProfiler(on_trace_ready="tensorboard", output_path=tmp_path, with_stack=True)
    pt_profiler.attach(trainer)
    trainer.run(range(20), max_epochs=epoch)
    assert epoch == len(os.listdir(tmp_path))


@pytest.mark.parametrize("n", [1, 5, 10])
def test_write_results(n, tmp_path):
    # File Length should be equal to n (row limit)
    trainer = get_engine()
    pt_profiler = PyTorchProfiler(on_trace_ready="tensorboard", output_path=tmp_path, file_name="testing_file")
    pt_profiler.attach(trainer)
    trainer.run(range(10), max_epochs=1)
    pt_profiler.write_results(n=n)

    fp = os.path.join(tmp_path, "testing_file.txt")
    assert os.path.isfile(fp)

    file_length = 0
    with open(fp, "r") as fp:
        for _ in fp:
            file_length += 1

    assert file_length == n + 5

import logging
import platform
import sys
from collections import namedtuple

import pytest
import torch
from packaging.version import Version

from ignite.engine import Engine, Events
from ignite.utils import _to_str_list, convert_tensor, deprecated, hash_checkpoint, setup_logger, to_onehot


def test_convert_tensor():
    x = torch.tensor([0.0])
    tensor = convert_tensor(x)
    assert torch.is_tensor(tensor)

    x = torch.tensor([0.0])
    tensor = convert_tensor(x, device="cpu", non_blocking=True)
    assert torch.is_tensor(tensor)

    x = torch.tensor([0.0])
    tensor = convert_tensor(x, device="cpu", non_blocking=False)
    assert torch.is_tensor(tensor)

    x = [torch.tensor([0.0]), torch.tensor([0.0])]
    list_ = convert_tensor(x)
    assert isinstance(list_, list)
    assert torch.is_tensor(list_[0])
    assert torch.is_tensor(list_[1])

    x = (torch.tensor([0.0]), torch.tensor([0.0]))
    tuple_ = convert_tensor(x)
    assert isinstance(tuple_, tuple)
    assert torch.is_tensor(tuple_[0])
    assert torch.is_tensor(tuple_[1])

    Point = namedtuple("Point", ["x", "y"])
    x = Point(torch.tensor([0.0]), torch.tensor([0.0]))
    tuple_ = convert_tensor(x)
    assert isinstance(tuple_, Point)
    assert torch.is_tensor(tuple_[0])
    assert torch.is_tensor(tuple_[1])

    x = {"a": torch.tensor([0.0]), "b": torch.tensor([0.0])}
    dict_ = convert_tensor(x)
    assert isinstance(dict_, dict)
    assert torch.is_tensor(dict_["a"])
    assert torch.is_tensor(dict_["b"])

    assert convert_tensor("a") == "a"

    with pytest.raises(TypeError):
        convert_tensor(12345)


@pytest.mark.parametrize(
    "input_data,expected",
    [
        (42, ["42.0000"]),
        ([{"a": 15, "b": torch.tensor([2.0])}], ["a: 15.0000", "b: [2.0000]"]),
        ({"a": 10, "b": 2.33333}, ["a: 10.0000", "b: 2.3333"]),
        ({"x": torch.tensor(0.1234), "y": [1, 2.3567]}, ["x: 0.1234", "y: 1.0000, 2.3567"]),
        (({"nested": [3.1415, torch.tensor(0.0001)]},), ["nested: 3.1415, 0.0001"]),
        (
            {"large_vector": torch.tensor(range(20))},
            ["large_vector: [0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000, ...]"],
        ),
        ({"large_matrix": torch.randn(5, 5)}, ["large_matrix: Shape[5, 5]"]),
        ({"empty": []}, ["empty: "]),
        ([], []),
        ({"none": None}, ["none: "]),
        ({1: 100, 2: 200}, ["1: 100.0000", "2: 200.0000"]),
    ],
)
def test__to_str_list(input_data, expected):
    assert _to_str_list(input_data) == expected


def test_to_onehot():
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    actual = to_onehot(indices, 4)
    expected = torch.eye(4, dtype=torch.uint8)
    assert actual.equal(expected)

    y = torch.randint(0, 21, size=(1000,))
    y_ohe = to_onehot(y, num_classes=21)
    y2 = torch.argmax(y_ohe, dim=1)
    assert y.equal(y2)

    y = torch.randint(0, 21, size=(4, 250, 255))
    y_ohe = to_onehot(y, num_classes=21)
    y2 = torch.argmax(y_ohe, dim=1)
    assert y.equal(y2)

    y = torch.randint(0, 21, size=(4, 150, 155, 4, 6))
    y_ohe = to_onehot(y, num_classes=21)
    y2 = torch.argmax(y_ohe, dim=1)
    assert y.equal(y2)

    # Test with `TorchScript`

    x = torch.tensor([0, 1, 2, 3])

    # Test the raw `to_onehot` function
    scripted_to_onehot = torch.jit.script(to_onehot)
    assert scripted_to_onehot(x, 4).allclose(to_onehot(x, 4))

    # Test inside `torch.nn.Module`
    class SLP(torch.nn.Module):
        def __init__(self):
            super(SLP, self).__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, x):
            x = to_onehot(x, 4)
            return self.linear(x.to(torch.float))

    eager_model = SLP()
    scripted_model = torch.jit.script(eager_model)

    assert eager_model(x).allclose(scripted_model(x))


def test_dist_setup_logger():
    logger = setup_logger("trainer", level=logging.CRITICAL, distributed_rank=1)
    assert logger.level != logging.CRITICAL


def test_setup_logger(capsys, dirname):
    trainer = Engine(lambda e, b: None)
    evaluator = Engine(lambda e, b: None)

    assert len(trainer.logger.handlers) == 0
    trainer.logger.addHandler(logging.NullHandler())
    trainer.logger.addHandler(logging.NullHandler())
    trainer.logger.addHandler(logging.NullHandler())

    fp = dirname / "log"

    def _test(stream):
        trainer.logger = setup_logger("trainer", stream=stream, filepath=fp, reset=True)
        evaluator.logger = setup_logger("evaluator", stream=stream, filepath=fp, reset=True)

        assert len(trainer.logger.handlers) == 2
        assert len(evaluator.logger.handlers) == 2

        @trainer.on(Events.EPOCH_COMPLETED)
        def _(_):
            evaluator.run([0, 1, 2])

        trainer.run([0, 1, 2, 3, 4, 5], max_epochs=5)

        captured = capsys.readouterr()
        if stream is sys.stdout:
            err = captured.out.split("\n")
        else:
            err = captured.err.split("\n")

        with open(fp, "r") as h:
            data = h.readlines()

        for source in [err, data]:
            assert "trainer INFO: Engine run starting with max_epochs=5." in source[0]
            assert "evaluator INFO: Engine run starting with max_epochs=1." in source[1]

    _test(stream=None)
    _test(stream=sys.stderr)
    _test(stream=sys.stdout)

    # Needed by windows to release FileHandler in the loggers
    logging.shutdown()


def _setup_a_logger_and_dump(name, message):
    logger = setup_logger(name)
    logger.info(message)


def test_override_setup_logger(capsys):
    _setup_a_logger_and_dump(__name__, "test_override_setup_logger")

    source = capsys.readouterr().err.split("\n")

    assert "tests.ignite.test_utils INFO: test_override_setup_logger" in source[0]

    # change the logger level of _setup_a_logger_and_dump
    setup_logger(name=__name__, level=logging.WARNING, reset=True)

    _setup_a_logger_and_dump(__name__, "test_override_setup_logger")

    source = capsys.readouterr().err.split("\n")

    assert source[0] == ""

    # Needed by windows to release FileHandler in the loggers
    logging.shutdown()


@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_setup_logger_encoding(encoding, dirname):
    fp = dirname / "log.txt"
    logger = setup_logger(name="logger", filepath=fp, encoding=encoding, reset=True)
    test_words = ["say hello", "say 你好", "say こんにちわ", "say 안녕하세요", "say привет"]
    for w in test_words:
        logger.info(w)
    logging.shutdown()

    with open(fp, "r", encoding=encoding) as h:
        data = h.readlines()

    if platform.system() == "Windows" and encoding is None:
        flatten_data = "\n".join(data)
        assert test_words[0] in flatten_data
        for word in test_words[1:]:
            assert word not in flatten_data
    else:
        assert len(data) == len(test_words)
        for expected, output in zip(test_words, data):
            assert expected in output


def test_deprecated():
    # Test on function without docs, @deprecated without reasons
    @deprecated("0.4.2", "0.6.0")
    def func_no_docs():
        return 24

    assert func_no_docs.__doc__ == "**Deprecated function**.\n\n    .. deprecated:: 0.4.2"

    # Test on function with docs, @deprecated without reasons
    @deprecated("0.4.2", "0.6.0")
    def func_no_reasons():
        """Docs are cool"""
        return 24

    assert func_no_reasons.__doc__ == "**Deprecated function**.\n\n    Docs are cool.. deprecated:: 0.4.2"

    # Test on function with docs, @deprecated with reasons
    @deprecated("0.4.2", "0.6.0", reasons=("r1", "r2"))
    def func_no_warnings():
        """Docs are very cool"""
        return 24

    assert (
        func_no_warnings.__doc__
        == "**Deprecated function**.\n\n    Docs are very cool.. deprecated:: 0.4.2\n\n\t\n\t- r1\n\t- r2"
    )

    # Tests that the function emits DeprecationWarning
    @deprecated("0.4.2", "0.6.0", reasons=("r1", "r2"))
    def func_check_warning():
        """Docs are very ..."""
        return 24

    with pytest.deprecated_call():
        assert func_check_warning() == 24
    with pytest.warns(
        DeprecationWarning,
        match="This function has been deprecated since version 0.4.2 and will be removed in version 0.6.0."
        + "\n Please refer to the documentation for more details.",
    ):
        # Trigger a warning.
        func_check_warning()

    # Test that the function raises Exception
    @deprecated("0.4.2", "0.6.0", reasons=("reason1", "reason2"), raise_exception=True)
    def func_with_everything():
        return 1

    with pytest.raises(Exception) as exec_info:
        func_with_everything()

    assert (
        str(exec_info.value)
        == "This function has been deprecated since version 0.4.2 and will be removed in version 0.6.0."
        + "\n Please refer to the documentation for more details."
    )


def test_smoke__utils():
    from ignite._utils import apply_to_tensor, apply_to_type, convert_tensor, to_onehot  # noqa: F401


@pytest.mark.skipif(Version(torch.__version__) < Version("1.5.0"), reason="Skip if < 1.5.0")
def test_hash_checkpoint(tmp_path):
    # download lightweight model
    from torchvision.models import squeezenet1_0

    model = squeezenet1_0()
    torch.hub.download_url_to_file(
        "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth", f"{tmp_path}/squeezenet1_0.pt"
    )
    hash_checkpoint_path, sha_hash = hash_checkpoint(f"{tmp_path}/squeezenet1_0.pt", str(tmp_path))
    model.load_state_dict(torch.load(str(hash_checkpoint_path), "cpu"), True)
    assert sha_hash[:8] == "b66bff10"
    assert hash_checkpoint_path.name == f"squeezenet1_0-{sha_hash[:8]}.pt"

    # test non-existent checkpoint_path
    with pytest.raises(FileNotFoundError, match=r"not_found.pt does not exist in *"):
        hash_checkpoint(f"{tmp_path}/not_found.pt", tmp_path)

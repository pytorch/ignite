import logging
import os
import sys
import warnings
from collections import namedtuple

import pytest
import torch

from ignite.engine import Engine, Events
from ignite.utils import convert_tensor, deprecated, setup_logger, to_onehot


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

    fp = os.path.join(dirname, "log")

    def _test(stream):

        trainer.logger = setup_logger("trainer", stream=stream, filepath=fp)
        evaluator.logger = setup_logger("evaluator", stream=stream, filepath=fp)

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


def test_deprecated():

    # Test on function without docs, @deprecated without reasons
    @deprecated("0.4.2", "0.6.0")
    def func_no_docs():
        return 24

    assert func_no_docs.__doc__ == "**Deprecated function**.\n\n    .. deprecated:: 0.4.2"

    # Test on function with docs, @deprecated without reasons
    @deprecated("0.4.2", "0.6.0")
    def func_no_reasons():
        """Docs are cool
        """
        return 24

    assert func_no_reasons.__doc__ == "**Deprecated function**.\n\n    Docs are cool\n        .. deprecated:: 0.4.2"

    # Test on function with docs, @deprecated with reasons
    @deprecated("0.4.2", "0.6.0", reasons=("r1", "r2"))
    def func_no_warnings():
        """Docs are very cool
        """
        return 24

    assert (
        func_no_warnings.__doc__
        == "**Deprecated function**.\n\n    Docs are very cool\n        .. deprecated:: 0.4.2\n\n\t\n\t- r1\n\t- r2"
    )

    # Tests that the function emits DeprecationWarning
    @deprecated("0.4.2", "0.6.0", reasons=("r1", "r2"))
    def func_check_warning():
        """Docs are very ...
        """
        return 24

    with pytest.deprecated_call():
        func_check_warning()
    assert func_check_warning() == 24
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        func_check_warning()
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert (
            "This function has been deprecated since version 0.4.2 and will be removed in version 0.6.0."
            + "\n Please refer to the documentation for more details."
            in str(w[-1].message)
        )

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

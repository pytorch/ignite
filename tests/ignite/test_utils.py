import logging
import os
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


def test_dist_setup_logger():

    logger = setup_logger("trainer", level=logging.CRITICAL, distributed_rank=1)
    assert logger.level != logging.CRITICAL


def test_setup_logger(capsys, dirname):

    trainer = Engine(lambda e, b: None)
    evaluator = Engine(lambda e, b: None)

    fp = os.path.join(dirname, "log")
    assert len(trainer.logger.handlers) == 0
    trainer.logger.addHandler(logging.NullHandler())
    trainer.logger.addHandler(logging.NullHandler())
    trainer.logger.addHandler(logging.NullHandler())

    trainer.logger = setup_logger("trainer", filepath=fp)
    evaluator.logger = setup_logger("evaluator", filepath=fp)

    assert len(trainer.logger.handlers) == 2
    assert len(evaluator.logger.handlers) == 2

    @trainer.on(Events.EPOCH_COMPLETED)
    def _(_):
        evaluator.run([0, 1, 2])

    trainer.run([0, 1, 2, 3, 4, 5], max_epochs=5)

    captured = capsys.readouterr()
    err = captured.err.split("\n")

    with open(fp, "r") as h:
        data = h.readlines()

    for source in [err, data]:
        assert "trainer INFO: Engine run starting with max_epochs=5." in source[0]
        assert "evaluator INFO: Engine run starting with max_epochs=1." in source[1]

    # Needed by windows to release FileHandler in the loggers
    logging.shutdown()


def test_deprecated():

    # No docs in orignal function, no reasons
    @deprecated("0.4.2", "0.6.0")
    def func_no_docs():
        return 24

    assert func_no_docs.__doc__ == "**Deprecated function**.\n\n    .. deprecated:: 0.4.2"

    # Docs but no reasons
    @deprecated("0.4.2", "0.6.0")
    def func_no_reasons():
        """Docs are cool
        """
        return 24

    assert func_no_reasons.__doc__ == "**Deprecated function**.\n\n    Docs are cool\n        .. deprecated:: 0.4.2"

    # Docs and reasons
    @deprecated("0.4.2", "0.6.0", reasons=["r1", "r2"])
    def func_no_warnings():
        """Docs are very cool
        """
        return 24

    assert (
        func_no_warnings.__doc__
        == "**Deprecated function**.\n\n    Docs are very cool\n        .. deprecated:: 0.4.2\n\n\t\n\t- r1\n\t- r2"
    )

    # Docs and Reasons and Warning
    @deprecated("0.4.2", "0.6.0", reasons=["r1", "r2"], raiseWarning=True)
    def func_with_everything():
        """Docs are very ...
        """
        return 24

    assert (
        func_with_everything.__doc__
        == "**Deprecated function**.\n\n    Docs are very ...\n        .. deprecated:: 0.4.2\n\n\t\n\t- r1\n\t- r2"
    )

    with pytest.deprecated_call():
        func_with_everything()
    assert func_with_everything() == 24
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        func_with_everything()
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert (
            "This function has been deprecated since version `0.4.2` and will be removed in version `0.6.0`."
            + "\n Please refer to the documentation for more details."
            in str(w[-1].message)
        )

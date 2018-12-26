
import pytest
import mock

from ignite.contrib.engines.fp16 import create_supervised_fp16_trainer, create_supervised_fp16_evaluator


def test_no_apex():

    # Mocking objects
    model = mock.MagicMock()
    # Necessary to unpack output
    model.return_value = (1, 1)
    optimizer = mock.MagicMock()
    loss = mock.MagicMock()

    with pytest.raises(RuntimeError):
        create_supervised_fp16_trainer(model, optimizer, loss)

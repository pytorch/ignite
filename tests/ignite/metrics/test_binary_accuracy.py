from ignite.metrics import BinaryAccuracy
import pytest


def test_warning():
    with pytest.warns(DeprecationWarning):
        BinaryAccuracy()

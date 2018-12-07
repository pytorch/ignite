from ignite.metrics import CategoricalAccuracy
import pytest


def test_warning():
    with pytest.warns(DeprecationWarning):
        CategoricalAccuracy()

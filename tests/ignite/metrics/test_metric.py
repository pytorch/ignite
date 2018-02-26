from ignite.metrics import Metric
import pytest


def test_abstract_base_class():
    class DummyMetric(Metric):
        pass

    with pytest.raises(TypeError):
        DummyMetric()

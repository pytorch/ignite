from ignite.engine import Engine
from ignite.metrics import Metric, MetricsLambda


class ListGatherMetric(Metric):

        def __init__(self, index):
            super(ListGatherMetric, self).__init__()
            self.index = index

        def reset(self):
            self.list_ = None

        def update(self, output):
            self.list_ = output

        def compute(self):
            return self.list_[self.index]


def test_metrics_lambda():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)

    def process_function(engine, data):
        return data

    engine = Engine(process_function)
    m0_plus_m1 = MetricsLambda(lambda x, y: x + y, m0, m1)
    m2_plus_2 = MetricsLambda(lambda x, y: x + y, m2, 2)
    m0_plus_m1.attach(engine, 'm0_plus_m1')
    m2_plus_2.attach(engine, 'm2_plus_2')

    engine.run([[1, 10, 100]])
    assert engine.state.metrics['m0_plus_m1'] == 11
    assert engine.state.metrics['m2_plus_2'] == 102
    engine.run([[2, 20, 200]])
    assert engine.state.metrics['m0_plus_m1'] == 22
    assert engine.state.metrics['m2_plus_2'] == 202


def test_metrics_lambda_reset():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    m2.update([1, 10, 100])

    m = MetricsLambda(lambda x, y, z, t: 1, m0, m1, m2, 0)

    # initiating a new instance of MetricsLambda must reset
    # its argument metrics
    assert m0.list_ is None
    assert m1.list_ is None
    assert m2.list_ is None

    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    m2.update([1, 10, 100])
    m.reset()
    assert m0.list_ is None
    assert m1.list_ is None
    assert m2.list_ is None

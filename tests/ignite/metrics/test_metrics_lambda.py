from ignite.metrics import Metric, MetricsLambda


class ListGatherMetric(Metric):

        def __init__(self, index):
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

    m0_plus_m1 = MetricsLambda(lambda x, y: x + y, m0, m1)
    m0_plus_m1.update([1, 10, 100])
    assert m0_plus_m1.compute() == 11
    m0_plus_m1.update([2, 20, 200])
    assert m0_plus_m1.compute() == 22

    m2_plus_2 = MetricsLambda(lambda x, y: x + y, m2, 2)
    m2_plus_2.update([1, 10, 100])
    assert m2_plus_2.compute() == 102


def test_metrics_lambda_reset():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    m2.update([1, 10, 100])

    m = MetricsLambda(lambda x, y, z, t: 1, m0, m1, m2, 0)
    m.reset()

    assert m0.list_ is None
    assert m1.list_ is None
    assert m2.list_ is None

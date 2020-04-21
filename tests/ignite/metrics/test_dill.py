import dill

from ignite.metrics import Metric


class Accumulation(Metric):
    def __init__(self):
        self.value = 0
        super(Accumulation, self).__init__()

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value

    def update(self, output):
        self.value += output


def test_metric():
    def _test(m, values, e):
        for v in values:
            m.update(v)
        assert m.compute() == e

    metric = Accumulation()

    m1 = dill.loads(dill.dumps(metric))

    values = list(range(10))
    expected = sum(values)

    _test(m1, values, expected)

    metric.update(5)

    m2 = dill.loads(dill.dumps(metric))

    _test(m2, values, expected + 5)

# Needed to collect coverage data
class MockFP16DeepSpeedZeroOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, closure=None):
        self.optimizer.step()

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

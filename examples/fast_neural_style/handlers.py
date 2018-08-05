import sys


class Progbar(object):

    def __init__(self, loader, metrics):
        self.num_iterations = len(loader)
        self.output_stream = sys.stdout
        self.metrics = metrics
        self.alpha = 0.98

    def _calc_running_avg(self, engine):
        for k, v in engine.state.output.items():
            old_v = self.metrics.get(k, v)
            new_v = self.alpha * old_v + (1 - self.alpha) * v
            self.metrics[k] = new_v

    def __call__(self, engine):
        self._calc_running_avg(engine)
        num_seen = engine.state.iteration - self.num_iterations * (engine.state.epoch - 1)

        percent_seen = 100 * float(num_seen) / self.num_iterations
        equal_to = int(percent_seen / 10)
        done = int(percent_seen) == 100

        bar = '[' + '=' * equal_to + '>' * (not done) + ' ' * (10 - equal_to) + ']'
        message = 'Epoch {epoch} | {percent_seen:.2f}% | {bar}'.format(epoch=engine.state.epoch,
                                                                       percent_seen=percent_seen,
                                                                       bar=bar)
        for key, value in self.metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=key, value=value)

        message += '\r'

        self.output_stream.write(message)
        self.output_stream.flush()

        if done:
            self.output_stream.write('\n')

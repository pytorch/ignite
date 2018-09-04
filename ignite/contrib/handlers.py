from tqdm import tqdm


class ProgressBar:
    def __init__(self, loader, metrics):
        self.num_iterations = len(loader)
        self.pbar = None
        self.metrics = metrics
        self.alpha = 0.98

    def _calc_running_avg(self, engine):
        for k, v in engine.state.output.items():
            old_v = self.metrics.get(k, v)
            new_v = self.alpha * old_v + (1 - self.alpha) * v
            self.metrics[k] = new_v

    def _reset(self):
        self.pbar = tqdm(
            total=self.num_iterations,
            bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')

    def _format_metrics(self):
        formatted_metrics = {}
        for key in self.metrics:
            formatted_metrics[key] = '{:.2e}'.format(self.metrics[key])
        return formatted_metrics

    def __call__(self, engine):
        if (engine.state.iteration - 1) % self.num_iterations == 0:
            self._reset()
        self._calc_running_avg(engine)
        self.pbar.set_description('Epoch {}'.format(engine.state.epoch))
        self.pbar.set_postfix(**self._format_metrics())
        self.pbar.update()

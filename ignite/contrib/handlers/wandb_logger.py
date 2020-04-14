from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOutputHandler,
    BaseOptimizerParamsHandler,
    global_step_from_engine,
)


__all__ = ["WandBLogger", "OutputHandler", "OptimizerParamsHandler",
           "global_step_from_engine"]


class OutputHandler(BaseOutputHandler):

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None, global_step_transform=None,
                 sync=None):
        super().__init__(tag, metric_names, output_transform, another_engine, global_step_transform)
        self.sync = sync

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, WandBLogger):
            raise RuntimeError("Handler '{}' works only with WandBLogger.".format(self.__class__.__name__))

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        metrics = self._setup_output_metrics(engine)
        if self.tag is not None:
            metrics = {"{tag}/{name}".format(tag=self.tag, name=name): value
                       for name, value in metrics.items()}

        logger.log(metrics, step=global_step, sync=self.sync)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):

    def __init__(self, optimizer, param_name="lr", tag=None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, WandBLogger):
            raise RuntimeError("Handler 'OptimizerParamsHandler' works only with WandBLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        logger.log(params, step=global_step)


class WandBLogger(BaseLogger):

    def __init__(self, *args, **kwargs):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise RuntimeError(
                "This contrib module requires wandb to be installed. "
                "You man install wandb with the command:\n pip install wandb\n"
            )
        if kwargs.get('init', True):
            wandb.init(*args, **kwargs)

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return getattr(self._wandb, attr)(*args, **kwargs)

        return wrapper

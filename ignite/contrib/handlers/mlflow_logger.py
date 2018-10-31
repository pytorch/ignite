from ignite.engine import Engine
from ignite.engine import Events
from typing import Callable, List

import mlflow


class MlflowLogger:
    """Handler that logs metrics using the `mlflow tracking` system.

    Examples:

    Plotting of trainer loss.

    .. code-block:: python

        import mlflow

        mlflow.set_tracking_uri(server_url)
        experiment_id = mlflow.set_experiment(MLFLOW_EXPERIMENT)

        #
        # Run the training under mlflow
        #
        with mlflow.start_run(experiment_id=experiment_id):

            trainer = create_supervised_trainer(model, optimizer, loss)

            mlflow_plotter = MlflowLogger()

            mlflow_plotter.attach(
                engine=trainer,
                prefix="Train ",
                plot_event=Events.ITERATION_COMPLETED,
                output_transform=lambda x: {"loss": x}
            )

            trainer.run(train_loader, max_epochs=epochs_num)

    """

    def __init__(self):

        self.metrics_step = []

    def _update(
            self,
            engine: Engine,
            attach_id: int,
            prefix: str,
            update_period: int,
            metric_names: List=None,
            output_transform: Callable=None):

        step = self.metrics_step[attach_id]
        self.metrics_step[attach_id] += 1
        if step % update_period != 0:
            return

        #
        # Get all the metrics
        #
        metrics = []
        if metric_names is not None:
            if not all(metric in engine.state.metrics for metric in metric_names):
                raise KeyError("metrics not found in engine.state.metrics")

            metrics.extend([(name, engine.state.metrics[name]) for name in metric_names])

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.extend([(name, value) for name, value in output_dict.items()])

        if not metrics:
            return

        for metric_name, new_value in metrics:
            mlflow.log_metric(prefix + metric_name, new_value)

    def attach(
            self,
            engine: Engine,
            prefix: str="",
            plot_event: str=Events.EPOCH_COMPLETED,
            update_period: int=1,
            metric_names: List=None,
            output_transform: Callable=None):
        """
        Attaches the mlflow plotter to an engine object

        Args:
            engine (Engine): engine object
            prefix (str, optional): A prefix to add before the metric name.
            plot_event (str, optional): Name of event to handle.
            update_period (int, optional): Can be used to limit the number of plot updates.
            metric_names (list, optional): list of the metrics names to log.
            output_transform (Callable, optional): a function to select what you want to plot from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
        """
        if metric_names is not None and not isinstance(metric_names, list):
            raise TypeError("metric_names should be a list, got {} instead".format(type(metric_names)))

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead"
                            .format(type(output_transform)))

        assert plot_event in (Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED), \
            "The plotting event should be either {} or {}".format(Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED)

        attach_id = len(self.metrics_step)
        self.metrics_step.append(0)

        engine.add_event_handler(
            plot_event,
            self._update,
            attach_id=attach_id,
            prefix=prefix,
            update_period=update_period,
            metric_names=metric_names,
            output_transform=output_transform
        )
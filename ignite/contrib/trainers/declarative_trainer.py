import copy
import numpy as np
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.mlflow_logger import (
    MLflowLogger,
    OutputHandler,
    global_step_from_engine,
)

import logging

log = logging.getLogger(__name__)

__all__ = ["NetworkTrain"]


class NetworkTrain:
    """Create a trainer for a supervised PyTorch model.

    Args:
        train_data_loader_params (dict): Parameters for data loader for training.
            Accepts args at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        val_data_loader_params (dict): Parameters for data loader for validation.
            Accepts args at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        epochs (int): Max epochs to train
        optimizer (torch.optim): Optimizer used to train.
            Accepts optimizers at https://pytorch.org/docs/stable/optim.html
        optimizer_params (dict): Parameters for optimizer.
        loss_fn (callable): Loss function used to train.
            Accepts an instance of loss functions at https://pytorch.org/docs/stable/nn.html#loss-functions
        evaluation_metrics (dict, optional): Metrics to compute for evaluation.
            Accepts dict of metrics at https://pytorch.org/ignite/metrics.html
        evaluate_train_data (str, optional): When to compute evaluation_metrics using training dataset.
            Accepts events at https://pytorch.org/ignite/engine.html#ignite.engine.Events
        evaluate_val_data (str, optional): When to compute evaluation_metrics using validation dataset.
            Accepts events at https://pytorch.org/ignite/engine.html#ignite.engine.Events
        progress_update (bool, optional): Whether to show progress bar using tqdm package
        scheduler (ignite.contrib.handle.param_scheduler.ParamScheduler, optional): Param scheduler
            Accepts a ParamScheduler at
            https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.param_scheduler
        scheduler_params (dict, optional): Parameters for scheduler
        model_checkpoint_params (dict, optional): Parameters for ModelCheckpoint at
            https://pytorch.org/ignite/handlers.html#ignite.handlers.ModelCheckpoint
        early_stopping_params (dict, optional): Parameters for EarlyStopping at
            https://pytorch.org/ignite/handlers.html#ignite.handlers.EarlyStopping
        time_limit (int, optioinal): Time limit for training in seconds.
        seed (int, optional): Random seed for training.
        mlflow_logging (bool, optional): If True and MLflow is installed, MLflow logging is enabled.

    Returns:
        trainer (callable): a callable to train a PyTorch model.

    """

    def __init__(
        self,
        train_data_loader_params=None,
        val_data_loader_params=None,
        epochs=None,
        optimizer=None,
        optimizer_params=None,
        loss_fn=None,
        evaluation_metrics=None,
        evaluate_train_data=None,
        evaluate_val_data=None,
        progress_update=None,
        scheduler=None,
        scheduler_params=None,
        model_checkpoint_params=None,
        early_stopping_params=None,
        time_limit=None,
        seed=None,
        mlflow_logging=True,  # type: bool
    ):
        self.train_params = dict(
            train_data_loader_params=train_data_loader_params,
            val_data_loader_params=val_data_loader_params,
            epochs=epochs,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            loss_fn=loss_fn,
            evaluation_metrics=evaluation_metrics,
            evaluate_train_data=evaluate_train_data,
            evaluate_val_data=evaluate_val_data,
            progress_update=progress_update,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            model_checkpoint_params=model_checkpoint_params,
            early_stopping_params=early_stopping_params,
            time_limit=time_limit,
            seed=seed,
        )
        self.mlflow_logging = mlflow_logging

    def __call__(self, model, train_dataset, val_dataset=None, parameters=None):
        """ Train a PyTorch model.

        Args:
            model (torch.nn.Module): PyTorch model to train.
            train_dataset (torch.utils.data.Dataset): Dataset used to train.
            val_dataset (torch.utils.data.Dataset, optional): Dataset used to validate.
            parameters (dict, optional) : Ignored.

        Returns:
            trained_model (torch.nn.Module): Trained PyTorch model.
        """
        assert train_dataset is not None
        train_params = self.train_params
        mlflow_logging = self.mlflow_logging

        if mlflow_logging:
            try:
                import mlflow  # NOQA
            except ImportError:
                log.warning("Failed to import mlflow. MLflow logging is disabled.")
                mlflow_logging = False

        train_dataset_size_limit = train_params.get("train_dataset_size_limit")
        val_dataset_size_limit = train_params.get("val_dataset_size_limit")
        if train_dataset_size_limit:
            train_dataset = PartialDataset(train_dataset, train_dataset_size_limit)
            log.info("train dataset size is set to {}".format(len(train_dataset)))

        if val_dataset_size_limit and (val_dataset is not None):
            val_dataset = PartialDataset(val_dataset, val_dataset_size_limit)
            log.info("val dataset size is set to {}".format(len(val_dataset)))

        train_data_loader_params = train_params.get("train_data_loader_params", dict())
        val_data_loader_params = train_params.get("val_data_loader_params", dict())
        epochs = train_params.get("epochs")
        progress_update = train_params.get("progress_update")

        optimizer = train_params.get("optimizer")
        assert optimizer
        optimizer_params = train_params.get("optimizer_params", dict())
        scheduler = train_params.get("scheduler")
        scheduler_params = train_params.get("scheduler_params", dict())
        loss_fn = train_params.get("loss_fn")
        assert loss_fn
        evaluation_metrics = train_params.get("evaluation_metrics")

        evaluate_train_data = train_params.get("evaluate_train_data")
        evaluate_val_data = train_params.get("evaluate_val_data")

        early_stopping_params = train_params.get("early_stopping_params")
        time_limit = train_params.get("time_limit")
        model_checkpoint_params = train_params.get("model_checkpoint_params")
        seed = train_params.get("seed")
        cudnn_deterministic = train_params.get("cudnn_deterministic")
        cudnn_benchmark = train_params.get("cudnn_benchmark")

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = cudnn_deterministic
        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = cudnn_benchmark

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        optimizer_ = optimizer(model.parameters(), **optimizer_params)
        trainer = create_supervised_trainer(model, optimizer_, loss_fn=loss_fn, device=device)

        train_data_loader_params.setdefault("shuffle", True)
        train_data_loader_params.setdefault("drop_last", True)
        train_data_loader_params["batch_size"] = _clip_batch_size(
            train_data_loader_params.get("batch_size", 1), train_dataset, "train"
        )
        train_loader = DataLoader(train_dataset, **train_data_loader_params)

        RunningAverage(output_transform=lambda x: x, alpha=0.98).attach(trainer, "ema_loss")

        RunningAverage(output_transform=lambda x: x, alpha=2 ** (-1022)).attach(trainer, "batch_loss")

        if scheduler:

            class ParamSchedulerSavingAsMetric(ParamSchedulerSavingAsMetricMixIn, scheduler):
                pass

            cycle_epochs = scheduler_params.pop("cycle_epochs", 1)
            scheduler_params.setdefault("cycle_size", int(cycle_epochs * len(train_loader)))
            scheduler_params.setdefault("param_name", "lr")
            scheduler_ = ParamSchedulerSavingAsMetric(optimizer_, **scheduler_params)
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_)

        if evaluate_train_data:
            evaluator_train = create_supervised_evaluator(model, metrics=evaluation_metrics, device=device)

        if evaluate_val_data:
            val_data_loader_params["batch_size"] = _clip_batch_size(
                val_data_loader_params.get("batch_size", 1), val_dataset, "val"
            )
            val_loader = DataLoader(val_dataset, **val_data_loader_params)
            evaluator_val = create_supervised_evaluator(model, metrics=evaluation_metrics, device=device)

        if model_checkpoint_params:
            assert isinstance(model_checkpoint_params, dict)
            minimize = model_checkpoint_params.pop("minimize", True)
            save_interval = model_checkpoint_params.get("save_interval", None)
            if not save_interval:
                model_checkpoint_params.setdefault("score_function", get_score_function("ema_loss", minimize=minimize))
            model_checkpoint_params.setdefault("score_name", "ema_loss")
            mc = ModelCheckpoint(**model_checkpoint_params)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, mc, {"model": model})

        if early_stopping_params:
            assert isinstance(early_stopping_params, dict)
            metric = early_stopping_params.pop("metric", None)
            assert (metric is None) or (metric in evaluation_metrics)
            minimize = early_stopping_params.pop("minimize", False)
            if metric:
                assert (
                    "score_function" not in early_stopping_params
                ), "Remove either 'metric' or 'score_function' from early_stopping_params: {}".format(
                    early_stopping_params
                )
                early_stopping_params["score_function"] = get_score_function(metric, minimize=minimize)

            es = EarlyStopping(trainer=trainer, **early_stopping_params)
            if evaluate_val_data:
                evaluator_val.add_event_handler(Events.COMPLETED, es)
            elif evaluate_train_data:
                evaluator_train.add_event_handler(Events.COMPLETED, es)
            elif early_stopping_params:
                log.warning(
                    "Early Stopping is disabled because neither "
                    "evaluate_val_data nor evaluate_train_data is set True."
                )

        if time_limit:
            assert isinstance(time_limit, (int, float))
            tl = TimeLimit(limit_sec=time_limit)
            trainer.add_event_handler(Events.ITERATION_COMPLETED, tl)

        pbar = None
        if progress_update:
            if not isinstance(progress_update, dict):
                progress_update = dict()
            progress_update.setdefault("persist", True)
            progress_update.setdefault("desc", "")
            pbar = ProgressBar(**progress_update)
            pbar.attach(trainer, ["ema_loss"])

        else:

            def log_train_metrics(engine):
                log.info("[Epoch: {} | {}]".format(engine.state.epoch, engine.state.metrics))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_metrics)

        if evaluate_train_data:

            def log_evaluation_train_data(engine):
                evaluator_train.run(train_loader)
                train_report = _get_report_str(engine, evaluator_train, "Train Data")
                if pbar:
                    pbar.log_message(train_report)
                else:
                    log.info(train_report)

            eval_train_event = (
                Events[evaluate_train_data] if isinstance(evaluate_train_data, str) else Events.EPOCH_COMPLETED
            )
            trainer.add_event_handler(eval_train_event, log_evaluation_train_data)

        if evaluate_val_data:

            def log_evaluation_val_data(engine):
                evaluator_val.run(val_loader)
                val_report = _get_report_str(engine, evaluator_val, "Val Data")
                if pbar:
                    pbar.log_message(val_report)
                else:
                    log.info(val_report)

            eval_val_event = Events[evaluate_val_data] if isinstance(evaluate_val_data, str) else Events.EPOCH_COMPLETED
            trainer.add_event_handler(eval_val_event, log_evaluation_val_data)

        if mlflow_logging:
            mlflow_logger = MLflowLogger()

            logging_params = {
                "train_n_samples": len(train_dataset),
                "train_n_batches": len(train_loader),
                "optimizer": _name(optimizer),
                "loss_fn": _name(loss_fn),
                "pytorch_version": torch.__version__,
                "ignite_version": ignite.__version__,
            }
            logging_params.update(_loggable_dict(optimizer_params, "optimizer"))
            logging_params.update(_loggable_dict(train_data_loader_params, "train"))
            if scheduler:
                logging_params.update({"scheduler": _name(scheduler)})
                logging_params.update(_loggable_dict(scheduler_params, "scheduler"))

            if evaluate_val_data:
                logging_params.update(
                    {"val_n_samples": len(val_dataset), "val_n_batches": len(val_loader),}
                )
                logging_params.update(_loggable_dict(val_data_loader_params, "val"))

            mlflow_logger.log_params(logging_params)

            batch_metric_names = ["batch_loss", "ema_loss"]
            if scheduler:
                batch_metric_names.append(scheduler_params.get("param_name"))

            mlflow_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="step", metric_names=batch_metric_names, global_step_transform=global_step_from_engine(trainer),
                ),
                event_name=Events.ITERATION_COMPLETED,
            )

            if evaluate_train_data:
                mlflow_logger.attach(
                    evaluator_train,
                    log_handler=OutputHandler(
                        tag="train",
                        metric_names=list(evaluation_metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.COMPLETED,
                )
            if evaluate_val_data:
                mlflow_logger.attach(
                    evaluator_val,
                    log_handler=OutputHandler(
                        tag="val",
                        metric_names=list(evaluation_metrics.keys()),
                        global_step_transform=global_step_from_engine(trainer),
                    ),
                    event_name=Events.COMPLETED,
                )

        trainer.run(train_loader, max_epochs=epochs)

        try:
            if pbar and pbar.pbar:
                pbar.pbar.close()
        except Exception as e:
            log.error(e, exc_info=True)

        model = load_latest_model(model_checkpoint_params)(model)

        return model


def get_score_function(metric, minimize=False):
    def _score_function(engine):
        m = engine.state.metrics.get(metric)
        return -m if minimize else m

    return _score_function


def load_latest_model(model_checkpoint_params=None):
    if model_checkpoint_params and "model_checkpoint_params" in model_checkpoint_params:
        model_checkpoint_params = model_checkpoint_params.get("model_checkpoint_params")

    def _load_latest_model(model=None):
        if model_checkpoint_params:
            try:
                dirname = model_checkpoint_params.get("dirname")
                assert dirname
                dir_glob = Path(dirname).glob("*.pth")
                files = [str(p) for p in dir_glob if p.is_file()]
                if len(files) >= 1:
                    model_path = sorted(files)[-1]
                    log.info("Model path: {}".format(model_path))
                    loaded = torch.load(model_path)
                    save_as_state_dict = model_checkpoint_params.get("save_as_state_dict", True)
                    if save_as_state_dict:
                        assert model
                        model.load_state_dict(loaded)
                    else:
                        model = loaded
                else:
                    log.warning("Model not found at: {}".format(dirname))
            except Exception as e:
                log.error(e, exc_info=True)
        return model

    return _load_latest_model


def _name(obj):
    return getattr(obj, "__name__", None) or getattr(obj.__class__, "__name__", "_")


def _clip_batch_size(batch_size, dataset, tag=""):
    dataset_size = len(dataset)
    if batch_size > dataset_size:
        log.warning("[{}] batch size ({}) is clipped to dataset size ({})".format(tag, batch_size, dataset_size))
        return dataset_size
    else:
        return batch_size


def _get_report_str(engine, evaluator, tag=""):
    report_str = "[Epoch: {} | {} | Metrics: {}]".format(engine.state.epoch, tag, evaluator.state.metrics)
    return report_str


def _loggable_dict(d, prefix=None):
    return {
        ("{}_{}".format(prefix, k) if prefix else k): ("{}".format(v) if isinstance(v, (tuple, list, dict, set)) else v)
        for k, v in d.items()
    }


class TimeLimit:
    def __init__(self, limit_sec=3600):
        self.limit_sec = limit_sec
        self.start_time = time.time()

    def __call__(self, engine):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            log.warning("Reached the time limit: {} sec. Stop training".format(self.limit_sec))
            engine.terminate()


class ParamSchedulerSavingAsMetricMixIn:
    """ Base code:
     https://github.com/pytorch/ignite/blob/v0.2.1/ignite/contrib/handlers/param_scheduler.py#L49
     https://github.com/pytorch/ignite/blob/v0.2.1/ignite/contrib/handlers/param_scheduler.py#L163
    """

    def __call__(self, engine, name=None):

        if self.event_index != 0 and self.event_index % self.cycle_size == 0:
            self.event_index = 0
            self.cycle_size *= self.cycle_mult
            self.cycle += 1
            self.start_value *= self.start_value_mult
            self.end_value *= self.end_value_mult

        value = self.get_param()

        for param_group in self.optimizer_param_groups:
            param_group[self.param_name] = value

        if name is None:
            name = self.param_name

        if self.save_history:
            if not hasattr(engine.state, "param_history"):
                setattr(engine.state, "param_history", {})
            engine.state.param_history.setdefault(name, [])
            values = [pg[self.param_name] for pg in self.optimizer_param_groups]
            engine.state.param_history[name].append(values)

        self.event_index += 1

        if not hasattr(engine.state, "metrics"):
            setattr(engine.state, "metrics", {})
        engine.state.metrics[self.param_name] = value  # Save as a metric


class PartialDataset:
    def __init__(self, dataset, size):
        size = int(size)
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "__len__")
        assert dataset.__len__() >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.dataset[item]


class CopiedPartialDataset:
    def __init__(self, dataset, size):
        size = int(size)
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "__len__")
        assert dataset.__len__() >= size
        self.dataset = [copy.deepcopy(dataset[i]) for i in range(size)]
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.dataset[item]


class GetPartialDataset:
    def __init__(self, size):
        self.size = size

    def __call__(self, dataset):
        return CopiedPartialDataset(dataset, self.size)

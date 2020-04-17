
import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler
from ignite.contrib.trainers.declarative_trainer import NetworkTrain


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


if __name__ == "__main__":

    train_params = {
        # loss_fn accepts a loss function at https://pytorch.org/docs/stable/nn.html#loss-functions
        "loss_fn": nn.NLLLoss(),
        # epochs accepts an integer
        "epochs": 2,
        # [Optional] seed (random seed) accepts an integer
        "seed": 42,
        # optimizer accepts optimizers at https://pytorch.org/docs/stable/optim.html
        "optimizer": SGD,
        # optimizer_params accepts parameters for the specified optimizer
        "optimizer_params": {"lr": 0.01, "momentum": 0.5},
        # train_data_loader_params accepts args at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        "train_data_loader_params": {"batch_size": 64, "num_workers": 0},
        # val_data_loader_params accepts args at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        "val_data_loader_params": {"batch_size": 1000, "num_workers": 0},
        # [Optional] evaluation_metrics accepts dict of metrics at https://pytorch.org/ignite/metrics.html
        "evaluation_metrics": {
            "accuracy": Accuracy(),
            "loss": Loss(loss_fn=nn.NLLLoss()),
        },
        # [Optional] evaluate_train_data (when to compute evaluation metrics using train_dataset)
        # accepts events at https://pytorch.org/ignite/engine.html#ignite.engine.Events
        "evaluate_train_data": "EPOCH_COMPLETED",
        # [Optional] evaluate_val_data (when to compute evaluation metrics using val_dataset)
        # accepts events at https://pytorch.org/ignite/engine.html#ignite.engine.Events
        "evaluate_val_data": "EPOCH_COMPLETED",
        # [Optional] progress_update (whether to show progress bar using tqdm package) accepts bool
        "progress_update": True,
        # [Optional] param scheduler at
        # https://pytorch.org/ignite/contrib/handlers.html#module-ignite.contrib.handlers.param_scheduler
        "scheduler": LinearCyclicalScheduler,
        # [Optional] scheduler_params accepts parameters for the specified scheduler
        "scheduler_params": {
            "param_name": "lr",
            "start_value": 0.001,
            "end_value": 0.01,
            "cycle_epochs": 2,
            "cycle_mult": 1.0,
            "start_value_mult": 1.0,
            "end_value_mult": 1.0,
            "save_history": False,
        },
        # [Optional] parameters for ModelCheckpoint at
        # https://pytorch.org/ignite/handlers.html#ignite.handlers.ModelCheckpoint
        "model_checkpoint_params": {
            "dirname": "../checkpoint",
            "filename_prefix": "model",
            "save_interval": None,
            "n_saved": 1,
            "atomic": True,
            "require_empty": False,
            "create_dir": True,
            "save_as_state_dict": True,
        },
        # [Optional] parameters for flexible version of EarlyStopping at
        # https://pytorch.org/ignite/handlers.html#ignite.handlers.EarlyStopping
        "early_stopping_params": {
            # metric (metric to monitor to determine whether to stop early) accepts str
            "metric": "loss",
            # minimize (if set to True, smaller metric value is considered better) accepts bool
            "minimize": True,
            # a parameter for ignite.handlers.EarlyStopping
            "patience": 1000,
            # a parameter for ignite.handlers.EarlyStopping
            "min_delta": 0.0,
            # a parameter for ignite.handlers.EarlyStopping
            "cumulative_delta": False,
        },
        # [Optional] time_limit (time limit for training in seconds) accepts an integer
        "time_limit": 3600,
        # [Optional] mlflow_logging: If True and MLflow is installed, MLflow logging is enabled.
        "mlflow_logging": False,
    }

    nn_train = NetworkTrain(**train_params)

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
    val_dataset = MNIST(download=False, root=".", transform=data_transform, train=False)

    initial_model = Net()

    trained_model = nn_train(initial_model, train_dataset, val_dataset)

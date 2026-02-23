import argparse
import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

import ray
from ray import tune
from ray.tune import Checkpoint, CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy

from utils import Net, get_data_loaders, get_test_loader, load_data


def train_with_ignite(config, data_dir=None, checkpoint_dir=None, num_epochs=10, num_workers=8):
    device = config.get("device", "cpu")
    net = Net(config["l1"], config["l2"])
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    start_epoch = 0
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_state = torch.load(checkpoint_path)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    train_loader, val_loader = get_data_loaders(config["batch_size"], data_dir, num_workers)

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device)

    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)

    state = {"epoch": start_epoch}

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_and_report(engine):
        state["epoch"] += 1
        epoch = state["epoch"]

        evaluator.run(val_loader)
        val_loss = evaluator.state.metrics["loss"]
        val_accuracy = evaluator.state.metrics["accuracy"]

        print(f"Epoch {epoch}: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}")

        with tempfile.TemporaryDirectory() as tmp_chkpt_dir:
            tmp_chkpt_path = os.path.join(tmp_chkpt_dir, "checkpoint.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                tmp_chkpt_path,
            )

            checkpoint = Checkpoint.from_directory(tmp_chkpt_dir)
            tune.report(
                {"loss": val_loss, "accuracy": val_accuracy},
                checkpoint=checkpoint,
            )

    if num_epochs - start_epoch > 0:
        trainer.run(train_loader, num_epochs - start_epoch)


def test_accuracy(model, device, data_dir):
    test_loader = get_test_loader(4, data_dir)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def tune_cifar(num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=2, data_dir="./data"):
    data_dir = os.path.abspath(data_dir)

    config = {
        "l1": tune.choice([2**i for i in range(2, 9)]),
        "l2": tune.choice([2**i for i in range(2, 9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "device": "cuda" if gpus_per_trial > 0 else "cpu",
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    print("Downloading CIFAR10 dataset...")

    load_data(data_dir)

    reporter = CLIReporter(
        metric_columns={"loss": "loss", "accuracy": "accuracy"},
        parameter_columns={"l1": "l1", "l2": "l2", "lr": "lr", "batch_size": "batch_size"},
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_with_ignite,
                data_dir=data_dir,
                num_epochs=num_epochs,
            ),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=ray.tune.RunConfig(
            storage_path=None,
            name="cifar10_ray_tune",
            progress_reporter=reporter,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result("accuracy", "max", filter_nan_and_inf=False)
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['accuracy']}")

    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = best_result.config["device"]
    best_trained_model = best_trained_model.to(device)

    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        best_checkpoint_data = torch.load(checkpoint_path, map_location=device)
        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

        test_acc = test_accuracy(best_trained_model, device, data_dir)
        print(f"Best trial test set accuracy: {test_acc}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ignite + Ray Tune CIFAR10 Example")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials for Ray Tune")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=0, help="GPUs per trial")
    parser.add_argument("--cpus_per_trial", type=int, default=2, help="CPUs per trial")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to CIFAR10 dataset",
    )
    args = parser.parse_args()

    ray.init(include_dashboard=False)
    tune_cifar(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        gpus_per_trial=args.gpus_per_trial,
        cpus_per_trial=args.cpus_per_trial,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

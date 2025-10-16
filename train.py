# train.py
import argparse
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from utils.display import (
    display_history,
    display_confusion_matrix_by_pred,
)
from utils.preprocessing import scale_data, add_noise_data
from utils.validate import validate_model
from utils.train import train, ensure_dir
from utils.results import count_parameters
from model_configs import get_model_config
from utils.callbacks import ModelCheckpoint
from experiment_configs import create_experiments, Experiment
import mlflow
import mlflow.pytorch

DEBUG = False

mlflow.autolog()


def _save_and_visualize_history(history, name, images_path, history_path):
    """Visualizes the training history and saves to MLFlow."""
    # Only create visualizations, don't save CSV files
    display_history(history, "acc", store=True, model_name=name, store_path=images_path)
    display_history(
        history, "loss", store=True, model_name=name, store_path=images_path
    )


def _log_to_mlflow(
    hyperparameters,
    history,
    validation_loss,
    model,
    experiment_name,
    figure,
    images_path,
):
    """Logs hyperparameters, metrics, and artifacts to MLFlow."""
    best_val_acc = np.max(history["val_acc"])

    # Log hyperparameters
    acceptable_hps = {}
    for key, value in hyperparameters.items():
        if (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, bool)
        ):
            acceptable_hps[key] = value
    mlflow.log_params(acceptable_hps)

    # Log metrics
    mlflow.log_metric("train_acc", float(np.max(history["train_acc"])))
    mlflow.log_metric("train_loss", float(np.min(history["train_loss"])))
    mlflow.log_metric("val_acc", float(best_val_acc))
    mlflow.log_metric("val_loss", float(validation_loss))
    mlflow.log_metric("num_params", count_parameters(model))

    # Log figure as artifact
    if figure is not None:
        confusion_matrix_path = f"{images_path}/confusion_matrix.png"
        figure.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path, artifact_path="images")

    # Log all images as artifacts
    if os.path.exists(images_path):
        for file in os.listdir(images_path):
            if file.endswith(".png"):
                mlflow.log_artifact(f"{images_path}/{file}", artifact_path="images")


def train_validate_model(
    model,
    train_loader,
    test_loader,
    name,
    optimizer,
    callbacks,
    epochs,
    device,
    hyperparameters,
    experiment_name,
    labels,
):
    timestamp = time.strftime("%Y-%m-%d_%H_%M_%S")
    checkpoint_path = f"checkpoints/{experiment_name}/{name}.pt"

    # Create temporary directory for images (will be logged to MLFlow)
    import tempfile

    temp_dir = tempfile.mkdtemp()
    images_path = temp_dir
    history_path = temp_dir

    # Prepare checkpoint directory
    ensure_dir(f"checkpoints/{experiment_name}")

    checkpoint_cb = ModelCheckpoint(model, checkpoint_path)
    callbacks.append(checkpoint_cb)

    # Set MLFlow experiment and start run
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=f"{name}_{timestamp}")

    # Log tags
    mlflow.set_tag("model_name", name)
    mlflow.set_tag("timestamp", timestamp)

    ## Validate Raw model
    model.eval()
    with torch.no_grad():
        y_pred, y_true, validation_loss, metrics = validate_model(
            model,
            test_loader,
            None,
            device,
            labels,
        )
        print(f"Raw model validation loss: {validation_loss}")
        print(f"Raw model validation metrics: {metrics}")

    ## Train the model
    history = train(
        model=model,
        epochs=epochs,
        dataloader_train=train_loader,
        dataloader_val=test_loader,
        optimizer=optimizer,
        callbacks=callbacks,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
    )
    _save_and_visualize_history(history, name, images_path, history_path)

    y_pred, y_true, validation_loss, metrics = validate_model(
        model,
        test_loader,
        checkpoint_path,
        device,
        labels,
    )

    print(f"Metrics: {metrics}")

    figure = display_confusion_matrix_by_pred(
        y_pred,
        y_true,
        labels=labels,
        save=True,
        normalize=True,
        model_name=name,
        decimas=6,
        store_path=images_path,
    )

    # Log to MLFlow
    _log_to_mlflow(
        hyperparameters,
        history,
        validation_loss,
        model,
        experiment_name,
        figure,
        images_path,
    )

    # Log model to MLFlow
    mlflow.pytorch.log_model(
        model.cpu(),
        name=f"{name}_model",
        input_example=np.random.randn(32, 1, 2048).astype(np.float32),
    )

    # Log model metrics as a JSON artifact
    mlflow.log_dict(metrics, "metrics.json")

    # Store run ID before ending
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    # End MLFlow run
    mlflow.end_run()

    # Clean up temporary directory
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "name": name,
        "validation_accuracy": np.max(history["val_acc"]) * 100,
        "validation_loss": validation_loss,
        "num_params": count_parameters(model),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "mlflow_run_id": run_id,
    }


def main(experiments: list[Experiment]):
    for experiment in experiments:
        if experiment.method == "cross_validation":
            run_cross_validation(experiment, DEBUG)
        else:
            run_train_validation(experiment, DEBUG)


def run_cross_validation(experiment: dict, debug: bool = False):
    raise NotImplementedError()


def _prepair_dataset(experiment: Experiment, debug: bool = False):
    train_dataset = experiment.get_train_dataset()
    test_dataset = experiment.get_test_dataset()

    if debug:
        X_train = train_dataset.inputs()[:100]
        y_train = train_dataset.targets()[:100]
        X_test = test_dataset.inputs()[:100]
        y_test = test_dataset.targets()[:100]
    else:
        X_train = train_dataset.inputs()
        y_train = train_dataset.targets()
        X_test = test_dataset.inputs()
        y_test = test_dataset.targets()

    if experiment.test_with_noise:
        X_test = add_noise_data(X_test, experiment.noise_level)
    if experiment.train_augmentation:
        X_train_noisy = add_noise_data(X_train, experiment.noise_level)
        X_train = np.vstack((X_train_noisy, X_train))
        y_train = np.hstack((y_train, y_train))
        indeces = np.arange(len(X_train))
        np.random.shuffle(indeces)
        X_train = X_train[indeces]
        y_train = y_train[indeces]
    elif experiment.train_with_noise:
        X_train = add_noise_data(X_train, experiment.noise_level)

    scale_data(X_train, X_test)

    del train_dataset, test_dataset

    train_set = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_set = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    del X_train, y_train, X_test, y_test

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    return train_loader, test_loader


def run_train_validation(experiment: Experiment, debug: bool = False):
    print(f"Running experiment: {experiment.name}")
    print(experiment.get_train_dataset().labels)
    num_classes = len(experiment.get_train_dataset().labels())
    model_configs = get_model_config(num_classes)

    experiment_name = experiment.name

    # Prepare checkpoint directory
    ensure_dir(f"checkpoints/{experiment_name}")

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = experiment.model
    cfg = model_configs[model_name]

    split_results = []
    for i in range(experiment.trials):
        model = cfg["model"](**cfg["hyperparameters"]).to(device)
        optimizer = cfg["optimizer"](model)
        callbacks = []
        # Create scheduler callbacks based on config
        if "callbacks" in cfg:
            callbacks = cfg["callbacks"](optimizer)
        print(
            f"Training {model_name}, number of parameters: {count_parameters(model):,}"
        )

        train_loader, test_loader = _prepair_dataset(experiment, debug)

        result_split = train_validate_model(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            name=model_name,
            optimizer=optimizer,
            callbacks=callbacks,
            epochs=cfg["epochs"],
            device=device,
            hyperparameters=cfg.get("hyperparameters", {}),  # Pass hyperparameters
            experiment_name=f"{experiment_name}_{i}",
            labels=experiment.get_train_dataset().labels(),
        )
        split_results.append(result_split)

    result = {
        "name": model_name,
        "validation_accuracy": np.mean(
            [res["validation_accuracy"] for res in split_results]
        ),
        "validation_loss": np.mean([res["validation_loss"] for res in split_results]),
        "num_params": count_parameters(model),
        "details": split_results,
    }

    # Results are already logged to MLFlow in train_validate_model
    print(f"Training complete for experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Average validation accuracy: {result['validation_accuracy']:.2f}%")
    print(f"Average validation loss: {result['validation_loss']:.4f}")
    print(f"Results logged to MLFlow experiment: {experiment_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bearing fault detection models")
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=["transformer_encoder_classifier"],
        help="List of models to train",
    )
    parser.add_argument(
        "--test_with_noise",
        action="store_true",
        help="Use noisy data for testing",
    )
    parser.add_argument(
        "--train_augmentation",
        action="store_true",
        help="Add noisy data to training set",
    )
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[0.0],
        help="List of noise levels in snr_dB",
    )
    parser.add_argument(
        "--scoring_method",
        default="train_validation",
        choices=["train_validation", "cross_validation"],
        help="Scoring method",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of random splits",
    )
    parser.add_argument(
        "--dataset",
        choices=["CWRU", "PU"],
        default="CWRU",
        help="Dataset to use",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--train_with_noise",
        action="store_true",
        help="Use noisy data for training",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=2048,
        help="Window size for the dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=0,
        help="Debug mode: When enabled, only subset of the data is used. Default is False",
    )

    args = parser.parse_args()
    DEBUG = args.debug if args.debug else False
    print(f"Debug mode: {DEBUG}")
    experiments = create_experiments(
        name=args.experiment,
        window_size=args.window_size,
        test_with_noise=args.test_with_noise,
        train_with_noise=args.train_with_noise,
        train_augmentation=args.train_augmentation,
        method=args.scoring_method,
        noise_levels=args.noise_levels,
        models=args.models,
        trials=args.trials,
        dataset_name=args.dataset,
    )
    main(experiments)

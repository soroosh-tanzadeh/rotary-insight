# train.py
import argparse
from datetime import datetime
import os
import torch
import shutil
import time
import numpy as np
from torch.utils.data import DataLoader
from datasets import CWRU, PU_DatasetProcessor, PU_Dataset
from utils.display import (
    display_history,
    display_confusion_matrix_by_pred,
)
from utils.validate import validate_model
from utils.train import train, ensure_dir
from model_configs import get_model_config
from utils.callbacks import ModelCheckpoint
from experiment_configs import create_experiments
import mlflow
import mlflow.pytorch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_seed = 42

DEBUG = False

mlflow.autolog()

# Define constants and labels
CWRU_LABELS = [
    "Normal",
    "0.007-Ball",
    "0.014-Ball",
    "0.021-Ball",
    "0.007-InnerRace",
    "0.014-InnerRace",
    "0.021-InnerRace",
    "0.007-OuterRace",
    "0.014-OuterRace",
    "0.021-OuterRace",
]

PU_LABELS = ["Healthy", "InnerRace", "OuterRace"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    dataset_name,
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
    mlflow.set_tag("dataset", dataset_name)
    mlflow.set_tag("timestamp", timestamp)

    labels = CWRU_LABELS if dataset_name == "CWRU" else PU_LABELS

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
    mlflow.pytorch.log_model(model, "model")

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


def main(experiments: list[dict]):
    # Get model configurations
    i = 0
    for experiment in experiments:
        if experiment["scoring_method"] == "cross_validation":
            run_cross_validation(i, experiment)
        else:
            run_train_validation(i, experiment, DEBUG)
        i += 1


def add_noise_data(signal, noise_level):
    """
    Adds Gaussian noise to the signal based on the specified noise level.

    Args:
        signal (np.ndarray): The input signal.
        noise_level (float): The noise level in dB.

    Returns:
        np.ndarray: The noisy signal.
    """
    snr = 10 ** (noise_level / 10)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    print(f"Adding noise with SNR: {snr:.2f} (level: {noise_level} dB)")

    return signal + noise


def scale_data(train_data, test_data):
    """
    Scales the training and testing data using Min-Max scaling.

    Args:
        train_data (np.ndarray): The training data.
        test_data (np.ndarray): The testing data.

    Returns:
        tuple: Scaled training and testing data.
    """
    scaler = StandardScaler()
    num_samples, num_channels, signal_length = train_data.shape
    train_data_reshaped = train_data.reshape(num_samples, -1)
    test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

    scaler.fit(train_data_reshaped)
    train_data_scaled = scaler.transform(train_data_reshaped).reshape(
        num_samples, num_channels, signal_length
    )
    test_data_scaled = scaler.transform(test_data_reshaped).reshape(
        test_data.shape[0], num_channels, signal_length
    )

    return train_data_scaled, test_data_scaled


def run_cross_validation(i, experiment: dict):
    # Existing implementation uses CWRU only; leaving unchanged
    num_classes = 10
    model_configs = get_model_config(num_classes)

    print(f"Running experiment: {experiment['name']}")
    experiment_name = experiment["name"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset_for_cross_validation(experiment)
    dataset.X = dataset.X[:, 1, :].view(dataset.X.shape[0], 1, -1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        fold_experiment_name = f"{experiment['name']}_fold_{fold + 1}"
        ensure_dir(f"checkpoints/{fold_experiment_name}")

        X_train, X_test = dataset.X[train_idx], dataset.X[val_idx]
        y_train, y_test = dataset.y[train_idx], dataset.y[val_idx]

        X_train, X_test = scale_data(X_train, X_test)

        print(f"Fold {fold + 1}/{kfold.n_splits}")
        if experiment["test_with_noise"]:
            print("Adding noise to test data", X_test.shape)
            test_subset = add_noise_data(
                X_test, experiment["test_noise_config"]["snr_dB"]
            )

            test_set = torch.utils.data.TensorDataset(
                torch.tensor(test_subset, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )
        else:
            test_set = torch.utils.data.TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )

        with_augmentation = experiment["train_augmentation"]
        if with_augmentation:
            train_subset = add_noise_data(
                X_train, experiment["train_noise_config"]["snr_dB"]
            )
            train_subset = np.vstack((train_subset, X_train))
            y_train_subset = np.hstack((y_train, y_train))
            indeces = np.arange(len(train_subset))
            np.random.shuffle(indeces)
            train_subset = train_subset[indeces]
            y_train_subset = y_train_subset[indeces]

            train_set = torch.utils.data.TensorDataset(
                torch.tensor(train_subset, dtype=torch.float32),
                torch.tensor(y_train_subset, dtype=torch.long),
            )
        else:
            train_set = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

        model_name = experiment["model_name"]
        cfg = model_configs[model_name]
        model = cfg["model"](**cfg["hyperparameters"]).to(device)
        optimizer = cfg["optimizer"](model)

        callbacks = []
        # Create scheduler callbacks based on config
        if "callbacks" in cfg:
            callbacks = cfg["callbacks"](optimizer)

        print(
            f"Training {model_name}, number of parameters: {count_parameters(model):,}"
        )
        result = train_validate_model(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            name=model_name,
            optimizer=optimizer,
            callbacks=callbacks,
            epochs=cfg["epochs"],
            device=device,
            hyperparameters=cfg.get("hyperparameters", {}),  # Pass hyperparameters
            experiment_name=fold_experiment_name,
            dataset_name=experiment["dataset"],
        )
        if not (model_name in results):
            results[model_name] = []
        results[model_name].append(result)

        fold += 1

    # Results are already logged to MLFlow in train_validate_model
    print(f"Cross-validation complete for experiment: {experiment_name}")
    print(f"Results logged to MLFlow experiment: {experiment_name}")


def prepare_cwru_data(
    experiment: dict, debug: bool = False
) -> tuple[DataLoader, DataLoader]:
    data_processor = load_cwru_dataset(experiment)
    data_processor.X = data_processor.X[:, 1, :].reshape(
        data_processor.X.shape[0], 1, -1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        data_processor.X, data_processor.y, test_size=0.2
    )

    if debug:
        X_train = X_train[:100]
        X_test = X_test[:100]
        y_train = y_train[:100]
        y_test = y_test[:100]

    X_train, X_test = scale_data(X_train, X_test)

    if experiment["test_with_noise"]:
        X_test = add_noise_data(X_test, experiment["test_noise_config"]["snr_dB"])
    with_augmentation = experiment["train_augmentation"]
    if with_augmentation:
        X_train_noisy = add_noise_data(
            X_train, experiment["train_noise_config"]["snr_dB"]
        )
        X_train = np.vstack((X_train_noisy, X_train))
        y_train = np.hstack((y_train, y_train))
        indeces = np.arange(len(X_train))
        np.random.shuffle(indeces)
        X_train = X_train[indeces]
        y_train = y_train[indeces]

    train_set = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_set = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    return train_loader, test_loader


def prepare_pu_data(
    experiment: dict, trial_count: int = 1, debug: bool = False
) -> tuple[DataLoader, DataLoader]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_processor = PU_DatasetProcessor(
        rdir="./data/dataset/PU",
        seed=train_seed,
        force_reload=False,
        resplit_train_test=True,
    )
    train_set = PU_Dataset(
        f"./data/dataset/PU/processed/data_train_{trial_count}",
        data_processor.data_train,
    )
    test_set = PU_Dataset(
        f"./data/dataset/PU/processed/data_test_{trial_count}",
        data_processor.data_test,
    )

    if debug:
        train_set = PU_Dataset(
            f"./data/dataset/PU/processed/data_train_{trial_count}_debug",
            data_processor.data_train[:10],
        )
        test_set = PU_Dataset(
            f"./data/dataset/PU/processed/data_test_{trial_count}_debug",
            data_processor.data_test[:10],
        )
        print(f"Debug mode: {debug}")
        print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    train_set.X, test_set.X = scale_data(train_set.X, test_set.X)
    train_set.X = torch.from_numpy(train_set.X).to(device)
    test_set.X = torch.from_numpy(test_set.X).to(device)

    if experiment["test_with_noise"]:
        test_set = add_noise_data(test_set, experiment["test_noise_config"]["snr_dB"])
    if experiment["train_augmentation"]:
        X_train = train_set.X
        y_train = train_set.y
        train_set = add_noise_data(
            train_set, experiment["train_noise_config"]["snr_dB"]
        )
        train_set = np.vstack((X_train, train_set))
        y_train = np.hstack((y_train, y_train))
        indeces = np.arange(len(train_set))
        np.random.shuffle(indeces)
        train_set.X = torch.from_numpy(X_train[indeces])
        train_set.y = torch.from_numpy(y_train[indeces])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    return train_loader, test_loader


def run_train_validation(i, experiment: dict, debug: bool = False):
    dataset_name = experiment.get("dataset", "CWRU")
    num_classes = 0
    if dataset_name == "PU":
        num_classes = 3
    else:
        num_classes = 10

    model_configs = get_model_config(num_classes)

    experiment_name = experiment["name"]
    # Prepare checkpoint directory
    ensure_dir(f"checkpoints/{experiment_name}")

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []

    model_name = experiment["model_name"]
    cfg = model_configs[model_name]

    split_results = []
    for i in range(experiment["number_of_random_splits"]):
        model = cfg["model"](**cfg["hyperparameters"]).to(device)
        optimizer = cfg["optimizer"](model)
        callbacks = []
        # Create scheduler callbacks based on config
        if "callbacks" in cfg:
            callbacks = cfg["callbacks"](optimizer)
        print(
            f"Training {model_name}, number of parameters: {count_parameters(model):,}"
        )

        if dataset_name == "PU":
            train_loader, test_loader = prepare_pu_data(experiment, i, debug)
        else:
            train_loader, test_loader = prepare_cwru_data(experiment, debug)

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
            dataset_name=experiment["dataset"],
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

    results.append(result)

    # Results are already logged to MLFlow in train_validate_model
    print(f"Training complete for experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Average validation accuracy: {result['validation_accuracy']:.2f}%")
    print(f"Average validation loss: {result['validation_loss']:.4f}")
    print(f"Results logged to MLFlow experiment: {experiment_name}")


def load_dataset_for_cross_validation(experiment: dict):
    return CWRU(
        rdir="./data/dataset/CWRU/",
        exps=experiment["exps"],
        rpms=experiment["rpms"],
        split_ratio=0.0,
        split="train",
        seed=44,
    )


def load_cwru_dataset(experiment: dict):
    """
    Loads train and test datasets based on experiment configuration.

    Args:
        experiment (dict): Experiment configuration

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    return CWRU(
        rdir="./data/dataset/CWRU/",
        exps=experiment["exps"],
        rpms=experiment["rpms"],
        split_ratio=0.2,
        split="train",
        seed=None,
    )


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
        test_with_noise=args.test_with_noise,
        train_augmentation=args.train_augmentation,
        scoring_method=args.scoring_method,
        noise_levels=args.noise_levels,
        models=args.models,
        number_of_random_splits=args.trials,
    )
    # Attach dataset selection to each experiment
    for exp in experiments:
        exp["dataset"] = args.dataset
    main(experiments)

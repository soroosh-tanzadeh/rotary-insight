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
from torch.utils.data import random_split
from preprocessings import AddNoise, StandardScaler, TrainAugmentation
from utils.validate import validate_model
from utils.train import train, ensure_dir
from utils.results import count_parameters
from model_configs import get_model_config
from utils.callbacks import ModelCheckpoint
from experiment_configs import create_experiments, Experiment
import dotenv
import mlflow
import mlflow.pytorch
from datasets.dataset import SimpleBearingDataset

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
    model_name,
    optimizer,
    callbacks,
    epochs,
    device,
    hyperparameters,
    experiment: Experiment,
    trial_number: int,
    labels,
):
    experiment_name = experiment.name
    run_name = f"{experiment_name}__{model_name}__{trial_number}"
    timestamp = time.strftime("%Y-%m-%d_%H_%M_%S")
    checkpoint_path = f"checkpoints/{experiment_name}/{model_name}.pt"

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
    s3_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    print(f"MLFlow S3 endpoint URL: {s3_url}")
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    if mlflow_experiment is None:
        mlflow_experiment = mlflow.create_experiment(
            name=experiment_name,
            # artifact_location=s3_url,
        )
        mlflow_experiment = mlflow.set_experiment(experiment_id=mlflow_experiment)
    print(f"Running run")

    with mlflow.start_run(
        run_name=run_name, experiment_id=mlflow_experiment.experiment_id
    ) as run:
        print(f"Running run: {run.info.run_id}")
        print(f"Running experiment: {mlflow_experiment.experiment_id}")
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
        _save_and_visualize_history(history, model_name, images_path, history_path)

        y_pred, y_true, validation_loss, metrics = validate_model(
            model,
            test_loader,
            checkpoint_path,
            device,
            labels,
        )

        figure = display_confusion_matrix_by_pred(
            y_pred,
            y_true,
            labels=labels,
            save=True,
            normalize=True,
            model_name=model_name,
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
            name=f"{model_name}_model",
            input_example=np.random.randn(32, 1, 2048).astype(np.float32),
        )

        # Log model metrics as a JSON artifact
        mlflow.log_dict(metrics, "metrics.json")

        # Store run ID before ending
        run_id = run.info.run_id

        # Clean up temporary directory
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "name": model_name,
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
    dataset = experiment.get_dataset()
    splits = random_split(dataset, [0.8, 0.2])
    train_dataset = dataset[splits[0].indices]
    test_dataset = dataset[splits[1].indices]

    if debug:
        train_dataset = train_dataset[:100]
        test_dataset = test_dataset[:100]

    train_dataset = SimpleBearingDataset(
        X=train_dataset[0],
        y=train_dataset[1],
        labels=dataset.labels(),
        window_size=dataset.window_size(),
    )
    test_dataset = SimpleBearingDataset(
        X=test_dataset[0],
        y=test_dataset[1],
        labels=dataset.labels(),
        window_size=dataset.window_size(),
    )

    print(f"Train dataset: {train_dataset.inputs().shape}")
    print(f"Test dataset: {test_dataset.inputs().shape}")

    preprocessings = []
    if experiment.train_augmentation:
        preprocessings.append(
            TrainAugmentation(train_dataset, test_dataset, experiment.noise_level)
        )
    if experiment.test_with_noise or experiment.train_with_noise:
        preprocessings.append(
            AddNoise(
                train_dataset,
                test_dataset,
                experiment.noise_level,
                to_training=experiment.train_with_noise,
                to_test=experiment.test_with_noise,
            )
        )

    for preprocessing in preprocessings:
        train_dataset, test_dataset = preprocessing.preprocess()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def run_train_validation(experiment: Experiment, debug: bool = False):
    print(f"Running experiment: {experiment.name}")
    num_classes = len(experiment.get_dataset().labels())
    window_size = experiment.get_dataset().window_size()
    model_configs = get_model_config(num_classes, window_size)

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
            experiment=experiment,
            trial_number=i,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            model_name=model_name,
            optimizer=optimizer,
            callbacks=callbacks,
            epochs=cfg["epochs"],
            device=device,
            hyperparameters=cfg.get("hyperparameters", {}),  # Pass hyperparameters
            labels=experiment.get_dataset().labels(),
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

    dotenv.load_dotenv()

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

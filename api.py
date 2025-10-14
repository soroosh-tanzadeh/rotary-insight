"""
API module for loading models from MLFlow and performing inference.
"""

import torch
import mlflow
import mlflow.pytorch
import numpy as np
from typing import Union, List, Dict, Any
import pandas as pd


class ModelInference:
    """
    Class for loading models from MLFlow and performing inference.
    """

    def __init__(
        self, experiment_name: str = None, run_id: str = None, model_uri: str = None
    ):
        """
        Initialize the inference class.

        Args:
            experiment_name: Name of the MLFlow experiment
            run_id: Specific run ID to load model from
            model_uri: Direct URI to the model (e.g., "runs:/<run_id>/model")
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.run_info = None

        if model_uri:
            self._load_model_from_uri(model_uri)
        elif experiment_name and run_id:
            self._load_model_from_run(experiment_name, run_id)
        elif experiment_name:
            self._load_best_model_from_experiment(experiment_name)

    def _load_model_from_uri(self, model_uri: str):
        """Load model directly from a URI."""
        print(f"Loading model from URI: {model_uri}")
        self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
        self.model.eval()
        print("Model loaded successfully")

    def _load_model_from_run(self, experiment_name: str, run_id: str):
        """Load model from a specific run."""
        model_uri = f"runs:/{run_id}/model"
        self._load_model_from_uri(model_uri)

        # Get run info
        client = mlflow.tracking.MlflowClient()
        self.run_info = client.get_run(run_id)
        print(f"Loaded model from run: {run_id}")
        print(f"Model metrics: {self.run_info.data.metrics}")

    def _load_best_model_from_experiment(self, experiment_name: str):
        """
        Load the best model from an experiment based on validation accuracy.

        Args:
            experiment_name: Name of the MLFlow experiment
        """
        client = mlflow.tracking.MlflowClient()

        # Get experiment by name
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search for runs in this experiment, sorted by val_acc descending
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_acc DESC"],
            max_results=1,
        )

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

        best_run = runs[0]
        self.run_info = best_run

        print(f"Best run ID: {best_run.info.run_id}")
        print(f"Validation accuracy: {best_run.data.metrics.get('val_acc', 'N/A')}")
        print(f"Model: {best_run.data.tags.get('model_name', 'N/A')}")

        # Load the model
        model_uri = f"runs:/{best_run.info.run_id}/model"
        self._load_model_from_uri(model_uri)

    def predict(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Perform inference on input data.

        Args:
            data: Input data as numpy array or torch tensor
                  Shape should be (batch_size, channels, signal_length)

        Returns:
            Predicted class labels as numpy array
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Initialize with experiment_name or model_uri."
            )

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Ensure data is on the correct device
        data = data.to(self.device)

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy()

    def predict_proba(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get prediction probabilities for input data.

        Args:
            data: Input data as numpy array or torch tensor
                  Shape should be (batch_size, channels, signal_length)

        Returns:
            Class probabilities as numpy array
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Initialize with experiment_name or model_uri."
            )

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Ensure data is on the correct device
        data = data.to(self.device)

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded run.

        Returns:
            Dictionary with run information
        """
        if self.run_info is None:
            return {}

        return {
            "run_id": self.run_info.info.run_id,
            "experiment_id": self.run_info.info.experiment_id,
            "status": self.run_info.info.status,
            "metrics": dict(self.run_info.data.metrics),
            "params": dict(self.run_info.data.params),
            "tags": dict(self.run_info.data.tags),
        }


def list_experiments() -> pd.DataFrame:
    """
    List all MLFlow experiments.

    Returns:
        DataFrame with experiment information
    """
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    exp_data = []
    for exp in experiments:
        exp_data.append(
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
            }
        )

    return pd.DataFrame(exp_data)


def list_runs(experiment_name: str, max_results: int = 10) -> pd.DataFrame:
    """
    List runs from an experiment.

    Args:
        experiment_name: Name of the experiment
        max_results: Maximum number of runs to return

    Returns:
        DataFrame with run information
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_acc DESC"],
        max_results=max_results,
    )

    run_data = []
    for run in runs:
        run_data.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "model_name": run.data.tags.get("model_name", "N/A"),
                "dataset": run.data.tags.get("dataset", "N/A"),
                "val_acc": run.data.metrics.get("val_acc", None),
                "val_loss": run.data.metrics.get("val_loss", None),
                "train_acc": run.data.metrics.get("train_acc", None),
                "num_params": run.data.metrics.get("num_params", None),
                "start_time": run.info.start_time,
                "status": run.info.status,
            }
        )

    return pd.DataFrame(run_data)


def get_best_model(experiment_name: str, metric: str = "val_acc") -> ModelInference:
    """
    Get the best model from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to use for selecting best model (default: "val_acc")

    Returns:
        ModelInference instance with the best model loaded
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Determine sort order (descending for accuracy, ascending for loss)
    order = "DESC" if "acc" in metric.lower() else "ASC"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]
    print(f"Loading best model based on {metric}")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"{metric}: {best_run.data.metrics.get(metric, 'N/A')}")

    return ModelInference(experiment_name=experiment_name, run_id=best_run.info.run_id)


def compare_runs(experiment_name: str, run_ids: List[str]) -> pd.DataFrame:
    """
    Compare multiple runs from an experiment.

    Args:
        experiment_name: Name of the experiment
        run_ids: List of run IDs to compare

    Returns:
        DataFrame comparing the runs
    """
    client = mlflow.tracking.MlflowClient()

    comparison_data = []
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            comparison_data.append(
                {
                    "run_id": run.info.run_id,
                    "model_name": run.data.tags.get("model_name", "N/A"),
                    "val_acc": run.data.metrics.get("val_acc", None),
                    "val_loss": run.data.metrics.get("val_loss", None),
                    "train_acc": run.data.metrics.get("train_acc", None),
                    "train_loss": run.data.metrics.get("train_loss", None),
                    "num_params": run.data.metrics.get("num_params", None),
                }
            )
        except Exception as e:
            print(f"Error loading run {run_id}: {e}")

    return pd.DataFrame(comparison_data)


# Example usage
if __name__ == "__main__":
    # Example 1: List all experiments
    print("Available experiments:")
    print(list_experiments())

    # Example 2: Load best model from an experiment
    # inference = ModelInference(experiment_name="my_experiment")

    # Example 3: Load model from specific run
    # inference = ModelInference(run_id="abc123")

    # Example 4: Perform inference
    # dummy_data = np.random.randn(1, 1, 2048)  # (batch_size, channels, signal_length)
    # predictions = inference.predict(dummy_data)
    # probabilities = inference.predict_proba(dummy_data)

    # Example 5: List runs from an experiment
    # runs_df = list_runs("my_experiment", max_results=10)
    # print(runs_df)

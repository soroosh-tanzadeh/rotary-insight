"""
Model loading and inference logic.
"""

import torch
import mlflow
import mlflow.pytorch
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

model_manager = None  # Global model manager instance

# Class names for CWRU dataset
CWRU_CLASS_NAMES = [
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

# Class names for PU dataset
PU_CLASS_NAMES = [
    "Healthy",
    "OuterRace",
    "InnerRace",
]


class ModelManager:
    """Manages multiple models for inference."""

    def __init__(self, config_path: str, mlflow_uri: str):
        """
        Initialize the model manager.

        Args:
            config_path: Path to the model configuration JSON file
            mlflow_uri: MLflow tracking URI
        """
        self.config_path = config_path
        self.mlflow_uri = mlflow_uri
        self.models: Dict[str, torch.nn.Module] = {}
        self.configs: Dict[str, dict] = {}

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load model configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Model configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.configs = config.get("models", {})

        if not self.configs:
            raise ValueError("No models found in configuration file")

        print(f"Loaded configuration for {len(self.configs)} model(s)")

    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Load a model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded PyTorch model

        Raises:
            ValueError: If model name is not found in configuration
            Exception: If model loading fails
        """
        if model_name not in self.configs:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {list(self.configs.keys())}"
            )

        # Check if already loaded
        if model_name in self.models:
            print(f"Model '{model_name}' already loaded")
            return self.models[model_name]

        config = self.configs[model_name]
        model_path = config["path"]

        print(f"Loading model '{model_name}' from '{model_path}'...")

        try:
            # Load model from MLflow
            if config["type"] == "pytorch":
                model = mlflow.pytorch.load_model(model_path)
                model.eval()  # Set to evaluation mode

                # Move to CPU (or GPU if available)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                self.models[model_name] = model
                print(f"Successfully loaded model '{model_name}' on {device}")
                return model
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")

        except Exception as e:
            raise Exception(f"Failed to load model '{model_name}': {str(e)}")

    def load_all_models(self):
        """Load all models defined in the configuration."""
        for model_name in self.configs.keys():
            try:
                self.load_model(model_name)
            except Exception as e:
                print(f"Warning: Failed to load model '{model_name}': {e}")

    def get_model(self, model_name: str) -> torch.nn.Module:
        """
        Get a loaded model or load it if not already loaded.

        Args:
            model_name: Name of the model

        Returns:
            The model
        """
        if model_name not in self.models:
            return self.load_model(model_name)
        return self.models[model_name]

    def get_model_config(self, model_name: str) -> dict:
        """
        Get configuration for a model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dictionary
        """
        if model_name not in self.configs:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return self.configs[model_name]

    def get_class_names(self, model_name: str) -> List[str]:
        """
        Get class names for a model based on its dataset.

        Args:
            model_name: Name of the model

        Returns:
            List of class names
        """
        config = self.get_model_config(model_name)
        dataset_name = config.get("dataset_name", "").upper()

        if dataset_name == "CWRU":
            return CWRU_CLASS_NAMES
        elif dataset_name == "PU":
            return PU_CLASS_NAMES
        else:
            # Default: return generic class names
            num_classes = config.get("num_classes", 10)
            return [f"Class_{i}" for i in range(num_classes)]

    def predict(
        self, model_name: str, data: np.ndarray, return_probabilities: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform inference on input data.

        Args:
            model_name: Name of the model to use
            data: Input data with shape (batch_size, channels, signal_length)
            return_probabilities: Whether to return class probabilities

        Returns:
            Tuple of (predictions, probabilities)
            - predictions: Predicted class indices (batch_size,)
            - probabilities: Class probabilities (batch_size, num_classes) or None
        """
        model = self.get_model(model_name)
        config = self.get_model_config(model_name)

        # Validate input shape
        expected_window_size = config["window_size"]
        if data.shape[2] != expected_window_size:
            raise ValueError(
                f"Invalid input shape. Expected signal length {expected_window_size}, "
                f"got {data.shape[2]}"
            )

        # Convert to tensor
        device = next(model.parameters()).device
        x = torch.tensor(data, dtype=torch.float32).to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(x)

            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # Get probabilities if requested
            probabilities = None
            if return_probabilities:
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        return predictions, probabilities

    def list_models(self) -> Dict[str, dict]:
        """
        List all available models with their configurations.

        Returns:
            Dictionary mapping model names to their info
        """
        models_info = {}
        for name, config in self.configs.items():
            models_info[name] = {
                "name": name,
                "type": config.get("type", "unknown"),
                "window_size": config.get("window_size", 0),
                "dataset_name": config.get("dataset_name", "unknown"),
                "task": config.get("task", "unknown"),
                "num_classes": len(self.get_class_names(name)),
                "class_names": self.get_class_names(name),
                "loaded": name in self.models,
            }
        return models_info

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            print(f"Unloaded model '{model_name}'")


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global model_manager
    return model_manager


def setup_model_manager(config_path: str, mlflow_uri: str):
    """Set up the global model manager instance."""
    global model_manager
    model_manager = ModelManager(config_path, mlflow_uri)
    model_manager.load_all_models()

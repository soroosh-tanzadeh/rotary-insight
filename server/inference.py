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
        stored_in = config.get("stored_in", "mlflow")  # Default to mlflow for backward compatibility

        print(f"Loading model '{model_name}' from '{model_path}' (stored_in: {stored_in})...")

        try:
            if config["type"] == "pytorch":
                # Determine device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Load model based on storage type
                if stored_in == "file":
                    # Load model from local file
                    model = torch.load(model_path, map_location=device)
                    model.eval()  # Set to evaluation mode
                    print(f"Successfully loaded model '{model_name}' from file on {device}")
                elif stored_in == "mlflow":
                    # Load model from MLflow
                    model = mlflow.pytorch.load_model(model_path)
                    model.eval()  # Set to evaluation mode
                    model = model.to(device)
                    print(f"Successfully loaded model '{model_name}' from MLflow on {device}")
                else:
                    raise ValueError(f"Unsupported stored_in value: {stored_in}. Must be 'file' or 'mlflow'")

                self.models[model_name] = model
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
            num_classes = config.get("num_classes", 3)
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

    def list_models(self, window_size: Optional[int] = None) -> Dict[str, dict]:
        """
        List all available models with their configurations.

        Args:
            window_size: Optional filter by window size

        Returns:
            Dictionary mapping model names to their info
        """
        models_info = {}
        for name, config in self.configs.items():
            model_window_size = config.get("window_size", 0)
            
            # Filter by window size if provided
            if window_size is not None and model_window_size != window_size:
                continue
                
            models_info[name] = {
                "name": config.get("name", name),
                "type": config.get("type", "unknown"),
                "window_size": model_window_size,
                "dataset_name": config.get("dataset_name", "unknown"),
                "task": config.get("task", "unknown"),
                "num_classes": len(self.get_class_names(name)),
                "class_names": self.get_class_names(name),
                "loaded": name in self.models,
            }
        return models_info

    def get_available_window_sizes(self) -> List[int]:
        """
        Get a list of all available window sizes across all models.

        Returns:
            List of unique window sizes, sorted in ascending order.
        """
        window_sizes = set()
        for config in self.configs.values():
            if "window_size" in config:
                window_sizes.add(config["window_size"])
        return sorted(list(window_sizes))

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

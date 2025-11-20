# Developer Guide for ML Engineers

This guide is designed to help Machine Learning Engineers understand the codebase, add new models, and run experiments effectively.

## Project Structure

```
rotary-insight/
├── models/               # Neural network architectures
│   ├── transformer.py    # Transformer-based models
│   ├── embedding.py      # Embedding layers
│   └── positional_encoder.py
├── datasets/             # Dataset loaders
│   ├── cwru_dataset.py   # CWRU Bearing dataset loader
│   └── pu_dataset.py     # PU Bearing dataset loader
├── utils/                # Utility functions
│   ├── train.py          # Training loop implementation
│   ├── validate.py       # Validation logic
│   └── callbacks.py      # Training callbacks (e.g., EarlyStopping)
├── server/               # Inference API server
├── train.py              # Main entry point for training
├── model_configs.py      # Model architecture configurations
└── experiment_configs.py # Experiment setup (hyperparams, etc.)
```

## Adding a New Model

To add a new model architecture:

1.  **Create the Model Class**: Add your PyTorch model class in `models/`.
    ```python
    # models/my_new_model.py
    import torch.nn as nn

    class MyNewModel(nn.Module):
        def __init__(self, num_classes, window_size):
            super().__init__()
            # Define layers...

        def forward(self, x):
            # Define forward pass...
            return logits
    ```

2.  **Register in `model_configs.py`**: Add a configuration function for your model.
    ```python
    # model_configs.py
    def my_new_model_config(window_size=1024, num_classes=10):
        return {
            "model_class": "models.my_new_model.MyNewModel",
            "args": {
                "num_classes": num_classes,
                "window_size": window_size
            }
        }
    ```

3.  **Update `train.py` (Optional)**: If your model requires specific initialization or handling, update the model loading logic in `train.py` or `utils/train.py`. Altenatively you can create a jupyter notebook to train the model and push the model to mlflow.

Here is an example of pushing model to mlflow:
```python
    import mlflow
    # set experiement and run
    mlflow.set_experiment("my_experiment")
    mlflow.start_run()
    # log model
    mlflow.pytorch.log_model(model, "model")
```

## Adding a New Dataset

1.  **Create Dataset Class**: Implement a `datasets.dataset.BearingDataset` in `datasets/`.
2.  **Update `train.py`**: Add the new dataset to the argument parser and dataset loading logic.

## Running Experiments

Experiments are tracked using MLFlow.

### Basic Training
```bash
python train.py -e experiment_name -m model_name --dataset CWRU
```

### Hyperparameter Tuning
You can modify `experiment_configs.py` to define different hyperparameter sets or use command-line arguments to override defaults.

### Cross-Validation
Use the `--scoring_method cross_validation` flag to run k-fold cross-validation.

```bash
python train.py -e cv_experiment --scoring_method cross_validation --trials 5
```

## MLFlow Integration

- **Tracking URI**: Set `MLFLOW_TRACKING_URI` in `.env` or as an environment variable.
- **Artifacts**: Models, plots, and configs are automatically logged.
- **Registry**: Best models can be registered in the MLFlow Model Registry for deployment.

## Testing

Run unit tests to ensure your changes don't break existing functionality:

```bash
pytest tests/
```

See `TESTING.md` for more details.

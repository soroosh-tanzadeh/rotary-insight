# MLFlow Setup and Usage Guide

This project uses MLFlow for experiment tracking and model management instead of TensorBoard.

**Important**: All experiment results are now stored exclusively in MLFlow. The `./results` folder is no longer used for storing experiment data, metrics, or artifacts. Everything is tracked through MLFlow, providing better organization, versioning, and accessibility.

## Installation

MLFlow is included in the requirements.txt file. Install it with:

```bash
pip install -r requirements.txt
```

## Configuration

### Local Setup (Default)

By default, MLFlow stores data locally in the `./mlruns` directory. No additional configuration is needed.

### Environment Variables

You can configure MLFlow using environment variables:

```bash
# For local file storage (default)
export MLFLOW_TRACKING_URI=file:./mlruns

# For remote MLFlow server
export MLFLOW_TRACKING_URI=http://localhost:5000

# For cloud storage (e.g., S3)
export MLFLOW_TRACKING_URI=s3://my-bucket/mlflow
```

Create a `.env` file in the project root with the following template:

```env
# MLFlow Tracking URI
MLFLOW_TRACKING_URI=file:./mlruns

# MLFlow Artifact Storage (optional)
MLFLOW_ARTIFACT_LOCATION=./mlartifacts

# AWS Credentials (if using S3)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key

# Azure Credentials (if using Azure Blob)
# AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

## Starting MLFlow UI

To view your experiments and models in the MLFlow UI:

```bash
mlflow ui
```

Then open your browser to http://localhost:5000

To specify a custom tracking URI or port:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

## Training with MLFlow

The training script automatically logs all experiments to MLFlow:

```bash
# Basic training
python train.py -e my_experiment -m transformer_encoder_classifier --dataset CWRU

# With multiple trials
python train.py -e my_experiment -m transformer_encoder_classifier --trials 5

# With noise augmentation
python train.py -e noisy_experiment --train_augmentation --noise_levels 10 20 30
```

All metrics, hyperparameters, and model artifacts are automatically logged to MLFlow. No local results files are created - everything is tracked in MLFlow.

## Using the Inference API

### Load Best Model from Experiment

```python
from api import ModelInference

# Load the best model from an experiment (based on validation accuracy)
inference = ModelInference(experiment_name="my_experiment")

# Perform inference
import numpy as np
data = np.random.randn(1, 1, 2048)  # (batch_size, channels, signal_length)
predictions = inference.predict(data)
probabilities = inference.predict_proba(data)

# Get run information
run_info = inference.get_run_info()
print(run_info)
```

### Load Specific Model by Run ID

```python
from api import ModelInference

# Load model from a specific run
inference = ModelInference(run_id="abc123...")

predictions = inference.predict(data)
```

### Load Model by URI

```python
from api import ModelInference

# Load model using direct URI
inference = ModelInference(model_uri="runs:/abc123.../model")

predictions = inference.predict(data)
```

## Utility Functions

### List All Experiments

```python
from api import list_experiments

experiments_df = list_experiments()
print(experiments_df)
```

### List Runs from an Experiment

```python
from api import list_runs

runs_df = list_runs("my_experiment", max_results=10)
print(runs_df)
```

### Get Best Model

```python
from api import get_best_model

# Get best model based on validation accuracy
best_model = get_best_model("my_experiment", metric="val_acc")

# Get best model based on validation loss (lower is better)
best_model = get_best_model("my_experiment", metric="val_loss")

predictions = best_model.predict(data)
```

### Compare Multiple Runs

```python
from api import compare_runs

comparison = compare_runs("my_experiment", ["run_id_1", "run_id_2", "run_id_3"])
print(comparison)
```

## What Gets Logged to MLFlow?

### Per Training Run

1. **Hyperparameters**: All model hyperparameters
2. **Metrics**:

   - `train_acc`: Final training accuracy
   - `train_loss`: Final training loss
   - `val_acc`: Best validation accuracy
   - `val_loss`: Validation loss
   - `num_params`: Number of model parameters
   - `epoch_train_loss`: Training loss per epoch
   - `epoch_train_acc`: Training accuracy per epoch
   - `epoch_val_loss`: Validation loss per epoch
   - `epoch_val_acc`: Validation accuracy per epoch

3. **Tags**:

   - `model_name`: Name of the model architecture
   - `dataset`: Dataset used (CWRU or PU)
   - `timestamp`: Training timestamp

4. **Artifacts**:
   - Trained model (PyTorch format)
   - Confusion matrix image
   - Training/validation accuracy plots
   - Training/validation loss plots
   - Metrics JSON file

### Experiment Summary

A summary run is created for each experiment containing:

- Aggregated metrics across all splits/folds
- Results CSV and JSON files

## Model Registry

To register a model in MLFlow:

```python
import mlflow

# Register the best model from a run
model_uri = "runs:/abc123.../model"
mlflow.register_model(model_uri, "BearingFaultDetector")
```

Then you can load registered models:

```python
from api import ModelInference

inference = ModelInference(model_uri="models:/BearingFaultDetector/1")
```

## Remote MLFlow Server Setup

To use a remote MLFlow server:

1. Start the MLFlow server:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts
```

2. Set the tracking URI in your environment:

```bash
export MLFLOW_TRACKING_URI=http://your-server:5000
```

3. Run training as normal

## Cloud Storage Integration

### AWS S3

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Azure Blob Storage

```bash
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

## Troubleshooting

### Issue: Models not appearing in UI

Check that the MLFlow UI is pointing to the correct backend store:

```bash
mlflow ui --backend-store-uri ./mlruns
```

### Issue: Cannot load model

Ensure the run ID is correct and the model artifact exists:

```python
from api import list_runs
runs = list_runs("my_experiment")
print(runs)
```

### Issue: Tracking URI not set

Set it explicitly in your code:

```python
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
```

## Migration from TensorBoard

The project has been fully migrated from TensorBoard to MLFlow. Key changes:

1. **Removed**: `SummaryWriter` and TensorBoard dependencies
2. **Added**: MLFlow tracking, logging, and model management
3. **Benefits**:
   - Better experiment organization
   - Model versioning and registry
   - Easy model loading for inference
   - Comparison of multiple runs
   - Cloud storage support
   - RESTful API for remote access

All previous functionality is maintained, with additional features for model management and inference.

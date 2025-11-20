# Rotary Insight

A deep learning framework for bearing fault detection using time-series vibration data.

## Overview

This project implements various neural network architectures for detecting bearing faults using the CWRU and PU datasets. All experiments are tracked using MLFlow for comprehensive experiment management and model versioning.

## Features

- **Multiple Model Architectures**: Support for Transformer, CNN, MLP, and KAN-based models
- **MLFlow Integration**: Complete experiment tracking, model management, and artifact storage
- **Multiple Datasets**: Support for CWRU and PU bearing fault datasets
- **Data Augmentation**: Noise injection for robust model training
- **Cross-Validation**: K-fold cross-validation support
- **REST API Server**: FastAPI-based inference server with authentication and auto-documentation
- **Docker Support**: Easy deployment with Docker and docker-compose

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd rotary-insight
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download datasets (CWRU and/or PU) to the `./data/dataset/` directory

## Quick Start

### Training a Model

```bash
# Basic training
python train.py -e my_experiment -m transformer_encoder_classifier --dataset CWRU

# With multiple trials
python train.py -e my_experiment -m transformer_encoder_classifier --trials 5

# With noise augmentation
python train.py -e noisy_experiment --train_augmentation --noise_levels 10 20 30

# Cross-validation
python train.py -e cv_experiment --scoring_method cross_validation
```

### Viewing Experiments in MLFlow UI

```bash
mlflow ui
```

Then open your browser to http://localhost:5000

### Using the REST API Server

The project includes a production-ready FastAPI server for model inference.

**Quick Start:**

```bash
# 1. Create .env file with API keys
echo "API_KEYS=your-secret-key" > .env

# 2. Copy model configuration
cp model_serve_config.example.json model_serve_config.json

# 3. Start MLflow
./start_mlflow.bash

# 4. Start API server
./start_server.bash
```

**Access the API:**

- Interactive docs: http://localhost:8000/docs
- API endpoint: http://localhost:8000/predict

**See full documentation:**

- [API Quick Start](API_QUICKSTART.md)
- [Environment Setup](ENV_SETUP.md)
- [Server Documentation](server/README.md)
- [Docker Deployment](DOCKER_DEPLOYMENT.md)
- [Developer Guide](DEVELOPER_GUIDE.md)

## Project Structure

```
rotary-insight/
├── train.py              # Main training script
├── api.py                # Legacy inference utilities
├── experiment_configs.py # Experiment configuration builder
├── model_configs.py      # Model architecture configurations
├── server/               # FastAPI inference server
│   ├── main.py          # API application
│   ├── auth.py          # Authentication
│   ├── models.py        # Pydantic models
│   ├── inference.py     # Model loading & inference
│   └── client_example.py # Example Python client
├── datasets/             # Dataset loaders
│   ├── cwru_dataset.py
│   └── pu_dataset.py
├── models/               # Neural network architectures
│   ├── transformer.py
│   ├── embedding.py
│   └── positional_encoder.py
├── utils/                # Utility functions
│   ├── train.py         # Training loop
│   ├── validate.py      # Validation functions
│   ├── callbacks.py     # Training callbacks
│   └── display.py       # Visualization utilities
├── Dockerfile.api        # Docker image for API
├── docker-compose.yml    # Docker compose configuration
└── checkpoints/          # Model checkpoints (local backup)
```

## MLFlow Integration

All experiments are automatically logged to MLFlow with:

- **Hyperparameters**: Model architecture settings
- **Metrics**: Training/validation accuracy and loss per epoch
- **Artifacts**: Trained models, confusion matrices, training plots
- **Tags**: Model name, dataset, timestamp

See [MLFLOW_SETUP.md](MLFLOW_SETUP.md) for detailed MLFlow configuration and usage.

## Command Line Arguments

```
python train.py [OPTIONS]

Options:
  -e, --experiment TEXT         Experiment name (required)
  -m, --models TEXT [TEXT ...]  List of models to train
  --dataset {CWRU,PU}          Dataset to use (default: CWRU)
  --trials INT                  Number of random train/test splits (default: 1)
  --test_with_noise            Add noise to test data
  --train_augmentation         Add noise to training data
  --noise_levels FLOAT [...]   Noise levels in SNR dB (default: [0.0])
  --scoring_method {train_validation,cross_validation}
  --debug                      Use subset of data for debugging
```

## Available Models

- `transformer_encoder_classifier`: Transformer-based encoder
- `mlp_classifier`: Multi-layer perceptron
- `kan_classifier`: Kolmogorov-Arnold Network
- `relu_kan_classifier`: ReLU-based KAN
- `mlp_conv_classifier`: CNN with MLP
- `kan_conv_classifier`: CNN with KAN layers
- And more variants...

## Datasets

### CWRU Dataset

- 10 classes (Normal + 9 fault types)
- Variable loads and fault sizes
- Single-channel vibration data

### PU Dataset

- 3 classes (Healthy, InnerRace, OuterRace)
- Multiple operating conditions
- Single-channel vibration data

## Examples

### Example 1: Train with Multiple Noise Levels

```bash
python train.py \
  -e noise_robustness_study \
  -m transformer_encoder_classifier \
  --dataset CWRU \
  --train_augmentation \
  --noise_levels 5 10 15 20 \
  --trials 3
```

### Example 2: Cross-Validation

```bash
python train.py \
  -e cv_baseline \
  -m transformer_encoder_classifier \
  --scoring_method cross_validation \
  --dataset CWRU
```

### Example 3: Compare Models

```bash
python train.py \
  -e model_comparison \
  -m transformer_encoder_classifier mlp_classifier kan_classifier \
  --trials 5
```

### Example 4: Load and Use Best Model

```python
from api import get_best_model, list_runs
import numpy as np

# List all runs from an experiment
runs = list_runs("my_experiment", max_results=10)
print(runs)

# Get the best model
model = get_best_model("my_experiment", metric="val_acc")

# Perform inference
data = np.random.randn(10, 1, 2048)  # Batch of 10 samples
predictions = model.predict(data)
print(f"Predictions: {predictions}")
```

## Model Registry

Register your best models in MLFlow for easy versioning:

```python
import mlflow

# Register a model
model_uri = "runs:/abc123.../model"
mlflow.register_model(model_uri, "BearingFaultDetector")

# Load registered model
from api import ModelInference
inference = ModelInference(model_uri="models:/BearingFaultDetector/1")
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a new branch for your feature
2. Test your changes thoroughly
3. Update documentation as needed
4. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Acknowledgments

- CWRU Bearing Data Center for the bearing fault dataset
- Paderborn University for the PU bearing dataset

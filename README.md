# Rotary Insight

A deep learning framework for bearing fault detection using time-series vibration data.

## Overview

This project implements various neural network architectures for detecting bearing faults using the CWRU and PU datasets. All experiments are tracked using MLFlow for comprehensive experiment management and model versioning.

## Features

- **Multiple Model Architectures**: Transformer, CNN, MLP, and KAN-based models
- **MLFlow Integration**: Experiment tracking, model registry, and artifact storage
- **Multiple Datasets**: Support for CWRU and PU bearing fault datasets
- **Data Augmentation**: Noise injection for robust training
- **Cross-Validation**: K‑fold cross‑validation support
- **REST API Server**: FastAPI inference server with authentication and OpenAPI docs
- **Docker Support**: Containerised deployment via Docker and docker‑compose

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/rotary-insight.git
   cd rotary-insight
   ```
2. Create a virtual environment and install Python dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Install frontend dependencies (if you plan to run the UI):
   ```bash
   cd frontend
   npm install
   cd ..
   ```
4. Download the CWRU and/or PU datasets into `./data/dataset/`.

## Quick Start

### Training a Model
```bash
# Basic training
python train.py -e my_experiment -m transformer_encoder_classifier --dataset CWRU

# Multiple trials
python train.py -e my_experiment -m transformer_encoder_classifier --trials 5

# With noise augmentation
python train.py -e noisy_experiment --train_augmentation --noise_levels 10 20 30

# Cross‑validation
python train.py -e cv_experiment --scoring_method cross_validation
```

### Viewing Experiments in MLflow UI
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### Running the REST API Server
```bash
# Set up environment variables (API keys, etc.)
echo "API_KEYS=your-secret-key" > .env

# Copy example configuration
cp model_serve_config.example.json model_serve_config.json

# Start services
./start_mlflow.bash   # launches MLflow tracking server
./start_server.bash   # launches FastAPI inference server
```
Access the interactive docs at `http://localhost:8000/docs` and the prediction endpoint at `http://localhost:8000/predict`.

## Project Structure
```
rotary-insight/
├── train.py                # Main training script
├── model_serve_config.json # Model serving configuration
├── model_serve_config.example.json
├── server/                 # FastAPI inference server
│   ├── main.py
│   ├── auth.py
│   ├── models.py
│   ├── inference.py
│   └── client_example.py
├── datasets/               # Dataset loaders
│   ├── cwru_dataset.py
│   └── pu_dataset.py
├── models/                 # Neural network architectures
│   ├── transformer.py
│   ├── embedding.py
│   └── positional_encoder.py
├── utils/                  # Utility functions
│   ├── train.py
│   ├── validate.py
│   ├── callbacks.py
│   └── display.py
├── Dockerfile.api          # Docker image for API
├── docker-compose.yml      # Docker compose configuration
└── checkpoints/            # Model checkpoints (local backup)
```

## MLflow Integration
All experiments are automatically logged to MLflow with hyperparameters, metrics, artifacts, and tags (model name, dataset, timestamp). See `MLFLOW_SETUP.md` for detailed configuration.

## Command‑Line Arguments
```bash
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

- `transformer_encoder_classifier`: Pure Transformer encoder with patch embeddings
- `resnet_classifier`: 2D ResNet with STFT preprocessing
- `resnet1d`: 1D ResNet with residual connections
- `densenet1d`: 1D DenseNet with dense connections
- `efficientnet1d`: 1D EfficientNet with MBConv blocks
- `wrn1d`: 1D Wide Residual Network
- `dsicnn`: Dilated Separable Inverted CNN
- `dpccnn`: Deep Pyramid CNN with SE and GAB blocks
- `cnn_bilstm`: Hybrid CNN + Bidirectional LSTM
- `cnn_selfattention`: Hybrid CNN + Self-Attention mechanism

## Datasets
### CWRU Dataset
- 10 classes (Normal + 9 fault types)
- Variable loads and fault sizes
- Single‑channel vibration data
### PU Dataset
- 3 classes (Healthy, InnerRace, OuterRace)
- Multiple operating conditions
- Single‑channel vibration data

## Examples
### Train with Multiple Noise Levels
```bash
python train.py \
  -e noise_robustness_study \
  -m transformer_encoder_classifier \
  --dataset CWRU \
  --train_augmentation \
  --noise_levels 5 10 15 20 \
  --trials 3
```
### Cross‑Validation
```bash
python train.py \
  -e cv_baseline \
  -m transformer_encoder_classifier \
  --scoring_method cross_validation \
  --dataset CWRU
```
### Compare Models
```bash
python train.py \
  -e model_comparison \
   -m transformer_encoder_classifier \\
  --trials 5
```
### Load and Use Best Model (Python)
```python
from api import get_best_model, list_runs
runs = list_runs("my_experiment", max_results=10)
print(runs)
model = get_best_model("my_experiment", metric="val_acc")
import numpy as np
data = np.random.randn(10, 1, 2048)  # Batch of 10 samples
predictions = model.predict(data)
print(f"Predictions: {predictions}")
```

## Model Registry
Register your best models in MLflow for versioning:
```python
import mlflow
model_uri = "runs:/abc123.../model"
mlflow.register_model(model_uri, "BearingFaultDetector")
from api import ModelInference
inference = ModelInference(model_uri="models:/BearingFaultDetector/1")
```

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository and create a new branch for your feature.
2. Write tests and ensure existing tests pass.
3. Update documentation as needed.
4. Open a pull request.

## License
See the `LICENSE` file for details.

## Citation
If you use this code in research, please cite:
```bibtex
@software{rotary_insight,
  author = {Mehdi Tanzadeh Mojarad, MohammadHasan Tavakoli, Majid Haidarasl, Seyede Zohre Mousavi
  title = {Rotary Insight: Bearing Fault Detection Framework},
  year = {2025},
  url = {https://github.com/soroosh-tanzadeh/rotary-insight}
}
```

## Acknowledgments
- CWRU Bearing Data Center for the bearing fault dataset
- Paderborn University for the PU bearing dataset
- A. Vaswani et al., “Attention Is All You Need,” Cornell University, Jun. 12, 2017.  
- G. Huang, Z. Liu, and Weinberger, Kilian Q, “Densely Connected Convolutional Networks, arXiv.org, 2016. 
- M. Zhao, S. Zhong, X. Fu, B. Tang, and M. Pecht, "Deep residual shrinkage networks for fault diagnosis," IEEE Trans. Ind. Informatics, vol. 16, no. 7, pp. 4681–4690, 2019.
- A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” arXiv:2010.11929 [cs], Oct. 2020.
- Zhang, J., Zhao, Z., Jiao, Y., Zhao, R., Hu, X., and Che, R., "DPCCNN: A new lightweight fault diagnosis model for small samples and high noise problem," Neurocomputing, vol. 626, p. 129526, 2025, doi: 10.1016/j.neucom.2025.129526.

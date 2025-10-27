# Rotary Insight Inference Server

FastAPI-based inference server for bearing fault classification models.

## Features

- 🔐 **API Key Authentication**: Secure access using API keys
- 🤖 **Multi-Model Support**: Serve multiple models simultaneously
- 📊 **Automatic API Documentation**: Interactive docs at `/docs`
- 🚀 **Fast Inference**: Optimized PyTorch model serving
- 📝 **Request Validation**: Pydantic V2 models for type safety
- ⚡ **Modern FastAPI**: Uses lifespan context manager for startup/shutdown
- 🔄 **Graceful Shutdown**: Proper cleanup on server shutdown

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```env
API_KEYS=your-secret-key-1,your-secret-key-2
MLFLOW_TRACKING_URI=http://localhost:5000
HOST=0.0.0.0
PORT=8000
MODEL_CONFIG_PATH=model_serve_config.json
```

### 3. Configure Models

Create or edit `model_serve_config.json`:

```json
{
	"models": {
		"transformer_encoder_cwru_512": {
			"path": "models:/transformer_encoder_cwru_512/1",
			"type": "pytorch",
			"window_size": 512,
			"dataset_name": "CWRU",
			"task": "classification"
		}
	}
}
```

### 4. Start MLflow Server

Make sure MLflow tracking server is running:

```bash
./start_mlflow.bash
```

Or manually:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### 5. Run the Server

```bash
python -m server.main
```

Or with uvicorn directly:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Examples

### Authentication

All requests (except `/health`) require an API key in the `X-API-Key` header:

```bash
curl -X GET "http://localhost:8000/models" \
  -H "X-API-Key: your-secret-key-1"
```

### List Available Models

```bash
curl -X GET "http://localhost:8000/models" \
  -H "X-API-Key: your-secret-key-1"
```

### Perform Inference

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "transformer_encoder_cwru_512",
    "data": [[[0.1, 0.2, ...]]],
    "return_probabilities": true
  }'
```

### Python Client Example

```python
import requests
import numpy as np

# Server configuration
API_URL = "http://localhost:8000"
API_KEY = "your-secret-key-1"

# Headers with API key
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Generate sample data (batch_size=1, channels=1, signal_length=512)
data = np.random.randn(1, 1, 512).tolist()

# Make prediction request
response = requests.post(
    f"{API_URL}/predict",
    headers=headers,
    json={
        "model_name": "transformer_encoder_cwru_512",
        "data": data,
        "return_probabilities": True
    }
)

result = response.json()
print(f"Predicted class: {result['predictions'][0]['class_name']}")
print(f"Confidence: {result['predictions'][0]['confidence']:.2%}")
```

### Load Model

Pre-load a model for faster first inference:

```bash
curl -X POST "http://localhost:8000/models/transformer_encoder_cwru_512/load" \
  -H "X-API-Key: your-secret-key-1"
```

### Unload Model

Free up memory by unloading a model:

```bash
curl -X DELETE "http://localhost:8000/models/transformer_encoder_cwru_512/unload" \
  -H "X-API-Key: your-secret-key-1"
```

## API Endpoints

| Endpoint                | Method | Auth Required | Description             |
| ----------------------- | ------ | ------------- | ----------------------- |
| `/`                     | GET    | No            | Root endpoint           |
| `/health`               | GET    | No            | Health check            |
| `/models`               | GET    | Yes           | List available models   |
| `/predict`              | POST   | Yes           | Perform inference       |
| `/models/{name}/load`   | POST   | Yes           | Load a specific model   |
| `/models/{name}/unload` | DELETE | Yes           | Unload a specific model |

## Input Data Format

The inference endpoint expects data in the following format:

```json
{
  "model_name": "transformer_encoder_cwru_512",
  "data": [
    [  // Batch dimension (can have multiple samples)
      [  // Channel dimension (typically 1 for this project)
        0.1, 0.2, 0.3, ...  // Signal values (length = window_size)
      ]
    ]
  ],
  "return_probabilities": true
}
```

Shape: `(batch_size, channels, signal_length)`

Example for window_size=512:

- `data` shape: `(1, 1, 512)` for single sample
- `data` shape: `(4, 1, 512)` for batch of 4 samples

## Response Format

```json
{
  "model_name": "transformer_encoder_cwru_512",
  "predictions": [
    {
      "predicted_class": 0,
      "class_name": "Normal",
      "confidence": 0.95,
      "probabilities": [0.95, 0.02, 0.01, ...]
    }
  ],
  "processing_time_ms": 15.3
}
```

## Supported Datasets

- **CWRU**: Case Western Reserve University Bearing Dataset (10 classes)
- **PU**: Paderborn University Bearing Dataset (3 classes)

## Class Names

### CWRU Dataset

0. Normal
1. 0.007-Ball
2. 0.014-Ball
3. 0.021-Ball
4. 0.007-InnerRace
5. 0.014-InnerRace
6. 0.021-InnerRace
7. 0.007-OuterRace
8. 0.014-OuterRace
9. 0.021-OuterRace

### PU Dataset

0. Healthy
1. OuterRace
2. InnerRace

## Troubleshooting

### Authentication Errors

If you get `401 Unauthorized` or `403 Forbidden`:

- Check that you're including the `X-API-Key` header
- Verify the API key matches one in your `.env` file

### Model Loading Errors

If models fail to load:

- Ensure MLflow server is running
- Check that model paths in `model_serve_config.json` are correct
- Verify models are registered in MLflow

### Input Shape Errors

If you get shape validation errors:

- Check that signal length matches the model's window_size
- Ensure data has correct shape: `(batch_size, channels, signal_length)`
- Verify channels=1 for most models in this project

## Production Deployment

For production deployment:

1. **Use environment variables for secrets**
2. **Enable HTTPS** (use a reverse proxy like nginx)
3. **Restrict CORS origins** in `main.py`
4. **Use a process manager** (e.g., systemd, supervisor)
5. **Scale with workers**: `uvicorn server.main:app --workers 4`
6. **Monitor with logs** and metrics
7. **Use a load balancer** for multiple instances

## License

Same as the main project.

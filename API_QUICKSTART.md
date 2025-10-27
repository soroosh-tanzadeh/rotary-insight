# API Quick Start Guide

Get your inference API running in 5 minutes!

## Prerequisites

- Python 3.8+ with dependencies installed
- Trained models registered in MLflow

## Step-by-Step Setup

### Step 1: Create `.env` File

```bash
cat > .env << 'EOF'
API_KEYS=dev-key-123,prod-key-456
MLFLOW_TRACKING_URI=http://localhost:5000
HOST=0.0.0.0
PORT=8000
MODEL_CONFIG_PATH=model_serve_config.json
EOF
```

Replace `dev-key-123` and `prod-key-456` with your own secret keys.

### Step 2: Create Model Configuration

```bash
cp model_serve_config.example.json model_serve_config.json
```

Edit `model_serve_config.json` if needed to match your registered models.

### Step 3: Install Dependencies

```bash
pip install python-dotenv pydantic uvicorn
```

(Or all dependencies: `pip install -r requirements.txt`)

### Step 4: Start MLflow

```bash
./start_mlflow.bash
```

Verify it's running: http://localhost:5000

### Step 5: Start the API Server

```bash
./start_server.bash
```

Or manually:

```bash
python -m server.main
```

### Step 6: Access the API

Open http://localhost:8000/docs in your browser to see the interactive API documentation!

## Quick Test

### Test 1: Health Check (No Auth)

```bash
curl http://localhost:8000/health
```

### Test 2: List Models (With Auth)

```bash
curl -H "X-API-Key: dev-key-123" http://localhost:8000/models
```

### Test 3: Make a Prediction

For a model with window_size=512:

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "transformer_encoder_cwru_512",
    "data": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]],
    "return_probabilities": true
  }'
```

Note: The data array should have 512 values for window_size=512 (shortened for example).

## Python Client Example

```python
import requests
import numpy as np

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "dev-key-123"

# Headers
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Generate sample data (1 sample, 1 channel, 512 time steps)
data = np.random.randn(1, 1, 512).tolist()

# Make prediction
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

## Using the Example Client

```bash
# Edit the API_KEY in the file first
nano server/client_example.py

# Run it
python server/client_example.py
```

## API Endpoints Summary

| Endpoint                | Method | Auth | Description      |
| ----------------------- | ------ | ---- | ---------------- |
| `/`                     | GET    | ❌   | Root endpoint    |
| `/health`               | GET    | ❌   | Health check     |
| `/models`               | GET    | ✅   | List all models  |
| `/predict`              | POST   | ✅   | Make predictions |
| `/models/{name}/load`   | POST   | ✅   | Pre-load a model |
| `/models/{name}/unload` | DELETE | ✅   | Unload a model   |

## Common Issues

**Problem**: `API_KEYS environment variable is not set`

- **Solution**: Create `.env` file with `API_KEYS=your-key`

**Problem**: `Model configuration file not found`

- **Solution**: Copy `model_serve_config.example.json` to `model_serve_config.json`

**Problem**: `Failed to load model`

- **Solution**: Check MLflow is running and model is registered

**Problem**: `401 Unauthorized`

- **Solution**: Add `X-API-Key` header to your request

**Problem**: `Invalid input shape`

- **Solution**: Ensure data length matches model's window_size (512/1024/2048)

## Next Steps

1. ✅ Read full documentation: `server/README.md`
2. ✅ Check environment setup: `ENV_SETUP.md`
3. ✅ Explore API docs: http://localhost:8000/docs
4. ✅ Try example client: `python server/client_example.py`
5. ✅ Integrate into your application

## File Checklist

Before starting, make sure you have:

- [x] `.env` file with API keys
- [x] `model_serve_config.json` with model paths
- [x] MLflow server running (port 5000)
- [x] Models registered in MLflow
- [x] Dependencies installed

## Production Deployment

For production:

1. Use strong API keys (e.g., generated with `openssl rand -hex 32`)
2. Enable HTTPS with nginx/traefik
3. Set up monitoring and logging
4. Use a process manager (systemd/supervisor)
5. Configure firewall rules
6. Set up backups for MLflow artifacts

See `ENV_SETUP.md` for detailed production deployment guide.

---

**Questions?** Check the full documentation in `server/README.md` or the API docs at `/docs`.

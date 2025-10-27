# Environment Setup for Rotary Insight API

This guide will help you set up the inference API server.

## Quick Start

### 1. Create `.env` File

Create a `.env` file in the project root with the following content:

```bash
# API Authentication Keys
# Add your API keys here (comma-separated for multiple keys)
API_KEYS=your-secret-key-1,your-secret-key-2,your-secret-key-3

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Model Configuration Path
MODEL_CONFIG_PATH=model_serve_config.json
```

**Important**: Replace `your-secret-key-1`, etc. with your actual secret keys. These can be any strings you want.

### 2. Create Model Configuration

If you don't have `model_serve_config.json`, copy the example:

```bash
cp model_serve_config.example.json model_serve_config.json
```

Or create it manually with your model configurations:

```json
{
	"models": {
		"transformer_encoder_cwru_2048": {
			"path": "models:/transformer_encoder_cwru_2048/1",
			"type": "pytorch",
			"window_size": 2048,
			"dataset_name": "CWRU",
			"task": "classification"
		},
		"transformer_encoder_cwru_1024": {
			"path": "models:/transformer_encoder_cwru_1024/1",
			"type": "pytorch",
			"window_size": 1024,
			"dataset_name": "CWRU",
			"task": "classification"
		},
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

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start MLflow Server

The API needs MLflow to load models. Start it with:

```bash
./start_mlflow.bash
```

Or manually:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///data/db.db --default-artifact-root ./mlartifacts
```

### 5. Start the API Server

```bash
./start_server.bash
```

Or with Python directly:

```bash
python -m server.main
```

Or with uvicorn:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Accessing the API

Once the server is running:

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Testing the API

### Using cURL

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# List models (requires API key)
curl -H "X-API-Key: your-secret-key-1" http://localhost:8000/models

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "transformer_encoder_cwru_512",
    "data": [[[0.1, 0.2, 0.3]]],
    "return_probabilities": true
  }'
```

### Using Python Client

Run the example client:

```bash
# Edit server/client_example.py to set your API key
python server/client_example.py
```

Or use it programmatically:

```python
from server.client_example import RotaryInsightClient
import numpy as np

# Initialize client
client = RotaryInsightClient(
    base_url="http://localhost:8000",
    api_key="your-secret-key-1"
)

# Generate sample data
data = np.random.randn(1, 1, 512).astype(np.float32)

# Make prediction
result = client.predict(
    data=data,
    model_name="transformer_encoder_cwru_512",
    return_probabilities=True
)

print(result)
```

## Troubleshooting

### Server Won't Start

1. **Check `.env` file exists**: Make sure you created `.env` with API keys
2. **Check MLflow is running**: Visit http://localhost:5000
3. **Check model config exists**: Verify `model_serve_config.json` is present
4. **Check dependencies**: Run `pip install -r requirements.txt`

### Authentication Errors

1. **401 Unauthorized**: You didn't include the `X-API-Key` header
2. **403 Forbidden**: The API key doesn't match any in your `.env` file

### Model Loading Errors

1. **Model not found**: Check the model path in `model_serve_config.json`
2. **MLflow connection error**: Ensure MLflow server is running
3. **Model not registered**: Use MLflow UI to verify the model is registered

## Production Deployment

For production deployment, consider:

1. **Use strong API keys**: Generate cryptographically secure random strings
2. **Enable HTTPS**: Use a reverse proxy (nginx, traefik)
3. **Restrict CORS**: Update allowed origins in `server/main.py`
4. **Use environment variables**: Don't commit `.env` to git
5. **Scale with workers**: `uvicorn server.main:app --workers 4`
6. **Set up monitoring**: Use logging and metrics
7. **Use a process manager**: systemd, supervisor, or Docker

## Security Best Practices

1. ✅ Never commit `.env` file to git (it's in `.gitignore`)
2. ✅ Use different API keys for different environments (dev/staging/prod)
3. ✅ Rotate API keys regularly
4. ✅ Use HTTPS in production
5. ✅ Restrict network access with firewall rules
6. ✅ Monitor API usage and set rate limits if needed

## File Structure

```
rotary-insight/
├── .env                          # Your API keys (DON'T COMMIT)
├── model_serve_config.json       # Model configuration (DON'T COMMIT)
├── model_serve_config.example.json  # Example configuration
├── start_server.bash             # Server startup script
├── server/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application
│   ├── auth.py                  # Authentication logic
│   ├── models.py                # Pydantic models
│   ├── inference.py             # Model loading and inference
│   ├── client_example.py        # Example Python client
│   └── README.md                # Server documentation
└── ...
```

## Next Steps

1. Read the API documentation at `/docs`
2. Try the example client in `server/client_example.py`
3. Integrate the API into your application
4. Monitor performance and adjust as needed

For more details, see `server/README.md`.

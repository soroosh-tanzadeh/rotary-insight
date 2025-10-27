# Getting Started with the Inference API

Welcome! This guide will help you get the API up and running in minutes.

## 📋 What You Need

Before starting, make sure you have:

- Python 3.8 or higher
- MLflow server running (or can start it)
- At least one trained model registered in MLflow

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Environment

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
API_KEYS=my-secret-api-key-123
MLFLOW_TRACKING_URI=http://localhost:5000
HOST=0.0.0.0
PORT=8000
MODEL_CONFIG_PATH=model_serve_config.json
EOF
```

**💡 Tip**: Replace `my-secret-api-key-123` with your own secret key!

### Step 2: Setup Model Configuration

Copy the example configuration:

```bash
cp model_serve_config.example.json model_serve_config.json
```

**💡 Tip**: Edit `model_serve_config.json` if you need to change model paths or add more models.

### Step 3: Start the Server

First, make sure MLflow is running:

```bash
./start_mlflow.bash
```

Then start the API server:

```bash
./start_server.bash
```

That's it! 🎉

## ✅ Verify Setup

Run the verification script to check everything is configured:

```bash
python verify_api_setup.py
```

This will check:

- ✓ .env file exists and has API keys
- ✓ model_serve_config.json is valid
- ✓ All dependencies are installed
- ✓ Server files are present
- ✓ MLflow is accessible
- ✓ Server can be imported

## 🌐 Access the API

Once the server is running, open your browser:

### Interactive API Documentation

**http://localhost:8000/docs**

This gives you:

- Complete API reference
- Interactive testing interface
- Request/response examples
- Try it out directly in the browser

### Alternative Documentation

**http://localhost:8000/redoc**

### Health Check

**http://localhost:8000/health**

## 🧪 Test the API

### Test 1: Check Health (No Auth Required)

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "mlflow_uri": "http://localhost:5000",
  "models_loaded": 0,
  "available_models": ["transformer_encoder_cwru_512", ...]
}
```

### Test 2: List Models (Auth Required)

```bash
curl -H "X-API-Key: my-secret-api-key-123" http://localhost:8000/models
```

Expected response:

```json
{
  "models": {
    "transformer_encoder_cwru_512": {
      "name": "transformer_encoder_cwru_512",
      "window_size": 512,
      "num_classes": 10,
      ...
    }
  },
  "total_count": 1
}
```

### Test 3: Make a Prediction (Auth Required)

```bash
# Note: This is a simplified example with fewer data points
# In reality, you need 512 values for window_size=512
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: my-secret-api-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "transformer_encoder_cwru_512",
    "data": [[[0.1, 0.2, 0.3, 0.4, 0.5]]],
    "return_probabilities": true
  }'
```

## 🐍 Use Python Client

The project includes a ready-to-use Python client:

```bash
# Edit the file to set your API key
nano server/client_example.py

# Run it
python server/client_example.py
```

Or use it in your code:

```python
from server.client_example import RotaryInsightClient
import numpy as np

# Initialize client
client = RotaryInsightClient(
    base_url="http://localhost:8000",
    api_key="my-secret-api-key-123"
)

# Check health
health = client.health_check()
print(f"Status: {health['status']}")

# List models
models = client.list_models()
print(f"Available models: {list(models['models'].keys())}")

# Generate sample data (1 sample, 1 channel, 512 time steps)
data = np.random.randn(1, 1, 512).astype(np.float32)

# Make prediction
result = client.predict(
    data=data,
    model_name="transformer_encoder_cwru_512",
    return_probabilities=True
)

# Print result
print(f"Predicted class: {result['predictions'][0]['class_name']}")
print(f"Confidence: {result['predictions'][0]['confidence']:.2%}")
```

## 🐳 Use Docker (Alternative)

If you prefer Docker:

```bash
# Start both MLflow and API with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop everything
docker-compose down
```

## 📚 Next Steps

1. **Read the full documentation**

   - [API_QUICKSTART.md](API_QUICKSTART.md) - Quick reference
   - [server/README.md](server/README.md) - Complete API docs
   - [ENV_SETUP.md](ENV_SETUP.md) - Environment setup
   - [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Docker guide

2. **Explore the API**

   - Visit http://localhost:8000/docs
   - Try the interactive examples
   - Test different endpoints

3. **Integrate into your application**

   - Use the Python client
   - Or make HTTP requests directly
   - See examples in documentation

4. **Deploy to production**
   - See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
   - See [ENV_SETUP.md](ENV_SETUP.md) production section

## 🔧 Troubleshooting

### "API_KEYS environment variable is not set"

→ Create `.env` file with `API_KEYS=your-key`

### "Model configuration file not found"

→ Run `cp model_serve_config.example.json model_serve_config.json`

### "Failed to connect to MLflow"

→ Start MLflow: `./start_mlflow.bash`

### "Port 8000 already in use"

→ Change port in `.env`: `PORT=8001`

### Import errors

→ Install dependencies: `pip install -r requirements.txt`

## 💡 Tips

1. **API Keys**: Use different keys for dev/staging/prod
2. **Models**: Models are loaded lazily (on first use)
3. **Batch Inference**: Send multiple samples at once
4. **GPU**: Automatically used if available
5. **Documentation**: Always check `/docs` for latest API info

## 🎯 Common Use Cases

### Use Case 1: Real-time Monitoring

```python
# Continuously monitor sensor data
while True:
    sensor_data = read_sensor()  # Your sensor reading function
    result = client.predict(
        data=sensor_data.reshape(1, 1, -1),
        model_name="transformer_encoder_cwru_512"
    )
    if result['predictions'][0]['class_name'] != 'Normal':
        send_alert(result)  # Your alerting function
```

### Use Case 2: Batch Processing

```python
# Process multiple samples at once
batch_data = np.random.randn(100, 1, 512)  # 100 samples
results = client.predict(
    data=batch_data,
    model_name="transformer_encoder_cwru_512",
    return_probabilities=True
)
# Analyze results...
```

### Use Case 3: Model Comparison

```python
# Compare predictions from different models
models = ["transformer_encoder_cwru_512",
          "transformer_encoder_cwru_1024"]

for model_name in models:
    result = client.predict(data=data, model_name=model_name)
    print(f"{model_name}: {result['predictions'][0]['class_name']}")
```

## 📝 File Checklist

Make sure you have these files:

- [x] `.env` - Your API keys and configuration
- [x] `model_serve_config.json` - Model paths and settings
- [x] `server/` - All server files (auto-created)
- [x] `requirements.txt` - Python dependencies (updated)
- [x] `start_server.bash` - Server startup script
- [x] `verify_api_setup.py` - Setup verification script

## 🆘 Need Help?

1. **Run verification**: `python verify_api_setup.py`
2. **Check logs**: Look at terminal output when starting server
3. **Visit docs**: http://localhost:8000/docs
4. **Read guides**: Check the documentation files
5. **Test MLflow**: Visit http://localhost:5000

## 🎉 Success!

If everything works, you should see:

```
Starting Rotary Insight Inference Server
==================================================
✓ Authentication initialized
✓ Model manager initialized
==================================================
Server ready at http://0.0.0.0:8000
API Documentation: http://0.0.0.0:8000/docs
==================================================
```

Visit http://localhost:8000/docs and start using your API! 🚀

---

**Quick Links**:

- 📖 [API Documentation](http://localhost:8000/docs)
- 🏥 [Health Check](http://localhost:8000/health)
- 📊 [MLflow UI](http://localhost:5000)
- 📚 [Complete Guides](API_SUMMARY.md)

#!/bin/bash

# Start script for Rotary Insight Inference Server

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create .env file with the following variables:"
    echo ""
    echo "API_KEYS=your-secret-key-1,your-secret-key-2"
    echo "MLFLOW_TRACKING_URI=http://localhost:5000"
    echo "HOST=0.0.0.0"
    echo "PORT=8000"
    echo "MODEL_CONFIG_PATH=model_serve_config.json"
    echo ""
    exit 1
fi

# Check if MLflow is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "Warning: MLflow server doesn't appear to be running at http://localhost:5000"
    echo "Start it with: ./start_mlflow.bash"
    echo ""
fi

# Check if model config exists
if [ ! -f model_serve_config.json ]; then
    echo "Warning: model_serve_config.json not found!"
    echo "Using model_serve_config.example.json as reference..."
    if [ -f model_serve_config.example.json ]; then
        cp model_serve_config.example.json model_serve_config.json
        echo "Created model_serve_config.json from example"
    fi
fi

# Start the server
echo "Starting Rotary Insight Inference Server..."
python -m server.main


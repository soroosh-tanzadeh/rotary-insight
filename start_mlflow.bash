#!/bin/bash
source .env

mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri=$MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_S3_ENDPOINT_URL/mlflow/
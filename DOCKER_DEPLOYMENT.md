# Docker Deployment Guide

Deploy the Rotary Insight API using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start with Docker Compose

### 1. Setup Environment

Create `.env` file:

```bash
cat > .env << 'EOF'
API_KEYS=your-secret-key-1,your-secret-key-2
EOF
```

Create model configuration:

```bash
cp model_serve_config.example.json model_serve_config.json
```

### 2. Build and Start Services

```bash
# Build the API image
docker compose build

# Start all services (MLflow + API)
docker compose up -d
```

This will start:

- MLflow server on http://localhost:5000
- API server on http://localhost:8000

### 3. Check Status

```bash
# View logs
docker compose logs -f

# Check health
curl http://localhost:8000/health
```

### 4. Stop Services

```bash
docker compose down
```

## Manual Docker Deployment

### Build API Image

```bash
docker build -f Dockerfile.api -t rotary-insight-api:latest .
```

### Run MLflow Container

```bash
docker run -d \
  --name rotary-mlflow \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/mlartifacts:/app/mlartifacts \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///app/data/db.db \
    --default-artifact-root /app/mlartifacts
```

### Run API Container

```bash
docker run -d \
  --name rotary-api \
  -p 8000:8000 \
  --link rotary-mlflow:mlflow \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e API_KEYS=your-secret-key-1 \
  -v $(pwd)/model_serve_config.json:/app/model_serve_config.json:ro \
  -v $(pwd)/mlartifacts:/app/mlartifacts:ro \
  rotary-insight-api:latest
```

## Docker Compose Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f api
docker compose logs -f mlflow

# Restart a service
docker compose restart api

# Rebuild after code changes
docker compose build api
docker compose up -d api

# Scale API instances
docker compose up -d --scale api=3
```

## Environment Variables

Set these in your `.env` file:

```env
# Required
API_KEYS=key1,key2,key3

# Optional (defaults shown)
MLFLOW_TRACKING_URI=http://mlflow:5000
HOST=0.0.0.0
PORT=8000
MODEL_CONFIG_PATH=model_serve_config.json
```

## Volume Mounts

The docker-compose setup mounts:

- `./data` → MLflow database
- `./mlartifacts` → MLflow artifacts (models)
- `./model_serve_config.json` → API model configuration

## Networking

Both containers are in the same network `rotary-insight-network`:

- MLflow accessible at `http://mlflow:5000` (internal)
- API accessible at `http://api:8000` (internal)
- Both exposed to host on ports 5000 and 8000

## Production Deployment

### Use Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml rotary

# List services
docker stack services rotary

# Remove stack
docker stack rm rotary
```

### Use Kubernetes

Convert compose file to k8s manifests:

```bash
# Install kompose
curl -L https://github.com/kubernetes/kompose/releases/download/v1.31.2/kompose-linux-amd64 -o kompose
chmod +x kompose
sudo mv kompose /usr/local/bin/

# Convert
kompose convert -f docker-compose.yml

# Apply
kubectl apply -f .
```

### Behind Nginx Reverse Proxy

Create nginx configuration:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

For HTTPS, use certbot:

```bash
sudo certbot --nginx -d api.yourdomain.com
```

## Monitoring

### Container Stats

```bash
docker stats rotary-api rotary-mlflow
```

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
```

### Logging

View logs with timestamps:

```bash
docker compose logs -f --timestamps api
```

Configure logging driver in `docker-compose.yml`:

```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Backup and Restore

### Backup MLflow Data

```bash
# Backup database and artifacts
tar -czf mlflow-backup-$(date +%Y%m%d).tar.gz data/ mlartifacts/
```

### Restore

```bash
# Extract backup
tar -xzf mlflow-backup-YYYYMMDD.tar.gz

# Restart services
docker compose restart mlflow
```

## Troubleshooting

### API Can't Connect to MLflow

Check network connectivity:

```bash
docker exec rotary-api ping mlflow
docker exec rotary-api curl http://mlflow:5000/health
```

### Container Won't Start

Check logs:

```bash
docker compose logs api
```

Common issues:

- Missing `.env` file
- Invalid model configuration
- Port already in use

### Permission Issues

If you get permission errors with volumes:

```bash
sudo chown -R $USER:$USER data/ mlartifacts/
```

### Out of Memory

Increase Docker memory limit in Docker Desktop settings or add swap:

```bash
# Add to docker-compose.yml
services:
  api:
    mem_limit: 4g
    memswap_limit: 8g
```

## Health Checks

Both containers have health checks:

```bash
# Check health status
docker inspect rotary-api | grep -A 10 Health
docker inspect rotary-mlflow | grep -A 10 Health

# Wait for healthy
docker compose up -d
until [ "$(docker inspect -f {{.State.Health.Status}} rotary-api)" == "healthy" ]; do
    echo "Waiting for API to be healthy..."
    sleep 2
done
echo "API is healthy!"
```

## Security Best Practices

1. ✅ Don't expose MLflow port (5000) to public internet
2. ✅ Use secrets management (Docker secrets, Kubernetes secrets)
3. ✅ Scan images for vulnerabilities
4. ✅ Run containers as non-root user (already configured)
5. ✅ Use read-only file systems where possible
6. ✅ Keep images updated

## Performance Optimization

1. Use multi-stage builds (already done)
2. Enable BuildKit: `export DOCKER_BUILDKIT=1`
3. Use layer caching
4. Minimize image size
5. Use GPU support if available:

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

For more information, see:

- `API_QUICKSTART.md` - Quick start guide
- `ENV_SETUP.md` - Environment setup
- `server/README.md` - API documentation

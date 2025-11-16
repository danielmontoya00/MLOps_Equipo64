# Docker Commands for MLOps Project

## Build and Run

### Build the Docker image
```bash
docker build -t obesity-estimation:latest .
```

### Run with Docker Compose (Recommended)
```bash
# Start all services (API, MLflow, Jupyter)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Run individual services
```bash
# Run API only
docker-compose up -d api

# Run MLflow only
docker-compose up -d mlflow

# Run Jupyter only
docker-compose up -d jupyter
```

## Development

### Run with local code mounted (for development)
```bash
docker run -it --rm \
  -v $(pwd)/Obesity_Estimation:/app/Obesity_Estimation \
  -p 8000:8000 \
  obesity-estimation:latest
```

### Access running container
```bash
docker exec -it obesity_api bash
```

### Run tests inside container
```bash
docker-compose exec api python -m pytest tests
```

## Training

### Run training pipeline in container
```bash
docker-compose exec api python Obesity_Estimation/Module/run_pipeline.py
```

### Train model with mounted data
```bash
docker run -it --rm \
  -v $(pwd)/Obesity_Estimation/data:/app/Obesity_Estimation/data \
  -v $(pwd)/Obesity_Estimation/models:/app/Obesity_Estimation/models \
  -v $(pwd)/mlruns:/app/mlruns \
  obesity-estimation:latest \
  python Obesity_Estimation/models/train_model.py
```

## Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Jupyter**: http://localhost:8888

## Production Deployment

### Tag and push to registry
```bash
docker tag obesity-estimation:latest your-registry/obesity-estimation:v1.0.0
docker push your-registry/obesity-estimation:v1.0.0
```

### Run in production mode
```bash
docker run -d \
  --name obesity-api \
  -p 8000:8000 \
  -e MLFLOW_MODEL_URI=models:/obesity_classifier/Production \
  -v /path/to/models:/app/Obesity_Estimation/models:ro \
  --restart unless-stopped \
  obesity-estimation:latest
```

## Maintenance

### Clean up
```bash
# Remove stopped containers
docker-compose down

# Remove images
docker rmi obesity-estimation:latest

# Remove all (containers, volumes, networks)
docker-compose down -v

# Clean build cache
docker builder prune
```

### Monitor resource usage
```bash
docker stats
```

### View container logs
```bash
docker-compose logs -f api
docker-compose logs -f mlflow
```

## Troubleshooting

### Check service health
```bash
docker-compose ps
```

### Inspect container
```bash
docker inspect obesity_api
```

### Debug failing container
```bash
docker-compose logs api
docker-compose exec api bash
```

### Restart specific service
```bash
docker-compose restart api
```

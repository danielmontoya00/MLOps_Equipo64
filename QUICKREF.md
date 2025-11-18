# Quick Reference - Docker Commands

## Essential Commands (As Requested)

### Build Image
```bash
docker build -t ml-service:latest .
```

### Run Container
```bash
docker run -p 8000:8000 ml-service:latest
```

## Access Services

- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/
- **MLflow UI**: http://localhost:5000
- **Jupyter Notebook**: http://localhost:8888

## Development Workflow

### 1. Build and Test Locally
```bash
# Build
docker build -t ml-service:latest .

# Test dependencies
docker run --rm ml-service:latest python -c "import fastapi, mlflow, pandas, sklearn"

# Run API
docker run -p 8000:8000 ml-service:latest

# Test API
curl http://localhost:8000/
curl http://localhost:8000/docs
```

### 2. Run with Docker Compose
```bash
# Start all services (API + MLflow + Jupyter)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Publish to DockerHub
```bash
# Login
docker login

# Tag
docker tag ml-service:latest yourusername/ml-service:1.0.0
docker tag ml-service:latest yourusername/ml-service:latest

# Push
docker push yourusername/ml-service:1.0.0
docker push yourusername/ml-service:latest

# Or use automated script
./publish-dockerhub.sh 1.0.0
```

### 4. Pull and Use Published Image
```bash
# Pull from DockerHub
docker pull yourusername/ml-service:1.0.0

# Run published image
docker run -p 8000:8000 yourusername/ml-service:1.0.0
```

## Advanced Usage

### Run with Environment Variables
```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e MLFLOW_MODEL_URI=models:/obesity_classifier/Production \
  ml-service:latest
```

### Run with Volume Mounts
```bash
docker run -p 8000:8000 \
  -v $(pwd)/Obesity_Estimation/models:/app/Obesity_Estimation/models:ro \
  -v $(pwd)/Obesity_Estimation/data:/app/Obesity_Estimation/data:ro \
  ml-service:latest
```

### Run in Background (Detached)
```bash
docker run -d -p 8000:8000 --name ml-api ml-service:latest
```

### View Container Logs
```bash
docker logs -f ml-api
```

### Execute Commands in Container
```bash
# Open bash shell
docker exec -it ml-api bash

# Run Python script
docker exec ml-api python Obesity_Estimation/Module/run_pipeline.py

# Run tests
docker exec ml-api python -m pytest tests
```

## Image Management

### List Images
```bash
docker images ml-service
```

### Remove Image
```bash
docker rmi ml-service:latest
```

### Check Image Size
```bash
docker images ml-service:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

### Inspect Image
```bash
docker inspect ml-service:latest
docker history ml-service:latest
```

## Container Management

### List Running Containers
```bash
docker ps
```

### List All Containers
```bash
docker ps -a
```

### Stop Container
```bash
docker stop ml-api
```

### Remove Container
```bash
docker rm ml-api
```

### Container Stats
```bash
docker stats ml-api
```

## Cleanup

### Remove Stopped Containers
```bash
docker container prune
```

### Remove Unused Images
```bash
docker image prune -a
```

### Remove Everything (Careful!)
```bash
docker system prune -a --volumes
```

## Troubleshooting

### Build Without Cache
```bash
docker build --no-cache -t ml-service:latest .
```

### Check Container Health
```bash
docker inspect --format='{{.State.Health.Status}}' ml-api
```

### View Build Output
```bash
docker build --progress=plain -t ml-service:latest .
```

### Test Network Connectivity
```bash
docker exec ml-api curl http://localhost:8000/
```

## Version Tags

```bash
# Semantic versioning
docker tag ml-service:latest yourusername/ml-service:1.0.0    # Specific version
docker tag ml-service:latest yourusername/ml-service:1.0      # Minor version
docker tag ml-service:latest yourusername/ml-service:1        # Major version
docker tag ml-service:latest yourusername/ml-service:latest   # Latest

# Date-based
docker tag ml-service:latest yourusername/ml-service:2025-11-16

# Environment-based
docker tag ml-service:latest yourusername/ml-service:production
docker tag ml-service:latest yourusername/ml-service:staging
```

## Make Commands (Shortcuts)

```bash
make docker-build      # Build image
make docker-up         # Start all services
make docker-down       # Stop all services
make docker-logs       # View logs
make docker-test       # Run tests
make docker-shell      # Open shell in container
make docker-clean      # Clean up resources
make docker-rebuild    # Rebuild and restart
```

## Automated Scripts

```bash
# Build and test
./build.sh

# Publish to DockerHub
./publish-dockerhub.sh 1.0.0

# Verify setup
./setup-verify.sh
```

## CI/CD (GitHub Actions)

### Manual Trigger
1. Go to Actions tab
2. Select "Publish to DockerHub"
3. Click "Run workflow"
4. Enter version (e.g., 1.0.0)

### Automatic on Tag
```bash
git tag v1.0.0
git push origin v1.0.0
```

## Production Deployment

```bash
# Pull from DockerHub
docker pull yourusername/ml-service:1.0.0

# Run with restart policy
docker run -d \
  --name obesity-api \
  -p 8000:8000 \
  -e MLFLOW_MODEL_URI=models:/obesity_classifier/Production \
  --restart unless-stopped \
  yourusername/ml-service:1.0.0

# Check status
docker ps
docker logs obesity-api
```

## Security

### Scan for Vulnerabilities
```bash
# Using Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image ml-service:latest
```

### Check User
```bash
docker exec ml-api whoami  # Should be 'appuser', not 'root'
```

## Documentation

- **Full Docker Guide**: [DOCKER.md](DOCKER.md)
- **DockerHub Publishing**: [DOCKERHUB.md](DOCKERHUB.md)
- **Project README**: [README.md](README.md)

---

**Quick Start:**
```bash
docker build -t ml-service:latest . && docker run -p 8000:8000 ml-service:latest
```

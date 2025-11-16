# MLOps - Obesity Estimation Project

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/mlflow-tracking-green.svg)](https://mlflow.org/)

Machine Learning Operations project for obesity classification with MLflow tracking, FastAPI REST API, and Docker deployment.

## Quick Start with Docker 🐳

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+ (optional, for multi-service setup)

### Build and Run (Simple)

**Build the image:**
```bash
docker build -t ml-service:latest .
```

**Run the container:**
```bash
docker run -p 8000:8000 ml-service:latest
```

**Access the API:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/

### Run All Services (Advanced)
```bash
# Start API, MLflow, and Jupyter
docker-compose up -d

# View logs
docker-compose logs -f
```

**Access Services:**
- API Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Jupyter Notebook: http://localhost:8888

### API Usage Example
```bash
# Health check
curl http://localhost:8000/

# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"feature1": value1, "feature2": value2, ...}]}'
```

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd MLOps_Equipo64
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Services Locally

**Start MLflow Tracking Server:**
```bash
mlflow ui --port 5000
```

**Start FastAPI Server:**
```bash
cd Obesity_Estimation
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Launch Jupyter:**
```bash
jupyter notebook
```

## Project Structure

```
MLOps_Equipo64/
├── Dockerfile              # Production Docker image
├── docker-compose.yml      # Multi-service orchestration
├── .dockerignore          # Docker build exclusions
├── requirements.txt        # Python dependencies
├── DOCKER.md              # Detailed Docker documentation
├── build.sh               # Build automation script
├── .env.example           # Environment variables template
│
└── Obesity_Estimation/
    ├── Module/            # Pipeline orchestration
    ├── models/            # Trained models & artifacts
    ├── data/              # Dataset storage
    ├── notebooks/         # Jupyter notebooks
    ├── src/
    │   ├── api/          # FastAPI application
    │   ├── data/         # Data loading utilities
    │   ├── features/     # Feature engineering
    │   └── visualization/ # Plotting utilities
    └── tests/            # Unit & integration tests
```

## Docker Configuration

### Services

**API Service** (`obesity_api`)
- FastAPI REST API for model inference
- Port: 8000
- Health checks enabled
- Non-root user for security

**MLflow Service** (`mlflow_server`)
- Experiment tracking and model registry
- Port: 5000
- SQLite backend storage
- Persistent volume for artifacts

**Jupyter Service** (`obesity_jupyter`)
- Interactive development environment
- Port: 8888
- Access token disabled for development
- Shared volume with project code

### Docker Best Practices Implemented

✅ **Multi-stage builds** for optimized image size  
✅ **Non-root user** for enhanced security  
✅ **Layer caching** for faster builds  
✅ **Health checks** for service monitoring  
✅ **Environment variables** for configuration  
✅ **.dockerignore** to exclude unnecessary files  
✅ **Volume mounts** for data persistence  
✅ **Network isolation** with bridge networking  
✅ **Restart policies** for reliability  
✅ **Resource optimization** with slim base images  

## Development Workflow

### 1. Train Model
```bash
# Using Docker
docker-compose exec api python Obesity_Estimation/Module/run_pipeline.py

# Or locally
python Obesity_Estimation/Module/run_pipeline.py
```

### 2. Track Experiments
View experiments and metrics in MLflow UI at http://localhost:5000

### 3. Test API
```bash
# Run tests in container
docker-compose exec api python -m pytest tests

# Or locally
pytest tests
```

### 4. Code Quality
```bash
# Format code
make format

# Run linters
make lint
```

## Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Key variables:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MLFLOW_MODEL_URI`: Model URI for inference
- `API_PORT`: API service port
- `JUPYTER_TOKEN`: Jupyter authentication token

## Production Deployment

### Pull from DockerHub
```bash
# Pull latest version
docker pull yourusername/ml-service:latest

# Pull specific version
docker pull yourusername/ml-service:1.0.0

# Run published image
docker run -p 8000:8000 yourusername/ml-service:latest
```

### Build and Publish to DockerHub

**Step 1: Build Production Image**
```bash
docker build -t ml-service:latest .
```

**Step 2: Login to DockerHub**
```bash
docker login
```

**Step 3: Tag and Push**
```bash
# Set your DockerHub username
export DOCKERHUB_USER=yourusername

# Tag with version
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:1.0.0
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:latest

# Push to registry
docker push $DOCKERHUB_USER/ml-service:1.0.0
docker push $DOCKERHUB_USER/ml-service:latest
```

**Or use automated script:**
```bash
# Publishes with semantic versioning
./publish-dockerhub.sh 1.0.0
```

### Version Tags Strategy
```
ml-service:1.0.0    # Specific version (recommended for production)
ml-service:1.0      # Minor version
ml-service:1        # Major version
ml-service:latest   # Latest stable release
```

### Deploy to Production
```bash
docker run -d \
  --name obesity-api \
  -p 8000:8000 \
  -e MLFLOW_MODEL_URI=models:/obesity_classifier/Production \
  -v /path/to/models:/app/Obesity_Estimation/models:ro \
  --restart unless-stopped \
  yourusername/ml-service:1.0.0
```

## Monitoring & Maintenance

### View Logs
```bash
docker-compose logs -f api
docker-compose logs -f mlflow
```

### Monitor Resources
```bash
docker stats
```

### Backup Data
```bash
# Backup MLflow artifacts
docker cp mlflow_server:/mlflow/mlruns ./backup/mlruns

# Backup models
docker cp obesity_api:/app/Obesity_Estimation/models ./backup/models
```

## Troubleshooting

### Service won't start
```bash
docker-compose ps
docker-compose logs <service-name>
```

### API returns 503
Check that MLflow is running and model is registered:
```bash
curl http://localhost:5000/api/2.0/mlflow/registered-models/list
```

### Permission issues
Ensure volumes have correct permissions:
```bash
sudo chown -R $(id -u):$(id -g) ./Obesity_Estimation/models
```

## Documentation

- [Docker Commands](DOCKER.md) - Comprehensive Docker usage guide
- [DockerHub Publishing](DOCKERHUB.md) - Image publishing and versioning guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Contributing

1. Create feature branch
2. Make changes
3. Run tests: `make test`
4. Format code: `make format`
5. Submit pull request

## License

[Specify license]

## Team

MLOps Equipo 64

---

For detailed Docker commands and workflows, see [DOCKER.md](DOCKER.md)

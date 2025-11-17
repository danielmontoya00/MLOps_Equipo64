#!/bin/bash
# Script to build and test Docker image locally

set -e

echo "Building Docker image..."
docker build -t ml-service:latest .

echo "Testing image..."
docker run --rm ml-service:latest python -c "import fastapi, mlflow, pandas, sklearn; print('All dependencies installed successfully')"

echo "Build completed successfully!"
echo ""
echo "Quick Start Commands:"
echo "  Build: docker build -t ml-service:latest ."
echo "  Run:   docker run -p 8000:8000 ml-service:latest"
echo ""
echo "To run all services:"
echo "  docker-compose up -d"
echo ""
echo "To access services:"
echo "  API Docs:   http://localhost:8000/docs"
echo "  MLflow UI:  http://localhost:5000"
echo "  Jupyter:    http://localhost:8888"
echo ""
echo "To publish to DockerHub:"
echo "  ./publish-dockerhub.sh 1.0.0"


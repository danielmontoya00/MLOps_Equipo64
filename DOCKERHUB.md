# Docker Image Publishing Guide - DockerHub

## Image Naming Convention

This project uses the following image naming:
- **Local/Development**: `ml-service:latest`
- **DockerHub**: `<your-dockerhub-username>/ml-service:<version>`

## Build Commands

### Build the Image
```bash
docker build -t ml-service:latest .
```

### Run the Container
```bash
docker run -p 8000:8000 ml-service:latest
```

### Run with Environment Variables
```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 \
  -e MLFLOW_MODEL_URI=models:/obesity_classifier/Production \
  ml-service:latest
```

### Run with Volume Mounts
```bash
docker run -p 8000:8000 \
  -v $(pwd)/Obesity_Estimation/models:/app/Obesity_Estimation/models:ro \
  ml-service:latest
```

## Versioning Strategy

This project follows Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (e.g., API restructure)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes and small improvements

### Version Tags

```bash
# Development/Testing
ml-service:latest
ml-service:dev

# Production Releases
ml-service:1.0.0        # Specific version
ml-service:1.0          # Minor version
ml-service:1            # Major version

# Feature Branches
ml-service:feature-name

# Date-based
ml-service:2025-11-16
```

## Publishing to DockerHub

### Step 1: Login to DockerHub
```bash
docker login
# Enter your DockerHub username and password/token
```

### Step 2: Tag the Image
```bash
# Replace 'yourusername' with your DockerHub username
export DOCKERHUB_USER=yourusername
export VERSION=1.0.0

# Tag with version
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:$VERSION
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:1.0
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:1
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:latest
```

### Step 3: Push to DockerHub
```bash
# Push all tags
docker push $DOCKERHUB_USER/ml-service:$VERSION
docker push $DOCKERHUB_USER/ml-service:1.0
docker push $DOCKERHUB_USER/ml-service:1
docker push $DOCKERHUB_USER/ml-service:latest
```

### Quick Script for Publishing
```bash
#!/bin/bash
# publish-docker.sh

VERSION=$1
DOCKERHUB_USER=${DOCKERHUB_USER:-yourusername}

if [ -z "$VERSION" ]; then
    echo "Usage: ./publish-docker.sh <version>"
    echo "Example: ./publish-docker.sh 1.0.0"
    exit 1
fi

echo "Building ml-service:latest..."
docker build -t ml-service:latest .

echo "Tagging version $VERSION..."
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:$VERSION
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:latest

echo "Pushing to DockerHub..."
docker push $DOCKERHUB_USER/ml-service:$VERSION
docker push $DOCKERHUB_USER/ml-service:latest

echo "✓ Successfully published $DOCKERHUB_USER/ml-service:$VERSION"
```

## Pull and Use Published Image

### Pull from DockerHub
```bash
docker pull yourusername/ml-service:latest
docker pull yourusername/ml-service:1.0.0
```

### Run Published Image
```bash
docker run -p 8000:8000 yourusername/ml-service:latest
```

## Complete Workflow Example

```bash
# 1. Build the image
docker build -t ml-service:latest .

# 2. Test locally
docker run -d -p 8000:8000 --name ml-service-test ml-service:latest

# 3. Test the API
curl http://localhost:8000/
curl http://localhost:8000/docs

# 4. Stop test container
docker stop ml-service-test
docker rm ml-service-test

# 5. Tag for DockerHub
export DOCKERHUB_USER=yourusername
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:1.0.0
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:latest

# 6. Login to DockerHub
docker login

# 7. Push to registry
docker push $DOCKERHUB_USER/ml-service:1.0.0
docker push $DOCKERHUB_USER/ml-service:latest

# 8. Verify on DockerHub
# Visit: https://hub.docker.com/r/yourusername/ml-service
```

## Image Information

### Check Image Details
```bash
# Size and layers
docker images ml-service:latest

# Image history
docker history ml-service:latest

# Detailed inspection
docker inspect ml-service:latest
```

### Image Layers
The image is built with the following layers:
1. Base: python:3.10-slim
2. System dependencies (gcc, g++)
3. Python dependencies (from requirements.txt)
4. Application code (Obesity_Estimation/)
5. User permissions (non-root user)

### Image Size Optimization
Current optimizations:
- ✓ Using slim Python base image
- ✓ Multi-stage build potential
- ✓ Cleaning apt cache
- ✓ No pip cache
- ✓ .dockerignore excludes unnecessary files

Expected image size: ~800-1200 MB (includes ML libraries)

## Automated Publishing with GitHub Actions

Add to `.github/workflows/docker-publish.yml`:

```yaml
name: Publish Docker Image

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Extract version from tag
        id: vars
        run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/ml-service:${{ steps.vars.outputs.VERSION }}
            ${{ secrets.DOCKERHUB_USERNAME }}/ml-service:latest
```

## Version History Template

Document your releases:

### Version 1.0.0 (2025-11-16)
- Initial production release
- FastAPI service for obesity classification
- MLflow integration
- Health checks and monitoring
- Non-root user security

**Pull command:**
```bash
docker pull yourusername/ml-service:1.0.0
```

### Version 0.9.0 (2025-11-10)
- Beta release
- Core ML model integration
- Basic API endpoints

**Pull command:**
```bash
docker pull yourusername/ml-service:0.9.0
```

## Security Considerations

### Image Security Best Practices
✓ Non-root user (appuser)
✓ Minimal base image (python:3.10-slim)
✓ No secrets in image
✓ Read-only mounts for sensitive data
✓ Regular security scanning

### Scan Image for Vulnerabilities
```bash
# Using Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image ml-service:latest

# Using Docker Scout (if available)
docker scout cves ml-service:latest
```

## Cleanup

### Remove Local Images
```bash
docker rmi ml-service:latest
docker rmi yourusername/ml-service:1.0.0
```

### Prune Unused Images
```bash
docker image prune -a
```

## Troubleshooting

### Image won't build
```bash
# Clean build without cache
docker build --no-cache -t ml-service:latest .
```

### Push fails
```bash
# Verify login
docker login

# Check credentials
cat ~/.docker/config.json
```

### Image too large
```bash
# Analyze layers
docker history ml-service:latest

# Use dive tool
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest ml-service:latest
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker build -t ml-service:latest .` | Build image |
| `docker run -p 8000:8000 ml-service:latest` | Run container |
| `docker tag ml-service:latest user/ml-service:1.0.0` | Tag image |
| `docker push user/ml-service:1.0.0` | Push to DockerHub |
| `docker pull user/ml-service:latest` | Pull from DockerHub |
| `docker images ml-service` | List local images |
| `docker rmi ml-service:latest` | Remove image |

## Support

For issues or questions:
- Check logs: `docker logs <container-id>`
- Inspect: `docker inspect ml-service:latest`
- DockerHub: https://hub.docker.com/r/yourusername/ml-service

---

**Note**: Replace `yourusername` with your actual DockerHub username throughout this guide.

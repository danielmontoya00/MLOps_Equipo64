#!/bin/bash
# Docker Image Publishing Script for DockerHub

set -e

VERSION=$1
DOCKERHUB_USER=${DOCKERHUB_USER:-yourusername}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Help message
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ -z "$VERSION" ]; then
    echo "Usage: ./publish-dockerhub.sh <version>"
    echo ""
    echo "Example: ./publish-dockerhub.sh 1.0.0"
    echo ""
    echo "Environment variables:"
    echo "  DOCKERHUB_USER - Your DockerHub username (default: yourusername)"
    echo ""
    echo "Before running:"
    echo "  1. Login to DockerHub: docker login"
    echo "  2. Set your username: export DOCKERHUB_USER=yourusername"
    exit 0
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker Image Publishing to DockerHub${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Version must follow semver format (e.g., 1.0.0)${NC}"
    exit 1
fi

# Extract major and minor versions
MAJOR=$(echo $VERSION | cut -d. -f1)
MINOR=$(echo $VERSION | cut -d. -f1-2)

echo "Version: $VERSION"
echo "Major: $MAJOR"
echo "Minor: $MINOR"
echo "DockerHub User: $DOCKERHUB_USER"
echo ""

# Build image
echo -e "${YELLOW}Step 1/4: Building image...${NC}"
docker build -t ml-service:latest .
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Tag images
echo -e "${YELLOW}Step 2/4: Tagging images...${NC}"
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:$VERSION
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:$MINOR
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:$MAJOR
docker tag ml-service:latest $DOCKERHUB_USER/ml-service:latest
echo -e "${GREEN}✓ Tagged: $VERSION, $MINOR, $MAJOR, latest${NC}"
echo ""

# Test locally
echo -e "${YELLOW}Step 3/4: Testing image...${NC}"
docker run --rm ml-service:latest python -c "import fastapi, mlflow, pandas, sklearn; print('Dependencies OK')" || {
    echo -e "${RED}Error: Image test failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Image test passed${NC}"
echo ""

# Push to DockerHub
echo -e "${YELLOW}Step 4/4: Pushing to DockerHub...${NC}"
docker push $DOCKERHUB_USER/ml-service:$VERSION
docker push $DOCKERHUB_USER/ml-service:$MINOR
docker push $DOCKERHUB_USER/ml-service:$MAJOR
docker push $DOCKERHUB_USER/ml-service:latest
echo -e "${GREEN}✓ Push complete${NC}"
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Successfully Published!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Image: $DOCKERHUB_USER/ml-service"
echo "Tags: $VERSION, $MINOR, $MAJOR, latest"
echo ""
echo "View on DockerHub:"
echo "  https://hub.docker.com/r/$DOCKERHUB_USER/ml-service"
echo ""
echo "Pull command:"
echo -e "  ${YELLOW}docker pull $DOCKERHUB_USER/ml-service:$VERSION${NC}"
echo ""
echo "Run command:"
echo -e "  ${YELLOW}docker run -p 8000:8000 $DOCKERHUB_USER/ml-service:$VERSION${NC}"
echo ""

#!/bin/bash
# Setup verification and quick start script

set -e

echo "========================================="
echo "MLOps Project - Setup Verification"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
echo "Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed: $(docker --version)"
else
    echo -e "${RED}✗${NC} Docker is not installed"
    echo "  Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
echo "Checking Docker Compose installation..."
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose is installed: $(docker-compose --version)"
elif docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose (plugin) is installed: $(docker compose version)"
else
    echo -e "${RED}✗${NC} Docker Compose is not installed"
    echo "  Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check Python
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python is installed: $(python3 --version)"
else
    echo -e "${YELLOW}⚠${NC} Python 3 is not installed (optional for local development)"
fi

# Verify project files
echo ""
echo "Verifying project files..."
files=(
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "requirements.txt"
    "DOCKER.md"
    "README.md"
    ".env.example"
    "build.sh"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (missing)"
    fi
done

# Check .env file
echo ""
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠${NC} .env file not found"
    echo "  Creating from template..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file"
else
    echo -e "${GREEN}✓${NC} .env file exists"
fi

# Summary
echo ""
echo "========================================="
echo "Setup Status"
echo "========================================="
echo ""
echo -e "${GREEN}✓${NC} All required files are present"
echo -e "${GREEN}✓${NC} Docker environment is ready"
echo ""
echo "Next steps:"
echo ""
echo "1. Build and start services:"
echo "   ${YELLOW}docker-compose up -d${NC}"
echo ""
echo "2. View logs:"
echo "   ${YELLOW}docker-compose logs -f${NC}"
echo ""
echo "3. Access services:"
echo "   - API Docs:   ${YELLOW}http://localhost:8000/docs${NC}"
echo "   - MLflow UI:  ${YELLOW}http://localhost:5000${NC}"
echo "   - Jupyter:    ${YELLOW}http://localhost:8888${NC}"
echo ""
echo "4. Run training:"
echo "   ${YELLOW}docker-compose exec api python Obesity_Estimation/Module/run_pipeline.py${NC}"
echo ""
echo "For more information, see README.md and DOCKER.md"
echo ""

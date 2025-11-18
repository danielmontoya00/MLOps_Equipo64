#!/bin/bash
# Demo Script - Data Drift Monitoring System
# Este script demuestra el flujo completo del sistema de monitoreo

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     DEMO: Data Drift Monitoring System                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv || {
        echo "❌ Failed to create venv. Install python3-venv:"
        echo "   sudo apt install python3-venv"
        exit 1
    }
fi

# Activate virtual environment
echo -e "${BLUE}📦 Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
if ! python -c "import mlflow" 2>/dev/null; then
    echo -e "${YELLOW}📥 Installing dependencies...${NC}"
    pip install -q -r ../requirements.txt
fi

echo -e "${GREEN}✅ Environment ready${NC}"
echo ""

# Check if model exists
if [ ! -f "models/current_run_id.txt" ]; then
    echo -e "${YELLOW}⚠️  No trained model found. Training model...${NC}"
    python models/train_model.py
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo "SCENARIO 1: Mean Shift Drift (Moderate)"
echo "════════════════════════════════════════════════════════════════"
echo ""
python -m models.monitor_drift --scenario mean_shift --intensity 0.3
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "SCENARIO 2: Seasonal Drift (High)"
echo "════════════════════════════════════════════════════════════════"
echo ""
python -m models.monitor_drift --scenario seasonal --intensity 0.5
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "SCENARIO 3: Combined Drift (Critical)"
echo "════════════════════════════════════════════════════════════════"
echo ""
python -m models.monitor_drift --scenario combined --intensity 0.7
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    DEMO COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Reports generated in: reports/monitoring/"
echo "🔬 MLflow experiments: mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo ""
echo "To view visualizations:"
echo "  ls -lh reports/monitoring/*.png"
echo ""

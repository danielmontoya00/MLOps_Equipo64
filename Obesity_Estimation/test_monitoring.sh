#!/bin/bash
# Test script for Data Drift Monitoring System

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Data Drift Monitoring - Test Suite                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if in Obesity_Estimation directory
if [ ! -f "models/train_model.py" ]; then
    echo "❌ Error: Must run from Obesity_Estimation directory"
    exit 1
fi

echo "📋 Test 1: Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
echo "✅ Python3 found: $(python3 --version)"

echo ""
echo "📋 Test 2: Checking required packages..."
REQUIRED_PACKAGES="numpy pandas scipy sklearn matplotlib seaborn mlflow"
MISSING_PACKAGES=""

for pkg in $REQUIRED_PACKAGES; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "⚠️  Missing packages:$MISSING_PACKAGES"
    echo "   Install with: pip install -r ../requirements.txt"
    echo "   (Skipping package tests, structure tests will continue)"
else
    echo "✅ All required packages installed"
fi

echo ""
echo "📋 Test 3: Checking directory structure..."
REQUIRED_DIRS=(
    "src/monitoring"
    "models"
    "data/processed"
    "reports"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
    else
        echo "❌ $dir missing"
        exit 1
    fi
done

echo ""
echo "📋 Test 4: Checking monitoring module files..."
REQUIRED_FILES=(
    "src/monitoring/__init__.py"
    "src/monitoring/drift_simulator.py"
    "src/monitoring/drift_detector.py"
    "src/monitoring/performance_monitor.py"
    "models/monitor_drift.py"
    "MONITORING.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists ($(wc -l < "$file") lines)"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo ""
echo "📋 Test 5: Checking baseline data..."
if [ -f "data/processed/X_test.csv" ] && [ -f "data/processed/y_test.csv" ]; then
    echo "✅ Baseline test data exists"
    echo "   X_test.csv: $(wc -l < data/processed/X_test.csv) rows"
    echo "   y_test.csv: $(wc -l < data/processed/y_test.csv) rows"
else
    echo "❌ Baseline test data missing"
    echo "   Run: python models/train_model.py to generate data"
    exit 1
fi

echo ""
echo "📋 Test 6: Checking model files..."
if [ -f "models/current_run_id.txt" ] || [ -f "models/model_info.json" ]; then
    echo "✅ Model metadata found"
    if [ -f "models/current_run_id.txt" ]; then
        RUN_ID=$(cat models/current_run_id.txt)
        echo "   Current run ID: $RUN_ID"
        if [ -f "models/obesity_classifier_${RUN_ID}.pkl" ]; then
            echo "✅ Model file exists: obesity_classifier_${RUN_ID}.pkl"
        else
            echo "⚠️  Model file not found for run ID: $RUN_ID"
        fi
    fi
else
    echo "⚠️  Model metadata not found"
    echo "   Train a model first: python models/train_model.py"
fi

echo ""
echo "📋 Test 7: Python syntax check..."
for file in "${REQUIRED_FILES[@]}"; do
    if [[ $file == *.py ]]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo "✅ $file - syntax OK"
        else
            echo "❌ $file - syntax error"
            exit 1
        fi
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ ALL TESTS PASSED                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Ready to run monitoring pipeline!"
echo ""
echo "Quick start:"
echo "  1. Ensure dependencies installed: pip install -r ../requirements.txt"
echo "  2. Train model (if not done): python3 models/train_model.py"
echo "  3. Run monitoring: python3 -m models.monitor_drift --scenario mean_shift"
echo ""
echo "For more info, see MONITORING.md"

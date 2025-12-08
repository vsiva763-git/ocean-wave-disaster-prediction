#!/bin/bash
# Quick start script for Ocean Wave Disaster Prediction Web Interface

set -e

echo "🌊 Ocean Wave Disaster Prediction System - Quick Start"
echo "======================================================="
echo ""

# Check if running from project root
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
if python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "✓ Dependencies already installed"
else
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
fi
echo ""

# Start the server
echo "Starting web server..."
echo "→ Web interface will be available at: http://localhost:8000"
echo "→ API documentation will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd src
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000

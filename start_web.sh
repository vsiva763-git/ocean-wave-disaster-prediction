#!/bin/bash
# ============================================================================
# Kanyakumari Ocean Wave & Tsunami Prediction System
# Linux/Mac Startup Script
# ============================================================================

echo ""
echo "================================================================================"
echo "   🌊 KANYAKUMARI OCEAN WAVE & TSUNAMI PREDICTION SYSTEM"
echo "================================================================================"
echo ""
echo "   📍 Location: Kanyakumari, Tamil Nadu, India"
echo "   🌐 Web Interface will be available at: http://localhost:8000"
echo "   📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

# Check/activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
    echo "   Consider creating one with: python3 -m venv venv"
fi

# Install dependencies if needed
echo ""
echo "📋 Checking dependencies..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies!"
        exit 1
    fi
fi

# Start the server
echo ""
echo "🚀 Starting the Ocean Wave Prediction Server..."
echo ""
echo "   Press Ctrl+C to stop the server."
echo ""
echo "================================================================================"
echo ""

# Run the API server
cd src
python3 -m uvicorn kanyakumari_api:app --host 0.0.0.0 --port 8000 --reload

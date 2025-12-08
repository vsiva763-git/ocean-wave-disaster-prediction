@echo off
REM ============================================================================
REM Kanyakumari Ocean Wave & Tsunami Prediction System
REM Windows Startup Script
REM ============================================================================

echo.
echo ================================================================================
echo    🌊 KANYAKUMARI OCEAN WAVE ^& TSUNAMI PREDICTION SYSTEM
echo ================================================================================
echo.
echo    📍 Location: Kanyakumari, Tamil Nadu, India
echo    🌐 Web Interface will be available at: http://localhost:8000
echo    📚 API Documentation: http://localhost:8000/docs
echo.
echo ================================================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Using system Python.
    echo    Consider creating one with: python -m venv venv
)

REM Install dependencies if needed
echo.
echo 📋 Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Start the server
echo.
echo 🚀 Starting the Ocean Wave Prediction Server...
echo.
echo    Press Ctrl+C to stop the server.
echo.
echo ================================================================================
echo.

REM Run the API server
cd src
python -m uvicorn kanyakumari_api:app --host 0.0.0.0 --port 8000 --reload

REM If server exits, pause before closing
echo.
echo Server stopped.
pause

@echo off
REM Quick start script for Ocean Wave Disaster Prediction Web Interface (Windows)

echo.
echo =======================================================
echo Ocean Wave Disaster Prediction System - Quick Start
echo =======================================================
echo.

REM Check if running from project root
if not exist "requirements.txt" (
    echo Error: Please run this script from the project root directory
    exit /b 1
)

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
python --version
echo.

REM Install dependencies if needed
echo Checking dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -q -r requirements.txt
    echo Dependencies installed
) else (
    echo Dependencies already installed
)
echo.

REM Start the server
echo Starting web server...
echo.
echo Web interface will be available at: http://localhost:8000
echo API documentation will be available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd src
python -m uvicorn api:app --host 0.0.0.0 --port 8000

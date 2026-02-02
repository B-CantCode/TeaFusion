@echo off
REM Tea Doctor - Quick Start for Windows
REM This batch file sets up and runs the app in one click

echo.
echo ======================================
echo   Tea Doctor - Setup & Launch
echo ======================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Python version OK
python --version

REM Create virtual environment if it doesn't exist
if not exist .venv (
    echo [2/5] Creating virtual environment...
    python -m venv .venv
    echo        Virtual environment created!
) else (
    echo [2/5] Virtual environment already exists
)

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call .venv\Scripts\Activate.bat

REM Install/upgrade dependencies
echo [4/5] Installing dependencies...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo        Dependencies installed!

REM Check model file
echo [5/5] Checking model file...
if not exist tea_doctor_v7_final.tflite (
    echo        WARNING: Model file not found!
    echo        App will run in DEMO mode
    echo        Place 'tea_doctor_v7_final.tflite' to enable predictions
) else (
    echo        Model file found! Ready for predictions
)

echo.
echo ======================================
echo   Launching Tea Doctor...
echo ======================================
echo.
echo The app will open in your browser at:
echo   http://localhost:8501
echo.
echo To access from other devices, use:
echo   Check terminal for Network URL
echo.

REM Run the app
streamlit run tea_doctor_TFLITE_fixed.py

pause

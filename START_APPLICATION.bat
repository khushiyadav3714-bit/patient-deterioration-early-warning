@echo off
title Patient Deterioration Early Warning System
color 0A

echo ==============================================================================
echo       PATIENT DETERIORATION EARLY WARNING SYSTEM - AUTO LAUNCHER
echo ==============================================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

echo [OK] Python detected!
echo.
echo [2/3] Checking and installing required packages...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] There was an issue installing dependencies. Attempting to start anyway...
)

echo.
echo [3/3] Launching Patient Deterioration Early Warning System...
echo The application will automatically open in your default browser.
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C in this terminal window whenever you wish to stop the application.
echo ==============================================================================
echo.

python -m streamlit run app.py

pause

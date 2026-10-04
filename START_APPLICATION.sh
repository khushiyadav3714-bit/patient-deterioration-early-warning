#!/usr/bin/env bash
echo "=============================================================================="
echo "      PATIENT DETERIORATION EARLY WARNING SYSTEM - AUTO LAUNCHER"
echo "=============================================================================="
echo ""
echo "[1/3] Checking Python installation..."
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[OK] Python detected!"
echo ""
echo "[2/3] Checking and installing required packages..."
python3 -m pip install -r requirements.txt

echo ""
echo "[3/3] Launching Patient Deterioration Early Warning System..."
echo "The application will open in your default browser at: http://localhost:8501"
echo ""
python3 -m streamlit run app.py

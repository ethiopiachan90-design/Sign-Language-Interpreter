@echo off
echo =======================================================
echo    SIGN LANGUAGE INTERPRETER - ONE-CLICK SETUP
echo =======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11 from python.org before running.
    pause
    exit /b
)

:: Create Virtual Environment if not exists
if not exist usb_env (
    echo [1/3] Creating virtual environment...
    python -m venv usb_env
)

echo [2/3] Installing/Updating dependencies (this may take a minute)...
usb_env\Scripts\python.exe -m pip install --upgrade pip
usb_env\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo [3/3] LAUNCHING DASHBOARD...
echo =======================================================
usb_env\Scripts\python.exe final_pred.py
pause

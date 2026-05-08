@echo off
setlocal EnableDelayedExpansion
title Sign Language Interpreter - Setup ^& Run
color 0A

echo.
echo  ================================================================
echo     SIGN LANGUAGE TO TEXT ^& SPEECH  ^|  AUTO SETUP ^& LAUNCH
echo  ================================================================
echo.

:: ── STEP 1: Check Python 3.11 ──────────────────────────────────────
echo  [1/5] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Python is NOT installed or not in PATH.
    echo.
    echo  Please install Python 3.11.9 from:
    echo  https://www.python.org/downloads/release/python-3119/
    echo.
    echo  IMPORTANT: During install, CHECK "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set PYVER=%%V
for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
    set PYMAJ=%%A
    set PYMIN=%%B
)
if not "!PYMAJ!"=="3" goto :wrongpython
if !PYMIN! LSS 11 goto :wrongpython
echo  [OK] Python !PYVER! detected.
goto :pythonok

:wrongpython
echo.
echo  [ERROR] Python 3.11 is required. You have Python !PYVER!
echo  Download Python 3.11.9 from: https://www.python.org/downloads/release/python-3119/
echo.
pause
exit /b 1

:pythonok

:: ── STEP 2: Create Virtual Environment ────────────────────────────
echo.
echo  [2/5] Setting up virtual environment...
if not exist "env\" (
    echo  Creating env\... (first time only)
    python -m venv env
    if %errorlevel% neq 0 (
        echo  [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
) else (
    echo  [OK] Virtual environment already exists.
)

:: ── STEP 3: Install Exact Dependencies ────────────────────────────
echo.
echo  [3/5] Installing dependencies...
echo  (First time: this takes 3-10 minutes depending on internet speed)
echo.

env\Scripts\python.exe -m pip install --upgrade pip --quiet --no-warn-script-location
env\Scripts\python.exe -m pip install -r requirements.txt --quiet --no-warn-script-location

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Some packages failed to install.
    echo  Try running manually: env\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo  [OK] All dependencies installed.

:: ── STEP 4: Check/Download MediaPipe Hand Model ───────────────────
echo.
echo  [4/5] Checking MediaPipe hand landmarker model...
if not exist "hand_landmarker.task" (
    echo  Downloading hand_landmarker.task (~7.8 MB)...
    env\Scripts\python.exe -c ^
        "import urllib.request, sys; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task'); print('  [OK] Downloaded.')"
    if %errorlevel% neq 0 (
        echo  [ERROR] Download failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
) else (
    echo  [OK] hand_landmarker.task found.
)

:: ── STEP 5: Check CNN Model File ──────────────────────────────────
echo.
echo  [5/5] Checking CNN sign language model...
if not exist "cnn8grps_rad1_model.h5" (
    echo.
    echo  ============================================================
    echo   [ACTION REQUIRED] cnn8grps_rad1_model.h5 is MISSING!
    echo.
    echo   Copy this file into the same folder as this .bat file:
    echo   %~dp0
    echo.
    echo   The model file is ~13 MB. Get it from:
    echo    - Your USB drive / Google Drive / OneDrive backup
    echo    - Your old laptop (copy it manually)
    echo  ============================================================
    echo.
    pause
    exit /b 1
) else (
    echo  [OK] cnn8grps_rad1_model.h5 found.
)

:: ── ALL GOOD - LAUNCH ─────────────────────────────────────────────
echo.
echo  ================================================================
echo   Setup complete! Launching Sign Language Interpreter...
echo  ================================================================
echo.
echo  Controls:
echo    - Show ASL hand gesture in the camera box
echo    - Hold steady for ~0.5 seconds to register a letter
echo    - Press SPEAK button to hear the sentence
echo    - Press CLEAR to reset
echo.

env\Scripts\python.exe final_pred.py

echo.
echo  ================================================================
echo   Application closed.
echo  ================================================================
pause

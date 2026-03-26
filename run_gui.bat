@echo off
cd /d "%~dp0"
title Mister Mobile Cropper - Local GUI

echo ============================================
echo   Mister Mobile Cropper GUI (Local Run)
echo   Version: Python Source Code
echo ============================================
echo.

if not exist ".\venv\Scripts\python.exe" (
    echo [ERROR] Missing virtual environment!
    echo         Please create it or test if Python paths are correct.
    echo.
    pause
    exit /b 1
)

echo Starting core layout...
.\venv\Scripts\python.exe gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The application crashed or was closed.
    pause
)

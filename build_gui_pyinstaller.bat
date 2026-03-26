@echo off
setlocal EnableDelayedExpansion
REM ============================================
REM   PyInstaller Build Script [GUI]
REM   Compiles gui.py into a standalone app
REM ============================================

cd /d "%~dp0"

set "PYTHON_EXE=.\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Missing venv Python at %PYTHON_EXE%
    pause
    exit /b 1
)

echo.
echo [1/4] Syncing dependencies from requirements_gui.txt...
"%PYTHON_EXE%" -m pip install -r requirements_gui.txt

echo.
echo [2/4] Checking PyInstaller...
"%PYTHON_EXE%" -m pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Missing PyInstaller in venv. Installing...
    "%PYTHON_EXE%" -m pip install pyinstaller
) else (
    echo   PyInstaller already installed.
)

echo.
echo [3/4] Cleaning stale build artifacts...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "MisterMobileCropper-GUI.spec" del /q "MisterMobileCropper-GUI.spec"
echo   Cleaned.

echo.
echo [4/4] Building standalone executable with PyInstaller...
"%PYTHON_EXE%" -m PyInstaller ^
    --onedir ^
    --noconsole ^
    --clean ^
    --name="MisterMobileCropper-GUI" ^
    --collect-all="ultralytics" ^
    --collect-all="customtkinter" ^
    --collect-all="zxingcpp" ^
    --add-data="MM Watermark.png;." ^
    gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo [POST] Setting up runtime folders...
set "DIST_APP=dist\MisterMobileCropper-GUI"

if not exist "%DIST_APP%\input" mkdir "%DIST_APP%\input"
if not exist "%DIST_APP%\output" mkdir "%DIST_APP%\output"

echo   Copying AI model [best.pt] to output dist...
mkdir "%DIST_APP%\runs\detect\carocrop_custom_fast4\weights" 2>nul
xcopy /Y "runs\detect\carocrop_custom_fast4\weights\best.pt" "%DIST_APP%\runs\detect\carocrop_custom_fast4\weights\"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to copy weights file explicitly.
)

echo.
echo ============================================
echo   Build complete! (PyInstaller)
echo ============================================
echo.
echo   %DIST_APP%\
echo     MisterMobileCropper-GUI.exe
echo     MM Watermark.png
echo     runs\   (AI model)
echo     input\   put images here
echo     output\  results appear here
echo.
echo Build Script Finished.

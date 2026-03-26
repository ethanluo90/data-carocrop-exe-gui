@echo off
setlocal
REM ============================================
REM   Nuitka Build Script [GUI]
REM   Compiles gui.py into a standalone app
REM ============================================

cd /d "%~dp0"

set "REQ_PY=3.12"
set "PYTHON_EXE=.\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Missing venv Python at %PYTHON_EXE%
    echo         Please ensure you have built the parent venv first.
    pause
    exit /b 1
)

echo.
echo [1/3] Checking build dependencies...
"%PYTHON_EXE%" -c "import nuitka, customtkinter" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Missing customtkinter or nuitka in venv. Installing...
    "%PYTHON_EXE%" -m pip install customtkinter nuitka
) else (
    echo   Build deps already installed.
)

echo.
echo [2/3] Cleaning stale build artifacts...
if exist "dist\gui.build" rmdir /s /q "dist\gui.build"
if exist "dist\gui.dist" rmdir /s /q "dist\gui.dist"
if exist "dist\MisterMobileCropper-GUI.dist" rmdir /s /q "dist\MisterMobileCropper-GUI.dist"
if exist "dist\MisterMobileCropper-GUI.exe" del /q "dist\MisterMobileCropper-GUI.exe"
echo   Cleaned.

echo.
echo [3/3] Building standalone executable...
"%PYTHON_EXE%" -m nuitka ^
    --standalone ^
    --assume-yes-for-downloads ^
    --low-memory ^
    --jobs=1 ^
    --output-dir=dist ^
    --output-filename=MisterMobileCropper-GUI.exe ^
    --include-package=customtkinter ^
    --include-package-data=ultralytics ^
    --include-package=PIL ^
    --include-package=pillow_heif ^
    --include-package=cv2 ^
    --include-package=numpy ^
    --include-package=numba ^
    --include-package=llvmlite ^
    --nofollow-import-to=scipy ^
    --nofollow-import-to=pytest ^
    --enable-plugin=tk-inter ^
    --windows-console-mode=disable ^
    gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo [POST] Setting up runtime folder...
set "DIST_APP=dist\MisterMobileCropper-GUI.dist"
if not exist "%DIST_APP%" set "DIST_APP=dist\gui.dist"

if not exist "%DIST_APP%" (
    echo [ERROR] Cannot find output app folder in dist.
    pause
    exit /b 1
)

echo   Copying assets...
copy "MM Watermark.png" "%DIST_APP%\MM Watermark.png" >nul 2>&1
if not exist "%DIST_APP%\runs" mkdir "%DIST_APP%\runs"
xcopy /E /I /Y "runs" "%DIST_APP%\runs" >nul 2>&1

if not exist "%DIST_APP%\input" mkdir "%DIST_APP%\input"
if not exist "%DIST_APP%\output" mkdir "%DIST_APP%\output"

if exist "dist\gui.build" rmdir /s /q "dist\gui.build"

echo.
echo ============================================
echo   Build complete! MisterMobileCropper-GUI
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

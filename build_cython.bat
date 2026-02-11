@echo off
REM Build script for Cython extensions (Windows)

echo Building Cython extensions for Windows...
python --version
echo Platform: %OS%

REM Change to the LIFECYCLE MODEL directory
cd /d "%~dp0LIFECYCLE MODEL"

REM Check if Cython is installed
python -c "import Cython" 2>nul
if errorlevel 1 (
    echo Error: Cython is not installed. Please install it with:
    echo   pip install cython
    exit /b 1
)

REM Check if numpy is installed
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo Error: NumPy is not installed. Please install it with:
    echo   pip install numpy
    exit /b 1
)

REM Build the extensions
echo Running setup.py build_ext --inplace...
python setup_cython.py build_ext --inplace

REM Move compiled modules to build directory
cd ..
if not exist "build" mkdir build

REM Find and copy .pyd files (Windows Python extensions)
for /r "LIFECYCLE MODEL" %%f in (*.pyd) do (
    copy "%%f" "build\" >nul
)

echo.
echo Build complete! Compiled modules are in: build
echo Files:
dir /b build\*.pyd 2>nul || echo   (No .pyd files found - check for errors above)

pause

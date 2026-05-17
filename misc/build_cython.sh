#!/bin/bash
# Build script for Cython extensions (Linux/macOS)

set -e

echo "Building Cython extensions for Linux/macOS..."

# Default: same Python as `streamlit` on PATH (pipx or venv), else python3
if [ -z "${PYTHON_CMD:-}" ]; then
    STREAMLIT_BIN="$(command -v streamlit 2>/dev/null || true)"
    if [ -n "$STREAMLIT_BIN" ]; then
        STREAMLIT_DIR="$(cd "$(dirname "$STREAMLIT_BIN")" && pwd)"
        if [ -x "$STREAMLIT_DIR/python" ]; then
            PYTHON_CMD="$STREAMLIT_DIR/python"
            echo "Auto-detected Streamlit Python: $PYTHON_CMD"
        fi
    fi
    PYTHON_CMD="${PYTHON_CMD:-python3}"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Python version: $PYTHON_VERSION"
echo "Python executable: $($PYTHON_CMD -c 'import sys; print(sys.executable)')"
echo "Platform: $(uname -s) $(uname -m)"
echo ""

# Repo root (parent of misc/), then LIFECYCLE MODEL (where setup_cython.py lives)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/LIFECYCLE MODEL"

# Check if Cython is installed
if ! $PYTHON_CMD -c "import Cython" 2>/dev/null; then
    echo "Error: Cython is not installed. Please install it with:"
    echo "  $PYTHON_CMD -m pip install cython"
    exit 1
fi

# Check if numpy is installed
if ! $PYTHON_CMD -c "import numpy" 2>/dev/null; then
    echo "Error: NumPy is not installed. Please install it with:"
    echo "  $PYTHON_CMD -m pip install numpy"
    exit 1
fi

# Build the extensions
echo "Running setup.py build_ext --inplace..."
$PYTHON_CMD setup_cython.py build_ext --inplace

# Move compiled modules to build directory at repo root
cd "$REPO_ROOT"
BUILD_DIR="$REPO_ROOT/build"
mkdir -p "$BUILD_DIR"

# Find and copy .so files (Linux/macOS shared libraries)
# Check both the LIFECYCLE MODEL directory and any lib.* subdirectories
find "$REPO_ROOT/LIFECYCLE MODEL" -name "*.so" -exec cp {} "$BUILD_DIR/" \;
# Also check for lib.* subdirectories created by setuptools
find "$REPO_ROOT/LIFECYCLE MODEL" -path "*/lib.*/*.so" -exec cp {} "$BUILD_DIR/" \;

echo ""
echo "✅ Build complete! Compiled modules are in: $BUILD_DIR"
echo ""
echo "Files:"
ls -lh "$BUILD_DIR"/*.so 2>/dev/null || echo "  (No .so files found - check for errors above)"
echo ""
echo "Note: If your app uses a different Python version, rebuild with:"
echo "  PYTHON_CMD=/path/to/python ./build_cython.sh"

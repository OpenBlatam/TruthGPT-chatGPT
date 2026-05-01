#!/bin/bash
# Bash script to install C++ dependencies for optimization_core
# Usage: ./install_deps.sh [--full] [--gpu] [--dev]

set -e

FULL=false
GPU=false
DEV=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full) FULL=true; shift ;;
        --gpu) GPU=true; shift ;;
        --dev) DEV=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "🚀 Installing optimization_core C++ dependencies"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Install system dependencies
echo "📦 Installing system dependencies..."

if [[ "$OS" == "linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        zip \
        unzip \
        pkg-config \
        python3-dev \
        libeigen3-dev
    
    if $FULL; then
        sudo apt-get install -y \
            libtbb-dev \
            liblz4-dev \
            libzstd-dev
    fi
    
    if $DEV; then
        sudo apt-get install -y \
            libgtest-dev \
            libbenchmark-dev
    fi
    
elif [[ "$OS" == "macos" ]]; then
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install \
        cmake \
        ninja \
        eigen \
        pybind11
    
    if $FULL; then
        brew install \
            tbb \
            lz4 \
            zstd \
            mimalloc
    fi
    
    if $DEV; then
        brew install \
            googletest \
            google-benchmark
    fi
fi

# Install vcpkg (optional, for more control)
VCPKG_ROOT="${VCPKG_ROOT:-$HOME/vcpkg}"

if [[ ! -d "$VCPKG_ROOT" ]]; then
    echo ""
    echo "📦 Installing vcpkg for additional dependencies..."
    
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    "$VCPKG_ROOT/bootstrap-vcpkg.sh"
    
    # Add to .bashrc
    echo "export VCPKG_ROOT=$VCPKG_ROOT" >> ~/.bashrc
    echo "export PATH=\$VCPKG_ROOT:\$PATH" >> ~/.bashrc
    
    echo "✅ vcpkg installed at $VCPKG_ROOT"
fi

# Install pybind11 via pip (ensures Python compatibility)
echo ""
echo "📦 Installing pybind11 Python package..."
pip3 install pybind11[global]

# GPU dependencies
if $GPU; then
    echo ""
    echo "⚠️ GPU dependencies require manual installation:"
    echo "  1. CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  2. CUTLASS:"
    echo "     git clone https://github.com/NVIDIA/cutlass.git"
    echo "     export CUTLASS_ROOT=/path/to/cutlass"
fi

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "📋 Next steps:"
echo "  1. cd cpp_core"
echo "  2. mkdir build && cd build"
echo "  3. cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  4. cmake --build . -j\$(nproc)"
echo ""

# Verify key dependencies
echo "🔍 Verifying installation..."

check_pkg() {
    if pkg-config --exists "$1" 2>/dev/null || ldconfig -p 2>/dev/null | grep -q "$1"; then
        echo "  ✅ $1"
    elif [[ -f "/usr/include/$1" ]] || [[ -d "/usr/include/$1" ]]; then
        echo "  ✅ $1"
    else
        # Check brew on macOS
        if [[ "$OS" == "macos" ]] && brew list "$1" &>/dev/null; then
            echo "  ✅ $1"
        else
            echo "  ⚠️ $1 (may need manual verification)"
        fi
    fi
}

check_pkg "eigen3"
check_pkg "pybind11"

if $FULL; then
    check_pkg "tbb"
    check_pkg "lz4"
    check_pkg "zstd"
fi

echo ""
echo "🎉 Setup complete!"













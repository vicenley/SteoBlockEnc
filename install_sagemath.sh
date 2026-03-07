#!/bin/bash
# SageMath Installation Script for Ubuntu 24.04

set -e

echo "============================================"
echo "  SageMath Installation Helper"
echo "============================================"
echo ""

# Check if already installed
if command -v sage &> /dev/null; then
    echo "✓ SageMath is already installed!"
    sage --version
    echo ""
    echo "To use SageMath with Jupyter:"
    echo "  sage -n jupyter"
    exit 0
fi

echo "SageMath not found. Checking installation options..."
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Option 1: Check if available in apt
echo "Option 1: System Package Manager (APT)"
echo "--------------------------------------"
if apt-cache show sagemath &> /dev/null; then
    echo "✓ SageMath available in apt"
    echo ""
    echo "To install:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install sagemath"
    echo ""
    read -p "Install via apt now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get update
        sudo apt-get install -y sagemath
        echo ""
        echo "✓ SageMath installed via apt!"
        sage --version
        exit 0
    fi
else
    echo "✗ SageMath not available in default apt repositories"
    echo "  (This is normal for Ubuntu 24.04)"
fi
echo ""

# Option 2: Conda/Mamba
echo "Option 2: Conda/Mamba (Recommended)"
echo "------------------------------------"
if command_exists conda || command_exists mamba; then
    if command_exists mamba; then
        CONDA_CMD="mamba"
    else
        CONDA_CMD="conda"
    fi
    
    echo "✓ $CONDA_CMD found"
    echo ""
    echo "To install:"
    echo "  $CONDA_CMD install -c conda-forge sage"
    echo ""
    read -p "Install via $CONDA_CMD now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $CONDA_CMD install -c conda-forge sage -y
        echo ""
        echo "✓ SageMath installed via $CONDA_CMD!"
        sage --version
        exit 0
    fi
else
    echo "✗ Conda/Mamba not found"
    echo ""
    echo "To install Miniforge (includes mamba):"
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  bash Miniforge3-Linux-x86_64.sh"
fi
echo ""

# Option 3: Docker
echo "Option 3: Docker (Isolated)"
echo "---------------------------"
if command_exists docker; then
    echo "✓ Docker found"
    echo ""
    echo "To run SageMath in Docker:"
    echo "  docker pull sagemath/sagemath:latest"
    echo "  docker run -it -p 8888:8888 sagemath/sagemath:latest"
    echo ""
    read -p "Pull Docker image now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker pull sagemath/sagemath:latest
        echo ""
        echo "✓ SageMath Docker image downloaded!"
        echo ""
        echo "To run:"
        echo "  docker run -it -p 8888:8888 sagemath/sagemath:latest"
        exit 0
    fi
else
    echo "✗ Docker not found"
fi
echo ""

# Option 4: Source build
echo "Option 4: Build from Source (Advanced)"
echo "---------------------------------------"
echo "This takes several hours and ~10GB disk space"
echo ""
echo "See: https://doc.sagemath.org/html/en/installation/source.html"
echo ""

# Summary
echo "============================================"
echo "  Installation Summary"
echo "============================================"
echo ""
echo "Recommended options (in order):"
echo ""
echo "1. Conda/Mamba (easiest, most compatible):"
echo "   Install Miniforge: https://github.com/conda-forge/miniforge"
echo "   Then: mamba install -c conda-forge sage"
echo ""
echo "2. Docker (isolated, good for testing):"
echo "   docker run -it -p 8888:8888 sagemath/sagemath:latest"
echo ""
echo "3. Use SymPy instead (already installed!):"
echo "   SymPy can handle most calculations needed for this project"
echo ""
echo "For this project, you can start with SymPy and add SageMath"
echo "later only if needed for advanced computations."

# Quick SageMath Installation Guide

## For Ubuntu 24.04 (Your System)

### ✅ Recommended: APT Installation

```bash
# Update package list
sudo apt-get update

# Install SageMath
sudo apt-get install sagemath

# Verify installation
sage --version
```

**Installation size:** ~2GB  
**Time:** 5-10 minutes (depending on internet speed)

### After Installation

#### Test SageMath:
```bash
sage
```
This opens the SageMath command line interface.

#### Use with Jupyter:
```bash
sage -n jupyter
```
This starts Jupyter with SageMath kernel available.

#### Run SageMath in this project:
```bash
cd ~/Documents/github/vicenley/SteoBlockEnc
sage -n jupyter
# Then navigate to notebooks/symbolic/
```

## Alternative: Conda/Mamba (If you prefer)

```bash
# Install miniforge (includes mamba)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# After miniforge is installed
mamba install -c conda-forge sage

# Or with conda
conda install -c conda-forge sage
```

## Using the Installation Script

I've created a helper script:

```bash
./install_sagemath.sh
```

This will guide you through the installation process interactively.

## Do You Need SageMath?

**For this project:**
- **SymPy (already installed)** - Can handle 90% of the calculations
- **SageMath (optional)** - Better for:
  - Complex Chebyshev polynomial manipulations
  - Advanced rational function analysis  
  - Projective geometry computations
  - Berry phase symbolic calculations

**Recommendation:** Start with SymPy, install SageMath later if needed.

## Quick Start Without Installation

You can use **SymPy** right now (already installed):

```bash
cd ~/Documents/github/vicenley/SteoBlockEnc
source venv/bin/activate
jupyter notebook notebooks/symbolic/01_stereographic_basics.ipynb
```

This will verify all the equations from your draft using SymPy!

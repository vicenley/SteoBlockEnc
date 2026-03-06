# SageMath Setup Instructions

SageMath is used for advanced symbolic computations in this project, particularly for:
- Möbius transformations and complex projective geometry
- Chebyshev polynomial manipulations
- Berry phase calculations
- Rational polynomial generation

## Installation Options

### Option 1: System Package Manager (Recommended for Linux)

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install sagemath
```

#### Fedora:
```bash
sudo dnf install sagemath
```

#### Arch Linux:
```bash
sudo pacman -S sagemath
```

### Option 2: Conda (Cross-platform)

```bash
conda install -c conda-forge sage
```

### Option 3: Docker (Isolated environment)

```bash
docker pull sagemath/sagemath:latest
docker run -it -p 8888:8888 sagemath/sagemath:latest
```

### Option 4: Build from source

See: https://doc.sagemath.org/html/en/installation/source.html

## Verify Installation

```bash
sage --version
```

## Running SageMath Notebooks

```bash
sage -n jupyter
```

This will start a Jupyter server with SageMath kernel available.

## Integration with Project

SageMath notebooks are stored in `notebooks/symbolic/` for analytical computations.
Results and formulas can be exported to Python/SymPy code for integration into the main codebase.

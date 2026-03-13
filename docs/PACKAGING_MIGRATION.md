# Packaging Migration to pyproject.toml

**Date**: 2026-03-06  
**Status**: ✅ Complete

## Summary

Successfully migrated the project from `setup.py` to modern `pyproject.toml` format, making it compatible with `uv` and other modern Python packaging tools.

## Changes Made

### 1. Created `pyproject.toml`
- Modern PEP 621 compliant configuration
- Defined all dependencies with version constraints
- Organized optional dependencies into groups:
  - `dev`: Development tools (pytest, jupyter, black, ruff)
  - `sage`: SageMath dependencies (cypari2, gmpy2, cython, etc.)
  - `docs`: Documentation tools (sphinx)
  - `all`: All optional dependencies combined

### 2. Updated Package Structure
- Added `__version__` attribute to `src/stereo_block_enc/__init__.py`
- Exported key classes for convenient imports
- Package can now be imported as:
  ```python
  from stereo_block_enc import StereographicEncoding, QSPStereographic
  ```

### 3. Configured Tools
- **pytest**: Test configuration with coverage reporting
- **black**: Code formatting with 100-char line length
- **ruff**: Fast linting with modern Python standards
- **mypy**: Type checking configuration

### 4. Created Documentation
- `USAGE.md`: Comprehensive usage guide with examples
- `docs/PACKAGING_MIGRATION.md`: This migration document

## Installation Methods

### Using uv (recommended)
```bash
uv sync                # Core dependencies
uv sync --extra dev    # Add development tools
uv sync --extra sage   # Add SageMath dependencies
uv sync --extra all    # Everything
```

### Using pip
```bash
pip install -e .              # Core only
pip install -e ".[dev]"       # With dev tools
pip install -e ".[all]"       # Everything
```

## Verification

All tests passed:
- ✅ Package builds successfully
- ✅ All modules import correctly
- ✅ Symbolic computation works (StereographicEncoding, QSP)
- ✅ Unitarity checks pass
- ✅ SageMath dependencies install without errors

## Dependencies Installed

**Core dependencies** (46 packages total):
- numpy 2.4.2
- scipy 1.17.1
- matplotlib 3.10.8
- sympy 1.14.0
- qiskit 2.3.0
- pennylane 0.44.0

**Optional (sage)** (+11 packages):
- cypari2 2.2.4
- cython 3.2.4
- gmpy2 2.3.0
- meson-python 0.19.0
- ninja 1.13.0

## Next Steps

1. **Option A**: Continue with symbolic verification
   - All symbolic modules are working
   - Can run verification notebooks
   
2. **Option B**: Write manuscript sections 4-8
   - Use verified equations from notebooks
   - LaTeX compilation is working
   
3. **Option C**: Add numerical implementations
   - Create numerical simulation modules
   - Implement quantum circuits with Qiskit/PennyLane

## Notes

- The `setup.py` file is now deprecated but kept for backward compatibility
- All LSP errors about "sympy not resolved" are false positives - sympy is installed
- The `.venv` directory contains the full environment with all dependencies
- Lock file is automatically managed by `uv`

## Rollback (if needed)

To revert to old setup.py approach:
```bash
pip install -e .  # Uses setup.py as fallback
```

However, this is not recommended as pyproject.toml is the modern standard.

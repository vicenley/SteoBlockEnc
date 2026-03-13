# Stereographic Projection for Quantum Block Encoding

A research repository exploring stereographic projection as a quantum encoding strategy for quantum block encoding, with applications to quantum signal processing (QSP).

## Features

- **Symbolic computation**: Analytical verification of encoding formulas using SymPy
- **Quantum circuits**: Implementation with Qiskit and PennyLane
- **QSP algorithms**: Chebyshev polynomial generation and rational approximations
- **Manuscript**: REVTeX format paper with verified equations
- **Verified equations**: All theoretical formulas symbolically verified

## Quick Start

### Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
# Install core dependencies
uv sync

# Install with optional dependencies
uv sync --extra dev     # Development tools
uv sync --extra sage    # SageMath dependencies
uv sync --extra all     # Everything
```

Or using pip:
```bash
pip install -e .                # Core dependencies
pip install -e ".[dev]"         # With dev tools
pip install -e ".[all]"         # Everything
```

### Basic Usage

```python
from sympy import symbols, simplify
from stereo_block_enc import StereographicEncoding, QSPStereographic

# Stereographic encoding
z = symbols('z', complex=True)
stereo = StereographicEncoding()
state = stereo.encoding_state(z)

# QSP with encoding unitary
r, theta = symbols('r theta', real=True, positive=True)
qsp = QSPStereographic()
U_z = qsp.encoding_unitary(r, theta)
print(f"Unitary? {simplify(U_z.H * U_z) == eye(2)}")  # True
```

See [USAGE.md](USAGE.md) for detailed examples.

## Project Structure

```
SteoBlockEnc/
├── ms/                          # Manuscript (REVTeX format)
│   ├── main.tex                 # Main document
│   ├── sections/                # Individual sections
│   └── draft.pdf                # Original theory document
├── src/stereo_block_enc/        # Main package
│   ├── symbolic/                # Symbolic computation (SymPy)
│   │   ├── stereographic.py     # Encoding/decoding formulas
│   │   ├── mobius.py            # Möbius transformations
│   │   └── qsp.py               # QSP algorithms
│   └── numerical/               # Numerical implementations
├── notebooks/                   # Jupyter notebooks
│   └── symbolic/                # Equation verification notebooks
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── VERIFICATION_REPORT.txt  # Symbolic verification results
│   └── PACKAGING_MIGRATION.md   # Packaging notes
└── examples/                    # Example scripts
```

## Research Focus

### Core Theory
- **Stereographic encoding**: |z⟩ = 1/√(|z|²+1) (z|0⟩ + |1⟩)
- **Möbius transformations**: Quantum gates act as Möbius maps on z
- **QSP**: Generate Chebyshev polynomials T_k, U_k via stereographic encoding
- **Applications**: Rational polynomial approximations with explicit control

### Key Results (Verified ✓)
- ✓ Encoding state and Bloch vector formulas
- ✓ Pauli gates as Möbius transformations (X→1/z, Y→-1/z, Z→-z)
- ✓ Encoding unitary U_z is unitary (U†U = I)
- ✓ Chebyshev polynomial generation for k=2,3,4,5
- ✓ Rational polynomial approximations

## Documentation

- [USAGE.md](USAGE.md) - Usage guide with examples
- [docs/VERIFICATION_REPORT.txt](docs/VERIFICATION_REPORT.txt) - Equation verification results
- [docs/SYMBOLIC_COMPUTATION.md](docs/SYMBOLIC_COMPUTATION.md) - Symbolic framework overview
- [ms/COMPILATION_GUIDE.md](ms/COMPILATION_GUIDE.md) - LaTeX compilation instructions

## Manuscript

Build the manuscript:
```bash
cd ms
make          # Compile PDF
make clean    # Clean auxiliary files
```

The manuscript is written in REVTeX 4.2 format (Physical Review A). Current status:
- ✅ Sections 1-3 complete (Introduction, Preliminaries, Encoding)
- 📝 Sections 4-8 in progress (Möbius, QSP, Dynamics, Applications, Discussion)

## Contributing

This is a research repository. For questions or collaboration:
- Email: vicenley@gmail.com
- Issues: [GitHub Issues](https://github.com/vicenley/SteoBlockEnc/issues)

## License

MIT (to be confirmed)

# Symbolic Computation Framework

This document describes the symbolic/analytical computation tools available for stereographic projection quantum encoding.

## Overview

We use a **hybrid approach** combining:
1. **SymPy** - Pure Python symbolic math (already installed)
2. **SageMath** - Advanced mathematical computations (optional, for heavy symbolic work)

## SymPy Modules

Located in `src/stereo_block_enc/symbolic/`

### 1. `stereographic.py` - Core Encoding Formulas

**Class: `StereographicEncoding`**

Implements fundamental formulas from theory (Sections 1-2):

- **Encoding state**: $|z\rangle = \frac{1}{\sqrt{|z|^2+1}}(z|0\rangle + |1\rangle)$
- **Bloch vector**: Compute $(\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$ from $z$
- **Decoding**: Recover $z = \frac{\langle X\rangle + i\langle Y\rangle}{1-\langle Z\rangle}$
- **Stereographic projection**: Map between $\mathbb{C}$ and $S^2$

**Example usage:**
```python
from stereo_block_enc.symbolic.stereographic import StereographicEncoding

stereo = StereographicEncoding()
state = stereo.encoding_state()  # Symbolic |z⟩
X, Y, Z = stereo.bloch_vector()   # Bloch components
```

### 2. `mobius.py` - Möbius Transformations

**Classes:**
- `MobiusTransformation` - General fractional linear transformations
- `PauliMobius` - Transformations from Pauli gates
- `U3Mobius` - General SU(2) gates
- `RotationMobius` - Rotation gates

Implements formulas from theory (Sections 2.2, 3):

- **Pauli X**: $z \to 1/z$
- **Pauli Y**: $z \to -1/z$
- **Pauli Z**: $z \to -z$
- **Hadamard**: $z \to (1+z)/(1-z)$

**Example usage:**
```python
from stereo_block_enc.symbolic.mobius import PauliMobius

H = PauliMobius.H()
z_transformed = H(z)  # Apply Hadamard transformation
```

### 3. `qsp.py` - Quantum Signal Processing

**Class: `QSPStereographic`**

Implements QSP formulas from theory (Section 5):

- **Encoding unitary**: $U_z = \frac{1}{\sqrt{1+r^2}}\begin{bmatrix}z & 1\\1 & -\bar{z}\end{bmatrix}$
- **Signal operator**: $U_z \sigma_z$
- **Chebyshev states**: $|\psi_k\rangle = T_k(\tilde{r})|0\rangle + \frac{U_{k-1}(\tilde{r})}{\sqrt{1+r^2}}|1\rangle$
- **Rational polynomials**: Decode to get $P(r)/Q(r)$

**Example usage:**
```python
from stereo_block_enc.symbolic.qsp import QSPStereographic

qsp = QSPStereographic()
U_z = qsp.encoding_unitary()
c0, c1 = qsp.qsp_state_coefficients(k=3)  # k=3 Chebyshev state
poly = qsp.rational_polynomial(k=3)        # Rational polynomial
```

## Jupyter Notebooks

Located in `notebooks/symbolic/`

### `01_stereographic_basics.ipynb`

Demonstrates:
- Stereographic encoding and decoding
- Bloch vector computations
- Möbius transformations from quantum gates
- QSP with Chebyshev polynomials
- Rational polynomial generation
- LaTeX export for manuscript

**To run:**
```bash
source venv/bin/activate
jupyter notebook notebooks/symbolic/01_stereographic_basics.ipynb
```

## SageMath (Optional)

For more advanced symbolic work, SageMath can be installed. See `SAGEMATH_SETUP.md` for instructions.

**Use SageMath for:**
- Complex projective geometry
- Advanced polynomial factorization
- Symbolic integration/differentiation that SymPy struggles with
- Berry phase calculations (Section 7.4)

**Use SymPy for:**
- Quick formula verification
- Integration with main Python codebase
- Automated code generation
- Lighter computations

## Key Formulas Implemented

From the theory document `ms/draft.pdf`:

| Section | Formula | Implementation |
|---------|---------|----------------|
| 2 | Encoding $\|z\rangle$ | `StereographicEncoding.encoding_state()` |
| 2 | Bloch vector (Eqs. 5-7) | `StereographicEncoding.bloch_vector()` |
| 2 | Decoding (Eq. 20) | `StereographicEncoding.decode_from_bloch()` |
| 2.2 | Pauli Möbius (Eqs. 24-28) | `PauliMobius.*()` |
| 3 | U3 transformation (Eq. 30) | `U3Mobius.transformation()` |
| 3.1 | Rotations (Eqs. 36-37) | `RotationMobius.*_formula()` |
| 5 | Encoding unitary (Eq. 41) | `QSPStereographic.encoding_unitary()` |
| 5 | Chebyshev states (Eq. 57) | `QSPStereographic.qsp_state_coefficients()` |
| 5 | Rational polynomials (Eq. 58) | `QSPStereographic.rational_polynomial()` |

## Workflow

1. **Analytical work** → Use symbolic modules or SageMath notebooks
2. **Derive formulas** → Export to LaTeX for manuscript
3. **Validate** → Compare symbolic vs numerical results
4. **Implement** → Convert formulas to numerical code in main package

## Export Options

### To LaTeX (for manuscript)
```python
from sympy import latex
latex_code = latex(expression)
```

### To NumPy code
```python
from sympy.utilities.lambdify import lambdify
f = lambdify((r, phi), expression, 'numpy')
```

### To Python code
```python
from sympy.printing.pycode import pycode
python_code = pycode(expression)
```

## Next Steps

- [ ] Implement Berry phase calculations (Section 7.4)
- [ ] Add Hamiltonian evolution formulas (Section 7.1-7.2)
- [ ] Kernel and feature map formulas (Section 8)
- [ ] Verify all formulas match theory numerically
- [ ] Generate figures for manuscript

"""
Stereographic Block Encoding for Quantum Signal Processing

This package provides symbolic and numerical tools for working with
stereographic projection-based quantum block encoding and quantum signal
processing (QSP).

Main modules:
- symbolic: Symbolic/analytical computations with SymPy
- numerical: Numerical simulations and implementations
- qsp: Quantum signal processing algorithms
"""

__version__ = "0.1.0"
__author__ = "vicenley@gmail.com"

# Import key classes for convenience
from stereo_block_enc.symbolic.stereographic import StereographicEncoding
from stereo_block_enc.symbolic.mobius import PauliMobius, U3Mobius, MobiusTransformation
from stereo_block_enc.symbolic.qsp import QSPStereographic

__all__ = [
    "StereographicEncoding",
    "PauliMobius",
    "U3Mobius",
    "MobiusTransformation",
    "QSPStereographic",
    "__version__",
]

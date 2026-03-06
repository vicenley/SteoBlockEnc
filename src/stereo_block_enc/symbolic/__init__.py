"""
Symbolic computation modules for stereographic encoding.

This module provides symbolic/analytical tools for working with:
- Stereographic projection mappings
- Möbius transformations
- Quantum gate representations
- Chebyshev polynomial generation
- Berry phase calculations
"""

from .stereographic import *
from .mobius import *
from .qsp import *

__all__ = [
    'stereographic',
    'mobius', 
    'qsp',
]

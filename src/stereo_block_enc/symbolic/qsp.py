"""
Quantum Signal Processing with stereographic encoding.

This module implements the analytical formulas for QSP using stereographic
projection, particularly the Chebyshev polynomial representations from Section 5.
"""

import sympy as sp
from sympy import symbols, Matrix, sqrt, exp, I, cos, sin, chebyshevt, chebyshevu
from sympy import simplify, atan, Rational


class QSPStereographic:
    """
    Symbolic QSP with stereographic encoding.
    
    From theory (Section 5):
    The encoding unitary is U_z = 1/√(1+|z|²) [[z, 1], [1, -z̄]]
    
    Applying (U_z σ_z)^k generates Chebyshev polynomials:
    |ψ_k⟩ = T_k(r̃)|0⟩ + U_{k-1}(r̃)/√(1+r²) |1⟩
    
    where r̃ = r/√(1+r²)
    """
    
    def __init__(self):
        self.r = symbols('r', real=True, positive=True)
        self.theta = symbols('theta', real=True)
        self.phi = symbols('phi', real=True)
        self.k = symbols('k', integer=True, positive=True)
    
    def encoding_unitary(self, r=None, theta=None):
        """
        Return the encoding unitary U_z for z = r*exp(iθ).
        
        From Eq. 41:
        U_z = 1/√(1+r²) [[r*exp(iθ), 1], [exp(-iθ), -r]]
        
        Parameters
        ----------
        r : sympy expression, optional
            Magnitude. If None, uses symbolic r.
        theta : sympy expression, optional
            Phase. If None, uses symbolic theta.
            
        Returns
        -------
        U : sympy Matrix
            2x2 encoding unitary
        """
        if r is None:
            r = self.r
        if theta is None:
            theta = self.theta
            
        norm = sqrt(r**2 + 1)
        U = Matrix([
            [r * exp(I*theta), 1],
            [exp(-I*theta), -r]
        ]) / norm
        
        return U
    
    def signal_operator(self, r=None, theta=None):
        """
        Return U_z σ_z (the signal operator).
        
        From Eq. 42:
        U_z σ_z = 1/√(1+r²) [[r, -exp(iθ)], [exp(-iθ), r]]
        
        Parameters
        ----------
        r, theta : sympy expressions, optional
            Polar parameters
            
        Returns
        -------
        U_sigma : sympy Matrix
            2x2 signal operator
        """
        if r is None:
            r = self.r
        if theta is None:
            theta = self.theta
            
        sigma_z = Matrix([[1, 0], [0, -1]])
        return self.encoding_unitary(r, theta) * sigma_z
    
    def r_tilde(self, r=None):
        """
        Return r̃ = r/√(1+r²), the rescaled parameter.
        
        Parameters
        ----------
        r : sympy expression, optional
            Magnitude
            
        Returns
        -------
        r_tilde : sympy expression
        """
        if r is None:
            r = self.r
        return r / sqrt(1 + r**2)
    
    def qsp_state_coefficients(self, k_val, r=None):
        """
        Return coefficients of |ψ_k⟩ = T_k(r̃)|0⟩ + U_{k-1}(r̃)/√(1+r²)|1⟩.
        
        Parameters
        ----------
        k_val : int
            Power k
        r : sympy expression, optional
            Magnitude
            
        Returns
        -------
        coeffs : tuple
            (coefficient of |0⟩, coefficient of |1⟩)
        """
        if r is None:
            r = self.r
            
        r_t = self.r_tilde(r)
        
        # T_k is Chebyshev polynomial of first kind
        coeff_0 = chebyshevt(k_val, r_t)
        
        # U_{k-1} is Chebyshev polynomial of second kind
        if k_val > 0:
            coeff_1 = chebyshevu(k_val - 1, r_t) / sqrt(1 + r**2)
        else:
            coeff_1 = 0
            
        return (coeff_0, coeff_1)
    
    def qsp_state(self, k_val, r=None):
        """
        Return the state vector |ψ_k⟩.
        
        Parameters
        ----------
        k_val : int
            Power k
        r : sympy expression, optional
            Magnitude
            
        Returns
        -------
        state : sympy Matrix
            2x1 state vector
        """
        coeff_0, coeff_1 = self.qsp_state_coefficients(k_val, r)
        return Matrix([coeff_0, coeff_1])
    
    def rational_polynomial(self, k_val, r=None):
        """
        Compute the rational polynomial z_k from |ψ_k⟩.
        
        From theory (Eq. 58):
        z_k = (⟨X⟩ + i⟨Y⟩) / (1 - ⟨Z⟩)
        
        Parameters
        ----------
        k_val : int
            Power k
        r : sympy expression, optional
            Magnitude (real-valued case, θ=0)
            
        Returns
        -------
        poly : sympy expression
            Rational polynomial P(r)/Q(r)
        """
        if r is None:
            r = self.r
            
        # For θ=0 (real case), we can compute explicitly
        c0, c1 = self.qsp_state_coefficients(k_val, r)
        
        # Density matrix elements
        rho_00 = c0 * sp.conjugate(c0)
        rho_11 = c1 * sp.conjugate(c1)
        rho_01 = c0 * sp.conjugate(c1)
        rho_10 = c1 * sp.conjugate(c0)
        
        # Pauli expectation values
        exp_X = rho_01 + rho_10  # Real part
        exp_Y = I * (rho_10 - rho_01)  # Imaginary part
        exp_Z = rho_00 - rho_11
        
        # Decode
        z_k = (exp_X + I * exp_Y) / (1 - exp_Z)
        
        return simplify(z_k)
    
    def rotation_angle(self, r=None):
        """
        Compute the rotation angle φ = arccos(r̃).
        
        From Eq. 46:
        φ = arccos(r/√(r²+1))
        
        Parameters
        ----------
        r : sympy expression, optional
            Magnitude
            
        Returns
        -------
        angle : sympy expression
        """
        if r is None:
            r = self.r
        return sp.acos(self.r_tilde(r))
    
    def qsp_sequence_formula(self, phases):
        """
        Symbolic formula for QSP sequence with given phases.
        
        From Eq. 44:
        (U_z e^{-iφ_0 σ_z} U_z e^{-iφ_1 σ_z} ... U_z e^{-iφ_k σ_z})|0⟩
        
        Parameters
        ----------
        phases : list of sympy expressions
            Phase angles [φ_0, φ_1, ..., φ_k]
            
        Returns
        -------
        result : str
            Symbolic description
        """
        k = len(phases) - 1
        return f"QSP sequence of length {k+1} with phases φ = {phases}"


class ChebyshevRationalPolynomials:
    """
    Generate and analyze rational polynomials from Chebyshev QSP.
    
    From theory (Section 5, page 9), for k=2,3,4,...:
    - Zeros and poles follow specific patterns
    - Odd k: one zero at r=0
    - Even k: one pole at r=0
    """
    
    @staticmethod
    def compute_zeros_poles(k_val):
        """
        Compute zeros and poles for the rational polynomial at order k.
        
        Parameters
        ----------
        k_val : int
            Order
            
        Returns
        -------
        info : dict
            Dictionary with 'zeros', 'poles', and 'expression' keys
        """
        r = symbols('r', real=True, positive=True)
        qsp = QSPStereographic()
        
        # Get rational polynomial
        poly = qsp.rational_polynomial(k_val, r)
        poly_simplified = simplify(poly)
        
        # For symbolic analysis, return the expression
        return {
            'k': k_val,
            'expression': poly_simplified,
            'zeros': 'compute numerically or with solve',
            'poles': 'compute numerically or with solve'
        }
    
    @staticmethod
    def pattern_analysis():
        """
        Return known patterns from theory.
        
        Returns
        -------
        patterns : dict
            Known patterns for small k
        """
        return {
            'k=2': {'zeros': 'r=1', 'poles': 'r=0'},
            'k=3': {'zeros': 'r=0, √3', 'poles': 'r=1/√3'},
            'k=4': {'zeros': 'r=√2±1', 'poles': 'r=0, 1'},
            'k=5': {'zeros': 'r=0, √(5±2√5)', 'poles': 'r=√((5±2√5)/5)'},
            'k=6': {'zeros': 'r=1, √3±2', 'poles': 'r=0, 1/√3, √3'},
        }


__all__ = [
    'QSPStereographic',
    'ChebyshevRationalPolynomials',
]

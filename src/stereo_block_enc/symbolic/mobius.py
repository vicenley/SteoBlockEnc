"""
Möbius transformations and their quantum gate representations.

This module implements the analytical formulas for Möbius (fractional linear)
transformations and their relationship to quantum gates, as described in Section 3
of the theory.
"""

import sympy as sp
from sympy import symbols, Matrix, simplify, sqrt, exp, I, cos, sin


class MobiusTransformation:
    """
    Symbolic representation of Möbius transformations.
    
    A Möbius transformation has the form:
    w = f(z) = (az + b)/(cz + d), where ad - bc ≠ 0
    
    These correspond to quantum gates acting on stereographically encoded states.
    """
    
    def __init__(self, a, b, c, d):
        """
        Initialize a Möbius transformation.
        
        Parameters
        ----------
        a, b, c, d : sympy expressions
            Coefficients satisfying ad - bc ≠ 0
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        # Verify non-degeneracy
        det = a*d - b*c
        if det == 0:
            raise ValueError("Determinant ad - bc must be non-zero")
        
        self.determinant = det
        
    def __call__(self, z):
        """
        Apply the Möbius transformation to z.
        
        Parameters
        ----------
        z : sympy expression
            Complex input
            
        Returns
        -------
        w : sympy expression
            Transformed value (az + b)/(cz + d)
        """
        return (self.a * z + self.b) / (self.c * z + self.d)
    
    def inverse(self):
        """
        Return the inverse Möbius transformation.
        
        The inverse of w = (az+b)/(cz+d) is z = (-dw+b)/(cw-a)
        
        Returns
        -------
        inv : MobiusTransformation
            Inverse transformation
        """
        return MobiusTransformation(-self.d, self.b, self.c, -self.a)
    
    def compose(self, other):
        """
        Compose this transformation with another: self ∘ other.
        
        Parameters
        ----------
        other : MobiusTransformation
            Transformation to compose with
            
        Returns
        -------
        composed : MobiusTransformation
            Composition self(other(z))
        """
        # Matrix multiplication
        a_new = self.a * other.a + self.b * other.c
        b_new = self.a * other.b + self.b * other.d
        c_new = self.c * other.a + self.d * other.c
        d_new = self.c * other.b + self.d * other.d
        
        return MobiusTransformation(a_new, b_new, c_new, d_new)
    
    def to_matrix(self):
        """
        Return the 2x2 matrix representation.
        
        Returns
        -------
        M : sympy Matrix
            [[a, b], [c, d]]
        """
        return Matrix([[self.a, self.b],
                      [self.c, self.d]])


class PauliMobius:
    """
    Möbius transformations induced by Pauli operators and common gates.
    
    From theory (Section 2.2):
    X|z⟩ → 1/z
    Y|z⟩ → -1/z
    Z|z⟩ → -z
    S|z⟩ → iz
    H|z⟩ → (1+z)/(1-z)
    """
    
    @staticmethod
    def X():
        """Pauli X: z → 1/z"""
        return MobiusTransformation(0, 1, 1, 0)
    
    @staticmethod
    def Y():
        """Pauli Y: z → -1/z"""
        return MobiusTransformation(0, -1, 1, 0)
    
    @staticmethod
    def Z():
        """Pauli Z: z → -z"""
        return MobiusTransformation(-1, 0, 0, 1)
    
    @staticmethod
    def S():
        """S gate: z → iz"""
        return MobiusTransformation(I, 0, 0, 1)
    
    @staticmethod
    def H():
        """Hadamard: z → (1+z)/(1-z)"""
        return MobiusTransformation(1, 1, 1, -1)


class U3Mobius:
    """
    Möbius transformation for the U3 gate.
    
    From theory (Eq. 30):
    U3(θ,φ,λ)|z⟩ → e^{-i(λ+φ)} (e^{iλ}cos(θ/2)z - sin(θ/2)) / (sin(θ/2)z + e^{iλ}cos(θ/2))
    """
    
    def __init__(self):
        self.theta = symbols('theta', real=True)
        self.phi = symbols('phi', real=True)
        self.lam = symbols('lambda', real=True)
    
    def transformation(self, theta=None, phi=None, lam=None):
        """
        Create Möbius transformation for U3(θ, φ, λ).
        
        Parameters
        ----------
        theta, phi, lam : sympy expressions, optional
            U3 parameters. If None, uses symbolic variables.
            
        Returns
        -------
        mobius : MobiusTransformation
            Corresponding Möbius transformation
        """
        if theta is None:
            theta = self.theta
        if phi is None:
            phi = self.phi
        if lam is None:
            lam = self.lam
        
        # Global phase
        global_phase = exp(-I * (lam + phi))
        
        # Coefficients (Eq. 30)
        a = global_phase * exp(I * lam) * cos(theta/2)
        b = -global_phase * sin(theta/2)
        c = sin(theta/2)
        d = exp(I * lam) * cos(theta/2)
        
        return MobiusTransformation(a, b, c, d)


class RotationMobius:
    """
    Möbius transformations for rotation gates.
    
    From theory (Section 3.1):
    - Rz(θ)|z⟩ → e^{iθ}z (simple phase)
    - Rx, Ry have more complex forms (Eqs. 36-37)
    """
    
    @staticmethod
    def Rz(theta):
        """
        Rotation about Z-axis: z → e^{iθ}z
        
        Parameters
        ----------
        theta : sympy expression
            Rotation angle
            
        Returns
        -------
        mobius : MobiusTransformation
        """
        return MobiusTransformation(exp(I*theta), 0, 0, 1)
    
    @staticmethod
    def Rx_formula(theta=None):
        """
        Return symbolic formula for Rx(θ) Möbius transformation.
        
        From theory (Eq. 36) - this is complex, returns the symbolic expression.
        
        Parameters
        ----------
        theta : sympy expression, optional
            Rotation angle
            
        Returns
        -------
        formula : sympy expression
            The transformation f_x(z; θ)
        """
        if theta is None:
            theta = symbols('theta', real=True)
        z = symbols('z', complex=True)
        
        numerator = I * (Abs(z)**2 - 1) * sin(theta) + 2*I*cos(theta)*sp.im(z) + 2*sp.re(z)
        denominator = sp.conjugate(z)*(z - z*cos(theta)) + cos(theta) + 2*sin(theta)*sp.im(z) + 1
        
        return numerator / denominator
    
    @staticmethod
    def Ry_formula(theta=None):
        """
        Return symbolic formula for Ry(θ) Möbius transformation.
        
        From theory (Eq. 37).
        
        Parameters
        ----------
        theta : sympy expression, optional
            Rotation angle
            
        Returns
        -------
        formula : sympy expression
            The transformation f_y(z; θ)
        """
        if theta is None:
            theta = symbols('theta', real=True)
        z = symbols('z', complex=True)
        
        numerator = z*(-sp.conjugate(z))*sin(theta) + sin(theta) + 2*(cos(theta)-1)*sp.re(z) + 2*z
        denominator = sp.conjugate(z)*(z - z*cos(theta)) + cos(theta) - 2*sin(theta)*sp.re(z) + 1
        
        return numerator / denominator


__all__ = [
    'MobiusTransformation',
    'PauliMobius',
    'U3Mobius',
    'RotationMobius',
]

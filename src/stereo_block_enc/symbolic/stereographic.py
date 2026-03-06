"""
Symbolic stereographic projection and encoding.

This module implements the analytical formulas for stereographic projection
between the complex plane and the Bloch sphere, as described in the theory.
"""

import sympy as sp
from sympy import symbols, sqrt, cos, sin, exp, I, conjugate, Matrix, simplify
from sympy import atan, atan2, re, im, Abs


class StereographicEncoding:
    """
    Symbolic representation of stereographic encoding.
    
    The stereographic encoding maps a complex number z to a quantum state:
    |z⟩ = 1/√(|z|² + 1) (z|0⟩ + |1⟩)
    
    This corresponds to encoding on the Bloch sphere via stereographic projection.
    """
    
    def __init__(self):
        # Define symbolic variables
        self.z = symbols('z', complex=True)
        self.r = symbols('r', real=True, positive=True)
        self.theta = symbols('theta', real=True)
        self.phi = symbols('phi', real=True)
        self.x = symbols('x', real=True)
        self.y = symbols('y', real=True)
        
    def encoding_state(self, z=None):
        """
        Return the symbolic quantum state |z⟩.
        
        Parameters
        ----------
        z : sympy expression, optional
            Complex number to encode. If None, uses symbolic z.
            
        Returns
        -------
        state : sympy Matrix
            2x1 column vector representing |z⟩
        """
        if z is None:
            z = self.z
            
        norm = sqrt(Abs(z)**2 + 1)
        state = Matrix([z, 1]) / norm
        return state
    
    def encoding_state_polar(self, r=None, phi=None):
        """
        Return |z⟩ in polar form: z = r*exp(i*phi).
        
        Parameters
        ----------
        r : sympy expression, optional
            Magnitude. If None, uses symbolic r.
        phi : sympy expression, optional
            Phase. If None, uses symbolic phi.
            
        Returns
        -------
        state : sympy Matrix
            2x1 column vector
        """
        if r is None:
            r = self.r
        if phi is None:
            phi = self.phi
            
        norm = sqrt(r**2 + 1)
        state = Matrix([r * exp(I * phi), 1]) / norm
        return state
    
    def density_matrix(self, z=None):
        """
        Return the density matrix ρ_z = |z⟩⟨z|.
        
        Parameters
        ----------
        z : sympy expression, optional
            Complex number. If None, uses symbolic z.
            
        Returns
        -------
        rho : sympy Matrix
            2x2 density matrix
        """
        state = self.encoding_state(z)
        return state * state.H
    
    def bloch_vector(self, z=None):
        """
        Compute the Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩) for state |z⟩.
        
        From theory:
        ⟨X⟩ = 2Re(z)/(1 + |z|²)
        ⟨Y⟩ = 2Im(z)/(1 + |z|²)
        ⟨Z⟩ = (|z|² - 1)/(|z|² + 1)
        
        Parameters
        ----------
        z : sympy expression, optional
            Complex number. If None, uses symbolic z.
            
        Returns
        -------
        bloch : tuple
            (X, Y, Z) components
        """
        if z is None:
            z = self.z
            
        z_abs_sq = Abs(z)**2
        X = 2 * re(z) / (1 + z_abs_sq)
        Y = 2 * im(z) / (1 + z_abs_sq)
        Z = (z_abs_sq - 1) / (z_abs_sq + 1)
        
        return (X, Y, Z)
    
    def decode_from_bloch(self, X, Y, Z):
        """
        Decode complex number z from Bloch vector components.
        
        From theory (Eq. 20):
        z = (⟨X⟩ + i⟨Y⟩) / (1 - ⟨Z⟩)
        
        Parameters
        ----------
        X, Y, Z : sympy expressions
            Bloch vector components
            
        Returns
        -------
        z : sympy expression
            Decoded complex number
        """
        return (X + I * Y) / (1 - Z)
    
    def stereographic_projection(self, x=None, y=None):
        """
        Map complex number z = x + iy to Riemann sphere coordinates (u, v, w).
        
        From theory:
        u = 2x/(|z|² + 1)
        v = 2y/(|z|² + 1)  
        w = (|z|² - 1)/(|z|² + 1)
        
        Parameters
        ----------
        x, y : sympy expressions, optional
            Real and imaginary parts. If None, uses symbolic x, y.
            
        Returns
        -------
        coords : tuple
            (u, v, w) coordinates on sphere
        """
        if x is None:
            x = self.x
        if y is None:
            y = self.y
            
        z_abs_sq = x**2 + y**2
        u = 2*x / (z_abs_sq + 1)
        v = 2*y / (z_abs_sq + 1)
        w = (z_abs_sq - 1) / (z_abs_sq + 1)
        
        return (u, v, w)
    
    def inverse_stereographic(self, u, v, w):
        """
        Map Riemann sphere coordinates (u, v, w) back to z = x + iy.
        
        From theory:
        x = u/(1 - w)
        y = v/(1 - w)
        
        Parameters
        ----------
        u, v, w : sympy expressions
            Sphere coordinates
            
        Returns
        -------
        z : sympy expression
            Complex number x + iy
        """
        x = u / (1 - w)
        y = v / (1 - w)
        return x + I * y
    
    def bloch_angles_from_z(self, r=None, phi=None):
        """
        Convert from z = r*exp(iφ) to Bloch sphere angles (θ, φ).
        
        From theory (Eq. 39-40):
        θ = 2*arctan(1/r)
        The azimuthal angle is -φ
        
        Parameters
        ----------
        r, phi : sympy expressions, optional
            Polar coordinates. If None, uses symbolic r, phi.
            
        Returns
        -------
        angles : tuple
            (theta, phi_bloch) angles
        """
        if r is None:
            r = self.r
        if phi is None:
            phi = self.phi
            
        theta = 2 * atan(1/r)
        phi_bloch = -phi
        
        return (theta, phi_bloch)


__all__ = ['StereographicEncoding']

import sys, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from like4like import ours
for r in (4, 8, 16, 32):
    f = lambda y: 1/((y-r)**2+1) + 1/((y+r)**2+1)
    L = np.sqrt(r*r+1)
    for eps in (1e-3, 1e-6):
        d, F = ours(f, L, eps)
        print(f"c/b={r:3d} eps={eps:.0e} d={d:5d} d/(2 r ln(1/eps))={d/(2*r*np.log(1/eps)):.2f} F={F:.3f} F/sqrt(b/2c)={F/np.sqrt(1/(2*r)):.2f} C/r^1.5={d/F/r**1.5:.1f}")

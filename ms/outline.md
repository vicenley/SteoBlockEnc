# Stereographic Encoding for Quantum Block Encoding - Manuscript Outline

## Proposed Structure

### 1. Introduction
- Motivation: Why stereographic projection for quantum encoding?
- Connection to quantum signal processing and block encoding
- Overview of results
- Related work

### 2. Mathematical Preliminaries
**2.1 Stereographic Projection**
- Definition and properties
- Map between C and S²
- Riemann sphere compactification

**2.2 Quantum States and the Bloch Sphere**
- Single-qubit states
- Bloch sphere representation
- Density matrices for pure states

### 3. Stereographic Encoding
**3.1 The Encoding Map**
- Definition: |z⟩ = 1/√(|z|²+1) (z|0⟩ + |1⟩)
- Density matrix ρ_z
- Bloch vector representation (Eqs. 5-7)

**3.2 Decoding via Measurement**
- Recovery formula: z = (⟨X⟩ + i⟨Y⟩)/(1 - ⟨Z⟩) (Eq. 20)
- Geometric series interpretation (Eq. 21)

**3.3 Properties of the Encoding**
- Unbounded encoding: r ∈ R⁺ vs traditional amplitude encoding
- Comparison to standard encodings

### 4. Möbius Transformations and Quantum Gates
**4.1 Möbius Transformations**
- Definition: w = (az+b)/(cz+d)
- Automorphisms of the Riemann sphere
- Composition and inverse

**4.2 Pauli Gates as Möbius Transformations**
- X: z → 1/z (Eq. 24)
- Y: z → -1/z (Eq. 25)
- Z: z → -z (Eq. 26)
- S: z → iz (Eq. 27)

**4.3 General Single-Qubit Gates**
- Hadamard: z → (1+z)/(1-z) (Eq. 28)
- U3(θ,φ,λ) representation (Eq. 30)
- Clifford group action

**4.4 Rotation Gates**
- Rz(θ): phase rotation
- Rx(θ) and Ry(θ) formulas (Eqs. 36-37)
- Driving states on the Bloch sphere (Section 4)

### 5. Quantum Signal Processing with Stereographic Encoding
**5.1 The QSP Protocol**
- Encoding unitary U_z (Eq. 41)
- Signal operator U_z σ_z (Eq. 42)
- QSP sequence (Eq. 44)

**5.2 Chebyshev Polynomial Representation**
- State after k applications: |ψ_k⟩ (Eq. 57)
- Connection to T_k and U_k Chebyshev polynomials
- Parameter r̃ = r/√(1+r²)

**5.3 Rational Polynomial Approximation**
- Decoding to rational polynomials (Eq. 58-59)
- Zeros and poles patterns for k=2,3,4,5,6 (page 9)
- Analysis and properties

### 6. Hamiltonian Dynamics and Time Evolution
**6.1 Signal Operation as Evolution Operator**
- U_SI(r,φ) = U_3(arctan(1/r), φ, -φ) (Eq. 70)
- Generator and eigenvalues (Eq. 66)

**6.2 Two-Level System Hamiltonian**
- Driven TLS model (Eq. 68-72)
- Hamiltonian components (Eq. 73)

**6.3 Trajectory Control**
- Linear motion in complex plane (Eqs. 74-76)
- Hamiltonian for specific trajectories

**6.4 Berry Phase**
- Geometric phase accumulation (Eqs. 80-81)
- Calculation via encoding operator (Eq. 83)

### 7. Applications
**7.1 Quantum Eigenvalue Transformation**
- Block encoding via stereographic projection (Eq. 79)

**7.2 Quantum Kernel Methods**
- Feature map G: χ → H (Eq. 90)
- Complex kernel κ(z,z')
- Reproducing kernel (Eq. 91-92)

**7.3 Classification**
- Classifier circuit (Eqs. 93-96)
- Decision boundary analysis

### 8. Discussion and Future Work
- Summary of contributions
- Advantages of stereographic encoding
- Open questions
- Future directions

### Appendices
**A. Mathematical Proofs**
**B. Additional Formulas**
**C. Numerical Examples**

---

## Sections to Merge/Reorganize

- Current Section 3 and 6 (both about Möbius) → New Section 4
- Current Section 7 subsections → Distributed to Sections 5 and 6
- Current Section 2.1 (QFT) → Include in Section 4.3 as example
- Current Section 4 (Driving) → Include in Section 4.4

## Sections That Need Expansion

1. Introduction - add motivation and context
2. Mathematical preliminaries - make self-contained
3. Applications - expand with concrete examples
4. Discussion - synthesize results

## Key Equations to Verify Symbolically

- [ ] Eq. 10: Encoding state |z⟩
- [ ] Eq. 20: Decoding formula
- [ ] Eqs. 24-28: Pauli Möbius transformations
- [ ] Eq. 30: U3 transformation
- [ ] Eqs. 36-37: Rx, Ry transformations
- [ ] Eq. 41: Encoding unitary
- [ ] Eq. 57: Chebyshev state
- [ ] Eq. 59: Rational polynomial formula
- [ ] Eqs. 72-73: Hamiltonian components
- [ ] Eq. 90: Kernel formula

---

## Final Paper Organization (as published)

### Section Map

| Section | Title | Role |
|---|---|---|
| I | Introduction | Motivation, problem statement, summary of results |
| II | Preliminaries | Background: stereographic projection, Bloch sphere, standard QSP, Chebyshev polynomials |
| III | Stereographic Encoding | Define encoding \|z>, decoding via measurement, encoding unitary, gates as Mobius transforms |
| IV | Stereographic QSP | Core theory: signal operator, rotation structure, Chebyshev states, rational Chebyshev basis, unbounded decoding, phase factors, basis change reduction, unnormalized formulation |
| V | Block Encoding with Unbounded Spectra | Applications: standard vs stereographic block encoding, 3 circuit constructions, QSVT protocol, cost analysis, eigenvalue inversion, error analysis, comparison with GQSP |
| VI | Discussion | Summary, standard QSP in stereographic picture, WZW identity, prior work, open problems, outlook |

### Paper Flowchart

```
                        I. INTRODUCTION
                   "Can QSP boundedness be lifted?"
                              |
                              v
                       II. PRELIMINARIES
          +----------+----------+----------+
          |          |          |          |
          v          v          v          v
    Stereographic  Bloch     Standard   Chebyshev
    Projection     Sphere    QSP        Polynomials
    (Eq. 1)        (Eq. 2)  (Eqs. 3-4) (Eqs. 5-6)
          |          |          |          |
          +----------+----+-----+----------+
                          |
                          v
                 III. STEREOGRAPHIC ENCODING
          +----------+----------+----------+
          |          |          |          |
          v          v          v          v
      Encoding    Decoding   Encoding   Gates as
      State |z>   via Pauli  Unitary    Mobius
      (Def. 2,   (Thm. 3,   (Def. 4,   Transforms
       Eq. 7)     Eq. 9)     Eq. 10)   (Eq. 11-12)
          |          |          |          |
          +-----+----+----+----+----------+
                |         |
       key insight:     key insight:
       r -> r~ maps     decoded value
       [0,inf)->[0,1)   is ratio alpha/beta
                |         |
                +----+----+
                     |
                     v
            IV. STEREOGRAPHIC QSP  [CORE THEORY]
                     |
     +-------+-------+-------+-------+-------+
     |       |       |       |       |       |
     v       v       v       v       v       v
   Signal  Rotation Cheby-  Rational Unbounded Basis
   Operator Structure shev   Cheby-  Decoding  Change
   S_z     (Lemma 7) States shev    (Thm 10) (Prop 16)
   (Def 5)    |     (Thm 8) Basis     |      S_z =
     |        |       |    (Eq.17-19) |    V^-1 W V
     |        v       v       |       v       |
     |    r~=cos(phi) TB_k,  Boyd's  z_k=     v
     |    S_z=R(phi)  SB_k   fns[10] cot(k   SAME phases
     |        |       |       |    arctan    as standard
     |        +---+---+---+---+    (1/r))    QSP!
     |            |       |           |        |
     |            v       v           v        v
     |      Orthogonality  Fig. 2:   Zeros/  Corollary 17:
     |      w/ Cauchy wt   plots of  Poles   Phases exist
     |      1/(1+r^2)      TB,SB,z_k (Prop12) for any
     |            |           |        |      valid (P,Q)
     +------+-----+-----------+--------+--------+
            |                                   |
            v                                   v
     Full QSP Protocol                   Unnormalized
     with phases (Eq. 23)                Formulation
     f(r) = P(r~)/Q(r~)                 (Thm 19, Eq. 30)
     (Thm 14: function class)           |P|^2+|Q|^2=(1+r^2)^d
            |
            v
   V. BLOCK ENCODING WITH UNBOUNDED SPECTRA  [APPLICATIONS]
            |
    +-------+-------+-------+
    |       |       |       |
    v       v       v       v
  Standard Stereo.  3 Circuit    QSVT
  Block    Block    Constructions Protocol
  Encoding Encoding (Sec V.C)    (Eq. 45-46)
  (Def 20) (Def 21)    |            |
  needs     no      +--+--+--+     |
  alpha>=   norm    |     |     |  |
  ||H||   needed   v     v     v  |
    |       |    Diag. Pauli-Z Heisen-
    |       |    Hamil. Sum H  berg
    |       |    (Ex.1) (Ex.2) (Ex.3)
    |       |    O(N)  O(poly  O(1)
    |       |    gates  (m))   gates
    |       |      |     |      |
    |       |      +--+--+------+
    |       |         |
    +---+---+---------+
        |             |
        v             v
   Cost Analysis    Eigenvalue Inversion (Sec V.F)
   (Table III)      Standard: depth O(alpha*kappa*log(1/eps))
        |           Stereo:   depth O(d), indep. of ||H||
        v             |
   Key tradeoff:      v
   depth O(d) vs    Error Analysis (Prop 29)
   O(d*alpha)       |z_hat - z| <= (1+r^2)(1+r)/2 * delta
   but sampling     Shots: O(r^6/eps^2)
   O(r^6/eps^2)       |
   vs O(1/eps^2)       v
        |           Fig. 7: Monte Carlo validation
        v
   Comparison: Standard QSP vs Stereographic vs GQSP
   (Table III, Sec V.H)
        |
        v
                    VI. DISCUSSION
    +-------+-------+-------+-------+
    |       |       |       |       |
    v       v       v       v       v
  Summary  Std QSP  WZW     Prior  Open
           in stereo Identity Work  Problems
           picture  Cheby.  [1-16] - Phase-finding
           (Eqs.    decomp.        - Approx. theory
           52-53)   (Eq.54-55)     - Error metrics
              |        |           - Practical block enc.
              v        v           - Generalized phases
           "Even std  P(a) =        (Eq. 56)
           QSP is     sum of      - Extension to QSVT
           unbounded  alpha_k *
           when       T_{d-2k}
           decoded
           stereo-
           graphically"
```

### Logical Flow

The paper follows a **define -> prove -> apply -> analyze** arc:

1. **Sec II** lays the mathematical foundations (4 independent prerequisites)
2. **Sec III** builds the encoding layer on top of those foundations (4 components)
3. **Sec IV** is the theoretical core -- combines encoding with QSP to prove main results,
   culminating in the basis change reduction (Prop 16): everything reduces to standard QSP
4. **Sec V** applies the theory to concrete problems, provides circuits, quantifies costs,
   and compares with alternatives
5. **Sec VI** reflects on what was learned and what remains open

### Two Key "Aha" Nodes

- **Theorem 10** -- decoding produces unbounded cot(k arctan(1/r))
- **Proposition 16** -- basis change V=diag(1,i) means standard QSP phases work directly

Everything before them builds toward them; everything after applies them.

### Figures and Tables

| Figure/Table | Section | Content |
|---|---|---|
| Fig. 1 | I | Schematic of stereographic QSP framework |
| Fig. 2 | IV.D | Rational Chebyshev functions TB_k, SB_k, and decoded z_k |
| Fig. 3 | IV.H | Phase recovery validation and bounded-to-unbounded mechanism |
| Table I | IV.F | Rational functions z_k with zeros and poles for k=2..6 |
| Table II | IV.I | Comparison of standard vs stereographic QSP |
| Fig. 4 | V.C.3 | End-to-end Heisenberg model demonstration |
| Fig. 5 | V.C.3 | Qulacs quantum circuit verification |
| Table III | V.E | Full cost comparison: Standard vs Stereographic vs GQSP |
| Fig. 6 | V.F | Eigenvalue inversion approximation |
| Fig. 7 | V.G | Measurement error amplification and shot cost |
| Fig. 8 | VI.E | Chebyshev expansion convergence and phase-finding cost |

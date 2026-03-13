# Paper 2 Material: Stereographic Block Encoding

Material saved from Paper 1 for development into a companion paper
on the full multi-qubit block encoding framework.

## Saved Files

- `qsp_05_block_encoding_FULL.tex` — Complete Section V from Paper 1,
  including:
  - Definition 13 (Standard Block Encoding)
  - Definition 14 (Stereographic Block Encoding)
  - Remark 5 (Caveats and scope / eigenbasis requirement)
  - Section V.B: Stereographic QSVT Protocol (full circuit, Eq. 94)
  - Proposition 8 (Extraction of Eigenvalues)
  - Section V.C.1: Diagonal Hamiltonians (uniformly controlled rotation)
  - Section V.C.2: Pauli-Z Sum Hamiltonians (reversible adder approach)
  - Section V.C.3: Two-Qubit Heisenberg Model (Givens rotation diag.)
  - Remarks 6-7 (Positivity shifts)

- `qsp_06_cost_analysis_ORIGINAL.tex` — Original cost analysis section
  (kept in Paper 1 but moved to Section V; saved here for reference)

- `qsp_07_discussion_ORIGINAL.tex` — Original discussion section
  (partially modified in Paper 1; saved here for reference)

## Issues to Address in Paper 2

From the analysis of Section V weaknesses:

1. **Eigenbasis requirement** — Needs deep examination. When is the
   eigenbasis known a priori? Connection to QPE. Class of tractable
   Hamiltonians.

2. **Decoding in multi-qubit setting** — What does f(H) mean here?
   Classical estimate of f(lambda_j) vs quantum state f(H)|psi>.
   Post-measurement state analysis.

3. **QSP phases lift to multi-qubit** — Show explicitly that phase
   gates commute with system register, signal operator decomposes
   as direct sum.

4. **Formalize Definition 14** — Exactness (no epsilon parameter),
   contrast with approximate standard block encoding.

5. **Heisenberg example** — Write down V matrix explicitly. Make
   circuit reproducible.

6. **Diagonal and Pauli-Z examples** — Expand with concrete gate
   counts, comparison tables.

7. **Depth advantage** — State as formal proposition/theorem.

8. **Connection to unnormalized formulation** — Use O_r in
   multi-qubit setting.

9. **Notation reconciliation** — U_z, S_z, W_j, O_r all need
   clear definitions and relationships.

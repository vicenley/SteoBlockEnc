# Verification log — ms-10p

## Phase 0 (T0.1): sech classification — RESOLVED: SUBGEOMETRIC (σ = 1/2)

`scripts/check_sech_class.py`, figure `scripts/out/sech_class.png`. Three independent
lines of evidence:

1. **Analytic (decisive).** The complex singularities of g(θ) = sech(L cot θ) sit at
   θ_m with |Im θ_m| = {0.752, 0.215, 0.128, 0.091, 0.071, 0.058, …} for m = 0,1,2,…
   (L = 1), decreasing monotonically to 0. They accumulate onto the real axis at θ = 0,
   so g admits no analyticity strip and its Fourier coefficients cannot decay
   geometrically (Paley–Wiener). This is the same essential-singularity-at-the-endpoint
   mechanism (g ~ 2e^{−L/θ} as θ→0+) that places the Gaussian in the subgeometric class.
2. **Envelope fits.** Upper-hull regression of |a_k|: R²(√k) > R²(k) for sech at every
   L ∈ {0.5,1,2}, matching the Gaussian control (analytically known σ = 1/2).
3. **Visual.** Coefficient envelope concave in k, straight in √k.

**Consequences (applied in the draft):**
- §5.2 Boyd class: sech is subgeometric (σ = 1/2), listed alongside the Gaussian; the
  claim "any function with exponential decay is geometric" is deleted.
- Complexity: d_δ = O((log 1/ε)^{1/σ}) = O(log²(1/ε)); total O(n log²(1/ε)).
- Table 1 sech row: O(n log²(1/ε)) in BOTH the stereographic and the QET columns
  (with L_trunc = O(log 1/ε) and degree O(L_trunc log 1/ε), the QET baseline is also
  O(n log²(1/ε)) — the two methods tie for sech).
- Abstract: sech is not advertised as a geometric-class O(n log 1/ε) example.

Verdict matches the plan's expected outcome; no plan contradiction, build proceeded.

## Phase 5 checks (T5.3) — DONE

`scripts/verify_thm1.py` (run in repo `.venv`).

- [x] **Theorem 1** numerically verified, d = 1–4, 3 random phase sets each: the raw
  product of unnormalized O_x equals [[P,−Q*],[Q,P*]] with |P|²+|Q|² = (1+x²)^d,
  correct degree (deg P ≤ d, deg Q ≤ d−1) and parity (d, d−1), all residuals < 1e-6.
  **Caught a statement bug:** the (1+x²)^{−d/2} prefactor does not belong on the matrix
  in Eq. (real-qsp-form) when O_x is unnormalized (it is the factor that normalizes the
  gate to SU(2)). Fixed in §2: matrix has no prefactor; added a sentence that the
  physical unitary is (1+x²)^{−d/2}U_φ and the factor cancels in the decoded ratio.
- [x] **Theorem 2** forward direction verified, d = 1–3 over a complex-z grid: the
  product G_d O_z ⋯ G_0 has the [[P,−Q*],[Q,P*]] structure with
  P P* + Q Q* = (1+|z|²)^d, residuals < 1e-9. Backward direction (Lemma 1 + gcd/unit
  analysis) checked by hand and written out in Appendix B; the algebraic core
  (DD* = −1 ⟹ D = c z^k, k odd) is elementary and self-contained.
- [x] **Fig. 1 commutes**: both routes θ→α equal cot θ to 8.9e-15 (top route
  cosθ/√(1−cos²θ); bottom route i(e^{2iθ}+1)/(e^{2iθ}−1)). The doubling z = e^{2iθ}
  is required.

## Open items flagged for author review
- Table 1 QSVT entries for sech (O(n log^{5/2})) and shifted Lorentzian are derived here
  under the uniform accounting (App. F); the Gaussian QSVT entry is cited from
  McArdle–Gilyén–Berta Thm 3. The sech/Gaussian QSVT exponents depend on the filling
  fraction of the uniform-grid method and should be cross-checked against MGB directly.
- `references_extra.bib`: `NguemtoLeytonOrtega2026` (Re-QGAN companion) needs its arXiv
  number once posted.

## Like-for-like comparison (2026-11-13)

Scripts in `scripts/like_for_like/`, run in the repo `.venv`. Results and the full table are
in `REFEREE_CHECK_2026-11-12.md`, Part D.

- [x] **Exact quadrature lemma** and **error identity of Theorem 4**: `check_weighted_theorem.py`.
- [x] **Coefficients of G as tail sums of those of g**: `check_G_coefficients.py`, to 1e-16.
- [x] **Pair of Lorentzians**, degree and filling fraction: `pair.py`, to 1%.
- [x] **Constant factors against the QSVT pipeline**, with the boundedness bracket:
  `qsvt_bounded.py`; figure from `fig_like_for_like.py`.
- [x] **Tuned L**: `tuned_L.py`; sech exponents 0.66, 0.34, 1.75 against 2/3, 1/3, 7/4.
- [ ] The "Phase 0" notes above predate this round: the Gaussian exponent is σ = 2/3, and
  Table 1 no longer has a QET column or a shifted-Lorentzian row.


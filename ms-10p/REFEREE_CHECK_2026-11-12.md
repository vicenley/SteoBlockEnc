# ms-10p: referee check and follow-up (2026-11-12)

Manuscript: "Stereographic quantum signal processing and state preparation on the real line".
Referee PDF: `~/Dropbox/temp/QSPExtended_PCL.pdf` (annotations on pp. 1–4).
State of the repo: commits `20b1760` (restructure), `f4819bd` (prose pass) and `d6203af`
(the item 1–3 fixes in Part B, with this report) on `paper-10p`. Nothing is pushed.

## Open questions to decide

1. **Do the like-for-like comparison (item 4)?** *Done 2026-11-13, see Part D.* A downstream task, most naturally the weighted
   quadrature of Sec. IV F, where the cotangent-grid state is used directly and its total cost
   is compared with the uniform-grid route. This is new analysis, not editing, and it decides
   whether the paper is submittable as more than a note. Scoped in Part C, which ends with
   three decisions.
2. **Cite the companion read-out paper?** `NguemtoLeytonOrtega2026` and `BarriosHita2026` dropped
   out when Remark 2 was rewritten for a single qubit. Re-adding one needs a sentence that fits.
3. **Keep the title?** "Stereographic quantum signal processing and state preparation on the real line".

---

## Part A. Verification report

### Verdict

All 23 referee annotations are answered in the text, and the paper now reads as one argument.
Two things remained open at the time of the report:

- **Correctness:** Appendix A had a wrong formula, and Remark 2 had a claim that could not be
  justified. (Both fixed in Part B.)
- **Significance:** the paper is now clear enough that its modest novelty is visible. This is
  the larger risk.

### A1. Referee comments

| # | Comment (page) | Status | Where |
|---|---|---|---|
| 1 | Title: what is the point? (p1) | Addressed | New title, abstract, intro: one question, two-part answer |
| 2 | "Four contributions" read as disconnected (p1) | Addressed | Contribution list replaced by the two-part answer |
| 3 | "Equivalent to rewriting a polynomial in another coordinate?" (p1) | Addressed | Intro para 2 uses this framing; Sec. III answers it |
| 4, 21 | "[? ]" citations (p1, p4) | Addressed | Build has no undefined citations |
| 5 | Commuting-square paragraph too technical (p2) | Mostly | Gone from the intro; one technical sentence on w = e^{2iθ} remains at the end of the Fig. 1 caption |
| 6 | QSP not introduced plainly (p2) | Addressed | Sec. II A |
| 7, 10 | "Signal operator", "QSP sequence" undefined (p2) | Addressed | Sec. II A defines both |
| 8 | Fubini–Study norm unexplained (p2) | Addressed | Removed |
| 9 | Define standard QSP first (p2) | Addressed | Sec. II A precedes the stereographic form |
| 11 | Why "completeness"? (p2) | Addressed | One sentence in Sec. II A |
| 12 | x = i proof looks invalid (p3) | Addressed | Main proof is now a corollary; Appendix A spells out the step (corrected in Part B) |
| 13 | Forward reference to Remark 4 (p3) | Addressed | The remark is now Eq. (8), before the theorem |
| 14, 15 | "Controls", "signal" ambiguous (p3) | Addressed | "Signal" defined; "controls" dropped from the main text |
| 16 | "For all z?" in Theorem 2 (p3) | Addressed | Stated in Appendix B |
| 17, 19 | ℂ[x], ℂ[z, z̄] notation (p3) | Addressed in the main text | Ring notation survives only in the Appendix B proof |
| 18 | "Bivariate structure is the image of…" (p3) | Addressed | Removed |
| 20 | Purpose of the decoding step (p4) | Addressed | Sec. II C, with a d = 1 example |
| 22 | Trade "relative to what?" (p4) | Addressed | Remark 2 names truncate-and-rescale as the alternative |
| 23 | r diverges? (p4) | Addressed | r defined; divergence located at the north pole |
| 24 | "Selects the signal operator" (p4) | Addressed | Remark removed |
| 25 | "Operators are the same geometry" (p4) | Addressed | Remark removed |
| 26 | "Approximation theory of this section" (p4) | Addressed | Removed |

The PDF has 26 highlights; three carry no note and belong to the items above. The request to
clarify later sections "accordingly" is also met.

### A2. Correctness problems found

- **Appendix A, backward direction: the phase formula was wrong.** It said
  e^{2iφ} = −iP(i)/Q*(i). A numerical test with random phases at d = 4 and d = 5 shows the
  correct formula is −iP(i)/Q(i). The stripping operator was also written in the wrong order.
  Theorem 1 itself is unaffected, because its main-text proof does not use this appendix.
  *Fixed in Part B.*
- **Remark 2 (read-out) did not fit where it sat.** It cited an "(n+1)-setting protocol per
  qubit" in a single-qubit section where n is undefined, and claimed the conjugated circuit
  "produces a state with real amplitudes", which fails for nonzero phases. *Fixed in Part B.*
- **"The results below need only Theorem 1" was not accurate.** Secs. III and IV use the
  coordinate and the identity T_k = TB_k, but neither Theorem 1 nor the ratio read-out.
  *Fixed in Part B.*
- **Proposition 2 is stated loosely.** Matrix elements after d queries are in general
  P(a) + √(1−a²)·Q(a), not a polynomial alone. *Still open.*
- **Two QSVT baseline entries in Table I were unverified.** *Checked in Part B.*
- **Attribution of the circuit to McArdle–Gilyén–Berta was unchecked.** *Checked in Part B.*

### A3. Comprehension

A QSP-literate reader can now follow it from start to finish. Remaining friction:

- "Pullback" is used in the introduction before it is defined in Sec. IV B.
- Sec. II C (read-out) is a side branch. Nothing later depends on it.
- Sec. IV is dense compared with Secs. I–III: Boyd class, filling fraction, algorithm, examples
  and comparison in quick succession.

### A4. Significance

- **Theorem 1:** a corollary of the standard theorem, by the paper's own statement. It is a
  restatement.
- **The no-go (Sec. III):** the QSVT theorem plus the identity T_k(a) = TB_k(α). Correct and
  worth one paragraph, but not a result a referee will credit as new.
- **State preparation (Sec. IV):** the only claimed advantage. Its weaknesses:
  - The circuit follows existing work; the new ingredient is the choice of grid.
  - The headline example is trivial. On this grid the Lorentzian's amplitudes are sin²θ_j,
    a three-term Fourier sum.
  - The comparison is between different states, as the paper now admits. No task is shown
    where the cotangent-grid state is what one needs.
  - Where the comparison is closest, the method loses (Gaussian) or pays for position
    (shifted Lorentzian, (|a|/b)^{3/2}).
- **Ratio read-out:** a possible second idea, but undeveloped and unused.

As it stands, the paper is honest and readable but incremental. The expected verdict from a
PRA referee is "correct, limited novelty". The single change that would raise it is a
like-for-like result (open question 1). Without that, the central claim is "a different state
is cheaper to prepare".

---

## Part B. Fixes applied after the report (items 1–3, commit `d6203af`)

The manuscript builds cleanly at 12 pages, with no undefined references or overfull boxes.

### B1. Appendix A

- Phase formula corrected to e^{2iφ} = −iP(i)/Q(i).
- Stripping step written out: operator order O_x⁻¹e^{−iφσ_z}, reduced pair displayed as an
  equation.
- Vanishing of Q̃ at x = i is shown from Q̃(i) = i·P̃(i); vanishing at x = −i follows from
  parity.

### B2. Sec. II

- Remark 2 rewritten for a single qubit: r = √(p₀/(1−p₀)) from computational-basis counts;
  the phase needs the σ_x and σ_y expectation values; cost O(r⁴/ε²) stated for r ≳ 1.
- The "real amplitudes / flag qubit / (n+1)-setting protocol" passage is removed, with its echo
  in the read-out appendix. Citations `BarriosHita2026` and `NguemtoLeytonOrtega2026` drop out.
- "The results below need only Theorem 1" now reads "The generalized form is not used below."

### B3. Check against McArdle–Gilyén–Berta (arXiv:2210.14892v2)

Only part of the paper could be read verbatim, so two points remain unconfirmed (see B4).

| What we had | What the source says | Action |
|---|---|---|
| Gaussian bound from "Thm. 3", written Õ | Theorem 2, Eq. (6), written O, valid for β ≥ log(1/ε) | Corrected in table and appendix |
| "Lem. III.1" for the normalized-state bound | No such lemma; the trace-distance bound is Lemma 6 (App. C.1) | Citation corrected; exact statement not read |
| "Filling fraction" attributed to them | Confirmed: "discretized L2-norm filling-fraction", Eq. (2) | None |
| Ancillas: 4 general, 3 under parity | Confirmed (abstract; Theorem 1 states 3 for definite parity) | None |
| Our circuit "is the construction of" theirs | They block-encode Σ sin(2x/N)\|x⟩⟨x\| and apply QSVT to a polynomial in sin(y/a) | Reworded: we follow their template but use a diagonal unitary directly as the QSP signal |
| Lorentzian, sech, shifted-Lorentzian entries in the "QSVT" column | Verbatim text gives only the Gaussian and the Kaiser window | Caption and text now say these are our estimates from their Theorem 1 |

**Wrong table entry found.** The QSVT estimate for the shifted Lorentzian was
O((n/b) ε⁻¹ log 1/ε). Rescaling y by b shows the cost cannot depend on b. The correct estimate
is O(n ε⁻¹ log 1/ε), valid while the window contains the peak (|a| ≲ b ε^{−2/3}). The comparison
text now adds that our cost stays below this estimate as long as (|a|/b)^{3/2} ≲ ε⁻¹.

### B4. Still open after the fixes

- ~~Lemma 6 statement and any sech/Lorentzian results in McArdle et al.~~ Checked against the
  v2 PDF on 2026-11-13: Lemma 6 exists (Appendix C.1), and the paper has no Lorentzian, Cauchy
  or sech result. See Part D.
- **Bib year:** the McArdle entry says 2025 (the v2 date) for a 2022 arXiv id. The arXiv page
  lists no journal reference, so the entry is acceptable as an arXiv v2 citation.
- **Proposition 2 looseness:** see A2.
- **Fig. 2 data** not regenerated since the restructure; `scripts/check_sech_class.py` still
  labels the Gaussian a σ ≈ 1/2 "control" (correct value: 2/3).
- **Voice:** the referee's main concern was authorship; the abstract and the introduction's
  first and third paragraphs are the places to rewrite in your own words.

Linter after the fixes: 5 HARD (all TikZ `--` false positives in Fig. 1), inversions 0.3 per
1000 words, closers 7/78.

---

## Part C. Scope of the like-for-like quadrature comparison

Analysis of what the manuscript already contains; no new numerics were run for this scope.
Summary: the comparison can be made exact with modest work, and its result is predictable.
The advantage survives only for heavy-tailed targets, and only against the QSVT pipeline.

### C1. The idea that makes it like-for-like

The quadrature-weighted state of Sec. IV F has amplitudes f(y_j)/sin θ_j. The map
f(y) → √L · f(L cot θ)/sin θ is a unitary from square-integrable functions on the real line
to square-integrable functions on (0, π). So the weighted cotangent-grid state is the ordinary
uniform-grid discretization of the same normalized function, written in the θ coordinate.

That gives a common yardstick: both methods approximate the same continuum state, and the
error is the L² distance to it. For a uniform grid the error is the truncated tail plus
discretization. For the cotangent grid it is the Fourier truncation in θ plus discretization,
with no tail term.

The weighted state would have to become the paper's primary object. The unweighted state,
currently the headline, is the one with no fair comparison.

### C2. Work items

**1. Analysis (pen and paper, plus Sage checks)**

- State and prove the unitary map and the error metric (short).
- Redo the cost theorem for the weighted target g(θ)/sin θ:
  - It needs f to decay at least as 1/|y|, otherwise the amplitude is unbounded.
  - It contains odd harmonics instead of even ones, so the "bounded on the full circle" step
    needs rechecking.
  - The filling fraction changes; for the Lorentzian the amplitude is sin θ and the fraction
    is 1/√2.
- Quadrature error on the cotangent grid: the midpoint rule on a smooth periodic integrand,
  which is the subject of the Boyd 1987 paper already cited. This fixes how many qubits are
  needed.
- Re-derive all four rows of Table I under the common metric, for both grids.

**2. Numerics (to be run by VLO)**

- Build both states by statevector for n up to about 16–20 and measure the L² error to the
  continuum function, and the error on a few test integrals.
- Confirm the predicted degree and filling-fraction scalings with ε.
- Regenerate Fig. 2 for the weighted targets. `scripts/convergence_comparison.py` and
  `scripts/qulacs_state_prep_sim.py` are the starting points; the Gaussian label in
  `scripts/check_sech_class.py` needs fixing at the same time.

**3. Manuscript**

- Sec. IV reorganized around the weighted state; Theorem 4, Table I, the comparison text, the
  abstract and the discussion updated; the table appendix rewritten.

Items 1 and 3 are a few days of work; item 2 depends on how far n is pushed.

### C3. Expected outcome

| Target | Against QSVT pipeline | Against the linear-depth circuits (GC, SH) |
|---|---|---|
| Centered Lorentzian and other algebraic tails | Polynomial saving in 1/ε, now on equal footing | Tie: their window costs only extra qubits, not gates |
| sech | A log^{1/2} factor against our own estimate of their cost | Not applicable |
| Gaussian | We lose by log^{1/4} | We lose |
| Shifted Lorentzian | We win only while (\|a\|/b)^{3/2} ≲ 1/ε | Tie or lose |

The defensible claim becomes: for targets with algebraic tails, the cotangent grid removes the
truncation cost that the QSVT pipeline pays, and matches specialized circuits with one general
algorithm. The paper's significance would stay modest, but the central claim would no longer
rest on comparing different states.

### C4. Limits that remain after the comparison

- **What the state can be used for.** Position-type observables V(y) and overlaps with states
  prepared the same way carry over exactly. Anything needing a uniform grid, such as a kinetic
  term applied through the quantum Fourier transform, does not.
- **The Lorentzian stays trivial.** In the θ coordinate the square root of a Cauchy density is
  the uniform superposition. That is a clean illustration, but a referee will still call it
  easy.

### C5. Optional larger step

A concrete application would raise the significance more than the comparison does: for
example, amplitude estimation of an expectation value under a heavy-tailed (Cauchy or
Student-t) distribution, where truncation is the known obstacle. It needs the payoff function
applied coherently in the θ coordinate and is closer to a separate paper than a revision.

### C6. Decisions

1. Make the weighted state the primary object? The comparison does not work otherwise.
2. Accept the narrowed claim (heavy tails, versus QSVT only) as the paper's result?
3. Keep the application out of this paper? Recommended: yes.

If 1 and 2 are agreed, the next step is work item 1, ending with a re-derived Table I before
any manuscript edit.

---

## Part D. Like-for-like comparison carried out (2026-11-13)

Commit `edb0980` put the weighted state and the new Theorem 4 into Sec. IV. The follow-up
edits described in D3 are not committed yet.

### D1. What the paper now claims

- **Common yardstick:** L²(ℝ) distance between the normalized target and the function the
  state represents. The weighted state has amplitudes f(y_j)/sin θ_j; L and the grid centre
  y₀ are free.
- **Theorem 4:** for a pure-parity trigonometric polynomial p of degree d < N, cost
  O(n·d/F_p), two ancillas, and L²(ℝ) error equal to the approximation error of G by p.
  No discretization term; F_p is exact.
- **Table I:** QSVT pipeline against this work only. Algebraic tails are exact at O(n);
  sech wins by log^(1/2); the Gaussian loses by log^(1/4); two peaks at ±a cost (|a|/b)^(3/2).
- **GC and SH** are out of the table and described in a paragraph.

### D2. Checks made

| Item | Result |
|---|---|
| Exact quadrature lemma | discrete and continuum norms agree to 4×10⁻¹⁶ for pure parity, d = N−1 |
| Error identity of Theorem 4 | L²(ℝ) and L²(0,π) errors agree to 4 digits (sech, Gaussian) |
| Pair of Lorentzians | d = 2(a/b)·ln(1/ε) and F = (b/2a)^(1/2), both to 1% |
| G against g decay rate | fitted σ = 0.50 (sech) and 0.65 (Gaussian) for G; same class as g |
| GQSP completeness | Motlagh–Wiebe Corollary 5: any P with \|P\| ≤ 1 on the circle, d signal calls, d+1 rotations |
| McArdle Theorem 1 (v2) | O(n·d/F) gates, at most 3 ancillas for definite parity (abstract: 4 in general) |
| McArdle boundedness | they multiply by a threshold function and "expect a modest increase in the degree"; not counted in their bound |

### D3. Constant factors (product d/F at each method's best parameter)

The QSVT value is bracketed: lower bound from a fit on the window only, upper bound from a
polynomial that is bounded on all of [−1, 1].

| ε | sech: ours / QSVT | Gaussian: ours / QSVT | Lorentzian: ours / QSVT |
|---|---|---|---|
| 10⁻² | 5.5 / 33–42 | 13 / 11–14 | 1.4 / 382–447 |
| 10⁻³ | 18 / 90–101 | 27 / 20–24 | 1.4 / 5500–6600 |
| 10⁻⁴ | 34 / 163–195 | 40 / 31–35 | not run |
| 10⁻⁵ | 48 / 281–330 | 56 / 42–51 | not run |
| 10⁻⁶ | 68 / 432–508 | 69 / 53–62 | not run |

These numbers are now quoted in Sec. IV G and Appendix D. Scripts: `scripts/like_for_like/`
(`like4like.py`, `qsvt_bounded.py`, `check_weighted_theorem.py`, `pair.py`, `rates.py`).

### D4. Second pass on the open items (2026-11-13, after commit `47af4fc`)

- **Tuned L, sech:** with L ∝ d^(1/3) the error decays as exp(−c·d^(2/3)) (Boyd Sec. 17.9) and
  the cost is O(n log^(7/4)). Numerics up to d = 85 (error 10⁻¹²) give exponents 0.66, 0.34
  and 1.75 against 2/3, 1/3 and 7/4. Now stated in Sec. IV E.
- **Tuned L, Gaussian:** numerics up to d = 57 give error exp(−c·d^0.90) and cost log^1.36.
  No asymptotic law claimed. Now stated in Sec. IV E as an observation.
- **Linear-depth methods:** the bib entry `GonzalezCondeEtAl2020` was wrong. Its arXiv id
  2005.04351 is Holmes and Matsuura, "Efficient Quantum Circuits for Accurate State
  Preparation of Smooth, Differentiable Functions", which is the matrix-product-state paper
  the text describes. A correct entry `HolmesMatsuura2020` is added to
  `references_extra.bib` (arXiv only; add the IEEE QCE 2020 proceedings details if wanted) and
  ms-10p cites it. The wrong entry is left in the shared `ms/references.bib` because
  `ms-short` still cites that key; it needs the same fix there. Neither that paper nor Sano–Hamamura states how cost
  grows with 1/ε or with the window; both show accuracy numerically at fixed bond dimension.
  The paragraph after Table I now says this.
- **Proposition 2:** restated for the QSVT circuit specifically, and the text now separates
  the encoded block (a polynomial) from the off-diagonal elements (√(1−a²) times a polynomial).
- **Figure:** new Fig. 3 (`figures/like_for_like.pdf`, from
  `scripts/like_for_like/fig_like_for_like.py`, data in `data/like_for_like.npz`) shows L²
  error against cost for sech and the Gaussian and cost against 1/ε for algebraic tails.
  Fig. 2 is unchanged.

### D5. Final pass (2026-11-13)

- **"G decays like g" is now proved.** The coefficient of e^{i(2j+1)θ} in G is 2i times the
  tail sum of the coefficients of g, when f vanishes at infinity. The identity holds to
  10⁻¹⁶ numerically (`scripts/like_for_like/check_G_coefficients.py`); the derivation is in
  the appendix proof of Theorem 4, and the theorem's hypothesis is stated on f.
- **Holmes–Matsuura** entry completed from Crossref: IEEE QCE 2020, pp. 169–179,
  doi 10.1109/QCE49297.2020.00030.

### D6. Still open

- **Gaussian with tuned L** has no proved or cleanly fitted exponent; the paper says so.
- **`ms-short`** still cites the wrong `GonzalezCondeEtAl2020` entry in the shared bib file.
- **Authorship voice, companion citations, title:** your decisions (see the open questions at
  the top). My recommendation is to keep the title and leave the companion citations out.

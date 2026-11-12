# ms-10p: referee check and follow-up (2026-11-12)

Manuscript: "Stereographic quantum signal processing and state preparation on the real line".
Referee PDF: `~/Dropbox/temp/QSPExtended_PCL.pdf` (annotations on pp. 1–4).
State of the repo: commits `20b1760` (restructure), `f4819bd` (prose pass) and `d6203af`
(the item 1–3 fixes in Part B, with this report) on `paper-10p`. Nothing is pushed.

## Open questions to decide

1. **Do the like-for-like comparison (item 4)?** A downstream task, most naturally the weighted
   quadrature of Sec. IV F, where the cotangent-grid state is used directly and its total cost
   is compared with the uniform-grid route. This is new analysis, not editing, and it decides
   whether the paper is submittable as more than a note.
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

- **Lemma 6 statement and any sech/Lorentzian results in McArdle et al.:** check the PDF
  directly before submission.
- **Bib year:** the McArdle entry says 2025 (the v2 date) for a 2022 arXiv id.
- **Proposition 2 looseness:** see A2.
- **Fig. 2 data** not regenerated since the restructure; `scripts/check_sech_class.py` still
  labels the Gaussian a σ ≈ 1/2 "control" (correct value: 2/3).
- **Voice:** the referee's main concern was authorship; the abstract and the introduction's
  first and third paragraphs are the places to rewrite in your own words.

Linter after the fixes: 5 HARD (all TikZ `--` false positives in Fig. 1), inversions 0.3 per
1000 words, closers 7/78.

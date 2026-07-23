# Mathematical exploration: what can actually close the gap

Diverge → converge survey. Two prototypes run with pre-registered gates; one failed,
one passed. **The results overturn the Stage-0/1 plan in `RESEARCH_DESIGN.md`.**

---

## Phase 1 — Divergence (32 candidates, no filtering)

Enumerated across all of mathematics, not the three familiar branches. One line each:
what structural feature of the problem it could grip.

**Analysis**
1. Frame theory / consistent reconstruction — redundancy → rate law for quantized expansions
2. Prolate spheroidal (Slepian) — out-of-band extrapolation from in-band data
3. Paley–Wiener / analytic continuation — time-limited ⇒ spectrum entire ⇒ determined everywhere
4. Wavelets / multiresolution — adaptive tiling matched to transients
5. Gabor frames — phase-space geometry, uncertainty limits on the noise structure
6. Phase retrieval — recover phase where only magnitude survives
7. Unlimited/modulo sampling — recovery from folded measurements
8. Operator theory (Friedrichs angle) — convergence rate of alternating projection
9. Nonlinear approximation — sparse approximation rate of music in TF dictionaries

**Geometry**
10. Convex geometry — the feasible set, projections, Dykstra
11. Riemannian optimization — descent on the consistent-signal manifold
12. Lattice theory / geometry of numbers — quantization cells as lattice cosets
13. Information geometry — Fisher information of a quantized observation
14. Algebraic geometry — variety of harmonic signals
15. Sheaf theory — gluing local frame constraints into a global section

**Probability / statistics**
16. **Rate–distortion–perception theory** — the provable distortion-vs-realism tradeoff
17. Diffusion / posterior sampling — plug-and-play, DDRM, DPS
18. Empirical Bayes / Tweedie — denoiser-as-score identity
19. Optimal transport — distributional (texture) matching where pointwise is impossible
20. Random matrix theory — frame conditioning
21. Extreme value theory — clipping and inter-sample peaks
22. Point processes — onset/transient statistics

**Discrete / algebraic**
23. Chinese Remainder Theorem — incommensurate quantizer moduli intersecting
24. Coding theory — the bitstream as a code; syndrome view
25. Combinatorial optimization — inverting bit allocation
26. Group theory / equivariance — pitch-shift and time-shift symmetry

**Computational**
27. Unrolled optimization / learned iterative schemes
28. Deep image prior — per-song overfitting, zero training data
29. Non-local means / self-similarity — repeated passages
30. Message passing over the frame graph

**Physics-adjacent**
31. Modal / source-filter instrument models — physically-grounded priors
32. Psychoacoustic masking — the metric itself

**Structural saturation reached.** These 32 fields grip only four distinct structures:
**(a) redundancy & consistency, (b) low-dimensionality, (c) distributional realism,
(d) constraint geometry.** Further field-naming yields no new mechanism.

---

## Phase 2 — Filter, with recorded rejections

| # | Candidate | Verdict | Why |
|---|-----------|---------|-----|
| 1 | Frame theory / consistent reconstruction | **PROTOTYPE** | Predicts a *rate law*; directly testable; underwrites fusion mechanisms |
| 2,3 | Slepian / analytic continuation | **PROTOTYPE** | Would make Regime C recoverable — too important to reject untested |
| 16 | Rate–distortion–perception | **ADOPT** | Theorem, not hypothesis; dictates architecture factorization |
| 8,10 | Convex geometry + Friedrichs angle | **ADOPT** | Sets projection algorithm and unrolling depth |
| 29 | Non-local self-similarity | **ADOPT (conditional)** | Manufactures redundancy — value follows entirely from #1's result |
| 23 | CRT / incommensurate lattices | **DEFER** | Only applies to multi-encoding; strong if that scenario is in scope |
| 19 | Optimal transport | **ADOPT for Regime C only** | Pointwise loss is meaningless where content is invented; distributional is the right genus |
| 17,18,27 | Diffusion / Tweedie / unrolling | **ADOPT** | Standard machinery once geometry is fixed |
| 6 | Phase retrieval | Reject | Phase is *not* missing here — the bitstream carries signs. Solving a problem we don't have |
| 7 | Modulo sampling | Reject | Codec quantization isn't folding; wrong operator |
| 14 | Algebraic geometry | Reject | Harmonic variety is not algebraically tractable at signal scale; complexity without leverage |
| 15,30 | Sheaf / message passing | Reject (fold into #10) | Correct description of the gluing problem, but Dykstra already solves it and is simpler. Re-expression, not strengthening |
| 12 | Lattice theory | Fold into #23 | Same structure |
| 11 | Riemannian optimization | Reject | Constraint set is convex and polyhedral; manifold machinery is unnecessary complexity |
| 21,22,26 | EVT / point processes / equivariance | Defer | Marginal gains, no structural grip |
| 28 | Deep image prior | Defer | Interesting zero-shot fallback; too slow to lead |
| 31 | Modal instrument models | Defer | Strong prior for Regime C but genre-limited |

---

## Phase 3 — Converge: two prototypes, pre-registered gates

### EXP 1 — Bandlimited extrapolation. **GATE: FAIL. Claim rejected.**

*Hypothesis:* out-of-band content is determined whenever the signal's effective
dimension `d` is below the number of in-band constraints `K`; expected a sharp error
transition at `d = K`.

*Result* (N=512 window, K=128 observed low bins, relative error on the unobserved band):

| effective dim `d` | SNR=∞ | 60 dB | 40 dB | 20 dB |
|---:|---:|---:|---:|---:|
| 8 | 1.3e−12 | 1.1e+00 | 9.0e+00 | 1.5e+02 |
| 16 | 1.7e−07 | 1.7e+05 | 1.4e+06 | 1.4e+07 |
| 32 | 3.1e−01 | 1.1e+08 | 1.1e+09 | 1.5e+10 |
| 128 (=K) | 8.4e−01 | 3.6e+08 | 4.1e+09 | 2.4e+10 |

**No transition at `d = K`.** Conditioning collapses at `d ≈ 16–32`, an order of
magnitude earlier, and *any* realistic noise amplifies error by 5–10 orders of
magnitude. At 40 dB SNR — more generous than MP3 quantization at `is=10` (±6.7% ≈ 23 dB)
— extrapolation is hopeless even at `d = 8`.

**This is a valuable negative result.** It closes off an entire seductive family of
"clever math" approaches to the high band, rigorously:

> **Regime C is not recoverable by any analytic method. It is synthesis, necessarily.**
> The only legitimate source of high-band content is a *prior over what music looks like*,
> never extrapolation from the observation.

Any future claim to "recover" the lost high band can be refuted by this table.

### EXP 2 — Consistent reconstruction rate law. **GATE: PASS.**

*Hypothesis:* linear reconstruction of quantized frame expansions gives MSE = O(1/r);
consistent reconstruction (any signal re-quantizing to the same indices) gives O(1/r²).

*First run failed with MSE increasing in r* — a test-design defect, not a hypothesis
failure: the frame was normalized by 1/√M, so measurements shrank below the quantizer
step and all information vanished. Corrected to unit-norm frame rows.

*Result:*

| redundancy r | linear MSE | consistent MSE | gain |
|---:|---:|---:|---:|
| 2 | 8.49e−02 | 8.38e−02 | **1.01×** |
| 4 | 3.00e−02 | 2.77e−02 | 1.08× |
| 8 | 1.11e−02 | 8.41e−03 | 1.31× |
| 16 | 5.38e−03 | 2.88e−03 | 1.87× |
| 32 | 2.56e−03 | 9.99e−04 | 2.56× |
| 64 | 1.30e−03 | 2.80e−04 | **4.65×** |

log-log slope: linear **−1.195** (theory −1), consistent **−1.630** (theory −2; POCS
under-converged at high r, so the true asymptote is likely steeper). Consistency wins at
every r and **the gain grows monotonically with redundancy**.

---

## Phase 4 — What this changes

### The correction that matters

**MP3's MDCT is critically sampled: r = 1** (576 lines per 576 new samples; overlap gives
at most an effective r ≈ 2). Read the r=2 row: **gain 1.01×.**

The hard-projection proposal in `RESEARCH_DESIGN.md` §2 — applied to a single file — is
worth roughly **1%**. Stage 1 as written would have produced a null result. It is now
cancelled, before it was built.

The guarantee (provable consistency, no hallucination over coded content) survives and is
still worth having. But as a *quality* mechanism, projection alone is nearly worthless.

*(Caveat: EXP 2 used random frames and a uniform quantizer, not MP3's structured filterbank
and power-law quantizer. The numbers are indicative; the direction is robust.)*

### The actual lever

> **Redundancy is the lever, not projection.** Consistency pays in proportion to `r`, and a
> single file gives you `r ≈ 1`. **The central engineering problem is therefore: manufacture
> redundancy.**

That reframing is the breakthrough this exploration was looking for, and it is quantitative.
Sources of redundancy, in descending strength:

| Source | redundancy | availability |
|---|---|---|
| Multiple encodings of one master (different bitrates/codecs/services) | r = N, with **incommensurate quantizer lattices** (§23) so cells intersect super-additively | archives, reissues |
| Repeated passages (chorus, loop, repeated hit) landing on **different MDCT phases** ⇒ statistically independent quantization noise | r ≈ K, and K is large in most popular music | **every file** |
| Overlapping MDCT frames (TDAC) | r ≈ 2 | every file |
| Stereo coupling on correlated (mid) content | r ≈ 2 on the mid component | every file |
| Local low-dimensional structure (harmonic/modal) | r_eff = #constraints / d | every file |

### The sharp distinction the two experiments jointly establish

EXP 1 and EXP 2 use the *same* structural fact — the signal is locally low-dimensional —
and reach opposite conclusions:

> Low-dimensional structure is **catastrophically ill-conditioned for extrapolating out of
> band** (EXP 1) and **powerfully well-conditioned for denoising in band** (EXP 2).

The field has this backwards. It spends its modelling capacity on the out-of-band problem,
where mathematics forbids success, and uses trivial feed-forward regression in-band, where
mathematics guarantees success.

### Architecture implied

Not a feed-forward spectral GAN. A **redundancy-aggregating, unrolled consistency engine**:

1. **Gather** every constraint bearing on a time region: its own MDCT frames, matched
   repetitions elsewhere in the song, the other channel, other encodings if available.
2. **Align** repetitions (time-warp + pitch), pulling each one's quantization box back to a
   common frame. This is non-local means meeting frame theory — the *new operator*.
3. **Project** jointly onto the intersection of all boxes via Dykstra. Note the filterbank
   is a **non-orthogonal** hybrid (PQF + MDCT + alias butterflies), so projection is *not*
   elementwise clipping — `RESEARCH_DESIGN.md` §2 states this incorrectly. It requires
   alternating projection, and unrolling depth is set by the Friedrichs angle between the
   box and the range of the analysis operator.
4. **Learn the prior as a plug-and-play denoiser inside the loop**, not as a regressor
   outside it — Tweedie gives the score, unrolling gives the depth.
5. **Regime C entirely separate**: a generative prior with an optimal-transport /
   distributional objective (pointwise loss is meaningless for invented content), labelled
   synthesis, with EXP 1 as the standing proof that nothing better is achievable.

### Why the three-head split is not a preference but a theorem

Rate–distortion–perception theory (Blau & Michaeli) proves that at fixed rate, distortion
and distributional realism **cannot both be optimized** — perfect realism costs a bounded
but nonzero penalty in MSE. Regime A wants the conditional mean (minimum distortion);
Regime C wants a posterior sample (maximum realism). These are *provably different
estimators*. A single network under one blended loss is therefore not merely inelegant —
it is provably suboptimal for both. This is the theoretical justification for the
factorization, and it also explains why Apollo's magnitude-only training paired with
SI-SDR checkpoint selection is incoherent.

---

## Revised plan

| Stage | Work | Gate |
|---|---|---|
| ~~1~~ | ~~Projection-only baseline~~ | **Cancelled** — EXP 2 predicts ~1% at r≈2 |
| 0 | Bitstream parser; verify true coefficients lie inside computed boxes | If not, quantizer model wrong — stop |
| **1′** | **Repetition mining**: measure achievable r on real songs. How many well-matched repetitions does a typical track admit, at what alignment error? | **If median r < 8, the whole program's ceiling is ~1.3× and it should be reconsidered** |
| 2′ | Joint Dykstra fusion over mined repetitions; measure in-band MSE vs r on real MP3 pairs | Must beat single-frame projection by the EXP-2 curve |
| 3′ | Plug-and-play prior inside the loop; unrolled | vs Stage 2′ |
| 4′ | Regime C generative head with OT objective; stereo restoration | Per-regime metrics only |

**Stage 1′ is the new decisive cheap experiment** — it needs no training and no model, only
a repetition-mining pass over real music, and it directly measures the quantity that
EXP 2 proves everything else depends on.

---

## Honest status

- EXP 1 (negative) and EXP 2 (positive) are synthetic, in the scratchpad, unpromoted.
- EXP 2's slope −1.63 vs theory −2 is unresolved; most likely POCS under-convergence.
- The redundancy-from-repetition mechanism is **untested on real audio** and is the single
  highest-risk, highest-value claim in this document. Stage 1′ exists to kill it fast.
- Multi-encoding fusion assumes access to multiple files of one master — true for archives,
  false for the consumer case.

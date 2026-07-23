# Stereo restoration — prototype and adversarial audit

Dated 2026-07-23. Same protocol as the repetition mechanism: pre-registered gates,
then attack the result. **The prototype passed its gates and the audit retracted it.**

---

## What intensity stereo actually destroys

Above the intensity bound, MP3 transmits only the sum signal plus a per-band position
parameter `is_pos`. Both channels are rebuilt as scaled copies of one signal, so
**inter-channel coherence is forced to exactly 1 and inter-channel phase to 0.**

Constraints that survive — and unlike the repetition case these are genuinely hard:

| | Status |
|---|---|
| `L+R` (mid) | **Transmitted — fully determined** |
| per-band \|L\|/\|R\| | **Transmitted via `is_pos`** |
| side channel fine structure | **Destroyed — the whole problem** |

## EXP 4 — the prototype. Gates passed.

Physical claim: for a spaced pair in a diffuse field, coherence follows the Cook law
`ICC(f) = sinc(2πfd/c)` — a *one-parameter* family. Estimate spacing `d` from the
surviving low band, and high-band coherence follows from physics rather than guesswork.

| Gate | Result |
|---|---|
| G1 soundness — mid preserved exactly | **PASS**, drift 6.2e−17 |
| G2 coherence error reduced >50% | **PASS**, 0.956 → 0.025 (97%) |
| G3 adversarial — waveform fidelity | NMSE 1.134 → 3.106, **worse as predicted** |

## The audit that killed it

**Tell:** in the generalization sweep, `d_hat` pinned to the grid edge (1.500) for true
d = 0.60 and 1.00 — the estimator failed completely — yet the reported reduction stayed
at **+97%**. A result immune to a broken estimator is not measuring what it claims.

### Probe 1 — the test band was vacuous. **EXP 4 is retracted.**

Mean true high-band coherence above 6 kHz, by spacing: d=0.05 → 0.054, d=0.15 → 0.019,
d=0.30 → 0.009, d=1.00 → 0.003. **Near zero for every spacing.** So "predict the
coherence" collapsed to "output decorrelated," which any parameterless widener achieves.
The physics story contributed nothing; the 97% was an artifact of the chosen band.

### Probe 2 — the method damages most of the catalogue. **Near-fatal.**

Amplitude panning (a pan pot) sends the *same* signal to both channels at different
gains, so coherence is **exactly 1**. On such material MP3's forced ICC=1 is *correct*,
and "restoring" decorrelation injects width that was never in the master.

| Treatment | resulting ICC | waveform NMSE |
|---|---:|---:|
| collapsed — what MP3 gives | 1.000 | **0.022** |
| mild decorrelation | 0.725 | 0.192 |
| full decorrelation | 0.038 | **1.081** |

**A 49× waveform degradation, and the coherence moves away from truth too.** Most
commercial pop, rock, hip-hop and electronic music is amplitude-panned multitrack, not
spaced-pair acoustic recording. The Cook law describes classical, jazz and live acoustic
recording — a minority of the catalogue and, notably, not the material most likely to
exist only as low-bitrate rips.

### Probe 3 — viability reduces to one unmeasured number

Dropping the sinc assumption, model ICC(f) as a smooth random profile and ask whether the
coded low band predicts the destroyed high band better than knowing nothing.

*(First attempt used R², which diverges as the profile flattens and target variance → 0 —
a metric artifact, discarded. Redone with mean absolute coherence error.)*

| ICC frequency-correlation length | low-band predictor beats uninformed by |
|---:|---:|
| 0.25 oct | **−7%** (worse than guessing) |
| 0.5 oct | −6% |
| 1 oct | −4% |
| 2 oct | **+20%** |
| 4 oct | +52% |
| 8 oct | +72% |

Sharp threshold near **1–2 octaves**. Below it the method has no restoration content at
all; above it there is real information in the coded band.

**That number cannot be obtained from simulation.** It requires real stereo masters.

---

## Prior art — the synthesis half is already standardized

**Parametric Stereo** (HE-AAC v2) and **MPEG Surround**, deployed since ~2004, transmit a
mono downmix plus spatial parameters — IID, **ICC**, IPD — and synthesize the stereo image
at the decoder with decorrelators. Schroeder all-pass decorrelation predates that by
decades, and the consumer stereo-widener industry is older still.

So the synthesis machinery is completely solved prior art. The only new element is
*estimating* the ICC that MP3 failed to transmit — and Probe 3 says that estimate may
carry no information. **The novel part is exactly the part that might not exist.**

---

## Verdict

| Question | Answer |
|---|---|
| Novel? | Only the ICC-estimation step; synthesis is 20-year-old standardized prior art |
| Solves a real problem? | Stereo collapse is genuinely audible at ≤128 kbps — **the problem is real** |
| Does this method solve it? | **Unproven, and it damages the majority of the catalogue as built** |
| Waveform-validatable? | **No.** Even in the favourable case G3 showed fidelity worsening. Definitionally perceptual, therefore easy to fool yourself with |

### The one repair worth keeping

The Probe-2 damage is avoidable by **gating on measured low-band coherence**: if low-band
ICC ≈ 1 the track is amplitude-panned — do nothing. Only intervene where the coded band
shows genuine decorrelation. That converts a catalogue-wide hazard into a
conservative, opt-in treatment, and it costs nothing to implement.

### Decisive next measurement

Measure the **frequency-correlation length of inter-channel coherence** across a real
stereo corpus, split by genre and production era. One number, no training, no model. If
it is below ~1–2 octaves, stereo restoration has no content and this direction closes
alongside the others.

---

## Meta-finding: four for four

EXP 2 (frame normalization), EXP 3 (wrong recovery target), EXP 4 (vacuous test band),
and Probe 3 (unstable metric) **all produced flattering first-pass results that were
artifacts** — every one caught only by attacking my own output.

The lesson is not that the individual tests were sloppy. It is that **synthetic validation
in this domain systematically flatters the hypothesis**, because the simulator embeds the
same assumptions the method relies on. No further synthetic experiment in this program
should be believed. Real audio is now a precondition, not an improvement.

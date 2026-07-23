# The melody scans the instrument past the cutoff

First mechanism in this program to survive its own adversarial test.
Dated 2026-07-23. Synthetic only. Read the scope conditions and open risks before
believing any of it.

---

## Diagnosis: why everything before this failed

| Attempt | Failure |
|---|---|
| EXP 1 — extrapolate one note's spectrum | Exponentially ill-conditioned; dead |
| EXP 3 — average repeated passages | Performances differ; fusion *hurts* 5–9× |
| EXP 4/5 — stereo coherence prediction | Vacuous test band; damages panned material |

**All three mined the observation.** The data processing inequality caps that at the
file's bitrate, so they were working a seam that is provably exhausted. Information can
come from only two places: the observation, or the prior. The prior was never seriously
examined.

And the first-principles question that reframes it: **how much information is actually in
a piece of music?** FLAC removes statistical redundancy and lands near 800 kbps. The
*generative* description — notes, instruments, room, mix — is orders of magnitude smaller.
That gap is the restoration capacity nobody exploits structurally.

Which exposes EXP 3's real error. Music's redundancy is not in repeated **waveforms** —
it is in repeated **generators**. The same guitar plays 500 notes: 500 different
waveforms, one identical instrument.

---

## The mechanism

A note is source-filter: `X(f) = S(h)·B(f)`, where `h = f/f0` is the harmonic index.

- `S(h)` — source spectrum, indexed by **harmonic number** (pitch-relative)
- `B(f)` — body/room filter, fixed in **absolute Hz** (pitch-invariant)

The codec zeroes everything above a cutoff `fc` **fixed in absolute frequency**. But the
instrument's structure is indexed by harmonic number. These are different axes, joined by
`f = h·f0`.

> **So as the melody moves, the instrument's harmonic structure slides across the fixed
> cutoff. A low note puts harmonic 40 at 3 kHz — observed. A high note puts harmonic 40 at
> 18 kHz — destroyed. The song's own melody scans the instrument's spectrum past the
> codec's window.**

Recovering the high band of a high note is then **not extrapolation along frequency**
(proven dead in EXP 1) but **interpolation along harmonic index**, using notes where that
index fell below the cutoff. Different axis, different conditioning, and it works.

It also repairs EXP 3 precisely: the shared object here — the instrument — genuinely *is*
identical across observations, unlike performances. And it finally gives EXP 2's rate law
a valid application: redundancy `r ≈ N_notes` on the shared generator, which is why the
result is nearly noise-invariant.

---

## Results (EXP 8, leak-free)

*EXP 7 scored a perfect 0.000 because the prediction step called the true body filter
above the cutoff — the one quantity declared unobservable. Retracted. EXP 8 jointly
estimates `S(h)`, `B(f)` and per-note gain from observed harmonics only, and extrapolates
`B` above `fc` low-order.*

Relative error on the destroyed band of high notes (1.000 = codec's zero-fill):

| quantization noise | zero-fill | per-note extrapolation | **cross-pitch transfer** |
|---|---:|---:|---:|
| none | 1.000 | 1.330 | **0.146** |
| −30 dB | 1.000 | 1.330 | **0.148** |
| −25 dB (realistic) | 1.000 | 1.330 | **0.134** |
| −20 dB | 1.000 | 1.330 | **0.146** |

Per-note extrapolation is *worse than doing nothing* — an independent confirmation of
EXP 1. Cross-pitch transfer cuts error ~85%, and is essentially noise-invariant because
the shared estimate averages ~60 notes. The residual ~0.15 is systematic **body-filter
extrapolation bias**, not noise — `B(f)` above `fc` is observed by no note and remains the
error floor.

### Identifiability caveat (structural)

`S(h) → S(h)·hᵃ` with `B(f) → B(f)·f⁻ᵃ` changes the product only by a per-note constant,
absorbed into the gain. **The power-law split between source and body is not
identifiable**, and since extrapolating `B` depends on exactly that slope, this ambiguity
sets the error floor. Exposed, not hidden.

---

## Scope conditions (EXP 9 — adversarial)

**T1 — model mismatch.** Real instruments vary brightness with velocity, breaking the
shared-`S` assumption:

| brightness jitter (dB/decade) | 0 | 2 | 4 | 6 | 10 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| transfer error | 0.155 | 0.172 | 0.226 | 0.405 | 0.568 | 0.931 |

Tolerates moderate expression; dies past ~10 dB/decade. **Concrete fix:** brightness
varies *with velocity*, which is observable — model `S(h, velocity)` and the nuisance
becomes a covariate rather than noise.

**T2 — pitch range.** The mechanism needs low notes to observe high harmonic indices:

| f0 span | 3.2 oct | 2.4 oct | 1.4 oct | 0.7 oct | 0.3 oct |
|---|---:|---:|---:|---:|---:|
| transfer error | 0.172 | 0.285 | 0.664 | 4.415 | 17.816 |

**Needs ≥ ~2 octaves.** Below 1 octave it is catastrophically worse than zero-fill.

**T3 — polyphony. Untested, and the largest open risk.** Real music sounds several
instruments at once, so an observed harmonic cannot be attributed to one generator without
source separation.

### Hard scope limit

**Percussion and noise-like content get nothing from this mechanism.** A cymbal has a
shared generator but no pitch diversity, so nothing ever moves across the cutoff.
Uncomfortable, because cymbals and hi-hats are among the most audibly damaged elements at
low bitrate.

So the honest domain is: **pitched, multi-octave, moderately-expressive, separable
instruments** — melodic and harmonic content, not percussion.

---

## Novelty — provisional, not established

Two searches found the adjacent prior art:

- **SBR (Spectral Band Replication)**, HE-AAC — copies low-band content upward guided by
  *transmitted* envelope parameters. Standardized, deployed. Differs in that MP3 transmits
  no such parameters, and SBR patches *within a frame*, translating spectrum to wrong
  absolute frequencies (its known roughness artifact). This mechanism places harmonics at
  exactly `h·f0` with amplitudes genuinely *measured* on that instrument at that harmonic
  index.
- **Speech bandwidth extension** via source-filter — long-established, but for a single
  speaker's vocal tract, not cross-pitch transfer within a polyphonic music recording.
- **DDSP** (Engel et al.) — a differentiable source-filter synthesis framework; the
  natural implementation substrate, and prior art for the synthesis half.

The specific framing — *the melody scans the instrument's harmonic structure across a
fixed-frequency cutoff, so pitch diversity supplies what extrapolation cannot* — did not
appear in these searches. **That is two searches, not a prior-art search.** Given that
five of this program's experiments produced flattering artifacts on first pass, treat the
novelty claim as unestablished pending a proper search.

---

## Why this is worth pursuing when the others were not

1. It is the **only** mechanism here that survived adversarial testing rather than
   collapsing or leaking.
2. It **evades EXP 1's impossibility proof** on a technicality that is real: EXP 1 forbids
   extrapolation *along frequency*; this interpolates *along harmonic index*.
3. It degrades **gracefully and believably** with mismatch — the signature of a real
   effect rather than a simulator artifact.
4. Its scope conditions are **computable per track** (pitch span, brightness variance), so
   it can be gated to where it works instead of applied blindly.
5. It attacks Regime C — the band that actually matters at the low bitrates where
   restoration is needed, correcting the self-defeating thesis in `MATH_SURVEY.md`.

## Next, in order

1. **Real monophonic instrument recordings.** Solo cello, sax, voice. Real source-filter
   violation, real quantization. This is where five previous results would have died.
2. **Measure the actual brightness jitter** of real instruments in dB/decade — T1's
   tolerance is known, the real value is not.
3. **Polyphony (T3)** via existing separation front-ends. The largest risk.
4. Only then: implementation on DDSP, joint with the codec constraint set.

## Standing caution

Six experiments this session; **five produced flattering first-pass results that were
artifacts** (frame normalization, wrong recovery target, vacuous test band, unstable
metric, and a leaked oracle). Every one was caught only by attacking my own output. The
simulator keeps embedding the assumptions the method needs. **No synthetic result in this
program should be believed. Real audio is a precondition, not an improvement.**

# Real-audio test of the cross-pitch generator mechanism — FAILED

Dated 2026-07-23. Real lossless sources, real LAME encode/decode, real pitch tracking.
**The mechanism fails its pre-registered gate. The synthetic result did not transfer.**

---

## Corpus (all lossless, downloaded)

| source | material | licence/origin |
|---|---|---|
| Flute B3–B6, nonvib, pp/mf/ff | chromatic scales, 44.1k/16 mono | Univ. of Iowa MIS |
| Cello C2–Bb4, arco mf | chromatic scales, 44.1k/16 mono | Univ. of Iowa MIS |
| **Sitar** — *Ocean of Milk*, Ranjit Makkuni | 44.1k/**24-bit** stereo, 56 s | Wikimedia Commons |
| **Carnatic vocal** — *Ninnukori/Mohanam* | 44.1k/16 mono, 6:19 | Wikimedia Commons |

## Real measured LAME cutoffs (white-noise probe, replacing earlier guesses)

| kbps | 24 | 32 | 64 | 96 | 128 | 192 | 320 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **measured** | 4377 | 5760 | 11267 | 15402 | 16780 | 18847 | 20225 |
| earlier guess | 8000 | 11000 | 13000 | 15500 | 16000 | 19000 | 20500 |

At 24 kbps only 4.4 kHz survives — **80% of the spectrum destroyed**, not the 64% documented
in `RESEARCH_DESIGN.md`. Low-bitrate damage is worse than previously stated.

---

## Results — relative error on the destroyed band (1.000 = codec's zero-fill)

| instrument | kbps | f0 span | frames | zero-fill | per-note extrap | **cross-pitch** |
|---|---:|---:|---:|---:|---:|---:|
| flute B3–B6 | 64 | 2.1 oct | 384 | 1.000 | **0.573** | 0.681 |
| flute B3–B6 | 128 | 2.1 oct | 367 | 1.000 | **0.392** | 0.859 |
| cello C2–Bb4 | 64 | 2.8 oct | 344 | 1.000 | **0.430** | 0.665 |
| cello C2–Bb4 | 128 | 2.8 oct | 174 | 1.000 | **0.433** | 0.786 |
| **sitar** | 64 | 3.0 oct | 181 | 1.000 | 1.028 | **0.758** |
| **sitar** | 128 | 3.0 oct | 133 | 1.000 | 40.371 | 12.171 |
| **Carnatic vocal** | 64 | 1.6 oct | 2039 | 1.000 | 1.031 | **0.736** |
| **Carnatic vocal** | 128 | 1.6 oct | 2031 | 1.000 | 1.399 | **0.757** |

**Pre-registered gate:** cross-pitch < 0.60 **and** beats per-note extrapolation.
**FAILS on both counts.** Synthetic promised 0.134; real audio delivers 0.67–0.86 — a
5–6× degradation — and on Western instruments it *loses to the baseline it was supposed to
replace*.

### Harness fairness

Two legitimate defects were found and fixed before accepting the verdict: the harmonic
peak-picking window could straddle neighbours at 65 Hz (capped at f0/3), and octave errors
polluted the shared fit (added a 3-harmonic energy validity check). f0 tracking was
validated against known chromatic pitches: 90% within 25 cents, 5% octave errors. Fixes
improved results marginally; the verdict is unchanged. The failure is real, not a bug.

---

## Artifact #6 — and this one flattered the method

**The synthetic experiment was unfair to the baseline.** EXP 8/9 reported per-note
extrapolation at 1.330 — *worse than doing nothing* — and I treated that as independent
confirmation of EXP 1's impossibility result.

On real Western instruments extrapolation scores **0.39–0.57**. It works well.

The cause: my synthetic source spectrum carried `1 + 0.5·sin(h/7) + 0.3·sin(h/3)` ripple,
which breaks power-law extrapolation. Real flute and cello spectra are far smoother than
that. **I built a simulator whose irregularity disabled the competing method**, then
reported my method beating it. Sixth artifact of the program, and the first that biased
*toward* my own proposal rather than merely inflating it.

This also partially rehabilitates EXP 1: bandlimited extrapolation is dead as an *exact*
inverse, but a smooth rolloff fit is a perfectly serviceable *estimator* for regular
spectra. Those are different claims and I conflated them.

---

## The one real finding: it is exactly the Indian instruments that behave differently

The pattern is consistent and, unlike everything else here, not an artifact:

- **Flute, cello** — smooth, regular harmonic rolloff. A power-law fit extrapolates well.
  Cross-pitch transfer adds nothing and loses.
- **Sitar, Carnatic vocal** — extrapolation **fails outright** (1.03, 1.03, 1.40: worse
  than leaving the band empty). Cross-pitch transfer is the *only* method that beats
  zero-fill, at 0.74–0.76.

The mechanism: sitar's **jawari** (curved bridge) produces a formant-rich, non-monotonic
spectrum with strong high partials, and Carnatic vocal has strong formant structure —
neither follows a power law, so rolloff extrapolation breaks. The shared-generator model
captures structure a rolloff fit cannot.

So the mechanism has a genuine domain — **spectrally irregular instruments** — and Indian
classical instruments sit squarely in it. But even there it is only ~25% better than doing
nothing, nowhere near the 0.60 gate, and far from the synthetic promise.

### The sitar 128 kbps anomaly

Both methods explode (12.2 and 40.4). At 128 kbps the cutoff is 16.8 kHz, so with
f0 ≈ 200–600 Hz the destroyed harmonics are index 28–84, where the shared `S(h)` estimate
is weakly constrained; sitar's sympathetic strings add strong non-harmonic content that the
harmonic model cannot represent at all. **Neither method is usable on sitar at 128 kbps.**

---

## Honest verdict

Six mechanisms attempted; six failed. This one failed on **real data** rather than on
self-audit, which is at least a better class of failure.

What is actually true after this test:

1. **For regular-spectrum instruments the best method is a smooth rolloff fit** — which is
   essentially what classical bandwidth extension and SBR already do. No advance.
2. **For spectrally irregular instruments (sitar, Carnatic voice) that baseline fails**, and
   cross-pitch transfer is the only thing that helps. Small, real, narrow.
3. The obvious repair is a **hybrid gated on measured spectral regularity below the cutoff** —
   observable at restoration time, no ground truth needed. Oracle-selecting the better of
   the two per instrument gives 0.39–0.76. That is an engineering combination of two
   mediocre estimators, **not a breakthrough**, and it should not be described as one.

## Open, not closed

- Only two Indian instruments tested, one recording each. The Indian-instrument finding
  rests on ~300 scored frames of sitar and one Carnatic performance. It needs a real corpus
  (Saraga) before it means anything.
- Polyphony still untested. Everything here is monophonic or near-monophonic.
- No perceptual evaluation whatsoever — all numbers are spectral-magnitude error.

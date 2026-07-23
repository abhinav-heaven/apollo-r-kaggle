# Evaluation that actually tests restoration

## The problem with every standard metric

A model that **outputs its input unchanged** scores respectably on all of them:

| metric | why it fails here |
|---|---|
| SI-SDR | strictly phase-sensitive; punishes inaudible phase error, meaningless over synthesized content. Apollo optimizes magnitude and *selects on SI-SDR* — incoherent |
| PESQ | speech only, not 44.1 kHz music |
| ViSQOL | opaque, narrow validation, not differentiable |
| MR-STFT | ignores phase; a model can match magnitude while emitting garbage phase |
| any absolute error | dominated by the loud low band **where the codec did no damage** — most of the number measures content the model never touched |

None of them are referenced to the damage, so none answer *"did you restore anything?"*

## Restoration Gain Ratio (RGR)

```
RGR = ||est − clean|| / ||degraded − clean||
```

**Verified properties on real flute @ mp3 64k:**

| case | RGR | required |
|---|---|---|
| identity (output = input) | **1.0000** | exactly 1.0 |
| oracle (output = clean) | **0.0000** | 0 |
| halfway to clean | **0.5007** | ~0.5 |
| degraded + noise | **22.72** | >1 |

And it is **comparable across bitrates**, which absolute error is not:

| kbps | absolute err | RGR identity | RGR halfway |
|---|---|---|---|
| 24 | 0.1264 | 1.0000 | 0.4998 |
| 64 | 0.0655 | 1.0000 | 0.5007 |
| 128 | 0.0501 | 1.0000 | 0.5000 |
| 192 | 0.0300 | 1.0000 | 0.5000 |

Absolute error varies 4× and is therefore meaningless to compare; RGR is anchored.
As a **loss** it also gives damage-weighting for free — samples the codec barely
touched cannot dominate. It includes a complex (phase-aware) term, closing the
magnitude-only loophole.

## Null test — the strongest "restoring or just styling?" check

Feed **already-lossless** audio. A true inverse-of-degradation must be ~identity. A
model that "restores" undamaged audio has learned a style it applies unconditionally,
not the inverse of a codec. Needs no reference, and almost nobody runs it.

## Hallucination detector (signed high-band energy bias, dB)

Generative restorers win listening tests by adding flattering treble that is not in the
master. Magnitude error cannot tell "added 3 dB of sizzle" from "missed 3 dB of real
content", so bias is reported **signed and separately**. Verified:

| input | bias | required |
|---|---|---|
| oracle | **+0.00 dB** | ~0 |
| clean + 6 dB fake sizzle | **+6.02 dB** | caught |
| codec output (band emptied) | **−48.75 dB** | very negative |

## Re-encode consistency — reference-free and falsifiable

Re-encode the restoration; below the cutoff it must reproduce the observed file, because
that band is *constrained by the bitstream*. Any deviation is hallucination, not
restoration. **Needs no clean reference**, so it works on real-world files with no master.

| case | value | required |
|---|---|---|
| honest (re-encode codec output) | **0.054** | ~0 |
| tampered in the coded band | **0.426** | ≫0, caught |

## Regime-split loss

Rate–distortion–perception theory: distortion and realism cannot both be optimized at
fixed rate, so the two bands need **different estimators** — one blended loss is provably
suboptimal for both.

- **below cutoff** — constrained → complex, phase-aware, damage-relative
- **above cutoff** — unrecoverable pointwise (EXP 1) → match the *distribution* of
  log-magnitude, not exact values

Both regimes are **anchored so identity == 1.0**. Un-anchored, identity scored 1.0 below
and **45.5** above, so training would have been ~45:1 dominated by the band that is
provably unrecoverable — the worst possible use of a fixed compute budget.

| codec/kbps | case | below-cut | above-cut |
|---|---|---|---|
| mp3 64 | identity | 1.000 | 1.000 |
| mp3 64 | halfway | 0.250 | 0.010 |
| mp3 64 | oracle | 0.000 | 0.000 |
| opus 64 | identity | 1.000 | 1.000 |

Stable across mp3/aac/opus and bitrate.

## Reporting rule

Never a single global number. Report **per group × per bitrate × per regime**, plus the
null test and the signed energy bias, and always against the rolloff baseline — not
against unprocessed audio.

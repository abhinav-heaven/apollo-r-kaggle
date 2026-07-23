# Apollo-R: a concrete, evidence-driven upgrade for 2×T4 / 24h

Every choice below traces to a measured result from this program. Where evidence is
absent, that is stated. **This is Apollo debugged, re-conditioned, and generalized — a
solid engineering advance with one genuinely novel angle. It is not a breakthrough, and
should not be written up as one.**

---

## Compute reality first

| | |
|---|---|
| Model | ~16.5 M params — small, fits T4 easily |
| Kaggle session cap | **12 h**, not 24 → two checkpointed sessions |
| GPU | 2×T4 16 GB, Turing: **fp16 only, no bf16** |
| Apollo's own config | 8 GPUs × 500 epochs × 40k samples — hundreds of GPU-hours |

**From-scratch training is impossible at this budget.** Fine-tune from
`JusperLee/Apollo`. Every change below is therefore constrained to preserve
checkpoint compatibility, or to add zero-initialized parameters so that step 0 of
fine-tuning reproduces the pretrained model exactly.

**Data:** MUSDB18-HQ ≈ 30 GB, MoisesDB ≈ 40 GB. Downloading both burns hours of a 12 h
session. **Search Kaggle Datasets for a pre-hosted MUSDB18-HQ and attach it — mounted
datasets cost zero download time.** Otherwise use MUSDB18-HQ alone.

---

## The changes, ranked by expected value per GPU-hour

### 1. Fix the bitrate randomization bug — **free, and probably the largest single win**

`codec_simu` resolved `'random'` **into the caller's dict**, so the bitrate froze at the
first draw per worker. Verified: 8 successive calls returned `[96,96,96,96,96,96,96,96]`.

**The released Apollo was trained on ~8 fixed bitrates, not the 24–128 kbps continuum its
README claims.** Fine-tuning with genuinely randomized bitrates directly attacks the
model's headline weakness and costs nothing. Already fixed in
`look2hear/datas/musdb_moisesdb_datamodule.py`.

### 2. Replace `apply_codec` — **required; the pipeline cannot run without it**

`torchaudio.functional.apply_codec` is deprecated. CORRECTION: it still works in torchaudio 2.2.2 (verified) -- an earlier claim that it was removed in 2.2 was wrong. The replacement is still required for multi-codec support.
Implemented as an ffmpeg-backed replacement in `codec_sim.py`.

### 3. Cutoff conditioning (FiLM, zero-init) — **cheap, evidence-backed**

Measured LAME cutoffs vary enormously and are *not* what I originally assumed:

| kbps | 24 | 32 | 64 | 96 | 128 | 192 | 320 |
|---|---:|---:|---:|---:|---:|---:|---:|
| measured cutoff (Hz) | 4377 | 5760 | 11267 | 15402 | 16780 | 18847 | 20225 |

At 24 kbps **80% of the spectrum is destroyed**. The model currently must infer this
blind — and per (1) it never even saw varied bitrates.

**Where the conditioning signal comes from — corrected.** I first built a blind spectral
cutoff detector. Validated on real music against the measured cutoffs, it is **not good
enough**: median error 2.5 kHz, worst case 4.9 kHz, and it systematically overestimates at
low bitrates, because a music signal's own spectral rolloff is confounded with the codec's
lowpass. (Two earlier versions were worse still — a −40 dB threshold merely measured the
music's rolloff, and zero-padded smoothing pinned every estimate to Nyquist.)

**The bitrate is carried in the MP3 frame header.** `ffprobe` returns it exactly —
verified: 64/128/320 k encode → header reports 64/128/320. And during training we *choose*
the bitrate, so conditioning is exact there too. Blind detection is a fallback for
already-decoded audio only, and is documented with its real accuracy.

Feed the cutoff as FiLM conditioning with zero-initialized projection, so init ≡
pretrained behaviour.

### 4. Learned per-band preservation gate — **implements the regime split safely**

Apollo regenerates all 442 bins from scratch, including the coded band that is already
nearly correct, then is penalized for changing it. The CCF AATC 2025 retrospective found
top discriminative models "approximate an identity mapping while subtly correcting
quantization artifacts" — i.e. they *learn* this behaviour the hard way.

Add a per-band gate `g` blending model output with the input spectrum, **initialized so
that g ≡ pretrained behaviour**, letting training discover the right blend per band. This
lets capacity move to the destroyed band without forfeiting Regime A/B denoising (which
my EXP 2 says is small but real, ~1% from consistency alone — a learned denoiser may do
better).

### 5. Indian-instrument generalization — **the one genuinely novel contribution**

This is the only claim in the entire program that survived real-data testing:

| instrument | 64 kbps: rolloff extrapolation | cross-pitch transfer |
|---|---:|---:|
| flute, cello (Western) | **0.43–0.57** (works) | 0.67 |
| **sitar** | **1.03** (fails — worse than nothing) | 0.76 |
| **Carnatic vocal** | **1.03** (fails) | 0.74 |

Sitar's **jawari** and Carnatic vocal formants produce non-monotonic spectra that
power-law extrapolation cannot represent. Apollo's training data (MUSDB18-HQ + MoisesDB)
is Western pop/rock. **The hypothesis — that codec restoration generalizes poorly to
non-Western timbres, and that adding such data fixes it — is testable in one fine-tune
and is publishable regardless of outcome.**

Add Saraga (CompMusic Carnatic/Hindustani) or equivalent to the fine-tune mix. **Hold out
sitar/Carnatic test material and report Western vs Indian separately.**

### 6. Drop the GAN for most of the run — **buys ~2× throughput**

Apollo's step runs **four discriminator forward passes** (two for D, two for G). For a
short fine-tune the adversarial term is unlikely to converge usefully and risks
destabilizing a good checkpoint. Fine-tune with multi-resolution STFT loss alone, then
optionally a short adversarial polish in session 2 if time remains.

### 7. Fix the objective/selection incoherence — **free**

Apollo trains a magnitude-only loss (`freq_MAE` compares `.abs()`) plus adversarial terms,
then selects checkpoints on **SI-SDR**, which is strictly phase-sensitive and never
optimized. Select on what you optimize, and **report per-regime** (below vs above the
measured cutoff) — a single global number hides whether the model is restoring or
synthesizing.

---

## Explicitly NOT doing (and why)

| Idea | Why not |
|---|---|
| Re-band onto the codec's scalefactor bands | Correct (they are Bark-like, and end at exactly 16002 Hz) but **breaks checkpoint compatibility** → from-scratch → out of budget |
| Bitstream parsing + hard projection | ~1% at r≈1 (EXP 2). High implementation cost, negligible quality return |
| Repetition fusion | **Refuted** (EXP 3): hurts 5–9× at realistic repetition mismatch |
| Stereo/ICC restoration | **Refuted** (EXP 5): damages amplitude-panned material 49× |
| Bandlimited extrapolation as exact inverse | **Refuted** (EXP 1) |

---

## Schedule

**Session 1 (12 h)**
1. (1 h) Attach data, build HDF5, smoke-test one training step end-to-end
2. (1 h) Baseline: evaluate released Apollo per-regime on held-out Western **and** Indian
3. (9 h) Fine-tune with fixes 1–4 + Indian data, checkpoint every 30 min
4. (1 h) Evaluate, save state

**Session 2 (12 h)**
5. (8 h) Continue best run
6. (2 h) Optional adversarial polish
7. (2 h) Final per-regime evaluation, Western vs Indian ablation

**Ablation that makes it publishable** — hold everything else fixed and vary only the
Indian data. Two runs, one number, a real claim.

---

## Honest expected outcome

A measurably better Apollo — chiefly from the bitrate bug fix and cutoff conditioning —
plus the first evidence on non-Western generalization in codec restoration.

**What it will not do:** recover the destroyed band exactly (EXP 1), or beat a simple
rolloff fit on regular-spectrum Western instruments where that baseline already scores
0.39–0.57. The realistic win is on irregular timbres and low bitrates.

**Measure against the rolloff baseline, not just against unprocessed audio.** That
baseline is stronger than the literature implies, and it is the honest comparison.

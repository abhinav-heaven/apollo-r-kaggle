# Toward provably-consistent, perceptually-transparent restoration of lossy-coded music

A first-principles design document. Supersedes the "make the GAN bigger" trajectory.

---

## 0. The target, stated honestly

The phrase "true FLAC studio quality" contains a category error that has to be cleared
before any engineering is meaningful.

**FLAC is not a quality tier.** It is a *lossless* compressor — LPC prediction plus Rice
coding — that reproduces its input bit-exactly. A FLAC file is exactly as good as whatever
was fed to it. Encoding a restored signal to FLAC is lossless *storage of a lossy-restored
signal*; it adds no quality and the label misleads. "FLAC quality" is undefined without
naming a reference.

The only well-posed target is: **the studio master**, and the only well-posed goals are

1. **Provable consistency** — the output could actually have produced the observed file.
2. **Perceptual transparency** — indistinguishable from the master under properly-run ABX.
3. **Honest accounting** — per-region disclosure of what was *restored* vs *synthesized*.

### 0.1 Bit-exact recovery is impossible. This is not a modelling failure.

Lossy encoding is a many-to-one map. An uncountable set of masters produces any given MP3
file. By the data processing inequality, no function of the file — however large the model —
can carry more information about the master than the file itself does:

| Bitrate | % of PCM 16/44.1 (1411 kbps) | % of a ~800 kbps FLAC master |
|--------:|-----------------------------:|-----------------------------:|
|  24 kbps |  1.7% |  3.0% |
|  64 kbps |  4.5% |  8.0% |
| 128 kbps |  9.1% | 16.0% |
| 320 kbps | 22.7% | 40.0% |

Anyone claiming recovery of the master from a 128 kbps MP3 is claiming to conjure 84% of
the information. What a model legitimately supplies is a **prior**. The task is therefore
not inversion but **posterior sampling**: draw a plausible master consistent with the
evidence. Bit-exactness is off the table; transparency is not.

**The gap between "not exactly recoverable" and "not perceptually distinguishable" is
large, and it is the entire opportunity.** Everything below is about occupying it rigorously
instead of decoratively.

---

## 1. Building-block audit: what each stage actually destroys

### 1.1 PCM and the master

44.1 kHz / 16-bit stereo. Dithered noise floor ≈ −93 dBFS; 24-bit masters ≈ −140 dBFS.
Relevant because generative restorers characteristically produce noise floors that are
*too clean* (an unnatural, "plastic" quiet) or too noisy. Reproducing the master's floor
and dither character is part of transparency and is essentially never evaluated.

### 1.2 The MP3 (MPEG-1 Layer III) chain, stage by stage

| # | Stage | What it does | What it destroys | Recoverable? |
|---|-------|--------------|------------------|--------------|
| 1 | 32-band polyphase filterbank (PQF) | 689 Hz subbands | Non-ideal filters alias between adjacent bands | Deterministic, structure known |
| 2 | MDCT, 18 lines (long) or 6×3 (short) | 576 lines/granule | Nothing yet — but critically sampled and *lapped* | — |
| 3 | Alias-reduction butterflies | Partially cancels (1) | Residual leakage | Known, invertible |
| 4 | Psychoacoustic model → scalefactors | Per-SFB noise allocation | Nothing directly; **decisions are transmitted** | Read from bitstream |
| 5 | Power-law quantizer `is = nint(\|xr\|^{3/4} / 2^{gain/4})` | Nonuniform quantization | **Bounded** error per coefficient | Interval-constrained |
| 6 | Huffman + count1 + **rzero** | Entropy coding; HF forced to zero | rzero region **identically zero**, unbounded loss | Generative only |
| 7 | Bit reservoir | Rate smoothing across frames | Nothing | — |
| 8 | Joint stereo: MS and **intensity** | Above a band, only sum + direction kept | **Side channel destroyed** — stereo image collapse | Constrained generative |

Three of these deserve emphasis because they overturn how the problem is usually posed.

**Stage 5 is bounded, and the bound is tight.** Given the transmitted integer `is`, the true
coefficient is pinned to a known interval:

| `is` | true value pinned to |
|-----:|---------------------:|
| 1 | ± 66% |
| 5 | ± 13.3% |
| 10 | ± 6.7% |
| 100 | ± 0.67% |
| 1000 | ± 0.07% |

Loud coefficients are known to a fraction of a percent. This is *not* an unknown quantity to
be hallucinated — it is a **constraint**, and it is being thrown away by every method that
takes decoded PCM as its input.

**Stage 2 imposes TDAC.** The MDCT is lapped and critically sampled: no block is
individually invertible. Only overlap-add of consecutive blocks cancels the time-domain
alias. Consequently **per-block spectral generation produces signals that are not valid
audio** — the alias doesn't cancel. Any method operating in a spectral domain must respect
this, and most GAN-style spectral models simply don't.

**Stage 6 is where the honesty lives.** The rzero region is not attenuated, it is *zero*.
Nothing in the file constrains it:

| Bitrate | LAME lowpass | fraction of 0–22.05 kHz that is exactly zero |
|--------:|-------------:|---------------------------------------------:|
|  24 kbps |  8.0 kHz | 63.7% |
|  64 kbps | 13.0 kHz | 41.0% |
| 128 kbps | 16.0 kHz | 27.4% |
| 320 kbps | 20.5 kHz |  7.0% |

### 1.3 Pre-echo

Quantization noise is spread over the entire analysis window. Before a sharp attack, that
window is quiet — so the noise is *audible ahead of the transient*. Block switching mitigates
it. Critically: **the bitstream transmits `block_type`**, so the encoder tells you exactly
where it detected every transient. This is a better onset annotation than any detector you
could train, it is free, and it is currently discarded.

### 1.4 The scalefactor bands are already a critical-band split

Computing the Bark width of each MPEG-1 LIII long-block scalefactor band at 44.1 kHz:

```
sfb 0-17 : Bark width 1.51 → 1.06, monotonically drifting, roughly constant
sfb 18-20: Bark width 0.89, 0.65, 0.59
21 bands cover 418 of 576 lines, ending at exactly 16002 Hz
```

Two consequences. First, MP3's own band table *is* approximately a critical-band (Bark)
layout — the psychoacoustic work is already done and transmitted. Second, the table
terminates at **16002 Hz**, which is precisely the 128 kbps lowpass: the codec's band
structure itself marks the restoration boundary.

Compare Apollo, which imposes 80 **uniform linear** bands, giving 0–2 kHz (where hearing is
sharpest) only 8 bands while spending ~10× the resolution above 10 kHz — and whose top band
is 47 bins wide exactly where the action is. The codec hands you the correct band layout for
free and Apollo declines it.

*(Widths used: 4,4,4,4,4,4,6,6,8,8,10,12,16,20,24,28,34,42,50,54,76 — the commonly-cited
table. **Re-verify against ISO/IEC 11172-3 Table B.8 before implementation.**)*

### 1.5 Grid misalignment

```
Apollo analysis window : 882 samples = 20.00 ms, hop 10.00 ms
MP3 granule            : 576 samples = 13.06 ms (long-block MDCT window 26.12 ms)
ratio = 1.3061  → incommensurate
```

Quantization noise is *structured*: constant within a scalefactor band, constant within a
granule, changing discontinuously at granule boundaries. Viewed through an incommensurate
20 ms Hann STFT, that clean block structure is smeared into something the model must expend
capacity un-smearing. **On the codec's native grid the noise structure is diagonal and
explicit.** This is a concrete, testable architectural claim.

---

## 2. The feasible set, and what it buys

Let `x` be the master's MDCT coefficients on the codec's grid, and `y` the observed
bitstream. Decode `y` to obtain, for every coefficient, the integer `is`, its scalefactor,
and `global_gain`. Then

> **The set of masters consistent with `y` is, in MDCT space, an axis-aligned box** —
> a product of independent per-coefficient intervals — intersected with the linear TDAC
> constraint. It is **convex**.

Three regimes fall out, with genuinely different mathematics:

| Regime | Definition | Constraint | Nature of the problem |
|--------|-----------|------------|----------------------|
| **A** | coded, \|is\| ≥ 1 | tight relative interval (±0.07%–66%) | *deterministic refinement*; MSE-optimal = conditional mean in box |
| **B** | coded to zero, below cutoff | bounded box containing 0 | *small-amplitude structured inference*; pre-echo lives here |
| **C** | rzero, above cutoff | **none** | *irreducibly generative* |

**This decomposition is the central contribution of this document.** It yields:

1. **A hard guarantee.** Projection onto a box is elementwise clipping — exact, O(1) per
   coefficient. Any generative output can be projected onto the feasible set. A projected
   output is *provably* consistent with the observed file: it could have produced it.
   Regimes A and B become **incapable of hallucination by construction**.
2. **A monotonicity property.** Projection never moves you further from the truth on A and B
   (projection onto a convex set containing the truth is non-expansive). You cannot damage
   already-good content — which is exactly the failure mode of regenerate-everything models.
3. **An honesty boundary.** A and B are *restoration*. C is *synthesis*. They must be
   evaluated separately and disclosed separately. A single global metric hides this and every
   paper in this area reports single global metrics.

### 2.1 The contrarian thesis

At 128 kbps, Regime C is 16–22 kHz: low-energy in most music, and above the hearing
threshold of most adult listeners. **Regime C is close to perceptually irrelevant at the
bitrates that dominate real archives** — while Regimes A and B (quantization noise, pre-echo,
stereo collapse) are audible and are exactly where provable guarantees are available.

The entire "SR-GAN / bandwidth extension" literature, Apollo included, optimizes Regime C.
It is optimizing the perceptually marginal region, using unbounded generation, in the one
place where no guarantee is possible — and ignoring the tractable regions.

Below ~64 kbps this inverts: the cutoff falls to 11–13 kHz, Regime C becomes dominant and
plainly audible, and there honesty demands stating that the output is *synthesized*, not
recovered.

Corroborating this from outside: the CCF AATC 2025 restoration challenge retrospective
reports that top discriminative models "approximate an identity mapping while subtly
correcting quantization artifacts" — i.e. the winning behaviour is precisely
near-identity-plus-residual on Regime A, arrived at empirically rather than by construction.

---

## 3. Proposed method

Name: **BLIP** — Bitstream-Locked Inverse Posterior.

### 3.1 Bitstream front-end (not a decoder)

Parse MP3/AAC directly. Emit, per coefficient and per granule:
`is`, scalefactor, `global_gain`, `block_type` (transient locations), window switching,
stereo mode per SFB, intensity-stereo boundary, cutoff, bit-reservoir state.

Everything here is free, exact, currently discarded by every method in the field, and
strictly more informative than decoded PCM. Decoding to PCM is a lossy summary of the
*evidence*, not just of the audio.

### 3.2 Native-grid, SFB-banded backbone

Operate on the codec's MDCT lattice; band-split on the codec's own scalefactor bands
(§1.4) rather than an invented uniform grid. Retain Apollo's band-attention × time-conv
factorization — that part is sound — but on the correct lattice.

### 3.3 Regime-factorized heads

- **A:** deterministic denoiser predicting the conditional mean *within* the box. Not a
  generator. Trained with MSE + complex-spectral loss. Phase matters and is constrained here.
- **B:** conditional distribution over a small box; models pre-echo and the noise floor.
  Conditioned on `block_type` for transient placement.
- **C:** genuinely generative bandwidth extension, conditioned on the harmonic structure of
  A/B below the cutoff. Adversarial or diffusion. **Labelled as synthesis in all reporting.**

### 3.4 Constrained posterior sampling with per-step projection

A diffusion/flow sampler over the master, with **exact projection onto the feasible box after
every denoising step**. Because the set is convex and projection is closed-form, this is the
well-understood plug-and-play / DDRM machinery with a projection that is *exact* rather than
approximated — no likelihood surrogate needed on A and B.

### 3.5 Analysis-by-synthesis: the re-encoding loss

The strongest available training signal is not waveform distance — it is **bitstream
distance**. Propose a master, re-encode it, compare the resulting quantization indices to
the observed ones.

This is normally blocked by the encoder's non-differentiability, but the observed file
removes the hard part: **you already know every psychoacoustic decision the encoder made**
(scalefactors, block types, and stereo modes are transmitted). You do not need to model the
encoder's judgement — only the deterministic quantize step, which is differentiable via a
straight-through estimator on the rounding.

### 3.6 Explicit stereo restoration

Above the intensity-stereo boundary the sum is known and the difference is destroyed — a
constrained generative problem with its own geometry. Stereo-image collapse is grossly
audible and, as far as the literature reviewed here goes, entirely unaddressed. Apollo
carries `nch=2` and does no stereo-specific modelling whatever.

### 3.7 Two mechanisms that create information from nothing

**Non-local self-similarity across repetitions.** Music is enormously self-similar — choruses,
loops, repeated drum hits. Different repetitions land on *different MDCT frame boundaries*
and therefore receive **statistically independent quantization noise realizations**, and
under VBR are coded at different local rates. Aligning and fusing repetitions reduces
Regime-A/B noise the way burst photography reduces sensor noise. Searches surfaced this
principle in image super-resolution but nothing applying it to codec restoration.

**Multi-encoding fusion.** In real archives the same master exists as several different lossy
files — different bitrates, codecs, services. Each imposes a *different* quantization lattice,
so each gives a different box. **The intersection of N boxes is dramatically smaller than any
one of them.** This is compressed sensing with multiple measurement operators, it is directly
practical for archival work, and no prior art surfaced.

---

## 4. Novelty positioning (honest)

The closest prior art is **Villasana & Villemoes, "Audio Decoding by Inverse Problem
Solving"** (arXiv 2409.07858, Dolby). It genuinely anticipates the core framing: it treats
decoding as an inverse problem, uses quantization-bin membership as the measurement, and
solves by posterior sampling. **The inverse-problem framing with quantization constraints is
therefore not novel and must be cited as prior art.**

What that work does *not* do, and what remains open:

| | Prior art (2409.07858) | Proposed |
|---|---|---|
| Constraint handling | **Soft** Gaussian relaxation of bin membership | **Hard, exact** projection + consistency guarantee |
| Codec | Single proprietary codec | Real legacy MP3/AAC |
| Audio | **Mono, 22.05 kHz**, 8–48 kbps | Stereo, 44.1 kHz, music bitrates |
| Zeroed HF (Regime C) | Not addressed | Explicit, separately evaluated |
| Stereo / intensity | Not addressed | Explicit |
| Side info beyond bins | No | block_type, SFBs, stereo modes, cutoff |
| Cost | 1500 Langevin iterations/segment | Practicality is a design constraint |
| Self-similarity / multi-encoding | No | Both |
| Status | "Proof of concept" | — |

Defensible novelty: the hard-projection guarantee, the three-regime factorization with
per-regime evaluation, native-grid + SFB banding, the re-encoding loss via known encoder
decisions, stereo restoration, and the two information-creating mechanisms of §3.7.

---

## 5. Evaluation — where this field is weakest

Existing practice is close to unusable for this problem, and Apollo inherits the flaw: it
trains a magnitude-only objective and then selects checkpoints on **SI-SDR**, a strictly
phase-sensitive waveform metric it never optimizes.

- **SI-SDR** punishes perceptually-inaudible phase differences and says nothing useful about
  synthesized content. On Regime C it is close to meaningless.
- **PESQ** is speech-only. **ViSQOL** is validated on narrow conditions, not 44.1 kHz music.

Proposed battery:

1. **Bitstream-consistency rate (new, and *provable*).** Re-encode the restoration with the
   same encoder and settings; check that quantization indices match the original bitstream.
   Objective, binary per coefficient, no reference master required. A method scoring 100% is
   *provably not hallucinating over known content*. This should become a required baseline
   number — and note that a projected method scores 100% **by construction**.
2. **Regime-decomposed metrics.** Log-spectral distance and complex-spectral distance
   reported *separately* for A, B, C. Never a single global number.
3. **Phase coherence on Regime A**, where phase is constrained and therefore meaningful.
4. **Noise-floor and dither-character match** vs the master (§1.1).
5. **Stereo-image metrics**: inter-channel coherence and width vs the master, above and below
   the intensity boundary.
6. **Proper ABX** with trained listeners plus MUSHRA with hidden reference and anchor —
   including an explicit **"pleasant sizzle" cheat check**, since models reliably win listening
   tests by adding flattering HF energy that is not in the master.
7. **Downstream-task probes**: do a source separator, a fingerprinter, and a transcriber
   behave identically on restored audio and on the master?

---

## 6. Staged plan with kill criteria

| Stage | Work | Kill criterion |
|---|---|---|
| 0 | Bitstream parser; dump `is`/scalefactors/block_type; verify boxes contain the true master on real encode/decode pairs | If true coefficients fall outside computed boxes, the quantizer model is wrong — stop and fix |
| 1 | Projection-only baseline: decode, project, no learning. Measure per-regime gain | If projection alone gives no measurable Regime-A/B gain, the premise is wrong |
| 2 | Regime-A denoiser on native grid + SFB bands vs Apollo, per-regime | If native grid doesn't beat 20 ms STFT, drop §1.5 claim |
| 3 | Re-encoding loss via STE | If STE is unstable, fall back to interval-hinge loss |
| 4 | Regime C generator + stereo restoration | — |
| 5 | Self-similarity fusion; multi-encoding fusion | Highest risk, highest novelty — run last |

Stage 1 is the decisive cheap experiment: **it needs no training at all** and it either
validates or destroys the central premise in a day.

---

## 7. Honest limits

- Bit-exact recovery is impossible (§0.1). Nothing here changes that.
- Regime C is synthesis. At ≤64 kbps most of the audible restoration *is* Regime C, and
  "studio quality" there means "convincing fabrication." This must be disclosed, not marketed.
- The consistency guarantee covers A and B only. It says the output *could* have produced the
  file; it does not say it is the master.
- Numbers in §1 are computed from the coding standard's arithmetic, not measured on a corpus.
  The scalefactor table needs verification against ISO/IEC 11172-3 (§1.4).
- Encoder diversity (LAME vs Fraunhofer vs iTunes) is a real generalization risk; the box
  geometry is encoder-specific.
- Two of the most novel mechanisms (§3.7) are unvalidated and carry the most risk.

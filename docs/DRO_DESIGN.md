# Distributional robustness for audio-agnostic restoration

The instinct — optimize worst-case across distributions rather than average-case — is
correct and is the right frame for "works on any audio file." **Wasserstein DRO is the
wrong instantiation for the content distribution, and the right one for the degradation
distribution.** Measured below, then a concrete formulation.

---

## Measurement: the Wasserstein ball does not scale

WDRO solves `min_θ sup_{Q : W(Q,P̂) ≤ ρ} E_Q[loss]`. It helps only if the ρ needed to
cover shifts we care about is *smaller* than the ρ at which the ball admits degenerate
distributions.

Sliced-Wasserstein distances between per-frame spectral-shape distributions of real
recordings (loudness removed; sitar excluded — proven lossy provenance):

| | max cross-instrument | min to white noise | min to near-silence |
|---|---:|---:|---:|
| sliced-W | **3.56** | **2.33** | 2.34 |

**White noise is closer to flute (2.38) than Carnatic vocal is (3.40).**

So any ball large enough to cover flute → Carnatic also contains white noise and
near-silence. The inner supremum would be attained on garbage, and the robust solution
would be uselessly conservative.

> **Wasserstein DRO over the content distribution is vacuous for this problem.**

*(Sliced-W is estimated by random projection, so the matrix is mildly asymmetric —
Monte-Carlo noise of ~0.01–0.05, far below the gaps that matter.)*

### Why — the structural reason

Wasserstein balls are geometric: they contain everything within a transport budget,
including pathological directions. Audio content shift is **semantic and
high-dimensional** — "sitar instead of guitar" is not a small perturbation in any
waveform or spectral metric. The geometry does not know which directions are meaningful.

f-divergence balls (KL, χ²) are worse still: they cannot change support at all, so they
can only reweight instruments already in the training set — exactly what fails.

---

## The correct decomposition

The training distribution factors into two parts with completely different geometry:

| | dimensionality | shift size | labels available? | right tool |
|---|---|---|---|---|
| **P(content)** — instrument, genre, speech, environmental | very high | **huge** (measured above) | **yes** — FSD50K categories, corpus of origin, Western/Indian | **Group DRO** |
| **P(degradation)** — bitrate, encoder, VBR/CBR, joint-stereo mode | **low** (a handful of parameters) | **small** — LAME vs Fraunhofer vs iTunes vs Opus are genuinely nearby operators | partially | **Wasserstein DRO** ✔ |

So:

> **Group DRO over content. Wasserstein DRO over degradation.**

Wasserstein is well-scaled exactly where the space is low-dimensional and parametric — the
encoder settings — and vacuous where the space is high-dimensional and semantic.

### Content: Group DRO

`min_θ max_g E_{P_g}[loss]` (Sagawa et al.). Nearly free: track an EMA of per-group loss
and reweight the batch online. Groups we already have for free: corpus (FSD50K / MUSDB /
VCTK / Saraga), FSD50K's own label taxonomy, Western vs non-Western, and bitrate bin.

This targets **subpopulation shift**, which is precisely what "audio agnostic" means, and
it is the failure mode our real-audio test exposed.

### Degradation: Wasserstein DRO

Here the ball is over a low-dimensional parameter vector (bitrate, encoder identity,
VBR/CBR, joint-stereo mode, lowpass). Shifts are genuinely small and the ball does not
contain nonsense. In practice this is the standard WDRO ≈ **gradient-penalty /
adversarial-perturbation** equivalence for Lipschitz losses: perturb the degradation
parameters adversarially within a small budget.

This is also where it matters practically — a model trained only on LAME should not
collapse on Fraunhofer, iTunes AAC, or Opus.

---

## ⚠ The failure mode that must be designed around

**Worst-group DRO chases irreducible error.** This program *proved* some groups have high
error floors for information-theoretic, not modelling, reasons:

- **EXP 1**: content above the codec cutoff is unrecoverable by any analytic method.
- **Percussion / noise-like content** gets nothing from harmonic structure — no pitch
  diversity, nothing crosses the cutoff.
- Achievable error differs ~2× across domains in our real-audio test: best achievable was
  **0.39 on flute** vs **0.74 on Carnatic vocal**.

Naive `max_g` will pour all capacity into the group with the highest *raw* loss — which
may simply be the one with the highest *floor* — and wreck the rest.

**Fix: equalize excess risk, not raw risk.**

```
min_θ  max_g  [ E_{P_g}[loss(θ)] − floor_g ]        (or the ratio form)
```

where `floor_g` is a measured per-group reference: the rolloff-extrapolation baseline
(0.39–0.57 on regular spectra, ≥1.0 on irregular), or a strong baseline's loss on that
group. This makes DRO equalize *how much better than baseline* we are, which is the
quantity we actually control.

Without this correction, DRO is actively harmful here — and this program has the
measurements to prove the floors differ.

---

## Practical recipe (fits the 12 h / 2×T4 budget)

1. **Group DRO over content groups**, with per-group floors from the rolloff baseline.
   Cost: one EMA per group, one reweighting — effectively free.
2. **Adversarial/Wasserstein perturbation over degradation parameters** — sample bitrate
   and encoder adversarially (bias toward whichever currently has the highest excess
   loss) rather than uniformly. Also free, and it composes with the existing finding that
   uniform bitrate sampling wastes ⅛ of the budget at 320 kbps.
3. **CVaR fallback** if group labels are unavailable for part of the corpus: optimize the
   worst α-fraction of per-sample losses (α ≈ 0.2). Label-free, cheap, and a reasonable
   middle ground between ERM and full group DRO.
4. **Report per-group**, never a single global number.

## Honest status

The measurement above (WDRO vacuous over content) is solid and reproducible. The
**Group-DRO-with-floors design is untested** — no training run has been done. Given this
program's record (six mechanisms attempted, six failed, most first-pass results
artifacts), treat it as a well-motivated design, not a validated one. The cheapest
decisive check is an ablation: ERM vs group DRO vs floor-corrected group DRO, same
budget, reported per group.

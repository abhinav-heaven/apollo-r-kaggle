# Distributional robustness methods — fit assessment for codec restoration

Ranked by fit to *this* problem, with rejections recorded. Two facts constrain everything:

- **Measured:** a Wasserstein ball over content is vacuous — white noise is closer to
  flute (2.33) than Carnatic vocal is (3.56).
- **Proved:** some groups have irreducible error floors (above-cutoff content, EXP 1;
  percussion, which has no pitch diversity). Any worst-case method must be protected
  against chasing them.

---

## Adopt

### 1. Floor-corrected Group DRO — *implemented in `train_dro.py`*
`min_θ max_g [L_g − floor_g]`. Targets subpopulation shift, which is exactly what
"audio agnostic" means. Labels are free (corpus, FSD50K taxonomy, Western/non-Western,
bitrate bin). Cost: one EMA per group. **The floor correction is not optional** — without
it the highest-*floor* group wins the max and starves everything improvable.

### 2. V-REx (variance of risks across environments) — **strongest cheap addition**
Penalty `λ·Var_g(L_g)` added to mean risk. Group DRO optimizes the *worst* group; V-REx
pushes all groups toward *equal* risk, which is closer to "works on any audio file" than
"works on the hardest audio." Simpler and empirically more stable than IRM, and it costs
one variance over the per-group EMAs you already track. **Compose with (1)** — they are
complementary, not alternatives.

### 3. DORO (outlier-robust DRO) — **directly addresses our known pathology**
CVaR-style DRO with the worst ε-fraction trimmed before taking the worst-case. Standard
DRO amplifies noisy or mislabelled samples; ours are guaranteed to exist — provenance-
filtered corpora still leak mild transcodes (measured: transcoded flute is
indistinguishable from the original at −12.7 vs −12.5 dB). Trimming makes worst-group
optimization survive that leakage. **Use if group DRO proves unstable.**

### 4. Domain randomization / adversarial sampling over *degradation*
This is Wasserstein DRO placed where it is correctly scaled: the low-dimensional
degradation parameters (bitrate, encoder identity, VBR/CBR, joint-stereo mode, resample,
dither). Shifts between LAME / Fraunhofer / iTunes AAC / Opus are genuinely small.
Partially implemented — `CodecDataset.set_bitrate_logits` lets the sampler chase whichever
bitrate currently has the highest excess loss. **Extend to multiple encoders**; this is
what stops a LAME-only model collapsing on AAC.

### 5. SWAD / weight averaging over the trajectory
Averaging weights in a flat region generalizes better out-of-distribution, at essentially
zero cost. Nearly free insurance on a short fine-tune where the final step may land badly.

---

## Consider

### 6. CVaR / superquantile DRO
Optimize the worst α-fraction of per-sample losses (α ≈ 0.2). **Label-free** — the right
fallback for corpus parts without usable group labels (much of FSD50K if the taxonomy is
too fine-grained to bin sensibly). Weaker than group DRO when labels exist.

### 7. Tilted ERM (TERM)
Exponential tilting `(1/t)·log E[e^{t·L}]` smoothly interpolates ERM (t→0) and max (t→∞).
One scalar knob, no groups, very cheap. Useful as a *tuning dial* if group DRO is too
aggressive and ERM too permissive.

### 8. Mixup / manifold mixup
Cheap OOD regularizer. Caveat specific to us: mixing *waveforms* is physically meaningful
(audio superposes), but the codec is **nonlinear** — `codec(a+b) ≠ codec(a)+codec(b)`. So
mix the clean targets and re-encode the mixture; never mix the degraded inputs.

### 9. Test-time adaptation via re-encode consistency — **most interesting, most caveated**
At inference you have the actual file and a free self-supervised signal: re-encoding the
restored output at the observed bitrate should reproduce the observed bitstream. That
adapts the model per-file with no ground truth, and reuses the consistency machinery from
`RESEARCH_DESIGN.md` (worth only ~1% as a projection, but potentially more as an
adaptation objective, since it moves the whole model rather than clipping the output).

**Degeneracy warning:** consistency alone is trivially satisfied by the identity map —
`model(y) = y` re-encodes perfectly and restores nothing. It is only usable as a
*regularizer* alongside the learned prior, with an anchor to the pretrained weights and
early stopping. Do not deploy it as a standalone objective.

---

## Reject (with reasons, for the record)

| Method | Why rejected |
|---|---|
| **Wasserstein DRO over content** | **Measured vacuous** — the ball covering flute→Carnatic (3.56) also contains white noise (2.33) and near-silence (2.34) |
| **f-divergence DRO (KL, χ²)** | Cannot change support — can only reweight instruments already in the training set. Structurally unable to address unseen timbres, which is the entire problem |
| **DANN / domain-adversarial features** | **Actively counterproductive.** It removes domain information from the representation, but we deliberately *condition on bitrate* via FiLM. DANN would erase the signal we are trying to inject |
| **IRM** | Notoriously hard to optimize, weak empirically vs V-REx, and full invariance is not what we want — the model *should* behave differently at 24 vs 320 kbps |
| **MAML / meta-learning** | Second-order cost is unjustifiable in a 12 h budget |
| **Fishr / gradient matching** | Plausible but expensive per-step; defer unless V-REx underperforms |

---

## Recommended stack for the 12 h budget

```
loss = GroupDRO_floor_corrected( per_sample_mrstft )   # primary
     + λ · Var_g( L_g )                                # V-REx, λ ≈ 1e-2
sampling: adversarial over bitrate AND encoder          # Wasserstein where it scales
weights:  SWAD tail-averaging over the final ~30%      # free insurance
fallback: DORO trimming if group DRO destabilizes
```

Everything here is O(groups) per step — negligible against the 786 GFLOP forward.

## Ablation that makes the result interpretable

Equal budget, reported per group:

1. ERM (`--no-dro`)
2. Group DRO, raw max (`--no-floors`)
3. Group DRO + floors
4. (3) + V-REx

**(2) vs (3) is the scientifically interesting comparison** — it directly tests whether
irreducible-error groups hijack worst-case optimization, which this program has the
measured floors to predict but has never verified.

## Status

`train_dro.py` and `eval_per_group.py` are **written and compile-checked but never
executed** — no GPU or torch in the authoring environment. Run `--smoke` first; it
asserts init-equivalence with pretrained Apollo before any training. V-REx, DORO, SWAD and
multi-encoder sampling are **specified here but not yet implemented**.

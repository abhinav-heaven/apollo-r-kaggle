# Apollo-R — checkpoint-compatible codec-restoration fine-tuning for 2×T4

Fine-tunes [Apollo](https://github.com/JusperLee/Apollo) (Kai Li, CC BY-SA 4.0) with
floor-corrected Group DRO, V-REx, DORO, SWAD and multi-encoder sampling, sized for a
**Kaggle 2×T4 / 12 h** session.

**Core safety property:** every addition is zero-initialized, so at step 0
`ApolloR(x) == Apollo(x)` **exactly** (verified: max abs diff `0.000e+00`). Fine-tuning
can only earn improvements; it cannot start by destroying a good checkpoint.
`train_dro.py` asserts this and refuses to run if it breaks.

## Kaggle quick start

```bash
!git clone https://github.com/abhinav-heaven/apollo-r-kaggle.git && cd apollo-r-kaggle && ./setup.sh
```

Then (2 GPUs):

```bash
!cd apollo-r-kaggle/Apollo && torchrun --nproc_per_node=2 train_dro.py \
    --groups ../configs/groups.example.json --pretrained pytorch_model.bin \
    --batch 4 --workers 8 --max-hours 10.5 --out /kaggle/working/exp
```

Always run `--smoke` first — it does a few steps per group and asserts init-equivalence.

## What it adds, and why

| | Rationale (each traces to a measurement in `docs/`) |
|---|---|
| **Bitrate randomization fix** | Upstream `codec_simu` mutated the caller's dict, freezing the bitrate per worker. Verified: 8 successive calls returned `[96]×8`. The released model saw ~8 fixed rates, not the advertised 24–128 kbps range. |
| **Log-frequency harmonic branch** | Harmonics sit at `h·f0` (multiplicative); Apollo's 80 bands are uniform in *linear* frequency, so the band offset between harmonics is pitch-dependent — the net must relearn the template per pitch. Log-frequency convolution is pitch-equivariant. |
| **FiLM codec/bitrate conditioning** | Measured cutoffs span 4.4 kHz (24 k) to 20.2 kHz (320 k). **Opus never lowpasses** (~20.4 kHz at every rate) — a different degradation, so codec identity is conditioned on, not just cutoff. |
| **Full-resolution refine branch** | Apollo's top band is 47 bins vs 5 elsewhere — ~10× coarser exactly where restoration happens. |
| **Floor-corrected Group DRO** | Naive `max_g` chases *irreducible* error. Measured floors differ ~2× across domains (0.39 flute vs 0.74 Carnatic vocal), so we equalize **excess** risk. |
| **V-REx / DORO / SWAD** | Equal-risk penalty; outlier trimming (the provenance filter provably leaks mild transcodes); trajectory weight averaging. |
| **Multi-encoder sampling** | Wasserstein-DRO over *degradation*, where the ball is correctly scaled. Over **content** it is vacuous — measured: white noise is closer to flute (2.33) than Carnatic vocal is (3.56). |

## Honest status

- Smoke-tested end-to-end on CPU (torch 2.2.2); **not yet run on GPU or at scale.**
- No claim that this beats Apollo on Western music at bitrates it already handles. The
  plausible wins are unseen bitrates, non-Western instruments, and non-music audio.
- Benchmark against the **rolloff baseline** (`eval_per_group.py`), not against
  unprocessed audio. It scores 0.39–0.57 on regular spectra and is stronger than the
  literature implies.
- `docs/` records the failures too — six mechanisms were tried and refuted, and the
  negative results are why the surviving design looks the way it does.

## License

CC BY-SA 4.0, inherited from upstream Apollo. See `LICENSE` and `NOTICE`.

# Run commands

Floors below are **measured**, in RGR units (identity == 1.0), from
`eval_per_group.py --mode calibrate` on real audio. **Recalibrate on your own corpus
before the long run** — floors are corpus-specific and Group DRO depends on them.

Measured on the local real-audio corpus (mp3, above-cutoff RGR of the rolloff baseline):

| group | rolloff RGR above cutoff |
|---|---|
| western_flute | 0.950 |
| western_cello | 0.891 |
| **indian_carnatic** | **1.007** — baseline achieves nothing |

## 0. Stage data FIRST, on a CPU-only notebook (zero GPU hours)

FSD50K and Sanidha are not on Kaggle. Stage them in a notebook with
**Accelerator = None** — the GPU quota is only consumed when an accelerator is
enabled. See `docs/DATA_STAGING.md`. Short version:

```bash
!cd /kaggle/working/Apollo && python prepare_kaggle_dataset.py \
    --repo Fhrozen/FSD50k --scratch /kaggle/tmp/dl \
    --out /kaggle/working/fsd50k_flac --budget-gb 18
```

Download goes to 60 GiB scratch; output is transcoded to **FLAC (lossless, ~50% of
WAV, verified bit-exact)** so ~34 GB fits the 20 GiB persist limit. Save Version, then
attach the output to the GPU notebook.

## 0a. GENERATE THE GROUPS CONFIG — do not hand-write paths

Kaggle mounts each dataset at `/kaggle/input/<slug>/`, but the layout INSIDE is
chosen by the uploader and the file tree is JS-rendered, so it cannot be read off
the dataset page. Guessed sub-paths are how you get `0 usable / 0 found`.

Do not guess — scan what is actually mounted:

```bash
!cd /kaggle/working/Apollo && python discover_groups.py --root /kaggle/input \
    --out /kaggle/working/groups.json
```

It walks each mounted dataset, reports file counts / sample rates / durations,
**refuses any all-lossy dataset** (Saraga, FMA — they cannot be ground truth), warns
about sub-44.1 kHz sources, and emits a ready groups JSON whose roots are the
DATASET ROOTS. `scan_files` recurses, so the internal layout never matters.

Use `/kaggle/working/groups.json` as `--groups` everywhere below.

### Datasets to attach

| corpus | Kaggle | note |
|---|---|---|
| MUSDB18-HQ | `quanglvitlm/musdb18-hq` | 44.1 kHz WAV stems |
| VCTK | `kynthesis/vctk-corpus` | 48 kHz — resampled automatically |
| FSD50K | **not on Kaggle** | upload from Zenodo/HF as a private dataset |
| Sanidha (Carnatic) | **not on Kaggle** | request access, then upload privately |
| ~~Saraga~~ | — | **mp3, unusable as ground truth** |

## 0. Kaggle setup

Accelerator → **GPU T4 ×2**. Attach datasets:
`musdb18-hq` (quanglvitlm), `vctk-corpus` (kynthesis). FSD50K is not on Kaggle —
upload from Zenodo/HuggingFace as a **private** dataset. **Do not use Saraga
(mp3) or FMA (mp3) as ground truth.**

```python
!git clone -q https://github.com/abhinav-heaven/apollo-r-kaggle.git /kaggle/working/apollo-r
!cd /kaggle/working/apollo-r && ./setup.sh /kaggle/working/Apollo
!pip -q install huggingface_hub
```

Fetch the pretrained checkpoint:

```python
from huggingface_hub import hf_hub_download
hf_hub_download("JusperLee/Apollo", "pytorch_model.bin",
                local_dir="/kaggle/working/Apollo")
```

## 1. Smoke test — ALWAYS FIRST

```bash
cd /kaggle/working/Apollo && python train_dro.py \
  --groups /kaggle/working/groups.json \
  --pretrained pytorch_model.bin --smoke --smoke-steps 8 --batch 2 --workers 2
```

Must print `max |ApolloR - Apollo| at init = 0.000e+00`. If not, **stop** — the
checkpoint is being corrupted from step 0.

## 2. Calibrate floors on YOUR corpus

```bash
cd /kaggle/working/Apollo && python eval_per_group.py \
  --groups /kaggle/working/groups.json \
  --mode calibrate --codec mp3 --bitrates 24,64,128 \
  --max-files 40 --max-segs 4 --out /kaggle/working/floors.json
```

Paste the printed floors into your groups JSON.

## 3. Baseline the pretrained model BEFORE training

```bash
cd /kaggle/working/Apollo && python eval_per_group.py \
  --groups /kaggle/working/groups.json \
  --ckpt pytorch_model.bin --codec mp3 --bitrates 24,64,128 \
  --out /kaggle/working/baseline.json
```

`model RGR` **< 1.0 = restoring, > 1.0 = making it worse.** This is the number the
fine-tune has to beat; without it you cannot claim an improvement.

## 4. Train — session 1 (both GPUs)

```bash
cd /kaggle/working/Apollo && torchrun --nproc_per_node=2 train_dro.py \
  --groups /kaggle/working/groups.json \
  --pretrained pytorch_model.bin \
  --loss rgr --batch 4 --workers 8 --seg 3.0 \
  --codecs mp3,aac,opus --vrex 1e-2 --doro-eps 0.05 --swad-start 0.6 \
  --max-hours 10.5 --ckpt-min 25 --out /kaggle/working/exp
```

Save `/kaggle/working/exp` as a Kaggle Dataset before the session ends, or it is lost.

## 5. Train — session 2 (resume)

```bash
cd /kaggle/working/Apollo && torchrun --nproc_per_node=2 train_dro.py \
  --groups /kaggle/working/groups.json \
  --resume /kaggle/working/exp/last.ckpt \
  --loss rgr --batch 4 --workers 8 --max-hours 10.5 --out /kaggle/working/exp
```

## 6. Evaluate — SWAD weights are the ones to score

```bash
cd /kaggle/working/Apollo && for C in mp3 aac opus; do python eval_per_group.py \
  --groups /kaggle/working/groups.json \
  --ckpt /kaggle/working/exp/swad.ckpt --codec $C \
  --bitrates 24,32,64,96,128,192 --out /kaggle/working/eval_$C.json; done
```

Read: **model RGR below/above cutoff per group per bitrate**, and the signed
**bias** column — a large positive bias means the model is adding flattering treble
that is not in the master, which is cheating, not restoring.

## 7. The ablation that makes the result meaningful

Equal budget, `--max-hours 2` each, everything else fixed:

```bash
python train_dro.py ... --no-dro                 # ERM
python train_dro.py ... --no-floors              # raw max_g
python train_dro.py ... --vrex 0 --doro-eps 0    # floor-corrected DRO only
python train_dro.py ...                          # full stack
```

**`--no-floors` vs floor-corrected is the scientifically interesting one** — it directly
tests whether irreducible-error groups hijack worst-case optimization.

## Sanity thresholds

| check | pass |
|---|---|
| init equivalence | `0.000e+00` exactly |
| model RGR (any group) | **< 1.0**, else worse than doing nothing |
| high-band energy bias | within a few dB of 0 |
| null test on lossless input | near 0 |

# Staging HuggingFace corpora into Kaggle — at zero GPU cost

## The one thing to get right

**Kaggle consumes the ~30 h/week GPU quota only when the ACCELERATOR IS ENABLED.**
Do all data staging in a notebook with **Accelerator = None**. You still get a 12 h
session; the GPU quota is untouched. Only the training run should ever hold a GPU.

## The storage constraint

| location | size | persisted? |
|---|---|---|
| `/kaggle/working` | **20 GiB** | yes — becomes the dataset |
| `/kaggle/tmp` (or any other dir) | **~60 GiB** | no — scratch |
| private dataset quota | **200 GB** | — |

FSD50K dev is ~108 h of 44.1 kHz/16-bit mono WAV ≈ **34 GB** — over the persist limit.

## The fix: download to scratch, transcode to FLAC into working

FLAC is **lossless**, so this costs nothing in quality. Measured here on real audio:

| file | WAV → FLAC | bit-exact? |
|---|---|---|
| flute | 2.8 → 1.1 MB (40%) | **YES** |
| cello | 7.1 → 1.4 MB (19%) | **YES** |
| carnatic | 31.9 → 18.6 MB (58%) | **YES** |
| **mean** | **50%** | verified by md5 of decoded PCM |

So ~34 GB WAV → **~17 GB FLAC**, which fits in 20 GiB.

This is the *only* compression that is safe in this project. Every lossy format would
put a codec hole in the ground truth — the exact failure the provenance filter exists
to catch. The pipeline reads FLAC natively (`scan_files`, `provenance_ok`, torchaudio).

## Recipe — CPU-only notebook, Internet ON

```python
!git clone -q https://github.com/abhinav-heaven/apollo-r-kaggle.git /kaggle/working/apollo-r
!cd /kaggle/working/apollo-r && ./setup.sh /kaggle/working/Apollo
!pip -q install huggingface_hub
```

```python
!cd /kaggle/working/Apollo && python prepare_kaggle_dataset.py \
    --repo Fhrozen/FSD50k --repo-type dataset \
    --scratch /kaggle/tmp/dl --out /kaggle/working/fsd50k_flac --budget-gb 18
```

Then **Save Version**. The output is attachable to the training notebook as a dataset.

Notes:
- `--allow 'dev_audio/*'` restricts what is pulled, if the repo is larger than needed.
- `--max-files N` caps the count. **Data volume is not the bottleneck**: a 10 h run on
  2×T4 sees ~100–200 h of audio, and FSD50K dev is 108 h — a subset loses little.
- Archives in the repo are auto-extracted and **deleted immediately** to reclaim scratch.
- Lossy files in the repo are counted and **ignored**; if a repo is entirely lossy the
  script exits rather than let it become ground truth.

## Sanidha (Carnatic) — same route, manual first step

Not on HuggingFace or Kaggle. Request access (email + GT guest account + VPN), download
locally, then either upload as a **private** Kaggle dataset, or push to a private HF repo
and stage it with the same script.

## Cost summary

| step | accelerator | GPU hours |
|---|---|---|
| stage data (this doc) | **None** | **0** |
| discover groups | None | 0 |
| calibrate floors | None (or GPU for speed) | ~0 |
| smoke test | GPU T4 ×2 | ~minutes |
| training | GPU T4 ×2 | the real spend |

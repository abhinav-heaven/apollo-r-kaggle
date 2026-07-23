# Kaggle setup — running Apollo-R, and securing your data

## 1. Pull the code (no copy-paste, no token)

The repo is **public**, so cloning needs no credential at all. In a Kaggle cell:

```python
!git clone -q https://github.com/abhinav-heaven/apollo-r-kaggle.git /kaggle/working/apollo-r
!cd /kaggle/working/apollo-r && ./setup.sh /kaggle/working/Apollo
```

To pick up later changes, re-run the clone cell (or `git -C /kaggle/working/apollo-r pull`).

**This is itself the safer design:** a public repo needs no Personal Access Token in the
notebook, so there is no credential to leak in shared notebook output.

## 2. Enable both GPUs

Settings → Accelerator → **GPU T4 ×2**. Then launch with `torchrun`:

```python
!cd /kaggle/working/Apollo && torchrun --nproc_per_node=2 train_dro.py \
    --groups /kaggle/working/apollo-r/configs/groups.example.json \
    --pretrained pytorch_model.bin --batch 4 --workers 8 \
    --max-hours 10.5 --out /kaggle/working/exp
```

`nvidia-smi` should show two busy GPUs. A single-process run uses only one.

## 3. Securing your data — do this yourself, never in code

**Never paste an API token into a notebook or commit one.** Anything printed by a public
notebook is public, and Kaggle output is retained.

**Private datasets (recommended for your corpus)**
1. Create the dataset on Kaggle and set visibility **Private**.
2. Attach it to the notebook via *Add Data*. It mounts read-only at
   `/kaggle/input/<slug>/` with **no token required** — attachment is the auth.
3. Keep the notebook itself **Private** until you intend to share it.

**If you genuinely need a secret** (e.g. pulling from a private mirror):
1. Notebook → *Add-ons* → **Secrets** → add the value there. It is stored by Kaggle, not
   in the notebook source.
2. Read it at runtime:
   ```python
   from kaggle_secrets import UserSecretsClient
   tok = UserSecretsClient().get_secret("MY_TOKEN")   # never print this
   ```
3. Rotate any token that has ever been pasted into a cell, committed, or shown in output.

`.gitignore` here already excludes `kaggle.json`, `*.env` and `.secrets` so a stray
credential file cannot be committed by accident.

## 4. Sessions are capped at 12 h

`--max-hours 10.5` stops with margin and writes `last.ckpt` + `swad.ckpt`. Resume in the
next session:

```python
!cd /kaggle/working/Apollo && torchrun --nproc_per_node=2 train_dro.py \
    --groups ... --resume /kaggle/working/exp/last.ckpt --max-hours 10.5
```

Save `/kaggle/working/exp` as a Kaggle Dataset (or *Save Version*) before the session
ends, or it is lost.

## 5. Always smoke-test first

```python
!cd /kaggle/working/Apollo && python train_dro.py --groups ... --smoke --workers 2
```

It asserts `max |ApolloR - Apollo| == 0` at init and refuses to proceed otherwise —
that assert has already caught one real bug that would have silently corrupted training.

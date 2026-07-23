"""
Stage a HuggingFace audio corpus into a Kaggle Dataset — using ZERO GPU hours.

WHY THIS COSTS NO GPU QUOTA
  Kaggle consumes the ~30 h/week GPU quota only when the ACCELERATOR IS ENABLED.
  Run this in a notebook with Accelerator = None: you still get a 12 h session,
  and the quota is untouched. Only the training run should ever hold a GPU.

THE STORAGE CONSTRAINT AND HOW WE BEAT IT
  /kaggle/working  20 GiB, PERSISTED  -> becomes the dataset
  /kaggle/tmp      ~60 GiB, scratch, NOT persisted
  private quota    200 GB total

  FSD50K dev is ~108 h of 44.1 kHz/16-bit mono WAV, roughly 34 GB — over the
  persist limit. So: download to SCRATCH, transcode to FLAC into WORKING.

  FLAC is LOSSLESS, so this is free. Measured on real audio here: 40%, 19%, 58%
  of original size, mean 50%, and byte-identical PCM on decode (verified by md5
  of the decoded stream). ~34 GB WAV -> ~17 GB FLAC, which fits.

  This is the one compression that is safe in this project: every other format
  would put a codec hole in the ground truth, which is exactly what we spend the
  provenance filter guarding against.

USAGE (CPU-only Kaggle notebook, Internet = ON)
    !python prepare_kaggle_dataset.py --repo Fhrozen/FSD50k --repo-type dataset \
        --scratch /kaggle/tmp/dl --out /kaggle/working/fsd50k_flac --budget-gb 18
Then: Save Version -> the output is attachable as a dataset to the training notebook.
"""
import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys

AUDIO_IN = (".wav", ".aiff", ".aif", ".flac")
LOSSY = (".mp3", ".m4a", ".ogg", ".opus", ".aac")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def to_flac(args):
    src, dst = args
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
            "-c:a", "flac", "-compression_level", "8", dst])
    if r.returncode != 0 or not os.path.exists(dst):
        return src, 0
    return src, os.path.getsize(dst)


def walk(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(exts):
                out.append(os.path.join(dp, f))
    return sorted(out)


def human(n):
    return f"{n/1e9:.2f} GB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HuggingFace repo id")
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--allow", default="", help="comma globs, e.g. 'dev_audio/*'")
    p.add_argument("--scratch", default="/kaggle/tmp/dl")
    p.add_argument("--out", default="/kaggle/working/corpus_flac")
    p.add_argument("--budget-gb", type=float, default=18.0,
                   help="stop before /kaggle/working's 20 GiB persist limit")
    p.add_argument("--max-files", type=int, default=0, help="0 = all")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--skip-download", action="store_true")
    a = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found")

    # ---- 1. download to SCRATCH (not persisted, ~60 GiB) ----
    if not a.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("pip install huggingface_hub")
        os.makedirs(a.scratch, exist_ok=True)
        print(f"[1/4] downloading {a.repo} -> {a.scratch} (scratch, not persisted)")
        snapshot_download(repo_id=a.repo, repo_type=a.repo_type,
                          local_dir=a.scratch,
                          allow_patterns=[x for x in a.allow.split(",") if x] or None,
                          max_workers=8)

    # auto-extract any archives the repo ships
    arcs = walk(a.scratch, (".zip", ".tar", ".tar.gz", ".tgz"))
    for i, z in enumerate(arcs):
        print(f"[2/4] extracting {os.path.basename(z)} ({i+1}/{len(arcs)})")
        try:
            shutil.unpack_archive(z, os.path.dirname(z))
            os.remove(z)                      # reclaim scratch immediately
        except Exception as e:
            print(f"      skip: {e}")

    # ---- 2. census ----
    src = walk(a.scratch, AUDIO_IN)
    lossy = walk(a.scratch, LOSSY)
    print(f"[3/4] found {len(src)} lossless-format, {len(lossy)} lossy")
    if lossy and not src:
        sys.exit("ALL audio in this repo is LOSSY. It cannot be ground truth — "
                 "training on it teaches the model to restore toward damaged targets.")
    if lossy:
        print(f"      ignoring {len(lossy)} lossy file(s); they are not ground truth")
    if not src:
        sys.exit(f"no lossless audio under {a.scratch}")
    if a.max_files:
        src = src[:a.max_files]

    # ---- 3. transcode to FLAC into WORKING, under budget ----
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for s in src:
        rel = os.path.relpath(s, a.scratch)
        jobs.append((s, os.path.join(a.out, os.path.splitext(rel)[0] + ".flac")))

    budget = a.budget_gb * 1e9
    written = 0
    done = 0
    print(f"[4/4] transcoding to FLAC ({a.workers} workers), budget {a.budget_gb} GB")
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(to_flac, j): j for j in jobs}
        for f in cf.as_completed(futs):
            _, sz = f.result()
            written += sz
            done += 1
            if done % 500 == 0:
                print(f"      {done}/{len(jobs)}  {human(written)}", flush=True)
            if written >= budget:
                print(f"      budget reached at {done} files — stopping")
                for g in futs:
                    g.cancel()
                break

    orig = sum(os.path.getsize(s) for s, _ in jobs[:done] if os.path.exists(s))
    print(f"\ndone: {done} files, {human(written)} FLAC "
          f"(from {human(orig)} source, {100*written/max(orig,1):.0f}%)")
    print(f"output: {a.out}")
    print("\nNext: Save Version on this CPU notebook, then attach its output to the\n"
          "training notebook as a dataset. No GPU quota was used.")


if __name__ == "__main__":
    main()

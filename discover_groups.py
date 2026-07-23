"""
Generate a valid --groups JSON from what is ACTUALLY mounted.

Guessing dataset sub-directories does not work: Kaggle mounts each dataset at
/kaggle/input/<slug>/ but the layout inside is chosen by whoever uploaded it, and
the file tree is JS-rendered so it cannot be read off the dataset page. Guessed
paths produce "0 usable / 0 found" with no other clue.

This scans the mount point instead, reports what is really there (count, sample
rates, channels, duration), flags anything unusable, and prints a groups JSON you
can paste. Roots are emitted as the DATASET ROOT -- scan_files() recurses, so the
internal layout does not matter.

    python discover_groups.py --root /kaggle/input
    python discover_groups.py --root /kaggle/input --probe 40 --out groups.json
"""
import argparse
import json
import os
import random
from collections import Counter

AUDIO = (".wav", ".flac", ".aiff", ".aif", ".ogg", ".opus", ".mp3", ".m4a")
LOSSLESS = (".wav", ".flac", ".aiff", ".aif")
SR_TARGET = 44100


def walk_audio(root, cap=200000):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(AUDIO):
                out.append(os.path.join(dp, f))
                if len(out) >= cap:
                    return out
    return out


def probe(paths, n):
    """Sample-rate / channel / duration census over a random subset."""
    try:
        import torchaudio
    except Exception:
        return None
    info = Counter()
    dur = 0.0
    ok = 0
    for p in random.sample(paths, min(n, len(paths))):
        try:
            i = torchaudio.info(p)
            info[(i.sample_rate, i.num_channels)] += 1
            dur += i.num_frames / max(i.sample_rate, 1)
            ok += 1
        except Exception:
            info[("unreadable", 0)] += 1
    return info, (dur / ok if ok else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--probe", type=int, default=25, help="files to probe per dataset")
    ap.add_argument("--min-files", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        raise SystemExit(f"{a.root} does not exist. On Kaggle, attach datasets first "
                         f"(Add Data), then re-run.")

    datasets = sorted(d for d in os.listdir(a.root)
                      if os.path.isdir(os.path.join(a.root, d)))
    if not datasets:
        raise SystemExit(f"No datasets mounted under {a.root}. Attach data first.")

    print(f"Scanning {a.root} — {len(datasets)} mounted dataset(s)\n")
    groups = []
    for d in datasets:
        p = os.path.join(a.root, d)
        files = walk_audio(p)
        if not files:
            print(f"  {d:34s} no audio found — skipped")
            continue
        loss = [f for f in files if f.lower().endswith(LOSSLESS)]
        lossy = len(files) - len(loss)
        pr = probe(loss or files, a.probe)
        srs = ""
        if pr:
            cnt, avgdur = pr
            top = ", ".join(f"{k[0]}Hz/{k[1]}ch×{v}" for k, v in cnt.most_common(3))
            srs = f"  [{top}]  avg {avgdur:.1f}s"
        print(f"  {d:34s} {len(files):7d} audio "
              f"({len(loss)} lossless, {lossy} lossy){srs}")

        if lossy and not loss:
            print(f"      ⚠ ALL LOSSY — unusable as ground truth. Training on it "
                  f"teaches the model to restore toward already-damaged targets.")
            continue
        if lossy:
            print(f"      ⚠ {lossy} lossy file(s) present; the provenance filter "
                  f"will drop them, but check the corpus is what you think it is.")
        if pr:
            low = [k for k in pr[0] if isinstance(k[0], int) and k[0] < SR_TARGET]
            if low:
                print(f"      ⚠ sample rates below {SR_TARGET} present {low} — "
                      f"those are band-limited and will be rejected.")
        if len(loss) < a.min_files:
            print(f"      too few lossless files ({len(loss)}) — skipped")
            continue
        groups.append({"name": d.replace("-", "_")[:28], "roots": [p], "floor": 1.0})

    if not groups:
        raise SystemExit("\nNo usable group could be built. Attach a LOSSLESS corpus "
                         "(MUSDB18-HQ, VCTK, FSD50K). Saraga and FMA are mp3 and "
                         "cannot serve as ground truth.")

    print("\n" + "=" * 68)
    print("groups JSON (roots are DATASET ROOTS — scan_files recurses, so the")
    print("internal layout does not matter). Floors are placeholders: run")
    print("`eval_per_group.py --mode calibrate` and paste the measured values.")
    print("=" * 68)
    js = json.dumps(groups, indent=1)
    print(js)
    if a.out:
        open(a.out, "w").write(js)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

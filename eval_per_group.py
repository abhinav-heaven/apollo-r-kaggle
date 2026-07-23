"""
Per-group, per-bitrate, per-regime evaluation + floor calibration.

Two jobs:

 1. `calibrate` -- measure each group's ACHIEVABLE error floor using the rolloff
    baseline. Floor-corrected Group DRO needs these numbers; without them worst-group
    optimization chases irreducible error. Measured floors differ ~2x across domains
    (0.39 flute vs 0.74 Carnatic vocal in this program's real-audio test).

 2. `eval` -- score a checkpoint per group x bitrate x regime, ALWAYS against the
    rolloff baseline, never against unprocessed audio alone. That baseline scores
    0.39-0.57 on regular spectra and is far stronger than the literature implies;
    beating "do nothing" is not evidence of anything.

Regimes are reported separately because they are different claims:
  BELOW cutoff -- restoration; the signal is constrained by the bitstream.
  ABOVE cutoff -- synthesis; EXP 1 proved analytic recovery is impossible.
A single global number conflates the two and hides hallucination.
"""
import argparse
import json
import os

import warnings

import numpy as np
import torch
import torchaudio

warnings.filterwarnings('ignore', message='All-NaN slice encountered')

from look2hear.models.apollo_r import ApolloR

# Import codec_sim by path: look2hear.datas.__init__ imports the HDF5/Lightning
# datamodule, which this script does not use.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "codec_sim", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "look2hear", "datas", "codec_sim.py"))
codec_sim = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(codec_sim)
BITRATES, MEASURED_CUTOFF = codec_sim.BITRATES, codec_sim.MEASURED_CUTOFF
CODECS, COND_DIM = codec_sim.CODECS, codec_sim.COND_DIM
codec_cutoff, condition_vector = codec_sim.codec_cutoff, codec_sim.condition_vector
multi_codec_simu = codec_sim.multi_codec_simu

_rs = _ilu.spec_from_file_location(
    "restoration", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "look2hear", "metrics", "restoration.py"))
restoration = _ilu.module_from_spec(_rs); _rs.loader.exec_module(restoration)
restoration_gain, energy_bias = restoration.restoration_gain, restoration.energy_bias
restoration_gain_regime = restoration.restoration_gain_regime

SR = 44100


def load_seg(path, seg, max_segs=6):
    info = torchaudio.info(path)
    if info.sample_rate != SR or info.num_frames < seg * 2:
        return []
    out, n = [], min(max_segs, info.num_frames // seg)
    for k in range(n):
        y, _ = torchaudio.load(path, frame_offset=k * seg, num_frames=seg)
        if y.shape[0] == 1:
            y = y.repeat(2, 1)
        y = y[:2]
        if float(y.pow(2).mean().sqrt()) < 1e-3:
            continue
        m = y.abs().max()
        out.append(y / m * 0.9 if m > 0 else y)
    return out


def stft_mag(x, n=2048):
    w = torch.hann_window(n, device=x.device)
    return torch.stft(x.reshape(-1, x.shape[-1]).float(), n, n // 2,
                      window=w, return_complex=True).abs()


def rolloff_baseline(deg, cutoff_hz, n=2048):
    """The honest baseline: fit a log-log rolloff below the cutoff, continue it above.
    On regular spectra this scores 0.39-0.57 -- it is the method to beat."""
    w = torch.hann_window(n, device=deg.device)
    D = torch.stft(deg.reshape(-1, deg.shape[-1]).float(), n, n // 2,
                   window=w, return_complex=True)
    mag, ph = D.abs(), torch.angle(D)
    f = torch.fft.rfftfreq(n, 1 / SR).to(deg.device)
    lo = (f > 500) & (f < cutoff_hz)
    hi = f >= cutoff_hz
    if lo.sum() < 8 or hi.sum() == 0:
        return deg
    # mag is (B, F, T): the regression runs along the FREQUENCY axis (dim 1),
    # independently per time frame. lf must broadcast as (1, F_lo, 1) -- indexing
    # it as a bare 1-D tensor silently aligns it with the TIME axis instead.
    lf = torch.log(f[lo] + 1e-9).view(1, -1, 1)          # 1, F_lo, 1
    lm = torch.log(mag[:, lo] + 1e-9)                    # B, F_lo, T
    fm = lf.mean()
    mm = lm.mean(dim=1, keepdim=True)                    # B, 1, T
    num = ((lf - fm) * (lm - mm)).sum(dim=1)             # B, T
    den = ((lf - fm) ** 2).sum() + 1e-9
    slope = num / den                                    # B, T
    inter = mm.squeeze(1) - slope * fm                   # B, T
    lhi = torch.log(f[hi] + 1e-9).view(1, -1, 1)         # 1, F_hi, 1
    pred = torch.exp(inter.unsqueeze(1) + slope.unsqueeze(1) * lhi)   # B, F_hi, T
    mag = mag.clone()
    mag[:, hi] = pred
    out = torch.istft(torch.polar(mag, ph), n, n // 2, window=w,
                      length=deg.shape[-1])
    return out.view_as(deg)


def regime_err(est, ref, cutoff_hz, n=2048, min_energy_frac=1e-4):
    """Relative spectral error below / above the cutoff.

    Returns NaN when the REFERENCE carries negligible energy in a regime. Without
    this guard the relative error divides by ~0 and explodes into meaningless
    numbers (>1 means "worse than leaving the band empty"). This is the same trap
    that silently invalidated the sitar result: that file was transcoded from
    128 kbps, so it had no true content above the cutoff to restore, and its error
    blew up to 12.2 / 40.4 rather than reporting honestly that the test was void.
    """
    E, R = stft_mag(est, n), stft_mag(ref, n)
    f = torch.fft.rfftfreq(n, 1 / SR).to(est.device)
    total = R.pow(2).sum()

    def rel(m):
        if m.sum() == 0:
            return float("nan")
        den = R[:, m].pow(2).sum()
        if float(den) < min_energy_frac * float(total):
            return float("nan")        # nothing there to restore; not a score of 1
        return float(((E[:, m] - R[:, m]).pow(2).sum() / (den + 1e-12)).sqrt())

    return rel(f < cutoff_hz), rel(f >= cutoff_hz)


def scan(roots, exts=(".wav", ".flac", ".aiff", ".aif")):
    out = []
    for r in roots:
        for dp, _, fns in os.walk(r):
            out += [os.path.join(dp, f) for f in fns if f.lower().endswith(exts)]
    return sorted(out)


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = json.load(open(args.groups))
    seg = int(args.seg * SR)

    model = None
    if args.ckpt:
        model = ApolloR(sr=SR, win=20, feature_dim=args.feature_dim,
                        layer=args.layer, cond_dim=COND_DIM)
        ck = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ck.get("model", ck), strict=False)
        model = model.to(device).eval()

    brs = [int(b) for b in args.bitrates.split(",")]
    rows = []
    for s in specs:
        files = scan(s["roots"])[:args.max_files]
        for br in brs:
            cut = codec_cutoff(args.codec, br)
            acc = {k: [] for k in ("deg_lo", "deg_hi", "base_lo", "base_hi",
                                   "mdl_lo", "mdl_hi", "base_rgr", "mdl_rgr",
                                   "mdl_bias", "base_rgr_lo", "base_rgr_hi",
                                   "mdl_rgr_lo", "mdl_rgr_hi")}
            for fp in files:
                for y in load_seg(fp, seg, args.max_segs):
                    y = y.to(device)
                    try:
                        d, _ = multi_codec_simu(y.cpu(), SR, args.codec, br)
                    except Exception:
                        continue
                    d = d[:, :y.shape[-1]].to(device)
                    if d.shape[-1] < y.shape[-1]:
                        d = torch.nn.functional.pad(d, (0, y.shape[-1] - d.shape[-1]))
                    a, b = regime_err(d, y, cut); acc["deg_lo"].append(a); acc["deg_hi"].append(b)
                    bl = rolloff_baseline(d, cut)
                    a, b = regime_err(bl, y, cut); acc["base_lo"].append(a); acc["base_hi"].append(b)
                    # RGR: identity == 1.0 exactly, and comparable across bitrates
                    acc["base_rgr"].append(float(restoration_gain(
                        bl.unsqueeze(0), y.unsqueeze(0), d.unsqueeze(0))))
                    rl, rh = restoration_gain_regime(bl.unsqueeze(0), y.unsqueeze(0),
                                                     d.unsqueeze(0), cut)
                    acc["base_rgr_lo"].append(rl); acc["base_rgr_hi"].append(rh)
                    if model is not None:
                        cf = condition_vector(args.codec, br, SR).to(device).unsqueeze(0)
                        e = model(d.unsqueeze(0), cond=cf).squeeze(0)
                        a, b = regime_err(e, y, cut)
                        acc["mdl_lo"].append(a); acc["mdl_hi"].append(b)
                        acc["mdl_rgr"].append(float(restoration_gain(
                            e.unsqueeze(0), y.unsqueeze(0), d.unsqueeze(0))))
                        acc["mdl_bias"].append(energy_bias(e.unsqueeze(0), y.unsqueeze(0), cut))
                        rl, rh = restoration_gain_regime(e.unsqueeze(0), y.unsqueeze(0),
                                                         d.unsqueeze(0), cut)
                        acc["mdl_rgr_lo"].append(rl); acc["mdl_rgr_hi"].append(rh)
            if not acc["deg_lo"]:
                continue
            m = {k: float(np.nanmedian(v)) if v else float("nan") for k, v in acc.items()}
            m.update(group=s["name"], bitrate=br, cutoff=cut, n=len(acc["deg_lo"]))
            rows.append(m)
            print(f"{s['name']:16s} {br:4d}k cut={cut:5d} n={m['n']:3d} | "
                  f"rolloff RGR lo/hi {m['base_rgr_lo']:.3f}/{m['base_rgr_hi']:.3f} | "
                  f"model RGR lo/hi {m['mdl_rgr_lo']:.3f}/{m['mdl_rgr_hi']:.3f} | "
                  f"bias {m['mdl_bias']:+.1f}dB   (1.000 == no restoration)",
                  flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)

    if args.mode == "calibrate":
        print("\nFLOORS in RGR UNITS for train_dro.py --groups.\n"
              "RGR: identity == 1.0 exactly, so a floor < 1.0 is the restoration the\n"
              "rolloff baseline already achieves -- the bar the model must clear.")
        floors = {}
        for s in specs:
            v = [r["base_rgr_hi"] for r in rows if r["group"] == s["name"]
                 and not np.isnan(r["base_rgr_hi"])]
            floors[s["name"]] = round(float(np.median(v)), 3) if v else 1.0
        for k, v in floors.items():
            print(f'  {{"name": "{k}", "roots": [...], "floor": {v}}},')
        print("\nNOTE: floors are the ACHIEVABLE error, not a target. Groups whose floor\n"
              "is high for information-theoretic reasons (percussion, very low bitrate)\n"
              "must not be allowed to dominate worst-group optimization.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--groups", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--mode", choices=["eval", "calibrate"], default="eval")
    p.add_argument("--bitrates", default="24,32,64,96,128,192")
    p.add_argument("--seg", type=float, default=3.0)
    p.add_argument("--max-files", type=int, default=40)
    p.add_argument("--max-segs", type=int, default=4)
    p.add_argument("--codec", default="mp3", choices=CODECS)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--out", default="eval_per_group.json")
    run(p.parse_args())

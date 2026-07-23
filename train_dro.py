"""
Apollo-R fine-tuning with floor-corrected Group DRO.

Designed for a Kaggle 2xT4 / 12 h session:
  * fp16 (Turing: no bf16), gradient checkpointing on the 6 BSNet layers
  * DDP over 2 GPUs
  * wall-clock-aware checkpointing + resume (sessions are capped at 12 h)
  * no GAN by default -- Apollo's step runs FOUR discriminator passes; for a short
    fine-tune the adversarial term is unlikely to converge and risks destabilizing
    a good checkpoint. Enable with --gan for a late polish phase.

WHY FLOOR-CORRECTED GROUP DRO
  Naive worst-group DRO chases IRREDUCIBLE error. This program measured that error
  floors differ ~2x across domains (best achievable 0.39 on flute vs 0.74 on
  Carnatic vocal), and proved some content is unrecoverable in principle (content
  above the codec cutoff; percussion, which has no pitch diversity). Optimizing
  raw max_g would pour all capacity into the highest-floor group.
  So we equalize EXCESS risk:  max_g [ L_g - floor_g ].

STATUS: SMOKE-TESTED end-to-end on CPU (torch 2.2.2) -- 40 steps, 3 groups,
mp3/aac/opus, checkpoint + SWAD save verified. Bugs found and fixed by that run:
  * gradient checkpointing called a non-existent API, and the obvious wrapper fix
    would have RENAMED parameters and broken checkpoint compatibility
  * the sigmoid preservation gate was not identity at init (1.2e-2 drift)
  * SWAD averaged aliased tensors (missing .clone()) -- a bug invisible on GPU fp16
  * AMP API differs across torch versions and must no-op on CPU
NOT yet run on GPU or at full scale. Run --smoke on Kaggle before the long job:
it asserts exact pretrained equivalence and will refuse to proceed if broken.
"""
import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from look2hear.models.apollo import Apollo
from look2hear.models.apollo_r import ApolloR


# Import codec_sim by path: look2hear.datas.__init__ pulls in the HDF5/Lightning
# datamodule, which this script does not use and which would force heavy deps
# (h5py, pytorch_lightning) onto anyone running the trainer.
_cs_spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
    "codec_sim", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "look2hear", "datas", "codec_sim.py"))
codec_sim = __import__("importlib.util", fromlist=["util"]).module_from_spec(_cs_spec)
_cs_spec.loader.exec_module(codec_sim)

# same reason: look2hear.metrics.__init__ imports wrapper.py -> librosa
_rs_spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
    "restoration", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "look2hear", "metrics", "restoration.py"))
restoration = __import__("importlib.util", fromlist=["util"]).module_from_spec(_rs_spec)
_rs_spec.loader.exec_module(restoration)
restoration_gain, regime_split_loss = restoration.restoration_gain, restoration.regime_split_loss
BITRATES, CODECS, COND_DIM = codec_sim.BITRATES, codec_sim.CODECS, codec_sim.COND_DIM
MEASURED_CUTOFF = codec_sim.MEASURED_CUTOFF
codec_cutoff, codec_rates = codec_sim.codec_cutoff, codec_sim.codec_rates
condition_vector, multi_codec_simu = codec_sim.condition_vector, codec_sim.multi_codec_simu

SR = 44100


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class GroupSpec:
    name: str
    roots: list
    floor: float = 1.0          # achievable error; measured, see calibrate_floors
    weight: float = 1.0
    files: list = field(default_factory=list)


def scan_files(roots, exts=(".wav", ".flac", ".aiff", ".aif")):
    out = []
    for r in roots:
        for dp, _, fns in os.walk(r):
            out += [os.path.join(dp, f) for f in fns if f.lower().endswith(exts)]
    return sorted(out)


def provenance_ok(path, thr_db=-55.0, sr=SR):
    """Reject files whose OWN content already has a codec hole.

    A WAV file is not proof of lossless provenance. Verified on real data: a
    24-bit "lossless" sitar download read -56.5 dB high-band energy (content
    stopping at 16.5 kHz -- a 128 kbps cutoff) against -6..-21 dB for genuinely
    clean files. Training on such a file teaches the model the hole is correct.

    LIMITATION (measured): this catches gross cases only. Deliberately transcoded
    flute at 96-320 kbps read -12.7..-14.9 dB vs -12.5 dB for the true original --
    indistinguishable. Prefer corpora with documented provenance.
    """
    try:
        import torchaudio
        y, s = torchaudio.load(path)
    except Exception:
        return False
    if y.shape[-1] < s:                      # under one second
        return False
    if s != sr:
        # RESAMPLE, don't reject. Rejecting on sample rate silently discarded
        # 100% of VCTK (48 kHz) even when the paths were right -- the group just
        # reported "0 files" with no reason given.
        if s < sr:
            return False                     # genuinely band-limited source
        import torchaudio.functional as AF
        y = AF.resample(y, s, sr)
    y = y.mean(0).double()
    n = 8192
    P = torch.zeros(n // 2 + 1, dtype=torch.float64)
    cnt = 0
    for i in range(0, min(len(y) - n, sr * 30), n):
        P += torch.fft.rfft(y[i:i + n] * torch.hann_window(n, dtype=torch.float64)).abs() ** 2
        cnt += 1
    if cnt == 0:
        return False
    P /= cnt
    f = torch.fft.rfftfreq(n, 1 / sr)
    lo = P[(f >= 10000) & (f < 15000)].mean()
    hi = P[(f >= 17500) & (f < 21000)].mean()
    return float(10 * torch.log10(hi / (lo + 1e-30) + 1e-30)) > thr_db


DEG_CELLS = [(c, r) for c in CODECS for r in codec_rates(c)]


class DegradationSampler:
    """Wasserstein-DRO over the DEGRADATION distribution, made practical.

    This is the one place a Wasserstein ball is correctly scaled. Measured: over
    CONTENT the ball is vacuous (white noise sits closer to flute, 2.33, than
    Carnatic vocal does, 3.56). Over degradation the space is low-dimensional and
    parametric -- codec identity x bitrate -- and mp3/aac/opus really are nearby
    operators. So we sample degradation adversarially: cells with the highest
    EXCESS loss get sampled more.

    Prior favours low bitrates: 24 kbps destroys ~80% of the spectrum, 320 kbps
    ~7%, so uniform sampling would waste a large share of the budget where there
    is almost nothing to learn.
    """

    def __init__(self, temp=3.0, ema=0.05):
        self.cells = DEG_CELLS
        self.L = torch.zeros(len(self.cells))
        self.seen = torch.zeros(len(self.cells))
        self.temp, self.ema = temp, ema
        prior = []
        for c, r in self.cells:
            prior.append(1.6 if r <= 32 else 1.2 if r <= 64 else
                         0.8 if r <= 96 else 0.4 if r <= 128 else 0.0)
        self.prior = torch.tensor(prior)

    def logits(self):
        return self.prior + self.temp * (self.L - self.L.mean())

    def observe(self, cell_idx, losses):
        for i in cell_idx.unique():
            m = cell_idx == i
            self.L[i] = (1 - self.ema) * self.L[i] + self.ema * float(losses[m].mean())
            self.seen[i] += int(m.sum())

    def state(self):
        return {"L": self.L.tolist(), "seen": self.seen.tolist()}


class CodecDataset(Dataset):
    """Self-supervised: ANY lossless audio is a training pair.

    Degradation is sampled over (codec, bitrate) cells, not bitrate alone -- see
    DegradationSampler. Opus in particular is a DIFFERENT degradation, not a
    harsher one: it never lowpasses (measured ~20.4 kHz at every rate), so its
    damage is broadband distortion. The conditioning vector carries codec identity
    so the model can tell which problem it is looking at.
    """

    def __init__(self, groups, seg_s=3.0, samples_per_epoch=20000,
                 deg_logits=None, codecs=None, stratify=True):
        self.groups = groups
        self.seg = int(seg_s * SR)
        self.n = samples_per_epoch
        self.stratify = stratify
        self.cells = ([c for c in DEG_CELLS if c[0] in codecs] if codecs else DEG_CELLS)
        self.cell_index = [DEG_CELLS.index(c) for c in self.cells]
        # SHARED MEMORY, not plain tensors. DataLoader workers get a COPY of the
        # dataset at fork; mutating self.deg_logits in the parent would never reach
        # them, so adversarial degradation sampling would silently do nothing
        # whenever num_workers > 0 -- exactly the configuration used for real runs.
        # share_memory_() makes the parent's writes visible to forked workers.
        base = (torch.zeros(len(self.cells)) if deg_logits is None
                else torch.as_tensor(deg_logits, dtype=torch.float))
        self.deg_logits = base.clone().share_memory_()
        self.gw = torch.tensor([g.weight for g in groups],
                               dtype=torch.float).share_memory_()

    def __len__(self):
        return self.n

    def set_group_weights(self, w):
        # copy_ IN PLACE: rebinding would break the shared-memory link to workers
        self.gw.copy_(torch.as_tensor(w, dtype=torch.float).clamp_min(1e-6))

    def set_deg_logits(self, l):
        full = torch.as_tensor(l, dtype=torch.float)
        self.deg_logits.copy_(full[torch.tensor(self.cell_index)])

    def _load_segment(self, path):
        import torchaudio
        info = torchaudio.info(path)
        sr_in = info.sample_rate
        need = int(self.seg * sr_in / SR) + 1        # frames at the FILE's rate
        if info.num_frames <= need:
            return None
        off = random.randint(0, info.num_frames - need - 1)
        y, s = torchaudio.load(path, frame_offset=off, num_frames=need)
        if s != SR:
            if s < SR:
                return None                          # band-limited source
            import torchaudio.functional as AF
            y = AF.resample(y, s, SR)                # e.g. VCTK 48k -> 44.1k
        y = y[:, :self.seg]
        if y.shape[-1] < self.seg:
            return None
        if y.shape[0] == 1:
            y = y.repeat(2, 1)
        return y[:2]

    def __getitem__(self, idx):
        for _try in range(12):
            if self.stratify and len(self.groups) > 1:
                # V-REx needs >=2 groups per batch for the variance to exist.
                # Round-robin the group, then let DRO weights bias within it.
                gi = int(idx % len(self.groups))
                if not self.groups[gi].files:
                    gi = int(torch.multinomial(self.gw, 1))
            else:
                gi = int(torch.multinomial(self.gw, 1))
            g = self.groups[gi]
            if not g.files:
                continue
            y = self._load_segment(random.choice(g.files))
            if y is None:
                continue
            # energy filter: silent crops teach nothing
            if float(y.pow(2).mean().sqrt()) < 1e-3:
                continue
            m = y.abs().max()
            if m > 0:
                y = y / m * 0.9
            ci = int(torch.multinomial(self.deg_logits.softmax(0), 1))
            codec, br = self.cells[ci]
            try:
                deg, cut = multi_codec_simu(y, SR, codec, br)
            except Exception:
                continue
            deg = deg[:, :y.shape[-1]]
            if deg.shape[-1] < y.shape[-1]:
                deg = F.pad(deg, (0, y.shape[-1] - deg.shape[-1]))
            cond = condition_vector(codec, br, SR)
            return y, deg, gi, self.cell_index[ci], cond, float(cut)
        z = torch.zeros(2, self.seg)
        return z, z, 0, 0, condition_vector("mp3", 128, SR), 16780.0


# --------------------------------------------------------------------------- #
# Loss  (magnitude-only was Apollo's; we add a phase-aware term because Apollo
# selected checkpoints on SI-SDR, a metric it never optimized)
# --------------------------------------------------------------------------- #
WINS = [32, 64, 128, 256, 512, 1024, 2048]
_WIN_CACHE = {}


def _hann(w, device):
    """Cache analysis windows. These were rebuilt on every call, for every one of
    7 resolutions, on every sample."""
    k = (w, str(device))
    if k not in _WIN_CACHE:
        _WIN_CACHE[k] = torch.hann_window(w, device=device, dtype=torch.float32)
    return _WIN_CACHE[k]


def mrstft_per_sample(est, ref, wins=WINS, eps=1e-8):
    """Multi-resolution STFT loss, returned PER SAMPLE in one batched pass.

    The previous call site looped `mrstft(est[i:i+1], ref[i:i+1])` over the batch,
    running 7 STFTs per sample separately -- B x 7 x 2 kernel launches per step
    instead of 7 x 2. Group DRO needs per-sample values, so we keep the per-sample
    axis and reduce over (freq, time) only.

    est/ref: (B, C, T) -> returns (B,)
    """
    B, C, T = est.shape
    e = est.reshape(B * C, T).float()
    r = ref.reshape(B * C, T).float()
    mag = e.new_zeros(B * C)
    cplx = e.new_zeros(B * C)
    for w in wins:
        win = _hann(w, e.device)
        E = torch.stft(e, w, w // 2, window=win, return_complex=True)
        R = torch.stft(r, w, w // 2, window=win, return_complex=True)
        rabs = R.abs()
        denom = rabs.mean(dim=(1, 2)) + eps                      # per sample
        mag = mag + (E.abs() - rabs).abs().mean(dim=(1, 2)) / denom
        cplx = cplx + (E - R).abs().mean(dim=(1, 2)) / denom
    per = (mag + cplx) / (2 * len(wins))
    return per.view(B, C).mean(1)                                # (B,)


def mrstft(est, ref, wins=WINS, eps=1e-8):
    return mrstft_per_sample(est, ref, wins, eps).mean()


def per_regime_error(est, ref, cutoff_hz, n_fft=2048):
    """Report BELOW vs ABOVE the codec cutoff separately.

    A single global number hides whether the model is restoring (below, where the
    signal is constrained) or synthesizing (above, where EXP 1 proved recovery is
    impossible in principle). These are different claims and must be reported apart.
    """
    e = est.reshape(-1, est.shape[-1]).float()
    r = ref.reshape(-1, ref.shape[-1]).float()
    win = torch.hann_window(n_fft, device=e.device)
    E = torch.stft(e, n_fft, n_fft // 2, window=win, return_complex=True).abs()
    R = torch.stft(r, n_fft, n_fft // 2, window=win, return_complex=True).abs()
    f = torch.fft.rfftfreq(n_fft, 1 / SR).to(e.device)
    lo, hi = f < cutoff_hz, f >= cutoff_hz
    def rel(m):
        num = (E[:, m] - R[:, m]).pow(2).sum()
        den = R[:, m].pow(2).sum() + 1e-12
        return float((num / den).sqrt())
    return rel(lo), rel(hi)


# --------------------------------------------------------------------------- #
# Group DRO
# --------------------------------------------------------------------------- #
class FloorCorrectedGroupDRO:
    """Sagawa-style online group DRO on EXCESS risk.

    q_g <- q_g * exp(eta * (L_g - floor_g));  normalize;  loss = sum_g q_g L_g

    Using (L_g - floor_g) rather than L_g is the whole point: raw max_g would
    chase groups whose error is irreducible (above-cutoff content, percussion),
    starving the groups we can actually improve.
    """

    def __init__(self, n_groups, floors, eta=0.01, ema=0.1, device="cuda"):
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.floors = torch.as_tensor(floors, dtype=torch.float, device=device)
        self.L = torch.zeros(n_groups, device=device)
        self.seen = torch.zeros(n_groups, device=device)
        self.eta, self.ema = eta, ema

    def update(self, losses, gidx, ddp=False):
        """DDP CORRECTNESS: per-group losses must be all-reduced before updating q.

        Without this each rank sees only its own shard, so `q` diverges across
        ranks and the two GPUs weight the SAME global batch differently -- the
        gradients being averaged by DDP would come from different objectives.
        Silent, and it gets worse the longer you train.
        """
        with torch.no_grad():
            n = self.q.numel()
            sums = torch.zeros(n, device=losses.device)
            cnts = torch.zeros(n, device=losses.device)
            sums.index_add_(0, gidx, losses.detach().float())
            cnts.index_add_(0, gidx, torch.ones_like(losses, dtype=torch.float))
            if ddp and dist.is_initialized():
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnts, op=dist.ReduceOp.SUM)
            seen = cnts > 0
            if seen.any():
                mean = sums[seen] / cnts[seen]
                self.L[seen] = (1 - self.ema) * self.L[seen] + self.ema * mean
                self.seen[seen] += cnts[seen]
            excess = (self.L - self.floors).clamp_min(0)
            self.q = self.q * torch.exp(self.eta * excess)
            self.q = self.q / self.q.sum()
        return (self.q[gidx] * losses).sum() / (self.q[gidx].sum() + 1e-12)

    def state(self):
        return {"q": self.q.tolist(), "L": self.L.tolist(), "seen": self.seen.tolist()}


class DoroMask:
    """DORO — trim the worst eps-fraction before the worst-case objective.

    Standard DRO amplifies outliers; ours are guaranteed to exist. The provenance
    filter provably leaks mild transcodes (measured: deliberately transcoded flute
    reads -12.7..-14.9 dB vs -12.5 dB for the true original -- indistinguishable),
    and a leaked transcode is a sample whose "clean" target already has the hole
    the model is being asked to fill. Without trimming, group DRO will chase those.

    DEVIATION FROM THE PAPER: DORO trims the top eps-fraction *within the batch*.
    Our batches are tiny (4 per GPU), so a per-batch quantile is meaningless. We
    instead track an EMA of the (1-eps) quantile of per-sample loss and mask
    samples above it. Equivalent in the large-step limit, usable at batch 4.

    Interaction note: eps must stay SMALL (0.02-0.10). Trimming too hard removes
    the hard-but-legitimate examples that group DRO exists to prioritise.
    """

    def __init__(self, eps=0.05, ema=0.02, warmup=200, device="cuda"):
        self.eps, self.ema, self.warmup = eps, ema, warmup
        self.q = torch.tensor(float("inf"), device=device)
        self.n = 0
        self.dropped = 0

    def __call__(self, losses):
        self.n += 1
        with torch.no_grad():
            hi = torch.quantile(losses.detach().float(), 1.0 - self.eps) \
                if losses.numel() > 1 else losses.detach().float().max()
            if torch.isinf(self.q):
                self.q = hi
            else:
                self.q = (1 - self.ema) * self.q + self.ema * hi
        if self.n < self.warmup:                 # don't trim before q is meaningful
            return torch.ones_like(losses, dtype=torch.bool)
        keep = losses.detach() <= self.q
        if not bool(keep.any()):                 # never drop the whole batch
            keep = torch.ones_like(losses, dtype=torch.bool)
        self.dropped += int((~keep).sum())
        return keep


def vrex_penalty(losses, gidx, n_groups):
    """V-REx — penalize the VARIANCE of per-group risks.

    Group DRO optimizes the WORST group; V-REx pushes all groups toward EQUAL
    risk, which is closer to "works on any audio file" than "works on the hardest
    audio". They are complementary, so we use both.

    Gradients must flow through the per-group means, so these are computed on the
    current batch (not the EMAs). With small batches not every group appears; the
    variance is defined only over groups PRESENT, and is skipped when fewer than
    two are present. Over many steps this averages out, but it is a real source of
    noise -- prefer >=2 groups per batch (see --stratify).
    """
    present = gidx.unique()
    if present.numel() < 2:
        return losses.sum() * 0.0
    means = torch.stack([losses[gidx == g].mean() for g in present])
    return means.var(unbiased=False)


class SWAD:
    """Weight averaging over the trajectory (SWAD, simplified).

    Averaging weights in a flat region generalizes better out-of-distribution, at
    ~zero cost: 16.5 M params in fp32 is 66 MB. Apollo uses RMSNorm, which has no
    running statistics, so no BN re-estimation pass is needed afterwards.

    Simplification: the paper selects the averaging window from a validation-loss
    criterion. We average densely over the final `start_frac` of the run, which is
    the standard cheap approximation.
    """

    def __init__(self, model, start_frac=0.6, every=10):
        self.start_frac, self.every = start_frac, every
        self.avg = None
        self.n = 0

    def maybe_update(self, model, step, total_steps):
        if total_steps <= 0 or step < self.start_frac * total_steps:
            return
        if step % self.every:
            return
        # .clone() is REQUIRED: .detach().float().cpu() is a no-op chain for a CPU
        # float32 tensor, so without it `avg` aliases the live parameters and the
        # running average updates to zero. On GPU fp16 the dtype cast copies and
        # hides the bug -- it only shows up locally, which is where it was caught.
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()
              if v.dtype.is_floating_point}
        if self.avg is None:
            self.avg = sd
            self.n = 1
        else:
            self.n += 1
            for k in self.avg:
                self.avg[k] += (sd[k] - self.avg[k]) / self.n

    def state_dict(self, model):
        """Averaged floats merged over the live state dict (ints/buffers kept)."""
        if self.avg is None:
            return None
        out = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        out.update(self.avg)
        return out


# --------------------------------------------------------------------------- #
def build_model(args, device):
    kw = dict(sr=SR, win=20, feature_dim=getattr(args, "feature_dim", 256),
              layer=getattr(args, "layer", 6))
    model = ApolloR(use_film=not args.no_film, use_logfreq=not args.no_logfreq,
                    use_refine=not args.no_refine, use_gate=not args.no_gate,
                    cond_dim=COND_DIM, **kw)
    if args.pretrained and os.path.exists(args.pretrained):
        sd = torch.load(args.pretrained, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.replace("audio_model.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        new = {n for n, _ in model.named_parameters()
               if n.split(".")[0] in ("film", "logfreq", "refine") or n == "band_gate"}
        stray = [k for k in missing if k not in new]
        assert not stray, f"pretrained weights missing: {stray[:8]}"
        print(f"[init] loaded pretrained; {len(new)} new tensors at zero-init")
    if args.grad_ckpt:
        enable_grad_checkpointing(model)
    return model.to(device)


def enable_grad_checkpointing(model):
    """Checkpoint the 6 BSNet layers to cut activation memory.

    The MLP's 8x expansion dominates: batch 8 stereo needs ~9.5 GB across 6 layers
    without this, which is tight on a 16 GB T4. Checkpointing keeps one layer live.

    IMPORTANT: this patches each block's BOUND FORWARD rather than wrapping the
    block in a new nn.Module. Wrapping would rename parameters (net.0.band_net.* ->
    net.0.mod.band_net.*) and silently break checkpoint compatibility, which is the
    one property this whole design rests on.
    """
    import torch.utils.checkpoint as ckpt

    def patch(block):
        original = block.forward

        def fwd(x, _o=original):
            if torch.is_grad_enabled() and x.requires_grad:
                return ckpt.checkpoint(_o, x, use_reentrant=False)
            return _o(x)
        block.forward = fwd

    for blk in model.net:
        patch(blk)


def verify_init_equivalence(model, device, feature_dim=256, layer=6):
    """Non-negotiable safety check: at init, ApolloR MUST equal Apollo exactly.
    If this fails the checkpoint is being corrupted from step 0 -- do not train."""
    base = Apollo(sr=SR, win=20, feature_dim=feature_dim, layer=layer).to(device).eval()
    sd = {k: v for k, v in model.state_dict().items()
          if k.split(".")[0] not in ("film", "logfreq", "refine") and k != "band_gate"}
    base.load_state_dict(sd, strict=False)
    x = torch.randn(1, 2, SR, device=device)
    was = model.training
    model.eval()
    with torch.no_grad():
        a = base(x)
        b = model(x, cond=condition_vector("mp3", 128, SR).to(device).unsqueeze(0))
    model.train(was)
    d = float((a - b).abs().max())
    print(f"[check] max |ApolloR - Apollo| at init = {d:.3e}")
    assert d < 1e-3, "branches are NOT no-ops at init -- checkpoint safety broken"


# --------------------------------------------------------------------------- #
def make_scaler(enabled):
    """GradScaler across torch versions: torch.amp.GradScaler is newer API,
    torch.cuda.amp.GradScaler is the 2.2-era one. Disabled on CPU."""
    if hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled):
    """fp16 autocast on CUDA (T4 is Turing: fp16 only, no bf16); no-op on CPU."""
    if not enabled:
        import contextlib
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def save_ckpt(args, model, ddp, opt, scaler, dro, degsampler, swad, step):
    core = model.module if ddp else model
    torch.save({"model": core.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "dro": dro.state(),
                "deg": degsampler.state(), "step": step, "args": vars(args)},
               os.path.join(args.out, "last.ckpt"))
    avg = swad.state_dict(core)
    if avg is not None:
        # SWAD weights are the ones to EVALUATE; Apollo uses RMSNorm (no running
        # statistics), so no BN re-estimation pass is needed.
        torch.save({"model": avg, "step": step, "swad_n": swad.n},
                   os.path.join(args.out, "swad.ckpt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", required=True, help="JSON: [{name, roots, floor}]")
    p.add_argument("--pretrained", default="")
    p.add_argument("--out", default="./exp_apollo_r")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seg", type=float, default=3.0)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--max-hours", type=float, default=10.5, help="stop before the 12h cap")
    p.add_argument("--ckpt-min", type=float, default=25.0)
    p.add_argument("--dro-eta", type=float, default=0.01)
    p.add_argument("--no-dro", action="store_true", help="ERM baseline for ablation")
    p.add_argument("--no-floors", action="store_true", help="raw max_g, for ablation")
    # --- V-REx / DORO / SWAD / multi-encoder ---
    p.add_argument("--loss", choices=["rgr", "mrstft"], default="rgr",
                   help="rgr = damage-relative (identity==1.0); mrstft = absolute")
    p.add_argument("--vrex", type=float, default=1e-2,
                   help="weight on Var_g(risk); 0 disables")
    p.add_argument("--doro-warmup", type=int, default=200,
                   help="steps before trimming starts (quantile must settle first)")
    p.add_argument("--doro-eps", type=float, default=0.05,
                   help="fraction trimmed as outliers; 0 disables. Keep small: "
                        "over-trimming removes the hard examples DRO exists to serve")
    p.add_argument("--swad-start", type=float, default=0.6,
                   help="begin weight averaging after this fraction of the run")
    p.add_argument("--swad-every", type=int, default=10)
    p.add_argument("--codecs", default="mp3,aac,opus",
                   help="comma list; opus is a DIFFERENT degradation (no lowpass)")
    p.add_argument("--deg-temp", type=float, default=3.0,
                   help="adversarial sharpness over (codec,bitrate) cells")
    p.add_argument("--deg-update", type=int, default=100)
    p.add_argument("--no-stratify", action="store_true",
                   help="disable round-robin groups (V-REx needs >=2 groups/batch)")
    p.add_argument("--total-steps", type=int, default=0,
                   help="0 = estimate from --max-hours and --sec-per-step (SWAD window)")
    p.add_argument("--sec-per-step", type=float, default=0.45)
    p.add_argument("--no-film", action="store_true")
    p.add_argument("--no-logfreq", action="store_true")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true", default=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--skip-provenance", action="store_true")
    args = p.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    if torch.cuda.is_available():
        # segment length is fixed, so cuDNN can autotune once and reuse.
        torch.backends.cudnn.benchmark = True
        # NOTE: no TF32 knob here on purpose -- T4 is Turing, TF32 is Ampere+.
        torch.set_float32_matmul_precision("high")
    os.makedirs(args.out, exist_ok=True)

    specs = json.load(open(args.groups))
    groups = [GroupSpec(name=s["name"], roots=s["roots"],
                        floor=float(s.get("floor", 1.0))) for s in specs]
    for g in groups:
        raw = scan_files(g.roots)
        if args.skip_provenance:
            fs, why = raw, ""
        else:
            fs = [f for f in raw if provenance_ok(f)]
            why = f"  ({len(raw) - len(fs)} rejected by provenance)" if raw else ""
        g.files = fs
        if rank == 0:
            print(f"[data] {g.name:22s} {len(fs):6d} usable / {len(raw):6d} found"
                  f"  floor={g.floor:.3f}{why}")
            # Distinguish "wrong path" from "everything rejected" -- printing only
            # the final count makes those two look identical, which is exactly how
            # this failed the first time.
            if not raw:
                for r in g.roots:
                    if not os.path.isdir(r):
                        print(f"        PATH NOT FOUND: {r}")
                        parent = os.path.dirname(r.rstrip("/")) or "/"
                        if os.path.isdir(parent):
                            sib = sorted(os.listdir(parent))[:12]
                            print(f"        {parent} contains: {sib}")
                    else:
                        top = sorted(os.listdir(r))[:12]
                        print(f"        dir exists but no audio under it; contains: {top}")
    if not any(g.files for g in groups):
        raise SystemExit(
            "\nNo usable files in ANY group.\n"
            "  * '0 found'    -> the roots in your --groups JSON are wrong. Kaggle mounts\n"
            "                    datasets at /kaggle/input/<slug>/... ; run\n"
            "                    `find /kaggle/input -maxdepth 3 -type d | head -50`\n"
            "                    and fix the paths.\n"
            "  * 'N found, 0 usable' -> the provenance filter rejected everything. Check\n"
            "                    the corpus is genuinely lossless, or re-run with\n"
            "                    --skip-provenance to confirm that is the cause.\n")

    ds = CodecDataset(groups, args.seg, args.samples_per_epoch,
                      codecs=[c.strip() for c in args.codecs.split(",") if c.strip()],
                      stratify=not args.no_stratify)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(), drop_last=True,
                    persistent_workers=args.workers > 0,
                    prefetch_factor=(4 if args.workers > 0 else None))

    model = build_model(args, device)
    if rank == 0:
        verify_init_equivalence(model, device, args.feature_dim, args.layer)
    if ddp:
        any_off = (args.no_film or args.no_logfreq or args.no_refine or args.no_gate)
        model = DDP(model, device_ids=[rank],
                    find_unused_parameters=any_off,      # costly; only when needed
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = make_scaler(use_amp)
    floors = [0.0] * len(groups) if args.no_floors else [g.floor for g in groups]
    dro = FloorCorrectedGroupDRO(len(groups), floors, eta=args.dro_eta, device=device)
    doro = None if args.doro_eps <= 0 else DoroMask(
        args.doro_eps, warmup=args.doro_warmup, device=device)
    swad = SWAD(model, start_frac=args.swad_start, every=args.swad_every)
    degsampler = DegradationSampler(temp=args.deg_temp)
    ds.set_deg_logits(degsampler.logits())
    # steps available in the wall-clock budget; only used to place the SWAD window
    total_steps = args.total_steps if args.total_steps > 0 else \
        int(args.max_hours * 3600 / max(args.sec_per_step, 1e-6))

    step0 = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        (model.module if ddp else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        dro.q = torch.tensor(ck["dro"]["q"], device=device)
        dro.L = torch.tensor(ck["dro"]["L"], device=device)
        step0 = ck["step"]
        if rank == 0:
            print(f"[resume] from step {step0}")

    t0 = time.time()
    last_ck = t0
    step = step0
    model.train()
    stop = False
    while not stop:
        for clean, deg, gidx, cidx, cond, cut in dl:
            clean = clean.to(device, non_blocking=True)
            deg = deg.to(device, non_blocking=True)
            gidx = gidx.to(device)
            cond = cond.to(device, dtype=torch.float32)

            with autocast_ctx(use_amp):
                est = model(deg, cond=cond)
                # Restoration Gain Ratio: error RELATIVE TO THE DAMAGE.
                # identity == exactly 1.0, so any value < 1 is genuine restoration
                # and the number means the same thing at every bitrate. A plain
                # MR-STFT loss is dominated by the loud low band the codec never
                # touched, so most of it measures content the model never changed.
                if args.loss == "rgr":
                    per = restoration_gain(est, clean, deg)
                else:
                    per = mrstft_per_sample(est, clean)

                # --- composition order matters ---
                # 1. DORO trims outliers FIRST, so leaked transcodes cannot drive
                #    the worst-case objective.
                keep = doro(per) if doro is not None else torch.ones_like(per, dtype=torch.bool)
                pk, gk = per[keep], gidx[keep]

                # 2. Group DRO on the surviving samples (floor-corrected excess risk)
                loss = pk.mean() if args.no_dro else dro.update(pk, gk, ddp=ddp)

                # 3. V-REx pushes groups toward EQUAL risk (DRO only fixes the worst)
                if args.vrex > 0:
                    loss = loss + args.vrex * vrex_penalty(pk, gk, len(groups))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            step += 1

            # 4. adversarial degradation sampling: feed excess loss back to the
            #    (codec, bitrate) sampler -- Wasserstein DRO where it is well-scaled
            degsampler.observe(cidx[keep.cpu()], per.detach().float().cpu()[keep.cpu()])
            if step % args.deg_update == 0:
                if ddp and dist.is_initialized():
                    # keep every rank sampling from the SAME degradation
                    # distribution, else the two GPUs drift onto different curricula
                    L = degsampler.L.to(device)
                    dist.all_reduce(L, op=dist.ReduceOp.SUM)
                    degsampler.L = (L / dist.get_world_size()).cpu()
                ds.set_deg_logits(degsampler.logits())

            # 5. SWAD weight averaging over the trajectory (free OOD insurance)
            swad.maybe_update(model.module if ddp else model, step, total_steps)

            if args.smoke and step >= args.smoke_steps:
                if rank == 0:
                    drop = doro.dropped if doro is not None else 0
                    print(f"[smoke] {step} steps OK | loss={float(loss):.4f} "
                          f"({args.loss}; identity==1.0 for rgr)")
                    print(f"[smoke] group q      = {[round(x,3) for x in dro.q.tolist()]}")
                    print(f"[smoke] group L(ema) = {[round(x,3) for x in dro.L.tolist()]}")
                    print(f"[smoke] doro dropped = {drop}")
                    top = torch.topk(degsampler.L, min(4, len(DEG_CELLS)))
                    print("[smoke] hardest degradation cells: " +
                          ", ".join(f"{DEG_CELLS[i][0]}@{DEG_CELLS[i][1]}k={v:.3f}"
                                    for v, i in zip(top.values.tolist(), top.indices.tolist())))
                    print("[smoke] SWAD snapshots =", swad.n)
                    print("[smoke] PASS")
                return

            if rank == 0 and step % 50 == 0:
                el = (time.time() - t0) / 3600
                print(f"step {step:7d} | {el:5.2f}h | loss {float(loss):.4f} | "
                      f"q {[round(x,3) for x in dro.q.tolist()]}", flush=True)

            if rank == 0 and (time.time() - last_ck) / 60 >= args.ckpt_min:
                save_ckpt(args, model, ddp, opt, scaler, dro, degsampler, swad, step)
                last_ck = time.time()
                print(f"[ckpt] step {step}", flush=True)

            if (time.time() - t0) / 3600 >= args.max_hours:
                stop = True
                break

    if rank == 0:
        save_ckpt(args, model, ddp, opt, scaler, dro, degsampler, swad, step)
        print("[done] step", step, "| SWAD snapshots:", swad.n)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

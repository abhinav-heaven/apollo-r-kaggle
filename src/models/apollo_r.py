"""
Apollo-R — checkpoint-compatible hybrid.

DESIGN CONSTRAINT
  Every pretrained parameter name and shape from `Apollo` is preserved (this is a
  subclass, so `BN.*`, `net.*`, `output.*` load unchanged). Every addition is a NEW
  module with NEW names, loaded with strict=False, and ZERO-INITIALIZED at its
  output so that at step 0:

      ApolloR(x)  ==  Apollo(x)     exactly, bit for bit.

  Fine-tuning can therefore only earn improvements; it cannot start by destroying a
  good checkpoint. Every branch is individually ablatable by zeroing its scale.

WHAT EACH ADDITION BUYS DOWN (each traces to a measured result)

  A. FiLM bitrate conditioning
     Measured LAME cutoffs span 4377 Hz (24k) to 20225 Hz (320k) -- at 24 kbps 80%
     of the spectrum is destroyed. Apollo must infer this blind, and because of the
     `codec_simu` dict-mutation bug it never even saw varied bitrates in training.
     The bitrate is exact and free from the MP3 frame header.

  B. Log-frequency (pitch-equivariant) harmonic branch  <- the main architectural fix
     A harmonic series is at h*f0: MULTIPLICATIVE spacing. Apollo's 80 bands are
     uniform in LINEAR frequency, so the band-offset between consecutive harmonics
     depends on pitch (0.44 bands at f0=110 Hz vs 0.88 at f0=220 Hz). A fixed
     pattern over linear bands cannot be pitch-equivariant -- the network must
     relearn the harmonic template at every pitch. In log frequency the same series
     is a fixed pattern that merely translates, so convolution along log-frequency
     IS pitch-equivariant. This is the structural reason to expect failure on
     harmonically dense irregular spectra, which is what the real-audio test found:
     rolloff extrapolation collapsed on sitar and Carnatic vocal (1.03, 1.03) while
     a generator-sharing model won (0.76, 0.74).

  C. Full-resolution refinement branch
     Apollo's top band is 47 bins wide while every other band is 5 -- ~10x coarser
     exactly where restoration happens (its band table ends at 19.7 kHz). The band
     split cannot be changed without breaking the checkpoint, so this branch adds
     fine detail at full linear resolution in parallel.

  D. Per-band preservation gate
     Apollo regenerates all 442 bins from scratch, including the coded band that is
     already nearly correct, then is penalized for changing it. Initialized to
     pretrained behaviour; training discovers the blend per band.

NOT FIXED (impossible without breaking the checkpoint)
  * band split -> codec scalefactor bands (Bark-like, ends at exactly 16002 Hz)
  * 20 ms window -> MP3's 13.06 ms granule grid (win determines every shape)
  Both are from-scratch changes. Documented in APOLLO_R_PLAN.md.

STATUS: verified on CPU (torch 2.2.2). `verify_init_equivalence` in train_dro.py
reports max |ApolloR - Apollo| = 0.000e+00 at init -- exact, bit for bit. Getting
there required replacing a sigmoid gate (sigmoid(-6)=0.0025 is NOT zero, and a
logit of -20 would zero it only by killing the gradient) with a clamped gate that
is exactly 0 at init and still has gradient 1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .apollo import Apollo, RMSNorm


def _zero(module):
    """Zero a module's output params so the branch starts as a no-op."""
    if getattr(module, "weight", None) is not None:
        nn.init.zeros_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
    return module


class LogFreqHarmonicBranch(nn.Module):
    """Pitch-equivariant branch: resample to log-frequency, convolve, map back.

    Convolution along the log-frequency axis is equivariant to pitch shift, so one
    kernel covers every f0 instead of one per pitch.
    """

    def __init__(self, n_bins, sr, n_log=160, ch=32, layers=4, f_lo=40.0):
        super().__init__()
        self.n_bins, self.n_log = n_bins, n_log
        f_lin = np.linspace(0, sr / 2, n_bins)
        f_log = np.geomspace(f_lo, sr / 2, n_log)
        # fixed (parameter-free) resampling matrices, both directions
        self.register_buffer("to_log", torch.tensor(
            self._interp_matrix(f_lin, f_log), dtype=torch.float32), persistent=False)
        self.register_buffer("to_lin", torch.tensor(
            self._interp_matrix(f_log, f_lin), dtype=torch.float32), persistent=False)

        c = [1] + [ch] * layers
        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(nn.Sequential(
                nn.Conv2d(c[i], c[i + 1], (5, 5), padding=(2, 2)),
                nn.GroupNorm(4, c[i + 1]),
                nn.SiLU()))
        self.out = _zero(nn.Conv2d(ch, 1, (5, 5), padding=(2, 2)))

    @staticmethod
    def _interp_matrix(src, dst):
        """Linear interpolation matrix mapping values on `src` grid to `dst` grid."""
        M = np.zeros((len(dst), len(src)), dtype=np.float32)
        idx = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
        w = (dst - src[idx]) / np.maximum(src[idx + 1] - src[idx], 1e-9)
        w = np.clip(w, 0, 1)
        M[np.arange(len(dst)), idx] = 1 - w
        M[np.arange(len(dst)), idx + 1] = w
        return M

    def forward(self, mag):
        # mag: (B, F, T) magnitude spectrum -> multiplicative log-domain correction
        x = torch.log(mag.clamp_min(1e-8))
        x = torch.einsum("lf,bft->blt", self.to_log, x).unsqueeze(1)   # B,1,L,T
        x = x - x.mean(dim=(2, 3), keepdim=True)
        for c in self.convs:
            x = c(x)
        x = self.out(x).squeeze(1)                                      # B,L,T
        return torch.einsum("fl,blt->bft", self.to_lin, x)              # B,F,T (log gain)


class RefineBranch(nn.Module):
    """Full-resolution linear-frequency refinement.

    Apollo's top band is 47 bins vs 5 elsewhere; this restores fine detail there
    without altering the (checkpoint-fixed) band split.
    """

    def __init__(self, ch=32, layers=4):
        super().__init__()
        c = [2] + [ch] * layers
        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(nn.Sequential(
                nn.Conv2d(c[i], c[i + 1], (3, 3), padding=(1, 1)),
                nn.GroupNorm(4, c[i + 1]),
                nn.SiLU()))
        self.out = _zero(nn.Conv2d(ch, 2, (3, 3), padding=(1, 1)))

    def forward(self, ri):
        # ri: (B, 2, F, T) real/imag -> additive real/imag correction
        x = ri
        for c in self.convs:
            x = c(x)
        return self.out(x)


class ApolloR(Apollo):
    def __init__(self, sr, win, feature_dim, layer,
                 use_film=True, use_logfreq=True, use_refine=True, use_gate=True,
                 logfreq_ch=32, refine_ch=32, cond_dim=1):
        super().__init__(sr=sr, win=win, feature_dim=feature_dim, layer=layer)
        self.use_film, self.use_logfreq = use_film, use_logfreq
        self.use_refine, self.use_gate = use_refine, use_gate
        self.cond_dim = cond_dim

        # --- A. FiLM on the bottleneck ---
        # cond_dim > 1 carries [cutoff, log-bitrate, one-hot(codec)]. Codec identity
        # matters: Opus does not lowpass at all (measured ~20.4 kHz at every rate),
        # so cutoff alone would misdescribe the damage. `film` is a NEW module, so
        # widening its input keeps full checkpoint compatibility.
        if use_film:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, 64), nn.SiLU(), nn.Linear(64, 2 * feature_dim))
            _zero(self.film[-1])          # gamma=0,beta=0 -> scale 1, shift 0

        # --- B / C. parallel branches ---
        if use_logfreq:
            self.logfreq = LogFreqHarmonicBranch(self.enc_dim, sr, ch=logfreq_ch)
        if use_refine:
            self.refine = RefineBranch(ch=refine_ch)

        # --- D. per-band preservation gate ---
        # Zero-init CLAMPED gate, not sigmoid. sigmoid(-6)=0.0025 is not zero, so a
        # sigmoid gate blends 0.25% of the raw input at init and breaks exact
        # pretrained equivalence (caught by verify_init_equivalence: 1.2e-2).
        # Pushing the logit to -20 would zero it but also kill the gradient
        # (sigmoid' ~ 2e-9). clamp(0,1) from 0 gives BOTH exact identity and grad 1.
        if use_gate:
            self.band_gate = nn.Parameter(torch.zeros(self.nband))

    # ------------------------------------------------------------------ #
    def forward(self, input, cond=None, bitrate_cutoff=None):
        """cond: (B, cond_dim) conditioning -- see datas/codec_sim.condition_vector.
        `bitrate_cutoff` is the legacy scalar alias, kept for older call sites."""
        if cond is None:
            cond = bitrate_cutoff
        B, nch, nsample = input.shape
        BN = B * nch

        win = self.stft_window.type_as(input)
        spec_in = torch.stft(input.view(BN, nsample), n_fft=self.win,
                             hop_length=self.stride, window=win, return_complex=True)

        # ---- pretrained core, unchanged ----
        subband_feature = self.feature_extractor(input)            # BN, nband, N, T

        if self.use_film and cond is not None:
            g, b = self.film(cond.to(input.dtype).view(B, self.cond_dim)).chunk(2, -1)
            if g.shape[0] == B and nch > 1:                        # expand over channels
                g = g.repeat_interleave(nch, 0); b = b.repeat_interleave(nch, 0)
            subband_feature = subband_feature * (1 + g[:, None, :, None]) \
                + b[:, None, :, None]

        feature = self.net(subband_feature)

        est_spec = []
        for i in range(self.nband):
            ri = self.output[i](feature[:, i]).view(BN, 2, self.band_width[i], -1)
            est_spec.append(torch.complex(ri[:, 0], ri[:, 1]))
        est_spec = torch.cat(est_spec, 1)                          # BN, F, T

        # ---- D. per-band preservation gate (blend toward the coded input) ----
        if self.use_gate:
            g = self.band_gate.clamp(0.0, 1.0)
            gmap = torch.repeat_interleave(
                g, torch.tensor(self.band_width, device=g.device)).view(1, -1, 1)
            est_spec = est_spec + gmap * (spec_in - est_spec)   # g=0 -> exact identity

        # ---- B. pitch-equivariant harmonic correction (multiplicative) ----
        if self.use_logfreq:
            gain = torch.exp(self.logfreq(est_spec.abs()).clamp(-4, 4))
            est_spec = est_spec * gain

        # ---- C. full-resolution refinement (additive, real/imag) ----
        if self.use_refine:
            ri = torch.stack([est_spec.real, est_spec.imag], 1)     # BN,2,F,T
            d = self.refine(ri)
            est_spec = torch.complex(ri[:, 0] + d[:, 0], ri[:, 1] + d[:, 1])

        return torch.istft(est_spec, n_fft=self.win, hop_length=self.stride,
                           window=win, length=nsample).view(B, nch, -1)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_apollo_checkpoint(cls, state_dict, **kw):
        """Load pretrained Apollo weights; new branches stay at their zero init."""
        m = cls(**kw)
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        new = {n for n, _ in m.named_parameters()
               if n.split(".")[0] in ("film", "logfreq", "refine") or n == "band_gate"}
        stray = [k for k in missing if k not in new]
        assert not stray, f"pretrained weights missing from checkpoint: {stray[:8]}"
        assert not unexpected, f"unexpected keys: {unexpected[:8]}"
        return m


def _selftest():
    torch.manual_seed(0)
    kw = dict(sr=44100, win=20, feature_dim=64, layer=2)
    base = Apollo(**kw).eval()
    m = ApolloR(**kw).eval()
    m.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 1, 44100)
    with torch.no_grad():
        a, b = base(x), m(x, bitrate_cutoff=torch.tensor([[0.5], [0.5]]))
    d = (a - b).abs().max().item()
    print("max |ApolloR - Apollo| at init = %.3e" % d)
    assert d < 1e-4, "branches are NOT no-ops at init -- checkpoint safety broken"
    n_new = sum(p.numel() for n, p in m.named_parameters()
                if n.split(".")[0] in ("film", "logfreq", "refine") or n == "band_gate")
    print("new params: %.2f M   total: %.2f M" %
          (n_new / 1e6, sum(p.numel() for p in m.parameters()) / 1e6))
    print("SELFTEST PASS")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()

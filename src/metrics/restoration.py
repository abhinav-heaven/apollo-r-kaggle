"""
Evaluation and loss criteria that actually test RESTORATION.

WHY THE USUAL METRICS ARE NOT ENOUGH
  SI-SDR       strictly phase-sensitive; punishes inaudible phase error and is
               meaningless over synthesized content. Apollo optimizes a magnitude
               loss and then selects checkpoints on SI-SDR -- incoherent.
  PESQ         speech only, not 44.1 kHz music.
  ViSQOL       opaque, validated on narrow conditions, not differentiable.
  MR-STFT      ignores phase entirely; a model can match magnitude while emitting
               garbage phase.
  absolute error of any kind
               dominated by the loud low band -- where the codec did NO damage --
               so most of the number is measuring content the model never touched.

The core problem: none of these answer "did you RESTORE anything?", because none
of them are referenced to the damage. A model that outputs its input unchanged
scores respectably on all of them.

THE FIX: measure error RELATIVE TO THE DAMAGE.

    RGR = || est - clean ||  /  || degraded - clean ||

  RGR = 1.0  -> exactly as good as doing nothing (the identity map)
  RGR < 1.0  -> genuine restoration, by that factor
  RGR > 1.0  -> the model made the audio WORSE than the codec left it

  It is scale-free across bitrate, content and loudness, and it automatically
  weights each sample by how much damage there was to undo -- samples the codec
  barely touched cannot dominate. As a training loss it is the damage-weighting
  we want, obtained for free.

  Crucially it is FALSIFIABLE and not hand-wavy: the identity solution has a known
  exact score, so "better than 1.0" is a real claim rather than a vibe.
"""
import torch

SR = 44100
WINS = [256, 512, 1024, 2048]
_W = {}


def _hann(n, device):
    k = (n, str(device))
    if k not in _W:
        _W[k] = torch.hann_window(n, device=device)
    return _W[k]


def _spec(x, n):
    return torch.stft(x.reshape(-1, x.shape[-1]).float(), n, n // 2,
                      window=_hann(n, x.device), return_complex=True)


# --------------------------------------------------------------------------- #
# 1. Restoration Gain Ratio  (metric AND loss)
# --------------------------------------------------------------------------- #
def restoration_gain_regime(est, clean, degraded, cutoff_hz, n=2048, eps=1e-12):
    """RGR restricted to BELOW and ABOVE the cutoff, returned separately.

    MEASURED FLAW IN THE GLOBAL VERSION: `restoration_gain` is energy-weighted, and
    the above-cutoff band carries a tiny share of total energy. On real audio the
    rolloff baseline scored a global RGR of 1.000 -- "no restoration" -- while its
    per-regime numbers showed it clearly helping above the cutoff (0.70-0.94 vs the
    codec's 1.0). The loud low band, which the codec barely damaged, swamps the
    ratio. That is the SAME energy-weighting flaw that makes absolute error
    useless here, so the global number must never be reported alone.

    Returns (rgr_below, rgr_above); 1.0 == identity in each regime, NaN where the
    reference carries no energy in that regime.
    """
    E, C, D = _spec(est, n), _spec(clean, n), _spec(degraded, n)
    f = torch.fft.rfftfreq(n, 1 / SR).to(est.device)
    total = C.abs().pow(2).sum()

    def band(m):
        if m.sum() == 0:
            return float("nan")
        cm = C[:, m].abs()
        if float(cm.pow(2).sum()) < 1e-4 * float(total):
            return float("nan")          # nothing there to restore
        num = (E[:, m].abs() - cm).pow(2).sum() + 0.5 * (E[:, m] - C[:, m]).abs().pow(2).sum()
        den = (D[:, m].abs() - cm).pow(2).sum() + 0.5 * (D[:, m] - C[:, m]).abs().pow(2).sum()
        return float((num / (den + eps)).sqrt())

    return band(f < cutoff_hz), band(f >= cutoff_hz)


def restoration_gain(est, clean, degraded, wins=WINS, eps=1e-8, complex_w=0.5):
    """RGR per sample. 1.0 == identity, <1 == restored, >1 == damaged further.

    Includes a complex (phase-aware) term: magnitude-only objectives let a model
    win while emitting wrong phase, which is precisely Apollo's train/select
    mismatch.
    """
    B = est.shape[0]
    num = est.new_zeros(B * est.shape[1])
    den = est.new_zeros(B * est.shape[1])
    for n in wins:
        E, C, D = _spec(est, n), _spec(clean, n), _spec(degraded, n)
        em, cm, dm = E.abs(), C.abs(), D.abs()
        num = num + (em - cm).pow(2).sum((1, 2)) + complex_w * (E - C).abs().pow(2).sum((1, 2))
        den = den + (dm - cm).pow(2).sum((1, 2)) + complex_w * (D - C).abs().pow(2).sum((1, 2))
    r = (num / (den + eps)).sqrt()
    return r.view(B, -1).mean(1)


# --------------------------------------------------------------------------- #
# 2. Null test -- the strongest "are you restoring or just styling?" check
# --------------------------------------------------------------------------- #
def null_test(model_fn, clean, cond=None):
    """Feed ALREADY-LOSSLESS audio. A true inverse-of-degradation must be ~identity.

    A model that "restores" undamaged audio has not learned the inverse of the
    codec -- it has learned a style it applies unconditionally. Almost nobody runs
    this, it needs no reference beyond the input, and it is decisive.

    Returns relative change; 0 = perfect pass-through.
    """
    with torch.no_grad():
        out = model_fn(clean) if cond is None else model_fn(clean, cond)
    return float((out - clean).pow(2).sum().sqrt() / (clean.pow(2).sum().sqrt() + 1e-9))


# --------------------------------------------------------------------------- #
# 3. Hallucination / "pleasant sizzle" detector
# --------------------------------------------------------------------------- #
def energy_bias(est, clean, cutoff_hz, n=2048):
    """Signed high-band energy bias in dB: positive = model ADDED energy.

    Generative restorers reliably win listening tests by adding flattering high
    frequencies that are not in the master. A magnitude error cannot distinguish
    "added 3 dB of sizzle" from "missed 3 dB of real content", so it must be
    reported separately and signed.
    """
    E, C = _spec(est, n).abs(), _spec(clean, n).abs()
    f = torch.fft.rfftfreq(n, 1 / SR).to(est.device)
    hi = f >= cutoff_hz
    if hi.sum() == 0:
        return float("nan")
    e = E[:, hi].pow(2).sum()
    c = C[:, hi].pow(2).sum()
    if float(c) <= 0:
        return float("nan")
    return float(10 * torch.log10(e / c + 1e-20))


# --------------------------------------------------------------------------- #
# 4. Re-encode consistency -- REFERENCE-FREE and falsifiable
# --------------------------------------------------------------------------- #
def reencode_consistency(est, observed_degraded, codec, bitrate, simu_fn, cutoff_hz):
    """Re-encode the restoration; it must reproduce the observed file BELOW cutoff.

    Needs NO clean reference, so it works on real-world files where no master
    exists. A model that invents content in the CODED band provably fails this:
    that band is constrained by the bitstream, so any deviation is hallucination
    rather than restoration.

    Returns relative deviation below the cutoff; lower is better, 0 is perfect.
    """
    with torch.no_grad():
        re, _ = simu_fn(est.detach().cpu(), SR, codec, bitrate)
    re = re[:, :observed_degraded.shape[-1]].to(est.device)
    if re.shape[-1] < observed_degraded.shape[-1]:
        re = torch.nn.functional.pad(re, (0, observed_degraded.shape[-1] - re.shape[-1]))
    n = 2048
    R, O = _spec(re, n).abs(), _spec(observed_degraded, n).abs()
    f = torch.fft.rfftfreq(n, 1 / SR).to(est.device)
    lo = f < cutoff_hz
    return float(((R[:, lo] - O[:, lo]).pow(2).sum() /
                  (O[:, lo].pow(2).sum() + 1e-12)).sqrt())


# --------------------------------------------------------------------------- #
# 5. Regime-split training loss
# --------------------------------------------------------------------------- #
def regime_split_loss(est, clean, degraded, cutoff_hz, n=2048, lam_hi=1.0):
    """Different estimators for different regimes -- required, not stylistic.

    Rate-distortion-perception theory says distortion and realism cannot both be
    optimized at a fixed rate, so the constrained and unconstrained bands need
    DIFFERENT objectives. Optimizing one blended loss over both is provably
    suboptimal for each.

      below cutoff : constrained by the bitstream -> complex (phase-aware) error,
                     i.e. aim at the conditional mean.
      above cutoff : unrecoverable pointwise (EXP 1) -> match the DISTRIBUTION of
                     log-magnitude (mean/std per band), not the exact values.
    """
    E, C, D = _spec(est, n), _spec(clean, n), _spec(degraded, n)
    f = torch.fft.rfftfreq(n, 1 / SR).to(est.device)
    lo, hi = f < cutoff_hz, f >= cutoff_hz

    lo_loss = est.new_zeros(())
    if lo.any():
        num = (E[:, lo] - C[:, lo]).abs().pow(2).sum()
        den = (D[:, lo] - C[:, lo]).abs().pow(2).sum() + 1e-8
        lo_loss = num / den                      # RGR-style: identity == 1.0

    hi_loss = est.new_zeros(())
    if hi.any():
        def dist_stat(X):
            lx = torch.log(X[:, hi].abs() + 1e-6)
            return lx.mean(-1), lx.std(-1)
        em, es = dist_stat(E)
        cm, cs_ = dist_stat(C)
        dm, ds = dist_stat(D)
        num = (em - cm).pow(2).mean() + (es - cs_).pow(2).mean()
        # ANCHOR THE HIGH BAND THE SAME WAY. Un-normalized, identity scored 1.0
        # below the cutoff but 45.5 above it, so training would have been ~45:1
        # dominated by the band EXP 1 proved is unrecoverable -- the worst possible
        # allocation of a fixed compute budget. Now identity == 1.0 in BOTH regimes
        # and lam_hi is a real, interpretable trade-off knob.
        den = (dm - cm).pow(2).mean() + (ds - cs_).pow(2).mean() + 1e-8
        hi_loss = num / den
    return lo_loss + lam_hi * hi_loss, float(lo_loss), float(hi_loss)

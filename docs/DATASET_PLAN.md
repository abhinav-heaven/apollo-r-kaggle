# Audio-agnostic dataset plan — and two measurement corrections

## The reframing that makes this easy

**Codec restoration is self-supervised: any lossless audio is a training pair.** Encode
it, and you have (damaged, clean). Apollo uses MUSDB18-HQ/MoisesDB *stems* only for its
stem-mixing augmentation — the task itself never needs stems, or music.

So the corpus can be anything, which is exactly what "works on any audio file" requires.
Apollo is trained on Western pop/rock stems; that is a *choice*, and it is the reason it
should be expected to generalize poorly off-distribution.

## Recommended mix (all lossless, all ≥44.1 kHz)

| corpus | content | format | size | role |
|---|---|---|---|---|
| **FSD50K** | 51k clips, 108.3 h — speech, music, animals, machines, environment | **PCM 16-bit 44.1 kHz** | ~50 GB | **the audio-agnostic backbone** |
| MUSDB18-HQ | 150 music tracks, stems | WAV 44.1 k | ~30 GB | music, matches Apollo's domain |
| VCTK | 110 speakers | WAV **48 kHz** | ~10 GB | full-band speech |
| Saraga (CompMusic) | Carnatic/Hindustani | mixed | varies | non-Western — the ablation that matters |

Prefer **Kaggle-hosted copies**: attached datasets mount instantly, while a 30–50 GB
download can consume a quarter of a 12 h session.

## ⚠ Never use FMA as ground truth

The Free Music Archive is the obvious "big music dataset" to reach for. **It is
distributed as MP3 at ~263 kbps average.** Training on it teaches the model to restore
*toward already-codec-damaged targets* — a silent, catastrophic failure with no error
message. The same applies to MTG-Jamendo (MP3), AudioSet (YouTube-sourced), and any
crowd-sourced "WAV" of unknown origin.

---

## Correction 1 — a WAV file is not proof of lossless provenance

Audited the corpus actually downloaded for the real-audio test. Energy in 17.5–21 kHz
relative to 10–15 kHz, decoded to **float** (16-bit requantization noise otherwise fills
the gap and hides the hole):

| file | ratio | verdict |
|---|---:|---|
| cello_hi / cello_lo | −6.6 / −5.8 dB | keep |
| flute (all) | −10.6 to −12.5 dB | keep |
| carnatic | −21.1 dB | keep |
| **sitar (24-bit "lossless" download)** | **−56.5 dB** | **REJECT — lossy origin** |

The sitar file's content stops at 16505 Hz — essentially the 128 kbps cutoff. **It was
transcoded from a 128 kbps MP3.**

This retroactively explains the unexplained sitar anomaly in `REAL_AUDIO_RESULTS.md`
(errors of 12.2 and 40.4 at 128 kbps): there was no content above the cutoff to restore,
so the relative error exploded. That result should be treated as invalid, not merely odd.

**Honest limits of this filter.** Controls show it catches *severe* cases and **misses
mild ones**: flute deliberately transcoded at 96/128/192/320 k reads −12.7 to −14.9 dB
against −12.5 dB for the true original — indistinguishable. So use it as a conservative
reject for gross cases, and otherwise **prefer corpora with documented provenance over
crowd-sourced WAV**. FSD50K, MUSDB18-HQ and VCTK all qualify; Wikimedia Commons does not.

## Correction 2 — LAME's cutoff is content-dependent, not a constant

My earlier table came from a **white-noise** probe. Measured on real flute at the same
bitrate:

| | 128 kbps cutoff |
|---|---:|
| white noise | 16780 Hz |
| flute | **19821 Hz** |

White noise is maximum-entropy: the encoder must lowpass hard to afford bits. Sparse
content leaves bits to spare and more survives. **The earlier table is a worst-case lower
bound, not a per-file constant.** Bitrate remains the strongest single predictor and is
still the right FiLM conditioning signal, but the model must not be told a hard cutoff —
condition on bitrate and let it infer the rest.

---

## Compute allocation matters more than corpus size

10 h on 2×T4 ≈ 122–244k stereo 3 s samples ≈ **100–200 hours of audio seen**. FSD50K
alone is 108 h. **Data volume is not the bottleneck — diversity and sampling are.**

Two free wins:

1. **Do not sample bitrates uniformly.** At 24 kbps ~80% of the spectrum is destroyed; at
   320 kbps ~7%. Uniform sampling spends ⅛ of the budget where there is almost nothing to
   learn. Weight toward 24–96 kbps.
2. **Energy-filter crops.** Apollo takes random crops with no energy check; silent crops
   teach nothing. FSD50K in particular has many short/quiet clips.

---

## What "beats SOTA" can honestly mean here

**It will not beat Apollo on Western music at bitrates Apollo already handles.** A 10 h
fine-tune cannot out-train the original run on its own distribution.

Where a win is plausible, and testable:

1. **Bitrates Apollo never saw.** The `codec_simu` dict-mutation bug froze the bitrate per
   worker — verified, 8 successive calls returned `[96]*8`. The released model was trained
   on ~8 fixed bitrates, not the advertised 24–128 kbps continuum. **Fine-tuning with
   correct randomization should beat it measurably at unseen bitrates.**
2. **Non-Western instruments** — Apollo never saw them; our only surviving real-data
   finding says this is where linear-band models fail.
3. **Non-music audio** — speech, environmental, mechanical. Apollo saw none.

(1)+(2)+(3) *is* "audio agnostic." That is a real, defensible contribution — "codec
restoration models are silently overfit to Western music at a handful of bitrates, here is
the evidence and the fix" — and it does not require out-computing anyone.

**Report per-domain and per-bitrate, never a single global number**, and benchmark against
the rolloff baseline (0.39–0.57 on regular spectra), which is stronger than the literature
implies.

#!/usr/bin/env bash
# Clone upstream Apollo (CC BY-SA 4.0, JusperLee) and overlay our additions.
# Upstream is NOT vendored here: it stays authoritative and correctly attributed.
set -euo pipefail
UP="${1:-./Apollo}"
[ -d "$UP" ] || git clone --depth 1 https://github.com/JusperLee/Apollo.git "$UP"
cp src/models/apollo_r.py "$UP/look2hear/models/"
cp src/datas/codec_sim.py "$UP/look2hear/datas/"
cp src/metrics/restoration.py "$UP/look2hear/metrics/"
cp train_dro.py eval_per_group.py discover_groups.py prepare_kaggle_dataset.py "$UP/"
# upstream bug fixes (see docs/): bitrate randomization was dead; roll() wrapped
python3 - "$UP" << 'PY'
import re,sys,pathlib
up=pathlib.Path(sys.argv[1])
p=up/"look2hear/datas/musdb_moisesdb_datamodule.py"; s=p.read_text()
if "options = dict(options" not in s:
    s=s.replace("def codec_simu(wav, sr=16000, options={'bitrate':'random','compression':'random', 'complexity':'random', 'vbr':'random'}):",
                "def codec_simu(wav, sr=16000, options=None):\n    # never mutate the caller's dict: resolving 'random' in place froze the\n    # bitrate at the first draw for the whole run (one rate per worker).\n    options = dict(options or {'bitrate':'random','compression':'random','complexity':'random','vbr':'random'})")
    s=s.replace("wav_encdec = torch.roll(wav_encdec, -tau[0], -1)",
                "shift = tau[0]  # shift-and-pad, not roll: roll wraps the tail to the head\n    if shift:\n        wav_encdec = torch.nn.functional.pad(wav_encdec, (0, shift))[..., shift:]")
    p.write_text(s); print("patched datamodule")
t=up/"test.py"; s=t.read_text()
if "GullFullband" in s: t.write_text(s.replace("GullFullband","Apollo")); print("patched test.py")
PY
echo "Apollo-R ready in $UP"

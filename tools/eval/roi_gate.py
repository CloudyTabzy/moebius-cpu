"""ROI decode gate: same seed and latents, ROI vs full decode, PSNR inside the mask.

    python tools/eval/roi_gate.py --manifest weights/eval/manifest.json \
        --full weights/eval/runs/full_exact --roi weights/eval/runs/roi_gate

`--roi` is an eval run of the same configuration as `--full` plus `--roi-decode true`.
Compares the final 8-bit outputs, so paste and rounding are included. Gate: >= 40 dB.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

GATE_DB = 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--roi", type=Path, required=True)
    args = ap.parse_args()
    worst = float("inf")
    for m in json.loads(args.manifest.read_text()):
        roi = args.roi / f"{m['id']}.png"
        if not roi.exists():
            continue
        a = np.array(Image.open(args.full / f"{m['id']}.png").convert("RGB"), dtype=np.float64)
        b = np.array(Image.open(roi).convert("RGB"), dtype=np.float64)
        inside = np.array(Image.open(m["mask"]).convert("L")) >= 128
        mse = np.mean((a[inside] - b[inside]) ** 2)
        db = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
        worst = min(worst, db)
        print(f"{m['id']:24s} {inside.mean() * 100:5.1f}% masked  {db:6.2f} dB")
    print(f"worst {worst:.2f} dB, gate {GATE_DB} dB: {'PASS' if worst >= GATE_DB else 'FAIL'}")


if __name__ == "__main__":
    main()

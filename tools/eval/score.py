"""Score eval runs against ground truth: LPIPS(alex), PSNR, median time.

    python tools/eval/score.py --manifest weights/eval/manifest.json --run weights/eval/runs/exact [--run ...]

Per run and subset (plus "all") prints mean LPIPS (full 512^2 image, inputs in
[-1, 1]), mean PSNR, n and median time from <run>/times.json. Writes
<run>/scores.json. With several runs, the LPIPS ratio to the first run is shown.
Only ids present in the run directory are scored (so --limit-per-subset runs work).
"""
import argparse
import json
import statistics
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def to_tensor(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float() / 127.5 - 1.0


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def score_run(run: Path, manifest, net):
    times_path = run / "times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}
    pairs = {}
    for m in manifest:
        out = run / f"{m['id']}.png"
        if not out.exists():
            continue
        gt, pred = load(m["image"]), load(out)
        assert gt.shape == pred.shape, (m["id"], gt.shape, pred.shape)
        with torch.no_grad():
            lp = float(net(to_tensor(pred), to_tensor(gt)).item())
        pairs[m["id"]] = {
            "subset": m["subset"],
            "lpips": lp,
            "psnr": psnr(pred, gt),
            "time": times.get(m["id"]),
        }
    summary = {}
    for name in ("scenes", "faces", "all"):
        rows = [v for v in pairs.values() if name == "all" or v["subset"] == name]
        if not rows:
            continue
        ts = [r["time"] for r in rows if r["time"] is not None]
        summary[name] = {
            "n": len(rows),
            "lpips": float(np.mean([r["lpips"] for r in rows])),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "median_time": statistics.median(ts) if ts else None,
        }
    res = {"summary": summary, "pairs": pairs}
    (run / "scores.json").write_text(json.dumps(res, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    net = lpips.LPIPS(net="alex", verbose=False).eval()
    base = None
    for run in args.run:
        summary = score_run(run, manifest, net)
        if base is None:
            base = summary
        print(f"== {run}")
        for name, s in summary.items():
            t = f"{s['median_time']:.2f}s" if s["median_time"] is not None else "n/a"
            ratio = ""
            if run != args.run[0] and name in base:
                ratio = f"  lpips/first {s['lpips'] / base[name]['lpips']:.3f}"
            print(f"  {name:6s} n={s['n']:3d}  LPIPS {s['lpips']:.4f}  PSNR {s['psnr']:.2f} dB  median time {t}{ratio}")


if __name__ == "__main__":
    main()

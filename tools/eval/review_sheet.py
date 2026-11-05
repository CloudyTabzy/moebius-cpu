"""Side-by-side review sheet: ground truth | mask | Exact | Fast.

    python tools/eval/review_sheet.py --manifest weights/eval/manifest.json \
        --exact weights/eval/runs/full_exact --fast weights/eval/runs/full_X \
        --out weights/eval/review_sheet.png

Takes every `--stride`th pair of each subset, so the selection does not depend on how
the results look. Two pairs per row: scenes on the left, faces on the right.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

TILE = 200


def tile(path, mode="RGB"):
    return Image.open(path).convert(mode).convert("RGB").resize((TILE, TILE), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--exact", type=Path, required=True)
    ap.add_argument("--fast", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cols = [[m for m in manifest if m["subset"] == s][:: args.stride] for s in ("scenes", "faces")]
    rows = max(len(c) for c in cols)
    sheet = Image.new("RGB", (8 * TILE + 20, rows * TILE), "white")
    for c, pairs in enumerate(cols):
        for r, m in enumerate(pairs):
            tiles = [tile(m["image"]), tile(m["mask"], "L"), tile(args.exact / f"{m['id']}.png"), tile(args.fast / f"{m['id']}.png")]
            for i, t in enumerate(tiles):
                sheet.paste(t, (c * (4 * TILE + 20) + i * TILE, r * TILE))
    sheet.save(args.out)
    print(args.out, sheet.size, sum(len(c) for c in cols), "pairs")


if __name__ == "__main__":
    main()

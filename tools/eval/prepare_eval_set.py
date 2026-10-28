"""Build the M8 eval set under weights/eval/: 60 COCO scenes + 60 FFHQ faces,
each with a deterministic random mask. Idempotent: existing files are kept.

    python tools/eval/prepare_eval_set.py [--out weights/eval]

Layout (paths in the manifest are relative to the repo root; run tools from there):
    raw/val2017.zip, raw/coco/*.jpg, raw/ffhq/NNNNN.png   downloads
    images/<id>.png (512^2 RGB), masks/<id>.png (512^2 L, 255 = repaint)
    manifest.json
"""
import argparse
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
N_PER_SUBSET = 60
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
FFHQ_URL = "https://huggingface.co/datasets/marcosv/ffhq-dataset/resolve/main/Part1/{:05d}.png"
MIN_SHORT_SIDE = 400
DILATE_PX = 6
COV_RANGE = (0.05, 0.25)


# ----------------------------------------------------------------- masks
def _ellipse(rng, d):
    cx, cy = rng.uniform(80, SIZE - 80, 2)
    rx, ry = rng.uniform(25, 110, 2)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)


def _rounded_rect(rng, mask):
    cx, cy = rng.uniform(80, SIZE - 80, 2)
    w, h = rng.uniform(50, 200, 2)
    r = rng.uniform(4, min(w, h) / 2)
    ang = rng.uniform(0, 180)
    side = int(math.hypot(w, h)) + 8
    tile = Image.new("L", (side, side), 0)
    x0, y0 = (side - w) / 2, (side - h) / 2
    ImageDraw.Draw(tile).rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=r, fill=255)
    tile = tile.rotate(ang, resample=Image.BICUBIC)
    mask.paste(255, (int(cx - side / 2), int(cy - side / 2)), tile)


def _stroke(rng, d):
    n = int(rng.integers(3, 7))  # segments
    width = float(rng.uniform(15, 45))
    x, y = rng.uniform(60, SIZE - 60, 2)
    ang = rng.uniform(0, 2 * math.pi)
    pts = [(x, y)]
    for _ in range(n):
        ang += rng.uniform(-0.9, 0.9)
        step = rng.uniform(30, 90)
        x = float(np.clip(x + step * math.cos(ang), 10, SIZE - 10))
        y = float(np.clip(y + step * math.sin(ang), 10, SIZE - 10))
        pts.append((x, y))
    d.line(pts, fill=255, width=int(width), joint="curve")
    r = width / 2
    for px, py in (pts[0], pts[-1]):
        d.ellipse([px - r, py - r, px + r, py + r], fill=255)


def make_mask(seed: int) -> np.ndarray:
    """Deterministic 512x512 uint8 mask (255 = repaint): union of 1-3 shapes,
    dilated by 6 px, coverage resampled into [5%, 25%]."""
    rng = np.random.default_rng(seed)
    while True:
        mask = Image.new("L", (SIZE, SIZE), 0)
        d = ImageDraw.Draw(mask)
        for _ in range(int(rng.integers(1, 4))):
            kind = int(rng.integers(0, 3))
            if kind == 0:
                _ellipse(rng, d)
            elif kind == 1:
                _rounded_rect(rng, mask)
            else:
                _stroke(rng, d)
        mask = mask.point(lambda v: 255 if v >= 128 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(2 * DILATE_PX + 1))  # square, 6 px each side
        arr = np.asarray(mask, dtype=np.uint8)
        cov = float((arr > 0).mean())
        if COV_RANGE[0] <= cov <= COV_RANGE[1]:
            return arr


# ------------------------------------------------------------ downloads
def download(url: str, dest: Path):
    """Streamed download with resume; atomic via .part."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "moebius-eval"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored Range
        with open(part, "ab" if have else "wb") as f:
            done = have
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if (done >> 20) % 50 == 0:
                    print(f"  {dest.name}: {done >> 20} MB", flush=True)
    part.replace(dest)


def square_512(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return img.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.LANCZOS)


def pick_coco(raw: Path) -> list[Path]:
    """First 60 val2017 files (sorted by name) with short side >= 400, extracted."""
    coco_dir = raw / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    chosen: list[Path] = []
    zpath = raw / "val2017.zip"
    cache = coco_dir / "selection.json"
    if cache.exists():
        names = json.loads(cache.read_text())
        if all((coco_dir / n).exists() for n in names):
            return [coco_dir / n for n in names]
    download(COCO_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".jpg"))
        for m in members:
            data = z.read(m)
            dest = coco_dir / Path(m).name
            dest.write_bytes(data)
            with Image.open(dest) as im:
                ok = min(im.size) >= MIN_SHORT_SIDE
            if ok:
                chosen.append(dest)
                if len(chosen) == N_PER_SUBSET:
                    break
            else:
                dest.unlink()
    cache.write_text(json.dumps([p.name for p in chosen]))
    return chosen


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("weights/eval"))
    out = ap.parse_args().out
    raw, img_dir, mask_dir = out / "raw", out / "images", out / "masks"
    for d in (raw, img_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, int, Path]] = []  # (subset, idx, file)
    for i, p in enumerate(pick_coco(raw)):
        sources.append(("scenes", i, p))
    for i in range(N_PER_SUBSET):
        p = raw / "ffhq" / f"{i:05d}.png"
        download(FFHQ_URL.format(i), p)
        sources.append(("faces", i, p))

    manifest = []
    for k, (subset, i, src) in enumerate(sources):
        pid = f"{'scene' if subset == 'scenes' else 'face'}_{i:03d}"
        ip, mp = img_dir / f"{pid}.png", mask_dir / f"{pid}.png"
        if not ip.exists():
            with Image.open(src) as im:
                square_512(im).save(ip)
        mask = make_mask(k)  # seed = pair index over the whole set (scenes 0..59, faces 60..119)
        if not mp.exists():
            Image.fromarray(mask, "L").save(mp)
        else:
            mask = np.asarray(Image.open(mp))
        manifest.append({
            "id": pid,
            "subset": subset,
            "image": ip.as_posix(),
            "mask": mp.as_posix(),
            "coverage": round(float((mask > 0).mean()), 5),
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    covs = np.array([m["coverage"] for m in manifest])
    print(f"{len(manifest)} pairs; coverage min {covs.min():.3f} mean {covs.mean():.3f} max {covs.max():.3f}")


if __name__ == "__main__":
    main()

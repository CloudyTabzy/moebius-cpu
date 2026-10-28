"""Convert upstream checkpoints to moebius-cpu safetensors (M1).

    python tools/convert_weights.py \
        --unet  weights/upstream/ft_places2/diffusion_pytorch_model.bin \
        --vae   weights/upstream/vae \
        --out   weights/ft_places2/ [--dtype f16|f32]

Writes `unet.safetensors` and `vae.safetensors` with metadata
`format = moebius-cpu/1` (src/weights.rs::FORMAT_VERSION), `dtype`, the
source checkpoint's SHA-256 (`source_sha256`) and, for the VAE, the
`scaling_factor` from config.json.

Transformations (the Rust side relies on every item; names are the upstream
names, UNet keeps its `diff_model.` / `embedding_layer.` prefixes):
  1. torch.load(map_location="cpu", mmap=True, weights_only=True); every name
     and shape is checked against docs/weights-manifest.txt (strict).
  2. Fold eval BatchNorms into the preceding conv/linear, in float64:
         scale = gamma / sqrt(var + 1e-5);   shift = beta - mean * scale
         w' = w * scale (per output row),    b' = b * scale + shift
       DWConv bn1 -> conv_dw, bn2 -> conv_pw; lambda norm_q -> to_q, norm_v ->
       to_v (cross-lambda norm_v is BatchNorm1d, still per channel).
     The norm tensors and num_batches_tracked are dropped, and the folded
     layers gain `.bias` tensors (conv_dw.bias, conv_pw.bias, to_q.bias,
     to_v.bias).
  3. Layouts: 1x1 conv / linear [o, i, (1, 1)] -> [o, i]; dense 3x3
     [o, i, 3, 3] -> [o, 3, 3, i]; depthwise [c, 1, 3, 3] -> [3, 3, c];
     self-lambda pos_conv.weight [k, 1, 1, 15, 15] -> [k, 225] (index
     ky*15 + kx); cross-lambda rel_pos_emb [N, 10, k, 1] -> [N, 10, k].
  4. With --dtype f16, GEMM weights (2-D linear/1x1 and 4-D dense 3x3) are
     stored f16, except tensors with max|w| > 65504 which stay f32 (and are
     reported). Everything else (norms, biases, depthwise, lambda tables,
     the token embedding) is f32.
  5. Cross-lambda constants are NOT precomputed here: they are cheap at load
     and keeping the raw tables keeps the format checkpoint-agnostic.

Golden tensors for parity tests come from tools/dump_golden.py.
"""
import argparse
import ast
import hashlib
import json
import os
import sys

import torch

FORMAT_VERSION = "moebius-cpu/1"
BN_EPS = 1e-5
F16_MAX = 65504.0
# BatchNorm module leaf -> the layer it is folded into (same parent module).
BN_TARGETS = {"bn1": "conv_dw", "bn2": "conv_pw", "norm_q": "to_q", "norm_v": "to_v"}
BN_KEYS = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "..", "docs", "weights-manifest.txt")


class Stats:
    """What the converter did; printed as the run summary."""

    def __init__(self):
        self.written = 0
        self.f16 = 0
        self.folded = 0
        self.prefold_over = []   # source tensors with max|w| > 65504 (before folding)
        self.overflow_f32 = []   # GEMM weights kept f32 because of overflow (after folding)

    def summary(self, label, dtype):
        lines = [
            f"[{label}] dtype={dtype}: {self.written} tensors written, {self.f16} f16, "
            f"{self.written - self.f16} f32, {self.folded} BatchNorms folded",
            f"[{label}] source tensors with max|w| > 65504 before folding: {len(self.prefold_over)}"
            + "".join(f"\n    {n}" for n in self.prefold_over),
            f"[{label}] GEMM weights kept f32 for f16 overflow after folding: {len(self.overflow_f32)}"
            + "".join(f"\n    {n}" for n in self.overflow_f32),
        ]
        return "\n".join(lines)


def _split(name):
    mod, _, leaf = name.rpartition(".")
    return mod, leaf


def _join(mod, leaf):
    return f"{mod}.{leaf}" if mod else leaf


def _find_batchnorms(sd):
    """module path of every BatchNorm in `sd` -> target layer path it folds into."""
    out = {}
    for k in sd:
        mod, leaf = _split(k)
        if leaf != "running_var":
            continue
        parent, bn = _split(mod)
        if bn not in BN_TARGETS:
            raise ValueError(f"unexpected BatchNorm {mod!r}: do not know which layer it folds into")
        out[mod] = _join(parent, BN_TARGETS[bn])
    return out


def _fold_params(sd, bn):
    """float64 (scale, shift) of an eval-mode BatchNorm."""
    g = sd[bn + ".weight"].double()
    b = sd[bn + ".bias"].double()
    mean = sd[bn + ".running_mean"].double()
    var = sd[bn + ".running_var"].double()
    scale = g / torch.sqrt(var + BN_EPS)
    return scale, b - mean * scale


def _relayout(name, w):
    """Apply the Rust-side layout. Returns (tensor, is_gemm_weight)."""
    leaf = name.rpartition(".")[2]
    if leaf == "rel_pos_emb":
        assert w.dim() == 4 and w.shape[-1] == 1, (name, tuple(w.shape))
        return w[..., 0].contiguous(), False
    if leaf != "weight":
        return w.contiguous(), False
    if w.dim() == 5:
        assert w.shape[1:3] == (1, 1), (name, tuple(w.shape))
        return w.reshape(w.shape[0], -1).contiguous(), False        # pos_conv [k, 15*15]
    if w.dim() == 4:
        o, i, kh, kw = w.shape
        if (kh, kw) == (1, 1):
            return w.reshape(o, i).contiguous(), True              # 1x1 -> [o, i]
        if (kh, kw) == (3, 3):
            if i == 1:                                              # depthwise [c,1,3,3] -> [3,3,c]
                return w[:, 0].permute(1, 2, 0).contiguous(), False
            return w.permute(0, 2, 3, 1).contiguous(), True        # dense -> [o,3,3,i]
        raise ValueError(f"{name}: unsupported conv kernel {tuple(w.shape)}")
    if w.dim() == 2:
        return w.contiguous(), not name.endswith("embedding_layer.weight")
    return w.contiguous(), False


def _convert(sd, dtype, stats):
    if dtype not in ("f16", "f32"):
        raise ValueError(f"dtype must be f16 or f32, got {dtype!r}")
    stats = stats if stats is not None else Stats()
    bns = _find_batchnorms(sd)
    targets = {t: bn for bn, t in bns.items()}
    skip = {f"{bn}.{k}" for bn in bns for k in BN_KEYS}

    # Pre-fold scan: any source tensor whose magnitude would not fit f16.
    for k, t in sd.items():
        if t.is_floating_point() and t.numel() and float(t.abs().max()) > F16_MAX:
            stats.prefold_over.append(k)

    out = {}
    for k in sd:
        if k in skip:
            continue
        mod, leaf = _split(k)
        t = sd[k]
        if mod in targets:                      # conv/linear that absorbs a BatchNorm
            if leaf == "bias":
                continue                        # merged with the weight below
            assert leaf == "weight", k
            scale, shift = _fold_params(sd, targets[mod])
            w = t.double() * scale.reshape(-1, *([1] * (t.dim() - 1)))
            old_b = sd.get(mod + ".bias")
            b = shift if old_b is None else old_b.double() * scale + shift
            out[mod + ".bias"] = b.float().contiguous()
            stats.folded += 1
            w, gemm = _relayout(k, w.float())
        else:
            w, gemm = _relayout(k, t.float() if t.is_floating_point() else t)
        if gemm and dtype == "f16":
            if float(w.abs().max()) > F16_MAX:
                stats.overflow_f32.append(k)
            else:
                w = w.half()
                stats.f16 += 1
        out[k] = w
    stats.written += len(out)
    return out


def convert_unet(sd, dtype, stats=None):
    """Upstream UNet (+ token embedding) state dict -> moebius-cpu tensors."""
    return _convert(sd, dtype, stats)


def convert_vae(sd, dtype, stats=None):
    """Upstream diffusers AutoencoderKL state dict (any float dtype) -> tensors.

    The VAE has no BatchNorms, so this is layout + dtype only; fp16 sources are
    widened to f32 before anything else."""
    return _convert(sd, dtype, stats)


# --------------------------------------------------------------------------- CLI


def read_manifest(path):
    """{section: {name: shape}} from docs/weights-manifest.txt."""
    sections, cur = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                cur = sections.setdefault(line.strip("[]"), {})
                continue
            name, shape = line.split("\t")
            cur[name] = tuple(ast.literal_eval(shape))
    return sections


def validate(sd, expected, label):
    got = {k: tuple(v.shape) for k, v in sd.items()}
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    bad = sorted(k for k in set(got) & set(expected) if got[k] != expected[k])
    if missing or extra or bad:
        msg = [f"{label}: checkpoint does not match docs/weights-manifest.txt"]
        msg += [f"  missing: {k} {expected[k]}" for k in missing[:20]]
        msg += [f"  unexpected: {k} {got[k]}" for k in extra[:20]]
        msg += [f"  shape: {k} got {got[k]} expected {expected[k]}" for k in bad[:20]]
        msg.append(f"  ({len(missing)} missing, {len(extra)} unexpected, {len(bad)} wrong shape)")
        raise SystemExit("\n".join(msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def _write(tensors, path, meta):
    from safetensors.torch import save_file

    save_file(tensors, path, metadata=meta)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unet", required=True, help="UNet diffusion_pytorch_model.bin")
    ap.add_argument("--vae", required=True, help="VAE directory (config.json + diffusion_pytorch_model.bin)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)

    manifest = read_manifest(args.manifest)
    vae_bin = os.path.join(args.vae, "diffusion_pytorch_model.bin")
    with open(os.path.join(args.vae, "config.json"), encoding="utf-8") as f:
        scaling = json.load(f).get("scaling_factor")
    if scaling is None:
        raise SystemExit("VAE config.json has no scaling_factor")
    os.makedirs(args.out, exist_ok=True)

    jobs = [
        ("unet", args.unet, convert_unet, {}),
        ("vae", vae_bin, convert_vae, {"scaling_factor": repr(float(scaling))}),
    ]
    for label, path, fn, extra in jobs:
        sd = _load(path)
        validate(sd, manifest[label], label)
        stats = Stats()
        tensors = fn(sd, args.dtype, stats)
        del sd
        meta = {"format": FORMAT_VERSION, "dtype": args.dtype, "source_sha256": sha256_file(path), **extra}
        print(stats.summary(label, args.dtype))
        _write(tensors, os.path.join(args.out, f"{label}.safetensors"), meta)
        del tensors


if __name__ == "__main__":
    sys.exit(main())

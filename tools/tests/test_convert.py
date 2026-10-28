"""Tests for tools/convert_weights.py against the upstream modules."""
import importlib
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
MOE = os.environ.get("MOEBIUS_SRC", os.path.join(HERE, "..", "..", "..", "Moebius"))
sys.path.insert(0, MOE)
# Skip model_lib/__init__.py (needs CUDA-only flash-linear-attention), as
# tools/count_flops.py does.
_pkg = types.ModuleType("model_lib")
_pkg.__path__ = [os.path.join(MOE, "model_lib")]
sys.modules.setdefault("model_lib", _pkg)

import convert_weights as cw  # noqa: E402

lam = importlib.import_module("model_lib.nets.layers.λ.vanillaλ")
from model_lib.nets.layers._efficientnet_blocks import DepthwiseSeparableConv  # noqa: E402

torch.manual_seed(0)


def randomize_bn(module):
    for m in module.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            n = m.num_features
            m.weight.data = torch.rand(n) + 0.5
            m.bias.data = torch.randn(n)
            m.running_mean = torch.randn(n)
            m.running_var = torch.rand(n) + 0.5
    return module.eval()


def sd_of(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


@pytest.mark.parametrize("cin,cout", [(12, 16), (16, 16)])
def test_dwconv_fold_matches_module(cin, cout):
    m = randomize_bn(DepthwiseSeparableConv(cin, cout, dw_kernel_size=3))
    out = cw.convert_unet(sd_of(m), "f32")
    assert not any(".bn" in k or k.startswith("bn") or "num_batches" in k for k in out)
    x = torch.randn(2, cin, 9, 11)
    dw = F.conv2d(x, out["conv_dw.weight"].permute(2, 0, 1).unsqueeze(1), out["conv_dw.bias"],
                  padding=1, groups=cin)
    h = F.relu(dw)
    y = F.conv2d(h, out["conv_pw.weight"][:, :, None, None], out["conv_pw.bias"])
    if cin == cout:  # skip connection
        assert m.has_skip
        y = y + x
    assert (y - m(x)).abs().max() < 1e-5


def test_self_lambda_fold():
    m = randomize_bn(lam.MultiQuerySelfLambda(dim=32, dim_k=4, heads=8, r=15, n=8))
    out = cw.convert_unet(sd_of(m), "f32")
    x = torch.randn(2, 32, 6, 7)
    for conv, norm in (("to_q", m.norm_q), ("to_v", m.norm_v)):
        ref = norm(getattr(m, conv)(x))
        got = F.conv2d(x, out[f"{conv}.weight"][:, :, None, None], out[f"{conv}.bias"])
        assert (ref - got).abs().max() < 1e-5
    assert tuple(out["pos_conv.weight"].shape) == (4, 225)
    # index ky*15 + kx
    w = m.pos_conv.weight.detach()
    assert torch.equal(out["pos_conv.weight"][:, 3 * 15 + 5], w[:, 0, 0, 3, 5])
    assert torch.equal(out["pos_conv.bias"], m.pos_conv.bias.detach())
    assert tuple(out["to_k.weight"].shape) == (4, 32)
    assert not any("norm_" in k for k in out)


def test_cross_lambda_fold():
    m = randomize_bn(lam.MultiQueryCrossLambda(dim=32, dim_k=4, dim_cross=24, n=4, m=10, heads=8))
    out = cw.convert_unet(sd_of(m), "f32")
    x = torch.randn(2, 32, 4, 4)
    ref = m.norm_q(m.to_q(x))
    got = F.conv2d(x, out["to_q.weight"][:, :, None, None], out["to_q.bias"])
    assert (ref - got).abs().max() < 1e-5
    h = torch.randn(2, 10, 24)  # [b, l, c]; BatchNorm1d runs over channels
    ref = m.norm_v(m.to_v(h).transpose(1, 2)).transpose(1, 2)
    got = F.linear(h, out["to_v.weight"], out["to_v.bias"])
    assert (ref - got).abs().max() < 1e-5
    assert tuple(out["rel_pos_emb"].shape) == (16, 10, 4)
    assert torch.equal(out["rel_pos_emb"], m.rel_pos_emb.detach()[..., 0])
    assert tuple(out["to_k.weight"].shape) == (4, 24)


def test_layouts():
    dense = torch.randn(5, 3, 3, 3)
    dw = torch.randn(7, 1, 3, 3)
    pw = torch.randn(6, 4, 1, 1)
    out = cw.convert_vae({"a.weight": dense, "b.weight": pw, "c.conv.weight": dw}, "f32")
    a = out["a.weight"]
    assert tuple(a.shape) == (5, 3, 3, 3) and a.is_contiguous()
    for o in range(5):
        for i in range(3):
            for ky in range(3):
                for kx in range(3):
                    assert a[o, ky, kx, i] == dense[o, i, ky, kx]
    d = out["c.conv.weight"]
    assert tuple(d.shape) == (3, 3, 7) and d.is_contiguous()
    for c in range(7):
        for ky in range(3):
            for kx in range(3):
                assert d[ky, kx, c] == dw[c, 0, ky, kx]
    assert tuple(out["b.weight"].shape) == (6, 4)
    assert torch.equal(out["b.weight"], pw[:, :, 0, 0])


def test_f16_overflow_kept_f32():
    big = torch.randn(8, 16)
    big[3, 5] = 7e4
    sd = {
        "big.weight": big,
        "ok.weight": torch.randn(8, 16),
        "conv.weight": torch.randn(4, 6, 3, 3),
        "ok.bias": torch.randn(8),
        "norm.weight": torch.randn(8),
        "dw.weight": torch.randn(8, 1, 3, 3),
        "embedding_layer.weight": torch.randn(20, 8),
    }
    stats = cw.Stats()
    out = cw.convert_unet(sd, "f16", stats)
    assert out["big.weight"].dtype == torch.float32
    assert out["big.weight"][3, 5] == 7e4
    assert out["ok.weight"].dtype == torch.float16
    assert out["conv.weight"].dtype == torch.float16
    for k in ("ok.bias", "norm.weight", "dw.weight", "embedding_layer.weight"):
        assert out[k].dtype == torch.float32, k
    assert stats.overflow_f32 == ["big.weight"]
    assert stats.prefold_over == ["big.weight"]
    assert stats.f16 == 2


def test_f32_mode_and_fp16_source():
    sd = {"w.weight": torch.randn(4, 4).half(), "b.bias": torch.randn(4).half()}
    out = cw.convert_vae(sd, "f32")
    assert all(v.dtype == torch.float32 for v in out.values())
    assert torch.equal(out["w.weight"], sd["w.weight"].float())
    out = cw.convert_vae(sd, "f16")
    assert out["w.weight"].dtype == torch.float16 and out["b.bias"].dtype == torch.float32

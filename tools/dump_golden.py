"""Dump golden tensors from the upstream PyTorch Moebius pipeline.

    python tools/dump_golden.py [--cases 0,1,...] [--out weights/golden]

For every test pair writes <out>/case{i}.safetensors (metadata
format=moebius-cpu-golden/1, all tensors NHWC f32) and <out>/case{i}.png (the
PyTorch result). Case 0 additionally holds every UNet / VAE / lambda-block
intermediate of the first denoising step. See the plan, Task 2, for the exact
tensor names; the Rust parity tests read them by name.

Needs torch, diffusers, accelerate, timm, einops, pyyaml, safetensors, Pillow,
numpy and a Moebius source checkout (default ../Moebius, override MOEBIUS_SRC).
"""
import argparse, builtins, gc, glob, math, os, sys, time, types

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOE = os.path.abspath(os.environ.get("MOEBIUS_SRC", os.path.join(ROOT, "..", "Moebius")))
sys.path.insert(0, MOE)
# Skip model_lib/__init__.py: it imports the PixelHacker teacher, which needs
# flash-linear-attention (CUDA only).
pkg = types.ModuleType("model_lib"); pkg.__path__ = [os.path.join(MOE, "model_lib")]
sys.modules["model_lib"] = pkg

UNET_CLS = "UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8"
GUIDANCE, NOISE_OFFSET, NUM_STEPS = 2.5, 0.0357, 20


def quiet(fn, *a, **k):
    """Call fn with print silenced (the UNet constructor prints per block)."""
    _print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        return fn(*a, **k)
    finally:
        builtins.print = _print


def build_pipeline():
    from diffusers import DDIMScheduler
    from diffusers.models import AutoencoderKL
    import model_lib.nets.unet_lambda_prune_lite as unet_mod
    import removal.v1_2.removal_model as rm
    from removal.v1_2.pipeline import RemovalSDXLPipeline_BatchMode

    # `from model_lib import *` imported nothing (stubbed package): make the
    # class name that build_removal_model eval()s resolvable.
    setattr(pkg, UNET_CLS, getattr(unet_mod, UNET_CLS))
    setattr(rm, UNET_CLS, getattr(unet_mod, UNET_CLS))

    model = quiet(rm.build_removal_model, os.path.join(MOE, "config/model_cfg/moebius.yaml"), 20)
    # load_removal_model, but with mmap so the 860 MB checkpoint is not read twice.
    ckpt = os.path.join(ROOT, "weights/upstream/ft_places2/diffusion_pytorch_model.bin")
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    print(model.load_state_dict(sd, strict=True))
    del sd

    vae = AutoencoderKL.from_pretrained(os.path.join(ROOT, "weights/upstream/vae"))
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              num_train_timesteps=1000, clip_sample=False)
    pipe = RemovalSDXLPipeline_BatchMode(removal_model=model, vae=vae, scheduler=scheduler,
                                         device="cpu", dtype=torch.float)
    assert next(pipe.vae.parameters()).dtype == torch.float32
    assert next(pipe.removal_model.parameters()).dtype == torch.float32
    return pipe


def first_tensor(args, kwargs):
    if args:
        return args[0]
    return next(iter(kwargs.values()))


def as_sample(x):
    """Module output -> the tensor (Transformer2DModelOutput / tuple / tensor)."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "sample"):
        return x.sample
    return x[0]


def nhwc(t):
    """NCHW -> NHWC; tokens [B, HW, C] -> [B, H, W, C] (row-major); else unchanged."""
    t = t.detach().float()
    if t.dim() == 4:
        return t.permute(0, 2, 3, 1).contiguous()
    if t.dim() == 3 and t.shape[1] in (64 * 64, 32 * 32, 16 * 16, 8 * 8):
        b, hw, c = t.shape
        h = math.isqrt(hw)
        assert h * h == hw
        return t.reshape(b, h, h, c).contiguous()
    return t.contiguous()


def add_unet_hooks(pipe, rec):
    """Forward hooks recording the first UNet call only (a name is never overwritten)."""
    rm = pipe.removal_model
    unet = rm.diff_model
    handles = []

    def hook(name, take_input):
        def fn(mod, args, kwargs, out):
            if name in rec:
                return
            t = first_tensor(args, kwargs) if take_input else as_sample(out)
            rec[name] = nhwc(t)
        return fn

    def reg(mod, name, take_input=False):
        handles.append(mod.register_forward_hook(hook(name, take_input), with_kwargs=True))

    reg(rm, "unet.in", take_input=True)
    reg(rm, "unet.out")
    reg(unet.conv_in, "unet.conv_in")
    reg(unet.time_embedding, "unet.temb")
    reg(unet.encoder_hid_proj, "unet.ehs")
    for kind, blocks in (("down", unet.down_blocks), ("up", unet.up_blocks)):
        for i, blk in enumerate(blocks):
            for j, r in enumerate(getattr(blk, "resnets", [])):
                reg(r, f"unet.{kind}.{i}.res.{j}")
            for j, a in enumerate(getattr(blk, "attentions", [])):
                reg(a, f"unet.{kind}.{i}.attn.{j}")
            reg(blk, f"unet.{kind}.{i}.out")  # tuple output -> hidden_states (after downsampler)
    tb = unet.down_blocks[0].attentions[0].transformer_blocks[0]
    for n in ("attn1", "attn2", "ff"):
        m = getattr(tb, n)
        reg(m, f"tf0.{n}.in", take_input=True)
        reg(m, f"tf0.{n}.out")
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", default=os.path.join(ROOT, "weights", "golden"))
    args = ap.parse_args()
    cases = [int(c) for c in args.cases.split(",")]
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    img_paths = sorted(glob.glob(os.path.join(MOE, "data", "images", "*.png")))
    msk_paths = sorted(glob.glob(os.path.join(MOE, "data", "masks", "*.png")))
    assert len(img_paths) == len(msk_paths) == 8, (len(img_paths), len(msk_paths))

    t0 = time.time()
    pipe = build_pipeline()
    import removal.v1_2.pipeline as pl
    orig_randn, orig_randn_like = torch.randn, torch.randn_like
    orig_encode, orig_decode = pl.encode_clean_latents, pipe.vae.decode
    sched = pipe.noise_scheduler
    orig_step = sched.step

    for ci in cases:
        tc = time.time()
        image = Image.open(img_paths[ci]).convert("RGB")
        mask = Image.open(msk_paths[ci]).convert("L")
        if mask.size != (512, 512):
            mask = mask.resize((512, 512), Image.NEAREST)
        assert image.size == (512, 512), image.size
        mask_thr = mask.point(lambda x: 0 if x < 255 / 2 else 255, "L")  # as pipeline._preprocess

        rec = {}
        draws, steps = [], []   # random draws in order; (t, step input, step output) per scheduler.step
        rec["input.image"] = torch.from_numpy(np.asarray(image).astype(np.float32))[None]
        rec["input.mask"] = torch.from_numpy(np.asarray(mask_thr).astype(np.float32))[None, :, :, None]

        def randn(*a, **k):
            r = orig_randn(*a, **k); draws.append(("randn", r.detach().clone())); return r

        def randn_like(*a, **k):
            r = orig_randn_like(*a, **k); draws.append(("randn_like", r.detach().clone())); return r

        def step(model_output, timestep, sample, *a, **k):
            r = orig_step(model_output, timestep, sample, *a, **k)
            steps.append((int(timestep.flatten()[0]), model_output.detach().clone(), r[0].detach().clone()))
            return r

        def encode(batch, vae, *a, **k):
            lat, mlat = orig_encode(batch, vae, *a, **k)
            if ci == 0:
                rec["vae.latents"], rec["vae.masked_latents"] = nhwc(lat), nhwc(mlat)
            return lat, mlat

        def decode(z, *a, **k):
            out = orig_decode(z, *a, **k)
            if ci == 0:
                rec["vae.decoded"] = nhwc((out.sample + 1) / 2)
            return out

        handles = add_unet_hooks(pipe, rec) if ci == 0 else []
        torch.randn, torch.randn_like = randn, randn_like
        pl.encode_clean_latents, pipe.vae.decode, sched.step = encode, decode, step
        try:
            out = pipe([image], [mask], image_size=512, guidance_scale=GUIDANCE, paste=True,
                       compensate=False, num_steps=NUM_STEPS, noise_offset=NOISE_OFFSET)[0]
        finally:
            torch.randn, torch.randn_like = orig_randn, orig_randn_like
            pl.encode_clean_latents = orig_encode
            del sched.step, pipe.vae.decode  # drop the instance overrides -> class methods
            for h in handles:
                h.remove()

        # --- random draws: VAE image, VAE masked, init noise, offset ---
        want = [("randn", (1, 4, 64, 64)), ("randn", (1, 4, 64, 64)),
                ("randn_like", (1, 4, 64, 64)), ("randn", (1, 4, 1, 1))]
        got = [(k, tuple(t.shape)) for k, t in draws]
        assert got == want, f"unexpected random draws: {got}"
        for name, (_, t) in zip(("noise.vae_img", "noise.vae_masked", "noise.init", "noise.offset"), draws):
            rec[name] = nhwc(t)

        # --- timesteps / latents ---
        ts = [s[0] for s in steps]
        assert ts == list(range(900, -1, -50)) and len(ts) == 19, ts
        rec["latents.final"] = nhwc(steps[-1][2])
        if ci == 0:
            rec["latents.step0"] = nhwc(steps[0][2])
            rec["unet.eps_cfg"] = nhwc(steps[0][1])
            o = rec["unet.out"]
            assert o.shape == (2, 64, 64, 4), o.shape
            recon = o[0:1] + GUIDANCE * (o[1:2] - o[0:1])
            err = (torch.linalg.norm(rec["unet.eps_cfg"] - recon) / torch.linalg.norm(recon)).item()
            print(f"  CFG sanity rel_l2 = {err:.3e}")
            assert err < 1e-6, err
            if rec["unet.temb"].shape[0] == 1:
                rec["unet.temb"] = rec["unet.temb"].repeat(2, 1)
            assert rec["unet.temb"].shape == (2, 1280), rec["unet.temb"].shape
            assert rec["unet.ehs"].shape == (2, 10, 768), rec["unet.ehs"].shape

        arr = np.asarray(out.convert("RGB"))
        assert arr.shape == (512, 512, 3), arr.shape
        rec["output.image"] = torch.from_numpy(arr.astype(np.float32))[None]
        out.save(os.path.join(args.out, f"case{ci}.png"))

        tensors = {k: v.float().contiguous() for k, v in rec.items()}
        path = os.path.join(args.out, f"case{ci}.safetensors")
        save_file(tensors, path, metadata={"format": "moebius-cpu-golden/1"})
        print(f"case {ci}: {os.path.basename(img_paths[ci])} + {os.path.basename(msk_paths[ci])}  "
              f"{time.time() - tc:.0f}s  {os.path.getsize(path) / 1e6:.1f} MB")
        print(f"  timesteps ({len(ts)}): {ts}")
        for n in ("noise.vae_img", "noise.vae_masked", "noise.init", "noise.offset"):
            print(f"  {n:18s} {list(rec[n].shape)}")
        if ci == 0:
            for k, v in rec.items():
                print(f"    {k:26s} {list(v.shape)}")
        del rec, tensors, draws, steps, out
        gc.collect()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

//! Layer / model parity against the upstream PyTorch pipeline.
//!
//! All tests need converted weights and golden dumps (`tools/convert_weights.py`,
//! `tools/dump_golden.py`), so they are `#[ignore]`d:
//! `cargo test --release --test parity -- --ignored`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use moebius_cpu::engine::{Scratch, Tensor};
use moebius_cpu::unet::blocks::{Downsample, DwResnet};
use moebius_cpu::unet::lambda::{CrossLambda, SelfLambda};
use moebius_cpu::unet::load::Loader;
use moebius_cpu::unet::transformer::MixTransformer;
use moebius_cpu::weights::Weights;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Golden dump of `case` (NHWC f32 tensors from the PyTorch pipeline).
pub fn golden(case: usize) -> Weights {
    let path = root().join(format!("weights/golden/case{case}.safetensors"));
    Weights::open_with_format(&path, "moebius-cpu-golden/1")
        .unwrap_or_else(|e| panic!("{e}; produce it with `python tools/dump_golden.py`"))
}

/// Converted UNet weights; `dtype` is "f16" or "f32".
pub fn model(dtype: &str) -> Weights {
    let dir = if dtype == "f32" { "ft_places2-f32" } else { "ft_places2" };
    let path = root().join(format!("weights/{dir}/unet.safetensors"));
    Weights::open(&path).unwrap_or_else(|e| panic!("{e}; produce it with `python tools/convert_weights.py --dtype {dtype}`"))
}

/// A tensor of any rank <= 4 as an NHWC `Tensor` (lower ranks are right-aligned
/// onto `[n, h, w, c]`, with leading axes of size 1).
pub fn tensor(w: &Weights, name: &str) -> Tensor {
    let shape = w.shape(name).unwrap_or_else(|| panic!("missing tensor {name}")).to_vec();
    assert!((1..=4).contains(&shape.len()), "{name}: rank {}", shape.len());
    let mut s = [1usize; 4];
    s[4 - shape.len()..].copy_from_slice(&shape);
    Tensor::from_vec(w.get_f32(name, &shape).unwrap(), s[0], s[1], s[2], s[3])
}

/// ||a - b|| / ||b||, accumulated in f64.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den).sqrt()
}

const BLOCK: &str = "diff_model.down_blocks.0.attentions.0.transformer_blocks.0";

fn ehs(g: &Weights) -> [Vec<f32>; 2] {
    let all = g.get_f32("unet.ehs", &[2, 10, 768]).unwrap();
    [all[..7680].to_vec(), all[7680..].to_vec()]
}

/// Runs self-λ on `tf0.attn1.in`; returns rel_l2 against `tf0.attn1.out`.
fn self_lambda_err(dtype: &str) -> f64 {
    let (m, g) = (model(dtype), golden(0));
    let layer = SelfLambda::load(&Loader::new(&m, BLOCK).sub("attn1"), 320).unwrap();
    let (x, want) = (tensor(&g, "tf0.attn1.in"), tensor(&g, "tf0.attn1.out"));
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    layer.forward(&x, &mut out, &mut s);
    assert_eq!(out.shape(), want.shape());
    let err = rel_l2(&out.data, &want.data);
    println!("self-lambda {dtype}: rel_l2 = {err:.3e}");
    err
}

/// Runs cross-λ on `tf0.attn2.in` with branches [0, 1]; returns (rel_l2, per-item rel_l2s).
fn cross_lambda_err(dtype: &str) -> (f64, [f64; 2]) {
    let (m, g) = (model(dtype), golden(0));
    let layer = CrossLambda::load(&Loader::new(&m, BLOCK).sub("attn2"), 320, &ehs(&g)).unwrap();
    let (x, want) = (tensor(&g, "tf0.attn2.in"), tensor(&g, "tf0.attn2.out"));
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    layer.forward(&x, &[0, 1], &mut out, &mut s);
    assert_eq!(out.shape(), want.shape());
    let err = rel_l2(&out.data, &want.data);
    let items = [0, 1].map(|i| rel_l2(out.item(i), want.item(i)));
    println!("cross-lambda {dtype}: rel_l2 = {err:.3e} (uncond {:.3e}, cond {:.3e})", items[0], items[1]);
    (err, items)
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn self_lambda_matches_pytorch_f32() {
    let err = self_lambda_err("f32");
    assert!(err <= 1e-4, "rel_l2 {err:e}");
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn cross_lambda_matches_pytorch_f32() {
    let (err, _) = cross_lambda_err("f32");
    assert!(err <= 1e-4, "rel_l2 {err:e}");

    // Batch items must not leak: identical inputs, different branches -> different outputs,
    // each matching its own golden half.
    let (m, g) = (model("f32"), golden(0));
    let layer = CrossLambda::load(&Loader::new(&m, BLOCK).sub("attn2"), 320, &ehs(&g)).unwrap();
    let (x, want) = (tensor(&g, "tf0.attn2.in"), tensor(&g, "tf0.attn2.out"));
    let mut same = Tensor::zeros(2, x.h, x.w, x.c);
    same.data[..x.item(0).len()].copy_from_slice(x.item(0));
    let half = same.data.len() / 2;
    same.data.copy_within(..half, half);
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    layer.forward(&same, &[0, 1], &mut out, &mut s);
    let differ = rel_l2(out.item(0), out.item(1));
    println!("cross-lambda identical inputs, branches [0,1]: rel_l2(item0, item1) = {differ:.3e}");
    assert!(differ > 1e-3, "branches did not differ: {differ:e}");
    assert!(rel_l2(out.item(0), want.item(0)) <= 1e-4);
    // Item 1 saw the cond branch on item-0's input, so compare against the same input run alone.
    layer.forward(&x, &[1, 0], &mut out, &mut s);
    assert!(rel_l2(out.item(1), want.item(1)) > 1e-3, "swapping branches had no effect");
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn lambdas_match_pytorch_f16() {
    let e = self_lambda_err("f16");
    assert!(e <= 2e-3, "self-lambda rel_l2 {e:e}");
    let (e, _) = cross_lambda_err("f16");
    assert!(e <= 2e-3, "cross-lambda rel_l2 {e:e}");
}

#[test]
#[ignore = "timing probe; run with --nocapture"]
fn lambda_timing() {
    let (m, g) = (model("f16"), golden(0));
    let l = Loader::new(&m, BLOCK);
    let selfl = SelfLambda::load(&l.sub("attn1"), 320).unwrap();
    let cross = CrossLambda::load(&l.sub("attn2"), 320, &ehs(&g)).unwrap();
    let (x1, x2) = (tensor(&g, "tf0.attn1.in"), tensor(&g, "tf0.attn2.in"));
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    for (name, f) in [
        ("self", Box::new(|o: &mut Tensor, s: &mut Scratch| selfl.forward(&x1, o, s)) as Box<dyn Fn(&mut Tensor, &mut Scratch)>),
        ("cross", Box::new(|o: &mut Tensor, s: &mut Scratch| cross.forward(&x2, &[0, 1], o, s))),
    ] {
        f(&mut out, &mut s);
        let t = Instant::now();
        for _ in 0..5 {
            f(&mut out, &mut s);
        }
        println!("{name}-lambda 2x64x64x320: {:.1} ms/forward", t.elapsed().as_secs_f64() * 200.0);
    }
}

// ---- Task 4: convolutional blocks and MixTransformer ----

const DOWN0: &str = "diff_model.down_blocks.0";
const DOWN1: &str = "diff_model.down_blocks.1";

/// `unet.temb` `[2, 1280]` as a `[2, 1, 1, 1280]` tensor.
fn temb(g: &Weights) -> Tensor {
    let t = tensor(g, "unet.temb");
    Tensor::from_vec(t.data, 2, 1, 1, 1280)
}

fn check(label: &str, got: &Tensor, want: &Tensor, tol: f64) -> f64 {
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    let err = rel_l2(&got.data, &want.data);
    println!("{label}: rel_l2 = {err:.3e}");
    assert!(err <= tol, "{label}: rel_l2 {err:e} > {tol:e}");
    err
}

fn resnet_checks(dtype: &str, tol: f64) {
    let (m, g) = (model(dtype), golden(0));
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    let t = temb(&g);

    let r = DwResnet::load(&Loader::new(&m, &format!("{DOWN0}.resnets.0")), 320, 320).unwrap();
    r.forward(&tensor(&g, "unet.conv_in"), &t, &mut out, &mut s);
    check(&format!("resnet down0.res0 (identity skip) {dtype}"), &out, &tensor(&g, "unet.down.0.res.0"), tol);

    let r = DwResnet::load(&Loader::new(&m, &format!("{DOWN1}.resnets.0")), 320, 640).unwrap();
    r.forward(&tensor(&g, "unet.down.0.out"), &t, &mut out, &mut s);
    check(&format!("resnet down1.res0 (320->640 shortcut) {dtype}"), &out, &tensor(&g, "unet.down.1.res.0"), tol);
}

fn transformer_checks(dtype: &str, tol: f64) {
    let (m, g) = (model(dtype), golden(0));
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    let tf = MixTransformer::load(&Loader::new(&m, &format!("{DOWN0}.attentions.0")), 320, &ehs(&g)).unwrap();

    tf.forward_ff(&tensor(&g, "tf0.ff.in"), &mut out, &mut s);
    check(&format!("ff {dtype}"), &out, &tensor(&g, "tf0.ff.out"), tol);

    tf.forward(&tensor(&g, "unet.down.0.res.0"), &[0, 1], &mut out, &mut s);
    check(&format!("transformer down0.attn0 {dtype}"), &out, &tensor(&g, "unet.down.0.attn.0"), tol);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn resnet_matches_pytorch() {
    resnet_checks("f32", 1e-4);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn resnet_matches_pytorch_f16() {
    resnet_checks("f16", 2e-3);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn transformer_matches_pytorch() {
    transformer_checks("f32", 1e-4);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn transformer_matches_pytorch_f16() {
    transformer_checks("f16", 2e-3);
}

fn downsample_check(dtype: &str, tol: f64) {
    let (m, g) = (model(dtype), golden(0));
    let d = Downsample::load(&Loader::new(&m, &format!("{DOWN0}.downsamplers.0")), 320).unwrap();
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    d.forward(&tensor(&g, "unet.down.0.attn.1"), &mut out, &mut s);
    check(&format!("downsample down0 {dtype}"), &out, &tensor(&g, "unet.down.0.out"), tol);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn downsample_matches_pytorch() {
    downsample_check("f32", 1e-4);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn downsample_matches_pytorch_f16() {
    downsample_check("f16", 2e-3);
}

#[test]
#[ignore = "needs weights/ and weights/golden/"]
fn lambda_batch_items_are_independent() {
    let (m, g) = (model("f32"), golden(0));
    let l = Loader::new(&m, BLOCK);
    let selfl = SelfLambda::load(&l.sub("attn1"), 320).unwrap();
    let cross = CrossLambda::load(&l.sub("attn2"), 320, &ehs(&g)).unwrap();
    let mut s = Scratch::default();

    // Two different items: the golden map, and the same map flipped left-right and halved.
    let x0 = tensor(&g, "tf0.attn1.in");
    let a = Tensor::from_vec(x0.item(0).to_vec(), 1, x0.h, x0.w, x0.c);
    let mut flipped = a.clone();
    for y in 0..a.h {
        for x in 0..a.w {
            let (src, dst) = ((y * a.w + x) * a.c, (y * a.w + (a.w - 1 - x)) * a.c);
            for c in 0..a.c {
                flipped.data[dst + c] = 0.5 * a.data[src + c];
            }
        }
    }
    let mut batch = Tensor::zeros(2, a.h, a.w, a.c);
    let half = a.data.len();
    batch.data[..half].copy_from_slice(&a.data);
    batch.data[half..].copy_from_slice(&flipped.data);
    assert!(rel_l2(&a.data, &flipped.data) > 0.1, "test items must differ");

    let alone = |layer: &dyn Fn(&Tensor, usize, &mut Tensor, &mut Scratch), x: &Tensor, br: usize, s: &mut Scratch| {
        let mut o = Tensor::default();
        layer(x, br, &mut o, s);
        o
    };
    let self_fn = |x: &Tensor, _br: usize, o: &mut Tensor, s: &mut Scratch| selfl.forward(x, o, s);
    let cross_fn = |x: &Tensor, br: usize, o: &mut Tensor, s: &mut Scratch| cross.forward(x, &[br], o, s);

    let mut both = Tensor::default();
    selfl.forward(&batch, &mut both, &mut s);
    for (i, x) in [&a, &flipped].into_iter().enumerate() {
        let solo = alone(&self_fn, x, 0, &mut s);
        let err = rel_l2(both.item(i), &solo.data);
        println!("self-lambda batch item {i} vs alone: rel_l2 = {err:.3e}");
        assert!(err <= 1e-6, "self-lambda item {i} leaks: {err:e}");
    }

    cross.forward(&batch, &[0, 1], &mut both, &mut s);
    for (i, x) in [&a, &flipped].into_iter().enumerate() {
        let solo = alone(&cross_fn, x, i, &mut s);
        let err = rel_l2(both.item(i), &solo.data);
        println!("cross-lambda batch item {i} vs alone: rel_l2 = {err:.3e}");
        assert!(err <= 1e-6, "cross-lambda item {i} leaks: {err:e}");
    }
}

#[test]
#[ignore = "timing probe; run with --nocapture"]
fn blocks_timing() {
    let (m, g) = (model("f16"), golden(0));
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    let t = temb(&g);
    let r = DwResnet::load(&Loader::new(&m, &format!("{DOWN0}.resnets.0")), 320, 320).unwrap();
    let tf = MixTransformer::load(&Loader::new(&m, &format!("{DOWN0}.attentions.0")), 320, &ehs(&g)).unwrap();
    let x = tensor(&g, "unet.conv_in");
    let x2 = tensor(&g, "unet.down.0.res.0");
    for (name, f) in [
        ("resnet", Box::new(|o: &mut Tensor, s: &mut Scratch| r.forward(&x, &t, o, s)) as Box<dyn Fn(&mut Tensor, &mut Scratch)>),
        ("transformer", Box::new(|o: &mut Tensor, s: &mut Scratch| tf.forward(&x2, &[0, 1], o, s))),
    ] {
        f(&mut out, &mut s);
        let t0 = Instant::now();
        for _ in 0..5 {
            f(&mut out, &mut s);
        }
        println!("{name} 2x64x64x320: {:.1} ms/forward", t0.elapsed().as_secs_f64() * 200.0);
    }
}

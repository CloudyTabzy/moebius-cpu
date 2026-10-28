//! Engine kernels vs PyTorch (`tools/dump_kernel_fixtures.py`).

use std::path::Path;

use moebius_cpu::engine::conv::{Conv1x1, Conv3x3};
use moebius_cpu::engine::depthwise::{Epilogue, depthwise3x3};
use moebius_cpu::engine::norm::{GroupNorm, layer_norm};
use moebius_cpu::engine::{Param, Scratch, Tensor};
use moebius_cpu::weights::Weights;

struct Fixtures(Weights);

impl Fixtures {
    fn open() -> Self {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels.safetensors");
        Self(Weights::open_with_format(&path, "moebius-cpu-fixture/1").expect("run tools/dump_kernel_fixtures.py"))
    }

    fn vec(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.0.get_f32(name, shape).unwrap()
    }

    fn nhwc(&self, name: &str, [n, h, w, c]: [usize; 4]) -> Tensor {
        Tensor::from_vec(self.vec(name, &[n, h, w, c]), n, h, w, c)
    }
}

#[track_caller]
fn assert_close(got: &Tensor, want: &Tensor, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "shape");
    let worst = got
        .data
        .iter()
        .zip(&want.data)
        .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
        .fold(0.0f32, f32::max);
    assert!(worst <= tol, "max scaled error {worst:e} > {tol:e}");
}

#[test]
fn conv1x1_bias_and_residual() {
    let f = Fixtures::open();
    let x = f.nhwc("c1.x", [2, 5, 7, 20]);
    let conv = Conv1x1 { c_in: 20, c_out: 36, weight: f.vec("c1.w", &[36, 20]).into(), bias: Some(f.vec("c1.b", &[36])) };
    let (mut out, mut s) = (Tensor::default(), Scratch::default());
    conv.forward(&x, &mut out, None, &mut s);
    assert_close(&out, &f.nhwc("c1.y", [2, 5, 7, 36]), 1e-5);
    let r = f.nhwc("c1.r", [2, 5, 7, 36]);
    conv.forward(&x, &mut out, Some(&r), &mut s);
    assert_close(&out, &f.nhwc("c1.yr", [2, 5, 7, 36]), 1e-5);
}

#[test]
fn conv1x1_f16_weights_match_rounded_f32() {
    let f = Fixtures::open();
    let x = f.nhwc("c1.x", [2, 5, 7, 20]);
    let w = f.vec("c1.w", &[36, 20]);
    let rounded: Vec<f32> = w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let (mut a, mut b, mut s) = (Tensor::default(), Tensor::default(), Scratch::default());
    Conv1x1 { c_in: 20, c_out: 36, weight: Param::f16_from(&w), bias: None }.forward(&x, &mut a, None, &mut s);
    Conv1x1 { c_in: 20, c_out: 36, weight: rounded.into(), bias: None }.forward(&x, &mut b, None, &mut s);
    assert_close(&a, &b, 1e-6);
}

#[test]
fn conv3x3_variants() {
    let f = Fixtures::open();
    let x = f.nhwc("c3.x", [2, 9, 11, 13]);
    let w = f.vec("c3.w", &[22, 3, 3, 13]);
    let b = f.vec("c3.b", &[22]);
    let mut s = Scratch::default();
    let cases = [
        ("c3.same", 1, [1, 1, 1, 1], false, [2, 9, 11, 22]),
        ("c3.s2p1", 2, [1, 1, 1, 1], false, [2, 5, 6, 22]),
        ("c3.s2vae", 2, [0, 0, 1, 1], false, [2, 4, 5, 22]),
        ("c3.up", 1, [1, 1, 1, 1], true, [2, 18, 22, 22]),
    ];
    for (name, stride, pad, upsample2x, shape) in cases {
        let conv = Conv3x3 { c_in: 13, c_out: 22, weight: w.clone().into(), bias: Some(b.clone()), stride, pad, upsample2x };
        let mut out = Tensor::default();
        conv.forward(&x, &mut out, &mut s);
        assert_close(&out, &f.nhwc(name, shape), 1e-5);
    }
}

#[test]
fn conv3x3_band_boundaries() {
    // More output pixels than one band holds (c_in = 1024 → 910-pixel bands).
    let (n, h, w, c) = (1, 40, 30, 1024);
    let x = Tensor::from_vec((0..n * h * w * c).map(|i| ((i * 7919) % 211) as f32 / 211.0 - 0.5).collect(), n, h, w, c);
    let wt: Vec<f32> = (0..2 * 9 * c).map(|i| ((i * 104_729) % 97) as f32 / 970.0 - 0.05).collect();
    let conv = Conv3x3::same(c, 2, wt.clone().into(), None);
    let mut out = Tensor::default();
    conv.forward(&x, &mut out, &mut Scratch::default());
    let mut want = Tensor::zeros(n, h, w, 2);
    for oy in 0..h {
        for ox in 0..w {
            for o in 0..2 {
                let mut acc = 0.0f64;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (iy, ix) = (oy as isize + ky as isize - 1, ox as isize + kx as isize - 1);
                        if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                            continue;
                        }
                        let px = (iy as usize * w + ix as usize) * c;
                        let tap = ((o * 3 + ky) * 3 + kx) * c;
                        acc += (0..c).map(|i| x.data[px + i] as f64 * wt[tap + i] as f64).sum::<f64>();
                    }
                }
                want.data[(oy * w + ox) * 2 + o] = acc as f32;
            }
        }
    }
    assert_close(&out, &want, 2e-5);
}

#[test]
fn depthwise_epilogues() {
    let f = Fixtures::open();
    let x = f.nhwc("dw.x", [2, 6, 9, 26]);
    let (w, b) = (f.vec("dw.w", &[3, 3, 26]), f.vec("dw.b", &[26]));
    let mut out = Tensor::default();
    depthwise3x3(&x, &w, &b, Epilogue::Relu, &mut out);
    assert_close(&out, &f.nhwc("dw.relu", [2, 6, 9, 26]), 1e-5);
    depthwise3x3(&x, &w, &b, Epilogue::GluSilu, &mut out);
    assert_close(&out, &f.nhwc("dw.glu", [2, 6, 9, 13]), 1e-5);
}

#[test]
fn group_norm_plain_and_fused() {
    let f = Fixtures::open();
    let x = f.nhwc("gn.x", [2, 5, 6, 96]);
    let (gamma, beta) = (f.vec("gn.gamma", &[96]), f.vec("gn.beta", &[96]));
    let add = f.vec("gn.add", &[2, 96]);
    let mut out = Tensor::default();
    GroupNorm { groups: 32, eps: 1e-6, gamma: gamma.clone(), beta: beta.clone() }.forward(&x, None, false, &mut out);
    assert_close(&out, &f.nhwc("gn.y", [2, 5, 6, 96]), 1e-5);
    GroupNorm { groups: 32, eps: 1e-5, gamma, beta }.forward(&x, Some(&add), true, &mut out);
    assert_close(&out, &f.nhwc("gn.y_add_silu", [2, 5, 6, 96]), 1e-5);
}

#[test]
fn layer_norm_tokens() {
    let f = Fixtures::open();
    let x = Tensor::from_vec(f.vec("ln.x", &[3, 17, 44]), 3, 17, 1, 44);
    let mut out = Tensor::default();
    layer_norm(&x, &f.vec("ln.gamma", &[44]), &f.vec("ln.beta", &[44]), 1e-5, &mut out);
    let want = Tensor::from_vec(f.vec("ln.y", &[3, 17, 44]), 3, 17, 1, 44);
    assert_close(&out, &want, 1e-5);
}

#[test]
fn winograd_matches_direct_on_multiband_odd_sizes() {
    // Odd output size (partial tiles), several bands, two batch items,
    // upsampled and plain, f16 weights, plus a c_out large enough to need
    // per-block kernel transforms (16·c_in·c_out > U_BUDGET).
    let cases = [(2, 37, 23, 24, 40, false), (1, 9, 13, 16, 8, true), (1, 6, 5, 768, 720, false)];
    for (n, h, w, ci, co, up) in cases {
        let x = Tensor::from_vec((0..n * h * w * ci).map(|i| ((i * 7919) % 401) as f32 / 401.0 - 0.5).collect(), n, h, w, ci);
        let wt: Vec<f32> = (0..co * 9 * ci).map(|i| ((i * 104_729) % 113) as f32 / 1130.0 - 0.05).collect();
        let b: Vec<f32> = (0..co).map(|i| i as f32 * 0.01).collect();
        let mut conv = Conv3x3::same(ci, co, Param::f16_from(&wt), Some(b));
        conv.upsample2x = up;
        let (mut s, mut fast, mut direct) = (Scratch::default(), Tensor::default(), Tensor::default());
        conv.forward(&x, &mut fast, &mut s);
        conv.forward_im2col(&x, &mut direct, &mut s);
        assert_close(&fast, &direct, 2e-5);
    }
}

#[test]
fn depthwise_register_blocks_and_tails() {
    // 32-multiples take the register-blocked path, 70 / 2·35 add a tail; odd
    // widths exercise the unpaired last pixel. Reference: the f64 definition.
    for (h, w, c, ep) in [(5, 7, 64, Epilogue::Relu), (4, 9, 70, Epilogue::None), (3, 5, 140, Epilogue::GluSilu), (2, 4, 64, Epilogue::GluSilu)] {
        let x = Tensor::from_vec((0..h * w * c).map(|i| ((i * 7919) % 211) as f32 / 211.0 - 0.5).collect(), 1, h, w, c);
        let wt: Vec<f32> = (0..9 * c).map(|i| ((i * 31) % 17) as f32 / 17.0 - 0.5).collect();
        let b: Vec<f32> = (0..c).map(|i| i as f32 * 0.01 - 0.2).collect();
        let mut out = Tensor::default();
        depthwise3x3(&x, &wt, &b, ep, &mut out);
        let y = |oy: usize, ox: usize, ch: usize| -> f64 {
            let mut a = b[ch] as f64;
            for ky in 0..3 {
                for kx in 0..3 {
                    let (iy, ix) = (oy as isize + ky as isize - 1, ox as isize + kx as isize - 1);
                    if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                        a += x.data[(iy as usize * w + ix as usize) * c + ch] as f64 * wt[(ky * 3 + kx) * c + ch] as f64;
                    }
                }
            }
            a
        };
        let c_out = if ep == Epilogue::GluSilu { c / 2 } else { c };
        let mut want = Tensor::zeros(1, h, w, c_out);
        for oy in 0..h {
            for ox in 0..w {
                for ch in 0..c_out {
                    let v = match ep {
                        Epilogue::None => y(oy, ox, ch),
                        Epilogue::Relu => y(oy, ox, ch).max(0.0),
                        Epilogue::GluSilu => {
                            let g = y(oy, ox, c_out + ch);
                            y(oy, ox, ch) * g / (1.0 + (-g).exp())
                        }
                    };
                    want.data[(oy * w + ox) * c_out + ch] = v as f32;
                }
            }
        }
        assert_close(&out, &want, 1e-5);
    }
}

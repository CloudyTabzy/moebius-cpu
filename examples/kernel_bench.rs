//! M2 speed gates on real Moebius layer shapes (PLAN §9).
//!
//!     cargo run --release --example kernel_bench [threads]

use std::time::Instant;

use moebius_cpu::engine::conv::{Conv1x1, Conv3x3};
use moebius_cpu::engine::depthwise::{Epilogue, depthwise3x3};
use moebius_cpu::engine::norm::{GroupNorm, layer_norm};
use moebius_cpu::engine::ops::silu_inplace;
use moebius_cpu::engine::{Param, Scratch, Tensor};

fn filled(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed)) % 1999) as f32 / 1999.0 - 0.5).collect()
}

fn tensor(n: usize, h: usize, w: usize, c: usize) -> Tensor {
    Tensor::from_vec(filled(n * h * w * c, c), n, h, w, c)
}

/// Median of `iters` timed runs after one warm-up, in seconds.
fn time(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..iters)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[iters / 2]
}

fn report(name: &str, secs: f64, work: f64, unit: &str, gate: Option<(f64, &str)>) {
    let rate = work / secs / 1e9;
    let verdict = match gate {
        Some((min, _)) if rate >= min => "PASS",
        Some(_) => "FAIL",
        None => "",
    };
    let gate_txt = gate.map(|(min, what)| format!("gate ≥ {min:.0} {unit} ({what})")).unwrap_or_default();
    println!("  {name:44} {:8.2} ms {rate:7.1} {unit:6} {verdict:4} {gate_txt}", secs * 1e3);
}

fn main() {
    let threads = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or_else(moebius_cpu::engine::cpu::default_threads);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    println!("threads = {threads}, avx2+fma = {}", moebius_cpu::engine::simd::avx2_fma());
    pool.install(run);
}

fn run() {
    let mut s = Scratch::default();
    let mut out = Tensor::default();

    println!("1x1 conv (f16 weights, widened per call):");
    for (name, [n, h, w, ci, co]) in [
        ("MixFFN in, lvl0  2×64²  320→1600", [2, 64, 64, 320, 1600]),
        ("MixFFN out, lvl0 2×64²  800→320", [2, 64, 64, 800, 320]),
        ("MixFFN in, lvl2  2×16² 1280→6400", [2, 16, 16, 1280, 6400]),
    ] {
        let x = tensor(n, h, w, ci);
        let conv = Conv1x1 { c_in: ci, c_out: co, weight: Param::f16_from(&filled(ci * co, 1)), bias: Some(filled(co, 2)) };
        let t = time(9, || conv.forward(&x, &mut out, None, &mut s));
        report(name, t, 2.0 * (n * h * w * ci * co) as f64, "GFLOPS", Some((450.0, "PLAN M2")));
    }

    println!("3x3 conv (f16 weights):");
    for (name, [n, h, w, ci, co], up, iters) in [
        ("VAE 128ch @256²", [1, 256, 256, 128, 128], false, 7),
        ("VAE 512ch @64²", [1, 64, 64, 512, 512], false, 7),
        ("VAE 128ch @512² (decoder last level)", [1, 512, 512, 128, 128], false, 3),
        ("UNet Upsample2D 1280ch 16²→32², B=2", [2, 16, 16, 1280, 1280], true, 7),
    ] {
        let x = tensor(n, h, w, ci);
        let mut conv = Conv3x3::same(ci, co, Param::f16_from(&filled(9 * ci * co, 3)), Some(filled(co, 4)));
        conv.upsample2x = up;
        let (ho, wo) = conv.output_hw(h, w);
        let t = time(iters, || conv.forward(&x, &mut out, &mut s));
        report(name, t, 2.0 * (n * ho * wo * 9 * ci * co) as f64, "GFLOPS", Some((420.0, "PLAN M2")));
    }

    println!("depthwise 3x3 (memory-bound: saturates at 4–6 threads; GB/s = input + output bytes):");
    for (name, [n, h, w, c], ep, torch_ms) in [
        ("MixFFN GLU lvl0 2×64²×1600", [2, 64, 64, 1600], Epilogue::GluSilu, 10.0),
        ("DWConv2d ReLU lvl0 2×64²×320", [2, 64, 64, 320], Epilogue::Relu, f64::NAN),
        ("MixFFN GLU lvl2 2×16²×6400", [2, 16, 16, 6400], Epilogue::GluSilu, f64::NAN),
    ] {
        let x = tensor(n, h, w, c);
        let (wt, b) = (filled(9 * c, 5), filled(c, 6));
        let t = time(15, || depthwise3x3(&x, &wt, &b, ep, &mut out));
        let bytes = 4.0 * (x.data.len() + out.data.len()) as f64;
        report(name, t, bytes, "GB/s", None);
        if !torch_ms.is_nan() {
            println!("      PyTorch {torch_ms:.1} ms → {:.1}× faster", torch_ms / (t * 1e3));
        }
    }

    println!("norms / elementwise (GB/s = input + output bytes):");
    {
        let x = tensor(2, 64, 64, 320);
        let gn = GroupNorm { groups: 32, eps: 1e-5, gamma: filled(320, 7), beta: filled(320, 8) };
        let add = filled(2 * 320, 9);
        let t = time(25, || gn.forward(&x, Some(&add), true, &mut out));
        report("GroupNorm+add+SiLU 2×64²×320", t, 8.0 * x.data.len() as f64, "GB/s", None);

        let x = tensor(1, 512, 512, 128);
        let gn = GroupNorm { groups: 32, eps: 1e-6, gamma: filled(128, 7), beta: filled(128, 8) };
        let t = time(7, || gn.forward(&x, None, true, &mut out));
        report("GroupNorm+SiLU VAE 512²×128", t, 8.0 * x.data.len() as f64, "GB/s", None);

        let x = tensor(2, 64, 64, 320);
        let t = time(25, || layer_norm(&x, &filled(320, 1), &filled(320, 2), 1e-5, &mut out));
        report("LayerNorm 8192 tokens × 320", t, 8.0 * x.data.len() as f64, "GB/s", None);

        let mut x = tensor(2, 64, 64, 1600);
        let t = time(15, || silu_inplace(&mut x.data));
        report("SiLU in place 2×64²×1600", t, 8.0 * x.data.len() as f64, "GB/s", None);
    }
}

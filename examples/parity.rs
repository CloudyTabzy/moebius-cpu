//! End-to-end parity against the upstream PyTorch pipeline.
//!
//!     cargo run --release --example parity [-- --f16 0,1,2,3,4,5,6,7 --f32 0,1]
//!
//! For each golden case (`tools/dump_golden.py`) runs `Engine::inpaint_with_noise`
//! with the recorded noise and compares the 8-bit result with `output.image`
//! and the final latents with `latents.final`. One engine is alive at a time.

use std::path::{Path, PathBuf};
use std::time::Instant;

use moebius_cpu::engine::Tensor;
use moebius_cpu::imaging::{ImageMut, MaskRef};
use moebius_cpu::pipeline::{Control, Engine, NoiseSet};
use moebius_cpu::weights::Weights;
use moebius_cpu::Preset;

const SIDE: usize = 512;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn tensor(g: &Weights, name: &str, shape: [usize; 4]) -> Tensor {
    Tensor::from_vec(g.get_f32(name, &shape).unwrap_or_else(|e| panic!("{name}: {e}")), shape[0], shape[1], shape[2], shape[3])
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den).sqrt()
}

fn cases(args: &[String], flag: &str, default: &str) -> Vec<usize> {
    let list = args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map_or(default, String::as_str);
    list.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().expect("case index")).collect()
}

struct Row {
    case: usize,
    dtype: &'static str,
    psnr: f64,
    max_diff: u8,
    differing: usize,
    latents: f64,
    secs: f64,
}

fn run_case(engine: &Engine, case: usize, dtype: &'static str) -> Row {
    let path = root().join(format!("weights/golden/case{case}.safetensors"));
    let g = Weights::open_with_format(&path, "moebius-cpu-golden/1")
        .unwrap_or_else(|e| panic!("{e}; produce it with `python tools/dump_golden.py`"));
    let to_u8 = |name: &str, c: usize| -> Vec<u8> {
        g.get_f32(name, &[1, SIDE, SIDE, c]).unwrap().iter().map(|&v| v.round() as u8).collect()
    };
    let mut data = to_u8("input.image", 3);
    let mask = to_u8("input.mask", 1);
    let want = to_u8("output.image", 3);
    let noise = NoiseSet {
        vae_img: tensor(&g, "noise.vae_img", [1, 64, 64, 4]),
        vae_masked: tensor(&g, "noise.vae_masked", [1, 64, 64, 4]),
        init: tensor(&g, "noise.init", [1, 64, 64, 4]),
        offset: tensor(&g, "noise.offset", [1, 1, 1, 4]),
    };
    let want_latents = g.get_f32("latents.final", &[1, 64, 64, 4]).unwrap();

    let params = Preset::Exact.params();
    let mut img = ImageMut { data: &mut data, width: SIDE, height: SIDE, stride: SIDE * 3, channels: 3 };
    let m = MaskRef { data: &mask, width: SIDE, height: SIDE, stride: SIDE };
    let mut latents = Vec::new();
    let t = Instant::now();
    engine
        .inpaint_traced(&mut img, m, &params, &noise, &mut |_| Control::Continue, Some(&mut latents))
        .expect("inpaint");
    let secs = t.elapsed().as_secs_f64();

    let mse = data.iter().zip(&want).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / data.len() as f64;
    Row {
        case,
        dtype,
        psnr: if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() },
        max_diff: data.iter().zip(&want).map(|(&a, &b)| a.abs_diff(b)).max().unwrap(),
        differing: data.iter().zip(&want).filter(|(a, b)| a != b).count(),
        latents: rel_l2(&latents, &want_latents),
        secs,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let plan = [
        ("f16", "weights/ft_places2", cases(&args, "--f16", "0,1,2,3,4,5,6,7")),
        ("f32", "weights/ft_places2-f32", cases(&args, "--f32", "0,1")),
    ];
    println!("| case | dtype | PSNR (dB) | max abs diff | differing bytes | rel_l2(latents.final) | wall (s) |");
    println!("|-----:|:-----:|----------:|-------------:|----------------:|----------------------:|---------:|");
    for (dtype, dir, list) in plan {
        if list.is_empty() {
            continue;
        }
        let engine = Engine::load(&root().join(dir), 0).unwrap_or_else(|e| panic!("{dir}: {e}"));
        for case in list {
            let r = run_case(&engine, case, dtype);
            println!(
                "| {} | {} | {:.2} | {} | {} / {} | {:.3e} | {:.1} |",
                r.case, r.dtype, r.psnr, r.max_diff, r.differing, SIDE * SIDE * 3, r.latents, r.secs
            );
        }
    }
}

//! M0 gate: does the `gemm` crate sustain the throughput the speed promise
//! assumes (~450 GFLOPS f32 on the benchmark CPU, i.e. on par
//! with PyTorch/MKL measured on the same shapes)?
//!
//!     cargo run --release --example gemm_probe [threads...]

use std::time::Instant;

use moebius_cpu::engine::gemm::{MatRef, matmul};

struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

const SHAPES: &[Shape] = &[
    // UNet level 0: MixFFN inverted conv, B=2 × 64² pixels, 320 → 1600.
    Shape { name: "UNet MixFFN-in  lvl0 8192x320x1600", m: 8192, k: 320, n: 1600 },
    // UNet level 2: MixFFN inverted conv, B=2 × 16² pixels, 1280 → 6400.
    Shape { name: "UNet MixFFN-in  lvl2  512x1280x6400", m: 512, k: 1280, n: 6400 },
    // UNet level 0: MixFFN point conv, 800 → 320.
    Shape { name: "UNet MixFFN-out lvl0 8192x800x320", m: 8192, k: 800, n: 320 },
    // VAE 3x3 conv 128→128 at 256², as one implicit-im2col tile row band.
    Shape { name: "VAE conv3x3 tile 4096x1152x128", m: 4096, k: 1152, n: 128 },
    // VAE 3x3 conv 512→512 at 64².
    Shape { name: "VAE conv3x3 512ch 4096x4608x512", m: 4096, k: 4608, n: 512 },
];

fn main() {
    let threads: Vec<usize> = {
        let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
        if args.is_empty() { vec![6, 10, 16] } else { args }
    };
    for &t in &threads {
        println!("threads={t}");
        for s in SHAPES {
            let a: Vec<f32> = (0..s.m * s.k).map(|i| ((i % 97) as f32 - 48.0) * 1e-2).collect();
            let w: Vec<f32> = (0..s.k * s.n).map(|i| ((i % 89) as f32 - 44.0) * 1e-2).collect();
            let mut c = vec![0.0f32; s.m * s.n];
            for (layout, b) in [
                ("B row-major ", MatRef::row_major(&w, s.k, s.n)),
                ("B [out,in]ᵀ ", MatRef::transposed(&w, s.k, s.n)),
            ] {
                let run = |c: &mut [f32]| matmul(c, MatRef::row_major(&a, s.m, s.k), b, false, t);
                run(&mut c);
                run(&mut c);
                let iters = 8;
                let t0 = Instant::now();
                for _ in 0..iters {
                    run(&mut c);
                }
                let dt = t0.elapsed().as_secs_f64() / iters as f64;
                let gflops = 2.0 * (s.m * s.k * s.n) as f64 / dt / 1e9;
                println!("  {:38} {layout} {:7.2} ms {:6.0} GFLOPS", s.name, dt * 1e3, gflops);
            }
        }
    }
}

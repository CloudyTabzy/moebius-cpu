//! Why Winograd runs its 16 GEMMs concurrently, single-threaded each:
//! compares that against running them in sequence with internal parallelism
//! on Moebius shapes (one GEMM per transformed tile position).
//!
//!     cargo run --release --example wino_probe

use std::time::Instant;
use moebius_cpu::engine::gemm::{MatRef, matmul};
use rayon::prelude::*;
fn main() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build().unwrap();
    pool.install(|| {
        // (name, tiles M, K=c_in, N=c_out) — one of the 16 Winograd GEMMs per band
        for (name, m, k, n) in [("VAE 256² 128ch band", 4096, 128, 128), ("VAE 512² 128ch band", 4096, 128, 128),
                                ("VAE 64² 512ch", 1024, 512, 512), ("VAE 128² 256ch", 4096, 256, 256), ("UNet up 1280", 512, 1280, 1280), ("UNet up 640 64²", 2048, 640, 640)] {
            let a: Vec<f32> = (0..16 * m * k).map(|i| (i % 13) as f32 * 0.01).collect();
            let b: Vec<f32> = (0..16 * k * n).map(|i| (i % 7) as f32 * 0.01).collect();
            let mut c = vec![0f32; 16 * m * n];
            for mode in ["16 seq × parallel gemm", "16 par × 1-thread gemm"] {
                let run = |c: &mut Vec<f32>| {
                    if mode.starts_with("16 seq") {
                        for (xi, cc) in c.chunks_mut(m * n).enumerate() {
                            matmul(cc, MatRef::row_major(&a[xi * m * k..], m, k), MatRef::transposed(&b[xi * k * n..], k, n), false, 10);
                        }
                    } else {
                        c.par_chunks_mut(m * n).enumerate().for_each(|(xi, cc)| {
                            matmul(cc, MatRef::row_major(&a[xi * m * k..], m, k), MatRef::transposed(&b[xi * k * n..], k, n), false, 1);
                        });
                    }
                };
                run(&mut c); run(&mut c);
                let t0 = Instant::now(); for _ in 0..5 { run(&mut c); }
                let dt = t0.elapsed().as_secs_f64() / 5.0;
                println!("{name:22} {mode:26} {:7.2} ms {:5.0} GFLOPS", dt * 1e3, 2.0 * 16.0 * (m * k * n) as f64 / dt / 1e9);
            }
        }
    });
}

//! Sustained GEMM throughput over ~70 s. On the benchmark CPU it
//! oscillates 340–600 GFLOPS as the power limit cycles, ~480 average.
//!
//!     cargo run --release --example sustained
use std::time::Instant;
use moebius_cpu::engine::gemm::{MatRef, matmul};
fn main() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build().unwrap();
    let (m, k, n) = (8192, 320, 1600);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01).collect();
    let mut c = vec![0f32; m * n];
    let start = Instant::now();
    pool.install(|| {
        let mut window = Instant::now();
        let mut flops = 0.0;
        while start.elapsed().as_secs() < 70 {
            matmul(&mut c, MatRef::row_major(&a, m, k), MatRef::transposed(&b, k, n), false, 10);
            flops += 2.0 * (m * k * n) as f64;
            if window.elapsed().as_secs_f64() >= 5.0 {
                println!("t={:5.1}s  {:6.0} GFLOPS", start.elapsed().as_secs_f64(), flops / window.elapsed().as_secs_f64() / 1e9);
                window = Instant::now();
                flops = 0.0;
            }
        }
    });
}

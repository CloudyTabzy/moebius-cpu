//! Wall time per UNet block (trace-callback deltas) for one CFG step, f16 weights.
//!
//!     cargo run --release --example unet_blocks -- [weights_dir] [threads]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use moebius_cpu::engine::{Scratch, Tensor};
use moebius_cpu::unet::{Guidance, UNet};
use moebius_cpu::weights::Weights;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "weights/ft_places2".into()));
    let threads: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let w = Weights::open(&dir.join("unet.safetensors")).unwrap();
    let unet = pool.install(|| UNet::load(&w).unwrap());
    let x = Tensor::from_vec((0..64 * 64 * 9).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect(), 1, 64, 64, 9);
    let mut s = Scratch::default();
    let mut acc: BTreeMap<String, f64> = BTreeMap::new();
    let runs = 4;
    for run in 0..=runs {
        let t0 = Instant::now();
        let mut last = Instant::now();
        let eps = pool.install(|| {
            unet.forward_traced(&x, 900, Guidance::Cfg, &mut s, &mut |name: &str, _t: &Tensor| {
                let now = Instant::now();
                if run > 0 {
                    *acc.entry(name.to_string()).or_default() += (now - last).as_secs_f64() / runs as f64;
                }
                last = now;
            })
        });
        s.recycle(eps);
        if run > 0 {
            println!("run {run}: {:.3} s", t0.elapsed().as_secs_f64());
        }
    }
    let mut rows: Vec<_> = acc.into_iter().collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    for (name, t) in rows {
        println!("  {name:20} {:7.1} ms  {:5.1}%", t * 1e3, 100.0 * t / total);
    }
}

//! Per-stage wall times of one Exact inpaint (case 0 image/mask, seed 0).
//!
//!     cargo run --release --example profile -- [weights_dir] [threads]

use std::path::PathBuf;
use std::time::Instant;

use moebius_cpu::imaging::{ImageMut, MaskRef};
use moebius_cpu::pipeline::{Control, Progress, Stage};
use moebius_cpu::{Engine, Preset};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "weights/ft_places2".into()));
    let threads: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let img = image::open("../Moebius/data/images/0.png").unwrap().into_rgb8();
    let mask = image::open("../Moebius/data/masks/000000.png").unwrap().into_luma8();
    let t = Instant::now();
    let engine = Engine::load(&dir, threads).unwrap();
    println!("load {:.2?}", t.elapsed());
    for run in 0..2 {
        let mut rgb = img.clone();
        let mut im = ImageMut { data: &mut rgb, width: 512, height: 512, stride: 1536, channels: 3 };
        let m = MaskRef { data: &mask, width: 512, height: 512, stride: 512 };
        let (t0, mut last) = (Instant::now(), Instant::now());
        let mut steps = Vec::new();
        engine
            .inpaint(&mut im, m, &Preset::Exact.params(), &mut |p: Progress| {
                let now = Instant::now();
                let dt = (now - last).as_secs_f64();
                match (p.stage, p.done) {
                    (Stage::Encode, 1) => println!("  encode      {dt:6.2} s"),
                    (Stage::Denoise, _) => steps.push(dt),
                    (Stage::Decode, 1) => println!("  decode      {dt:6.2} s"),
                    _ => {}
                }
                last = now;
                Control::Continue
            })
            .unwrap();
        let total = t0.elapsed().as_secs_f64();
        let sum: f64 = steps.iter().sum();
        println!(
            "run {run}: total {total:.2} s | denoise {sum:.2} s over {} steps (first {:.2}, median {:.2}, max {:.2}) | other {:.2} s",
            steps.len(),
            steps[0],
            { let mut s = steps.clone(); s.sort_by(f64::total_cmp); s[s.len() / 2] },
            steps.iter().cloned().fold(0.0, f64::max),
            total - sum
        );
    }
}

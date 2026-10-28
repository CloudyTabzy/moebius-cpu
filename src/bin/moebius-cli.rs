//! Developer CLI: PNG in, PNG out, same code path as the plugin.
//!
//!     cargo run --release --features cli -- --weights weights --image in.png --mask m.png --output out.png

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use moebius_cpu::imaging::{ImageMut, MaskRef};
use moebius_cpu::pipeline::{Control, Progress};
use moebius_cpu::{Engine, Preset};

#[derive(Clone, Copy, ValueEnum)]
enum PresetArg {
    Exact,
    Fast,
}

#[derive(Parser)]
#[command(name = "moebius-cli", about = "Moebius inpainting on CPU")]
struct Args {
    /// Directory with unet.safetensors and vae.safetensors.
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    image: PathBuf,
    /// Grayscale mask, white = inpaint.
    #[arg(long)]
    mask: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = PresetArg::Exact)]
    preset: PresetArg,
    /// Override the preset's step count.
    #[arg(long)]
    steps: Option<usize>,
    /// Override the preset's guidance scale.
    #[arg(long)]
    cfg: Option<f32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Worker threads (0 = physical cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut params = match args.preset {
        PresetArg::Exact => Preset::Exact.params(),
        PresetArg::Fast => Preset::Fast.params(),
    };
    params.steps = args.steps.unwrap_or(params.steps);
    params.guidance = args.cfg.unwrap_or(params.guidance);
    params.seed = args.seed;

    let mut rgb = image::open(&args.image)?.into_rgb8();
    let mask = image::open(&args.mask)?.into_luma8();
    let (w, h) = rgb.dimensions();

    let t0 = Instant::now();
    let engine = Engine::load(&args.weights, args.threads)?;
    eprintln!("loaded in {:.2?}", t0.elapsed());

    let t1 = Instant::now();
    let mut img = ImageMut { data: &mut rgb, width: w as usize, height: h as usize, stride: 3 * w as usize, channels: 3 };
    let mask = MaskRef { data: &mask, width: w as usize, height: h as usize, stride: w as usize };
    let changed = engine.inpaint(&mut img, mask, &params, &mut |p: Progress| {
        eprint!("\r{:?} {}/{}   ", p.stage, p.done, p.total);
        Control::Continue
    })?;
    eprintln!("\ninpainted {changed:?} in {:.2?}", t1.elapsed());
    rgb.save(&args.output)?;
    Ok(())
}

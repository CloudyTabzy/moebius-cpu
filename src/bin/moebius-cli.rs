//! PNG in, PNG out. The GIMP plugin runs this as its worker, with `--plugin`.
//!
//!     cargo run --release --features cli -- --weights weights --image in.png --mask m.png --output out.png

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use moebius_cpu::imaging::{ImageMut, MaskRef};
use moebius_cpu::pipeline::{Control, Progress, Stage};
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
    /// Run as a plugin worker: print `PROGRESS <stage> <done> <total>` lines on stdout, and
    /// exit when stdin closes, so that a killed plugin leaves no process behind.
    #[arg(long)]
    plugin: bool,
}

/// Exit code when the parent went away.
const EXIT_ORPHANED: i32 = 3;

fn exit_when_stdin_closes() {
    std::thread::spawn(|| {
        let mut sink = [0u8; 64];
        let mut stdin = std::io::stdin();
        while matches!(stdin.read(&mut sink), Ok(n) if n > 0) {}
        std::process::exit(EXIT_ORPHANED);
    });
}

fn report(p: Progress) {
    let stage = match p.stage {
        Stage::Encode => "encode",
        Stage::Denoise => "denoise",
        Stage::Decode => "decode",
    };
    let mut out = std::io::stdout().lock();
    // A closed pipe means the parent is gone: the stdin watchdog ends the process.
    let _ = writeln!(out, "PROGRESS {stage} {} {}", p.done, p.total);
    let _ = out.flush();
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

    if args.plugin {
        exit_when_stdin_closes();
    }

    let source = image::open(&args.image)?;
    let channels = if source.color().has_alpha() { 4 } else { 3 };
    let (w, h) = (source.width(), source.height());
    let mut pixels = if channels == 4 { source.into_rgba8().into_raw() } else { source.into_rgb8().into_raw() };
    let mask = image::open(&args.mask)?.into_luma8();
    if mask.dimensions() != (w, h) {
        return Err(format!("mask is {:?}, image is {:?}", mask.dimensions(), (w, h)).into());
    }

    let t0 = Instant::now();
    let engine = Engine::load(&args.weights, args.threads)?;
    eprintln!("loaded in {:.2?}", t0.elapsed());

    let t1 = Instant::now();
    let (wu, hu) = (w as usize, h as usize);
    let mut img = ImageMut { data: &mut pixels, width: wu, height: hu, stride: channels * wu, channels };
    let mask = MaskRef { data: &mask, width: wu, height: hu, stride: wu };
    let changed = engine.inpaint(&mut img, mask, &params, &mut |p: Progress| {
        if args.plugin {
            report(p);
        } else {
            eprint!("\r{:?} {}/{}   ", p.stage, p.done, p.total);
        }
        Control::Continue
    })?;
    eprintln!("\ninpainted {changed:?} in {:.2?}", t1.elapsed());
    let color = if channels == 4 { image::ExtendedColorType::Rgba8 } else { image::ExtendedColorType::Rgb8 };
    image::save_buffer(&args.output, &pixels, w, h, color)?;
    Ok(())
}

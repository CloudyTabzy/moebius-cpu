//! Eval runner: inpaint every pair of an eval manifest with one Engine per
//! subset, writing `<out>/<id>.png`, `<out>/times.json` and `<out>/config.json`.
//! Score the output with `tools/eval/score.py`. Run from the repo root.
//!
//!     cargo run --release --features cli --example eval -- \
//!         --weights-scenes weights/ft_places2 --weights-faces weights/ft_celebahq \
//!         --manifest weights/eval/manifest.json --out weights/eval/runs/exact --preset exact
//!
//! Seed = the pair's index in the manifest (the same for every run, so
//! Exact/Fast comparisons share noise). Pairs whose output already exists are
//! skipped, so an interrupted run resumes; the per-image times in `times.json`
//! are inpaint wall time only (no engine load, no PNG I/O).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use moebius_cpu::imaging::{ImageMut, MaskRef};
use moebius_cpu::pipeline::{Control, Progress};
use moebius_cpu::scheduler::Spacing;
use moebius_cpu::preset::Solver;
use moebius_cpu::{Engine, InpaintParams, Preset};
use serde_json::{Value, json};

type BoxErr = Box<dyn std::error::Error>;

#[derive(Clone, Copy, ValueEnum)]
enum PresetArg {
    Exact,
    Fast,
}

#[derive(Clone, Copy, ValueEnum)]
enum SpacingArg {
    Leading,
    Trailing,
}

#[derive(Clone, Copy, ValueEnum)]
enum SolverArg {
    Ddim,
    Dpm,
    DpmContinuous,
}

/// `--cfg-steps N` or `--cfg-steps none` (guidance on every step).
#[derive(Clone, Copy)]
struct CfgSteps(Option<usize>);

impl FromStr for CfgSteps {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("none") {
            Ok(CfgSteps(None))
        } else {
            s.parse().map(|n| CfgSteps(Some(n))).map_err(|_| format!("expected N or `none`, got `{s}`"))
        }
    }
}

#[derive(Parser)]
#[command(name = "eval", about = "Run the M8 eval set through the engine")]
struct Args {
    /// Checkpoint directory for the `scenes` subset.
    #[arg(long)]
    weights_scenes: PathBuf,
    /// Checkpoint directory for the `faces` subset.
    #[arg(long)]
    weights_faces: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum, default_value_t = PresetArg::Exact)]
    preset: PresetArg,
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    cfg_steps: Option<CfgSteps>,
    #[arg(long, value_enum)]
    spacing: Option<SpacingArg>,
    #[arg(long, value_enum)]
    solver: Option<SolverArg>,
    #[arg(long)]
    strength: Option<f64>,
    #[arg(long)]
    single_encode: Option<bool>,
    #[arg(long)]
    roi_encode: Option<bool>,
    #[arg(long)]
    roi_decode: Option<bool>,
    /// Only the first N pairs of each subset (smoke tests, dev sweeps).
    #[arg(long)]
    limit_per_subset: Option<usize>,
    /// Worker threads (0 = physical cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

struct Pair {
    index: usize,
    id: String,
    image: PathBuf,
    mask: PathBuf,
}

fn config_json(args: &Args, p: &InpaintParams) -> Value {
    json!({
        "preset": match args.preset { PresetArg::Exact => "exact", PresetArg::Fast => "fast" },
        "steps": p.steps,
        "strength": p.strength,
        "spacing": format!("{:?}", p.spacing).to_lowercase(),
        "solver": format!("{:?}", p.solver),
        "guidance": p.guidance,
        "noise_offset": p.noise_offset,
        "cfg_steps": p.cfg_steps,
        "single_encode": p.single_encode,
        "roi_encode": p.roi_encode,
        "roi_decode": p.roi_decode,
        "paste": format!("{:?}", p.paste),
        "mask_dilate": p.mask_dilate,
        "limit_per_subset": args.limit_per_subset,
    })
}

fn main() -> Result<(), BoxErr> {
    let args = Args::parse();
    let mut base = match args.preset {
        PresetArg::Exact => Preset::Exact.params(),
        PresetArg::Fast => Preset::Fast.params(),
    };
    if let Some(v) = args.steps {
        base.steps = v;
    }
    if let Some(CfgSteps(v)) = args.cfg_steps {
        base.cfg_steps = v;
    }
    if let Some(v) = args.spacing {
        base.spacing = match v {
            SpacingArg::Leading => Spacing::Leading,
            SpacingArg::Trailing => Spacing::Trailing,
        };
    }
    if let Some(v) = args.solver {
        base.solver = match v {
            SolverArg::Ddim => Solver::Ddim,
            SolverArg::Dpm => Solver::DpmPp2M,
            SolverArg::DpmContinuous => Solver::DpmPp2MContinuous,
        };
    }
    if let Some(v) = args.strength {
        base.strength = v;
    }
    if let Some(v) = args.roi_encode {
        base.roi_encode = v;
    }
    if let Some(v) = args.single_encode {
        base.single_encode = v;
    }
    if let Some(v) = args.roi_decode {
        base.roi_decode = v;
    }

    // Pairs of the manifest, grouped by subset; seed = index in the manifest.
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(&args.manifest)?)?;
    let mut subsets: Vec<(&str, &Path, Vec<Pair>)> =
        vec![("scenes", &args.weights_scenes, vec![]), ("faces", &args.weights_faces, vec![])];
    for (index, e) in manifest.as_array().ok_or("manifest is not a list")?.iter().enumerate() {
        let field = |k: &str| e[k].as_str().map(str::to_owned).ok_or(format!("manifest entry {index}: missing `{k}`"));
        let subset = field("subset")?;
        let pair = Pair { index, id: field("id")?, image: field("image")?.into(), mask: field("mask")?.into() };
        subsets
            .iter_mut()
            .find(|s| s.0 == subset)
            .ok_or(format!("unknown subset `{subset}`"))?
            .2
            .push(pair);
    }

    std::fs::create_dir_all(&args.out)?;
    // Refuse to mix outputs of different configurations in one directory.
    let cfg = config_json(&args, &base);
    let cfg_path = args.out.join("config.json");
    if let Ok(old) = std::fs::read_to_string(&cfg_path) {
        if serde_json::from_str::<Value>(&old)? != cfg {
            return Err(format!("{} holds a different configuration; use a fresh --out", cfg_path.display()).into());
        }
    }
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;

    let times_path = args.out.join("times.json");
    let mut times: BTreeMap<String, f64> = match std::fs::read_to_string(&times_path) {
        Ok(s) => serde_json::from_str(&s)?,
        Err(_) => BTreeMap::new(),
    };

    let mut new_images = 0usize;
    for (name, weights, mut pairs) in subsets {
        if let Some(n) = args.limit_per_subset {
            pairs.truncate(n);
        }
        pairs.retain(|p| !args.out.join(format!("{}.png", p.id)).exists());
        if pairs.is_empty() {
            eprintln!("[{name}] nothing to do");
            continue;
        }
        let t = Instant::now();
        let engine = Engine::load(weights, args.threads)?;
        eprintln!("[{name}] engine loaded in {:.2?} ({} pairs to run)", t.elapsed(), pairs.len());
        for p in &pairs {
            let mut rgb = image::open(&p.image)?.into_rgb8();
            let mask = image::open(&p.mask)?.into_luma8();
            let (w, h) = rgb.dimensions();
            let params = InpaintParams { seed: p.index as u64, ..base };
            let t = Instant::now();
            {
                let mut img =
                    ImageMut { data: &mut rgb, width: w as usize, height: h as usize, stride: 3 * w as usize, channels: 3 };
                let m = MaskRef { data: &mask, width: w as usize, height: h as usize, stride: w as usize };
                engine.inpaint(&mut img, m, &params, &mut |_: Progress| Control::Continue)?;
            }
            let secs = t.elapsed().as_secs_f64();
            // Temp name first: a killed run must not leave a truncated PNG that resume would skip.
            let dst = args.out.join(format!("{}.png", p.id));
            let tmp = args.out.join(format!("{}.tmp.png", p.id));
            rgb.save(&tmp)?;
            std::fs::rename(&tmp, &dst)?;
            times.insert(p.id.clone(), secs);
            std::fs::write(&times_path, serde_json::to_string_pretty(&times)?)?;
            new_images += 1;
            eprintln!("[{name}] {} (seed {}) {secs:.2} s  [{new_images} done this run]", p.id, p.index);
        }
    }
    eprintln!("finished: {new_images} new images, {} timed in total, out = {}", times.len(), args.out.display());
    Ok(())
}

//! End-to-end inpainting, mirroring `RemovalSDXLPipeline_BatchMode` (M7).
//!
//! ```text
//!  1  bbox(mask) → CropPlan (square window)                imaging::crop
//!  2  window → 512² RGB [-1,1] + binary mask (Lanczos)      imaging::resize
//!  3  masked = rgb · (1 − mask)
//!  4  z_img, z_masked = VAE.encode([rgb, masked]) · 0.13025 vae::encoder
//!       (Fast/single_encode: z_img := z_masked)
//!  5  mask_lat = nearest(mask → 64²)
//!  6  noise = N(0,1) + noise_offset · N(0,1)[per channel]   seeded
//!  7  x = noise if strength ≥ 1 else add_noise(z_img, noise, t₀)
//!  8  for t in timesteps:                                   scheduler
//!        ε = UNet([x ‖ mask_lat ‖ z_masked], t)             unet (B=2 / B=1)
//!        ε = ε_u + g·(ε_c − ε_u)                            (CFG steps)
//!        x = DDIM.step(ε, t, x)
//!        progress(); bail out if cancelled
//!  9  rgb = VAE.decode(x / 0.13025)  (Fast: ROI window)     vae::decoder
//! 10  resize back to window, paste with feathered mask      imaging::blend
//! ```
//! Everything runs inside the engine's own rayon pool, sized to physical
//! cores; nothing touches the global pool of a host process.

use std::path::Path;

use crate::imaging::{ImageMut, MaskRef, Rect};
use crate::preset::InpaintParams;
use crate::{Error, Result};

/// Progress report: denoising steps done / total, plus a coarse stage.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub stage: Stage,
    pub done: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Encode,
    Denoise,
    Decode,
}

/// Returned by the progress callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Cancel,
}

pub struct Engine {
    pool: rayon::ThreadPool,
}

impl Engine {
    /// `weights_dir` holds `unet.safetensors` and `vae.safetensors` from
    /// `tools/convert_weights.py`. `threads = 0` → one per physical core.
    pub fn load(weights_dir: &Path, threads: usize) -> Result<Self> {
        crate::engine::cpu::require_avx2_fma()?;
        let threads = if threads == 0 { crate::engine::cpu::default_threads() } else { threads };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("moebius-{i}"))
            .build()
            .map_err(|e| Error::InvalidInput(format!("thread pool: {e}")))?;
        let _ = weights_dir;
        todo!("M5/M6: open weights, build UNet + VAE, precompute constants (pool {})", pool.current_num_threads())
    }

    /// Inpaints `img` in place where `mask` ≥ 128. Returns the rectangle of
    /// pixels that changed. `Err(Error::Cancelled)` leaves `img` untouched.
    pub fn inpaint(
        &self,
        img: &mut ImageMut<'_>,
        mask: MaskRef<'_>,
        params: &InpaintParams,
        progress: &mut dyn FnMut(Progress) -> Control,
    ) -> Result<Rect> {
        if (img.width, img.height) != (mask.width, mask.height) {
            return Err(Error::InvalidInput("image and mask sizes differ".into()));
        }
        if !matches!(img.channels, 3 | 4) {
            return Err(Error::InvalidInput("image must be RGB or RGBA".into()));
        }
        let Some(_bbox) = mask.bbox() else {
            return Ok(Rect::default());
        };
        let _ = (params, progress);
        self.pool.install(|| todo!("M7: steps 1-10 above"))
    }
}

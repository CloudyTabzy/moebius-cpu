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

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

use crate::config::{IMAGE_SIZE, LATENT_CHANNELS, LATENT_SIZE, UNET_IN_CHANNELS, VAE_DOWNSAMPLE};
use crate::engine::{Scratch, Tensor};
use crate::imaging::blend::paste;
use crate::imaging::crop::CropPlan;
use crate::imaging::resize::{from_model, to_model};
use crate::imaging::{ImageMut, MaskRef, Rect, dilate};
use crate::preset::InpaintParams;
use crate::scheduler::Ddim;
use crate::unet::{Guidance, UNet};
use crate::vae::Vae;
use crate::weights::Weights;
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

/// Gaussian noise the pipeline consumes. `vae_*` are `[1,64,64,4]` (VAE
/// posterior sampling for the image / masked image), `init` is `[1,64,64,4]`
/// and `offset` is the per-channel `[1,1,1,4]` term added with
/// `noise_offset`, drawn in this order from one seeded generator.
pub struct NoiseSet {
    pub vae_img: Tensor,
    pub vae_masked: Tensor,
    pub init: Tensor,
    pub offset: Tensor,
}

impl NoiseSet {
    /// Seeded Rust generator (torch's cannot be reproduced here); parity
    /// runs feed the recorded PyTorch noise instead.
    pub fn sample(seed: u64) -> NoiseSet {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut normal = |n, h, w, c| {
            let data = (0..n * h * w * c).map(|_| rng.sample(StandardNormal)).collect();
            Tensor::from_vec(data, n, h, w, c)
        };
        let vae_img = normal(1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS);
        let vae_masked = normal(1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS);
        let init = normal(1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS);
        let offset = normal(1, 1, 1, LATENT_CHANNELS);
        NoiseSet { vae_img, vae_masked, init, offset }
    }

    fn check(&self) -> Result<()> {
        let want = [1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS];
        for (name, t, shape) in [
            ("vae_img", &self.vae_img, want),
            ("vae_masked", &self.vae_masked, want),
            ("init", &self.init, want),
            ("offset", &self.offset, [1, 1, 1, LATENT_CHANNELS]),
        ] {
            if t.shape() != shape {
                return Err(Error::InvalidInput(format!("noise.{name} has shape {:?}, expected {shape:?}", t.shape())));
            }
        }
        Ok(())
    }
}

pub struct Engine {
    pool: rayon::ThreadPool,
    unet: UNet,
    vae: Vae,
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
        // Weight packing and constant precomputation are parallel work too.
        let (unet, vae) = pool.install(|| -> Result<_> {
            let unet = UNet::load(&Weights::open(&weights_dir.join("unet.safetensors"))?)?;
            let vae = Vae::load(&Weights::open(&weights_dir.join("vae.safetensors"))?)?;
            Ok((unet, vae))
        })?;
        Ok(Self { pool, unet, vae })
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
        self.inpaint_with_noise(img, mask, params, &NoiseSet::sample(params.seed), progress)
    }

    /// Like [`Engine::inpaint`] with caller-supplied noise (parity runs feed
    /// the recorded PyTorch noise).
    pub fn inpaint_with_noise(
        &self,
        img: &mut ImageMut<'_>,
        mask: MaskRef<'_>,
        params: &InpaintParams,
        noise: &NoiseSet,
        progress: &mut dyn FnMut(Progress) -> Control,
    ) -> Result<Rect> {
        self.inpaint_traced(img, mask, params, noise, progress, None)
    }

    /// Like [`Engine::inpaint_with_noise`], also storing the final scaled
    /// latents (`[1,64,64,4]`, before decoding) in `latents` when given.
    pub fn inpaint_traced(
        &self,
        img: &mut ImageMut<'_>,
        mask: MaskRef<'_>,
        params: &InpaintParams,
        noise: &NoiseSet,
        progress: &mut dyn FnMut(Progress) -> Control,
        latents: Option<&mut Vec<f32>>,
    ) -> Result<Rect> {
        if (img.width, img.height) != (mask.width, mask.height) {
            return Err(Error::InvalidInput("image and mask sizes differ".into()));
        }
        if !matches!(img.channels, 3 | 4) {
            return Err(Error::InvalidInput("image must be RGB or RGBA".into()));
        }
        if img.width == 0 || img.height == 0 {
            return Err(Error::InvalidInput("empty image".into()));
        }
        if img.stride < img.width * img.channels
            || mask.stride < mask.width
            || img.data.len() < (img.height - 1) * img.stride + img.width * img.channels
            || mask.data.len() < (mask.height - 1) * mask.stride + mask.width
        {
            return Err(Error::InvalidInput("buffer smaller than width, height and stride imply".into()));
        }
        if params.steps == 0 || params.steps > 1000 {
            return Err(Error::InvalidInput(format!("steps must be 1..=1000, got {}", params.steps)));
        }
        noise.check()?;

        let dilated;
        let mask = if params.mask_dilate > 0 {
            dilated = dilate(mask, params.mask_dilate as usize);
            MaskRef { data: &dilated, width: mask.width, height: mask.height, stride: mask.width }
        } else {
            mask
        };
        let Some(bbox) = mask.bbox() else {
            return Ok(Rect::default());
        };
        let plan = CropPlan::new(bbox, img.width, img.height);

        // The model only reads `img`; it is written once, by the paste, after
        // the last cancellation point. `run` stays on the caller's thread (the
        // progress callback must not hop threads) and sends each heavy call
        // to the engine's pool.
        let result = self.run(img, mask, &plan, params, noise, progress, latents)?;
        Ok(paste(img, mask, &result, &plan, params.paste))
    }

    /// Steps 2-10a (heavy calls on the engine's pool): window → model → window-sized 8-bit RGB result.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        img: &ImageMut<'_>,
        mask: MaskRef<'_>,
        plan: &CropPlan,
        params: &InpaintParams,
        noise: &NoiseSet,
        progress: &mut dyn FnMut(Progress) -> Control,
        latents_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u8>> {
        let mut scratch = Scratch::default();
        let mut report = |stage, done, total| match progress(Progress { stage, done, total }) {
            Control::Continue => Ok(()),
            Control::Cancel => Err(Error::Cancelled),
        };
        let lat = LATENT_SIZE * LATENT_SIZE;

        // 2-3: window → 512² [-1,1], binary mask, masked image.
        report(Stage::Encode, 0, 1)?;
        let (rgb, bin) = to_model(img, mask, plan);
        let masked: Vec<f32> = rgb
            .chunks_exact(3)
            .zip(&bin)
            .flat_map(|(p, &m)| {
                let keep = if m >= 128 { 0.0 } else { 1.0 }; // image · (1 − mask)
                [p[0] * keep, p[1] * keep, p[2] * keep]
            })
            .collect();

        // 4: VAE-encode [image, masked] as one batch (posterior sampled with the given noise).
        let (z_img, z_masked) = if params.single_encode {
            let masked = Tensor::from_vec(masked, 1, IMAGE_SIZE, IMAGE_SIZE, 3);
            let z = self.pool.install(|| self.vae.encode(&masked, &noise.vae_masked, &mut scratch));
            (z.data.clone(), z.data)
        } else {
            let images = Tensor::from_vec([rgb, masked].concat(), 2, IMAGE_SIZE, IMAGE_SIZE, 3);
            let noises = Tensor::from_vec(
                [noise.vae_img.data.as_slice(), noise.vae_masked.data.as_slice()].concat(),
                2,
                LATENT_SIZE,
                LATENT_SIZE,
                LATENT_CHANNELS,
            );
            let z = self.pool.install(|| self.vae.encode(&images, &noises, &mut scratch));
            (z.item(0).to_vec(), z.item(1).to_vec())
        };
        report(Stage::Encode, 1, 1)?;

        // 5: nearest downsample of the mask (torch `interpolate` picks pixel (8y, 8x)).
        let mask_lat: Vec<f32> = (0..lat)
            .map(|i| {
                let (y, x) = (i / LATENT_SIZE, i % LATENT_SIZE);
                if bin[(y * VAE_DOWNSAMPLE) * IMAGE_SIZE + x * VAE_DOWNSAMPLE] >= 128 { 1.0 } else { 0.0 }
            })
            .collect();

        // 6: init = N(0,1) + noise_offset · N(0,1)[per channel].
        let init: Vec<f32> = noise
            .init
            .data
            .chunks_exact(LATENT_CHANNELS)
            .flat_map(|p| std::array::from_fn::<f32, LATENT_CHANNELS, _>(|c| p[c] + params.noise_offset * noise.offset.data[c]))
            .collect();

        // 7: x = noise if strength ≥ 1 else add_noise(z_img, noise, t₀).
        let ddim = Ddim::new(params.steps, params.strength, params.spacing);
        let steps = ddim.timesteps();
        let mut x = vec![0.0f32; lat * LATENT_CHANNELS];
        if ddim.pure_noise_start() {
            x.copy_from_slice(&init);
        } else {
            ddim.add_noise(&z_img, &init, steps[0], &mut x);
        }

        // 8: denoise. UNet input per pixel = [x(4) ‖ mask(1) ‖ masked latent(4)].
        let mut input = Tensor::zeros(1, LATENT_SIZE, LATENT_SIZE, UNET_IN_CHANNELS);
        let mut eps = vec![0.0f32; x.len()];
        for (i, &t) in steps.iter().enumerate() {
            for (p, o) in input.data.chunks_exact_mut(UNET_IN_CHANNELS).enumerate() {
                o[..4].copy_from_slice(&x[p * 4..p * 4 + 4]);
                o[4] = mask_lat[p];
                o[5..].copy_from_slice(&z_masked[p * 4..p * 4 + 4]);
            }
            if params.cfg_steps.is_none_or(|n| i < n) {
                let e = self.pool.install(|| self.unet.forward(&input, t, Guidance::Cfg, &mut scratch));
                let (u, c) = (e.item(0), e.item(1));
                for ((o, &u), &c) in eps.iter_mut().zip(u).zip(c) {
                    *o = u + params.guidance * (c - u);
                }
            } else {
                let e = self.pool.install(|| self.unet.forward(&input, t, Guidance::CondOnly, &mut scratch));
                eps.copy_from_slice(e.item(0));
            }
            ddim.step(&eps, t, &mut x);
            report(Stage::Denoise, i + 1, steps.len())?;
        }
        if let Some(out) = latents_out {
            out.clone_from(&x);
        }

        // 9: decode (M8: params.roi_decode decodes only a window around the mask).
        report(Stage::Decode, 0, 1)?;
        let latents = Tensor::from_vec(x, 1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS);
        let decoded = self.pool.install(|| self.vae.decode(&latents, &mut scratch));
        report(Stage::Decode, 1, 1)?;

        // 10 (first half): 8-bit truncation, resized back to the window.
        Ok(from_model(&decoded.data, plan.side))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_seeded_shaped_and_standard_normal() {
        let a = NoiseSet::sample(7);
        let b = NoiseSet::sample(7);
        let c = NoiseSet::sample(8);
        assert_eq!(a.vae_img.shape(), [1, 64, 64, 4]);
        assert_eq!(a.vae_masked.shape(), [1, 64, 64, 4]);
        assert_eq!(a.init.shape(), [1, 64, 64, 4]);
        assert_eq!(a.offset.shape(), [1, 1, 1, 4]);
        assert_eq!(a.init, b.init);
        assert_eq!(a.offset, b.offset);
        assert_ne!(a.init, c.init);
        assert_ne!(a.vae_img, a.vae_masked, "tensors are drawn sequentially, not copied");
        let d = &a.init.data;
        let mean = d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
        let var = d.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d.len() as f64;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.05, "mean {mean} var {var}");
    }
}

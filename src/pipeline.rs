//! End-to-end inpainting, mirroring `RemovalSDXLPipeline_BatchMode` (M7).
//!
//! ```text
//!  1  bbox(mask) → CropPlan (square window)                imaging::crop
//!  2  window → 512² RGB [-1,1] + binary mask (Lanczos)      imaging::resize
//!  3  masked = rgb · (1 − mask)
//!  4  z_img, z_masked = VAE.encode([rgb, masked]) · 0.13025 vae::encoder
//!       (Fast/roi_encode: only the image around the mask is
//!       encoded, pasted over z_masked)
//!  5  mask_lat = nearest(mask → 64²)
//!  6  noise = N(0,1) + noise_offset · N(0,1)[per channel]   seeded
//!  7  x = noise if strength ≥ 1 else add_noise(z_img, noise, t₀)
//!  8  for t in timesteps:                                   scheduler
//!        ε = UNet([x ‖ mask_lat ‖ z_masked], t)             unet (B=2 / B=1)
//!        ε = ε_u + g·(ε_c − ε_u)                            (CFG steps)
//!        x = solver.step(ε, t, x)                           DDIM / DPM++(2M)
//!        progress(); bail out if cancelled
//!  9  rgb = VAE.decode(x / 0.13025)                         vae::decoder
//!       (roi_decode, in no preset: decode latents[bbox(mask_lat) ± 8],
//!       rest = input image)
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
use crate::preset::{InpaintParams, Solver};
use crate::scheduler::{Ddim, Dpm2M};
use crate::unet::{Guidance, UNet};
use crate::vae::{LatentRect, ROI_MARGIN, Vae, roi_window};
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
        crate::engine::cpu::disable_power_throttling();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("moebius-{i}"))
            .start_handler(|_| crate::engine::cpu::disable_power_throttling())
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

        // 5: nearest downsample of the mask (torch `interpolate` picks pixel (8y, 8x)).
        let mask_lat: Vec<f32> = (0..lat)
            .map(|i| {
                let (y, x) = (i / LATENT_SIZE, i % LATENT_SIZE);
                if bin[(y * VAE_DOWNSAMPLE) * IMAGE_SIZE + x * VAE_DOWNSAMPLE] >= 128 { 1.0 } else { 0.0 }
            })
            .collect();

        // 4: VAE-encode [image, masked] as one batch (posterior sampled with the given noise).
        let window = roi_window(&mask_lat, ROI_MARGIN).filter(|r| r.w < LATENT_SIZE || r.h < LATENT_SIZE);
        let (z_img, z_masked) = if let (true, Some(r)) = (params.roi_encode, window) {
            let masked = Tensor::from_vec(masked, 1, IMAGE_SIZE, IMAGE_SIZE, 3);
            let z = self.pool.install(|| self.vae.encode(&masked, &noise.vae_masked, &mut scratch));
            let z_masked = z.data.to_vec();
            scratch.recycle(z);
            let (px, c) = (VAE_DOWNSAMPLE, LATENT_CHANNELS);
            let crop = crop_rows(&rgb, IMAGE_SIZE, 3, r.x * px, r.y * px, r.w * px, r.h * px);
            let crop = Tensor::from_vec(crop, 1, r.h * px, r.w * px, 3);
            let crop_noise = crop_rows(&noise.vae_img.data, LATENT_SIZE, c, r.x, r.y, r.w, r.h);
            let crop_noise = Tensor::from_vec(crop_noise, 1, r.h, r.w, c);
            let part = self.pool.install(|| self.vae.encode(&crop, &crop_noise, &mut scratch));
            // The window's border latents see zero padding: keep only the part nearer the mask.
            let inner = roi_window(&mask_lat, ROI_MARGIN / 2).expect("the outer window exists");
            let mut z_img = z_masked.clone();
            for y in inner.y..inner.y + inner.h {
                let src = ((y - r.y) * r.w + inner.x - r.x) * c;
                z_img[(y * LATENT_SIZE + inner.x) * c..][..inner.w * c].copy_from_slice(&part.data[src..][..inner.w * c]);
            }
            scratch.recycle(part);
            (z_img, z_masked)
        } else {
            let images = Tensor::from_vec([rgb.as_slice(), masked.as_slice()].concat(), 2, IMAGE_SIZE, IMAGE_SIZE, 3);
            let noises = Tensor::from_vec(
                [noise.vae_img.data.as_slice(), noise.vae_masked.data.as_slice()].concat(),
                2,
                LATENT_SIZE,
                LATENT_SIZE,
                LATENT_CHANNELS,
            );
            let z = self.pool.install(|| self.vae.encode(&images, &noises, &mut scratch));
            let pair = (z.item(0).to_vec(), z.item(1).to_vec());
            scratch.recycle(z);
            pair
        };
        // The encoder's 512² feature maps (3 x 134 MB) are dead until the decoder: do not
        // keep them resident through the whole denoising loop.
        scratch.release_below(usize::MAX);
        report(Stage::Encode, 1, 1)?;

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
        let mut dpm = Dpm2M::default();
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
                scratch.recycle(e);
            } else {
                let e = self.pool.install(|| self.unet.forward(&input, t, Guidance::CondOnly, &mut scratch));
                eps.copy_from_slice(e.item(0));
                scratch.recycle(e);
            }
            match params.solver {
                Solver::Ddim => ddim.step(&eps, t, &mut x),
                Solver::DpmPp2M => {
                    if params.cfg_steps == Some(i) {
                        dpm.restart();
                    }
                    dpm.step(&ddim, &eps, i, &mut x);
                }
            }
            report(Stage::Denoise, i + 1, steps.len())?;
        }
        if let Some(out) = latents_out {
            out.clone_from(&x);
        }

        // 9: decode. ROI: only a window around the mask (its latent bbox ± ROI_MARGIN);
        // the rest of the 512² frame is the model-input image, which the paste never
        // reads (outside mask + feather the source wins) but which keeps every blend
        // input defined. An empty latent mask (tiny mask between sample points) or a
        // window covering the whole latent decodes in full.
        report(Stage::Decode, 0, 1)?;
        let latents = Tensor::from_vec(x, 1, LATENT_SIZE, LATENT_SIZE, LATENT_CHANNELS);
        let roi = if params.roi_decode { roi_window(&mask_lat, ROI_MARGIN) } else { None };
        let decoded = self.pool.install(|| match roi {
            Some(r) if r.w < LATENT_SIZE || r.h < LATENT_SIZE => {
                let part = self.vae.decode_roi(&latents, r, &mut scratch);
                let frame = roi_frame(&rgb, r, &part.data);
                scratch.recycle(part);
                Tensor::from_vec(frame, 1, IMAGE_SIZE, IMAGE_SIZE, 3)
            }
            _ => self.vae.decode(&latents, &mut scratch),
        });
        report(Stage::Decode, 1, 1)?;

        // 10 (first half): 8-bit truncation, resized back to the window.
        let result = from_model(&decoded.data, plan.side);
        scratch.recycle(decoded);
        Ok(result)
    }
}

/// The `w x h` window at (`x`, `y`) of a row-major image `stride` pixels wide, `c` channels.
fn crop_rows(data: &[f32], stride: usize, c: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<f32> {
    (y..y + h).flat_map(|row| &data[(row * stride + x) * c..][..w * c]).copied().collect()
}

/// 512² model-space frame `(x + 1) / 2` of the input image `rgb` (NHWC, [-1, 1]) with the
/// decoded ROI `part` (`[8·roi.h, 8·roi.w, 3]`, already `(x + 1) / 2`) placed at `roi`.
fn roi_frame(rgb: &[f32], roi: LatentRect, part: &[f32]) -> Vec<f32> {
    let (px, row) = (VAE_DOWNSAMPLE, IMAGE_SIZE * 3);
    let (wx, wy, ww, wh) = (roi.x * px, roi.y * px, roi.w * px, roi.h * px);
    assert_eq!(part.len(), ww * wh * 3, "decoded ROI does not match its window");
    let mut frame: Vec<f32> = rgb.iter().map(|&v| (v + 1.0) * 0.5).collect();
    for y in 0..wh {
        frame[(wy + y) * row + wx * 3..][..ww * 3].copy_from_slice(&part[y * ww * 3..][..ww * 3]);
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roi_frame_places_the_window_and_keeps_the_input_elsewhere() {
        let rgb: Vec<f32> = (0..IMAGE_SIZE * IMAGE_SIZE * 3).map(|i| (i % 7) as f32 / 3.5 - 1.0).collect();
        let roi = LatentRect { x: 2, y: 3, w: 4, h: 5 };
        let part = vec![0.25f32; 32 * 40 * 3];
        let frame = roi_frame(&rgb, roi, &part);
        let at = |x: usize, y: usize| (y * IMAGE_SIZE + x) * 3;
        // Inside: the decoded window, first and last pixel.
        assert_eq!(frame[at(16, 24)], 0.25);
        assert_eq!(frame[at(16 + 31, 24 + 39) + 2], 0.25);
        // Just outside on every side, and the far corner: (rgb + 1) / 2.
        for (x, y) in [(15, 24), (48, 24), (16, 23), (16, 64), (0, 0), (511, 511)] {
            assert_eq!(frame[at(x, y)], (rgb[at(x, y)] + 1.0) * 0.5, "({x}, {y})");
        }
    }

    #[test]
    fn crop_rows_takes_the_window() {
        let data: Vec<f32> = (0..4 * 3 * 2).map(|i| i as f32).collect(); // 4 wide, 3 high, 2 channels
        assert_eq!(crop_rows(&data, 4, 2, 1, 1, 2, 2), [10.0, 11.0, 12.0, 13.0, 18.0, 19.0, 20.0, 21.0]);
    }

    #[test]
    fn roi_frame_of_the_full_window_is_the_decoded_image() {
        let rgb = vec![-1.0f32; IMAGE_SIZE * IMAGE_SIZE * 3];
        let part: Vec<f32> = (0..rgb.len()).map(|i| i as f32).collect();
        let full = LatentRect { x: 0, y: 0, w: LATENT_SIZE, h: LATENT_SIZE };
        assert_eq!(roi_frame(&rgb, full, &part), part);
    }

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

//! Inference presets. `Exact` reproduces upstream `infer/infer_moebius.py`
//! defaults; `Fast` trades exactness for time within the budget in
//! `docs/PLAN.md` §5. Fast values are provisional until the M8 quality gate.

use crate::scheduler::Spacing;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Exact,
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paste {
    /// Return the raw 512² result resized back (upstream `pst=False`).
    None,
    /// Blend through the mask blurred with σ = 3 (upstream `pst=True`).
    Feather,
    /// Feather plus mean-colour compensation (upstream `cps=True`).
    FeatherCompensate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Upstream's sampler.
    Ddim,
    /// DPM-Solver++(2M), restarting its history where guidance switches off.
    DpmPp2M,
    /// DPM-Solver++(2M) keeping its history across that switch.
    DpmPp2MContinuous,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InpaintParams {
    pub steps: usize,
    pub strength: f64,
    pub spacing: Spacing,
    pub solver: Solver,
    pub guidance: f32,
    pub noise_offset: f32,
    pub seed: u64,
    /// Run classifier-free guidance (batch 2) for only the first N denoising
    /// steps, conditional-only (batch 1) afterwards. `None` = every step.
    pub cfg_steps: Option<usize>,
    /// Encode only the masked image and reuse its latents for the initial
    /// `add_noise` (saves one of the two VAE encodes).
    pub single_encode: bool,
    /// Encode the original image only around the mask: `z_img` is `z_masked` with the
    /// mask's neighbourhood replaced. Ignored with `single_encode`.
    pub roi_encode: bool,
    /// Decode only the latent window around the mask (+ margin).
    pub roi_decode: bool,
    pub paste: Paste,
    /// Square-kernel mask dilation in pixels before inference (0 = off).
    pub mask_dilate: u32,
}

impl Preset {
    pub fn params(self) -> InpaintParams {
        match self {
            Preset::Exact => InpaintParams {
                steps: 20,
                strength: 0.99,
                spacing: Spacing::Leading,
                solver: Solver::Ddim,
                guidance: 2.5,
                noise_offset: 0.0357,
                seed: 0,
                cfg_steps: None,
                single_encode: false,
                roi_encode: false,
                roi_decode: false,
                paste: Paste::Feather,
                mask_dilate: 0,
            },
            Preset::Fast => InpaintParams {
                steps: 7,
                cfg_steps: Some(3),
                single_encode: true,
                roi_decode: true,
                ..Preset::Exact.params()
            },
        }
    }
}

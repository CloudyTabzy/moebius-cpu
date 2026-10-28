//! DDIM, matching diffusers `DDIMScheduler` exactly as Moebius builds it:
//! `beta_start=0.00085, beta_end=0.012, scaled_linear, 1000 steps,
//! clip_sample=False` with the remaining defaults (`set_alpha_to_one=True`,
//! `steps_offset=0`, `eta=0`, epsilon prediction), plus the
//! `strength` slicing of `removal/v1_2/pipeline.py::get_timesteps`.
//!
//! Hand-written instead of borrowed: candle's DDIM defaults to
//! `steps_offset=1` and uses `alphas_cumprod[0]` on the final step where
//! diffusers uses 1.0. Arithmetic is f32 in torch's operation order.
//! Expected values in the tests come from `tools/dump_ddim_reference.py`.

const TRAIN_STEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spacing {
    /// Upstream Moebius (diffusers default): 0, r, 2r, … reversed.
    Leading,
    /// Starts at t = 999; candidate for the Fast preset's short schedules.
    Trailing,
}

pub struct Ddim {
    alphas_cumprod: Vec<f32>,
    step_ratio: usize,
    timesteps: Vec<usize>,
    pure_noise_start: bool,
}

impl Ddim {
    pub fn new(num_inference_steps: usize, strength: f64, spacing: Spacing) -> Self {
        assert!((1..=TRAIN_STEPS).contains(&num_inference_steps));
        let alphas_cumprod = scaled_linear_alphas_cumprod();
        let step_ratio = TRAIN_STEPS / num_inference_steps;
        let all: Vec<usize> = match spacing {
            Spacing::Leading => (0..num_inference_steps).rev().map(|i| i * step_ratio).collect(),
            Spacing::Trailing => {
                let r = TRAIN_STEPS as f64 / num_inference_steps as f64;
                (0..num_inference_steps)
                    .map(|i| (TRAIN_STEPS as f64 - i as f64 * r).round() as usize - 1)
                    .collect()
            }
        };
        // get_timesteps(): keep the last int(steps * strength) entries.
        let init = ((num_inference_steps as f64 * strength) as usize).min(num_inference_steps);
        let t_start = num_inference_steps - init;
        Self {
            alphas_cumprod,
            step_ratio,
            timesteps: all[t_start..].to_vec(),
            pure_noise_start: strength >= 1.0,
        }
    }

    /// Timesteps actually denoised (20 steps at strength 0.99 → 19, from t=900).
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Upstream starts from `noise` alone when strength >= 1, otherwise from
    /// `add_noise(latents, noise, timesteps[0])`.
    pub fn pure_noise_start(&self) -> bool {
        self.pure_noise_start
    }

    pub fn alpha_cumprod(&self, t: usize) -> f32 {
        self.alphas_cumprod[t]
    }

    /// `out = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·noise`.
    pub fn add_noise(&self, x0: &[f32], noise: &[f32], t: usize, out: &mut [f32]) {
        let a = self.alphas_cumprod[t];
        let (sa, sb) = (a.sqrt(), (1.0 - a).sqrt());
        for ((o, &x), &n) in out.iter_mut().zip(x0).zip(noise) {
            *o = sa * x + sb * n;
        }
    }

    /// One deterministic (eta = 0) DDIM update of `sample` in place.
    pub fn step(&self, eps: &[f32], t: usize, sample: &mut [f32]) {
        let a_t = self.alphas_cumprod[t];
        let a_prev = t.checked_sub(self.step_ratio).map_or(1.0, |p| self.alphas_cumprod[p]);
        let (sqrt_b_t, sqrt_a_t) = ((1.0 - a_t).sqrt(), a_t.sqrt());
        let (sqrt_a_prev, sqrt_b_prev) = (a_prev.sqrt(), (1.0 - a_prev).sqrt());
        for (x, &e) in sample.iter_mut().zip(eps) {
            let x0 = (*x - sqrt_b_t * e) / sqrt_a_t;
            *x = sqrt_a_prev * x0 + sqrt_b_prev * e;
        }
    }
}

/// torch: `linspace(√β0, √β1, 1000, f32)² → cumprod(1 - β)` in f32.
fn scaled_linear_alphas_cumprod() -> Vec<f32> {
    let (start, end) = (BETA_START.sqrt() as f32, BETA_END.sqrt() as f32);
    let n = TRAIN_STEPS;
    let step = (end - start) / (n - 1) as f32;
    let mut acc = 1.0f32;
    (0..n)
        .map(|i| {
            // torch's CPU linspace fills the two halves from opposite ends.
            let s = if i < n / 2 { start + step * i as f32 } else { end - step * (n - 1 - i) as f32 };
            acc *= 1.0 - s * s;
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: &[f32], want: &[f64], rel: f64) {
        for (g, w) in got.iter().zip(want) {
            assert!(((*g as f64) - w).abs() <= rel * w.abs().max(1e-3), "got {g}, want {w}");
        }
    }

    #[test]
    fn timesteps_match_diffusers() {
        let d = Ddim::new(20, 0.99, Spacing::Leading);
        let want: Vec<usize> = (0..=18).rev().map(|i| i * 50).collect();
        assert_eq!(d.timesteps(), want.as_slice());
        assert!(!d.pure_noise_start());
        assert_eq!(Ddim::new(7, 0.99, Spacing::Leading).timesteps(), &[710, 568, 426, 284, 142, 0]);
        let full = Ddim::new(20, 1.0, Spacing::Leading);
        assert_eq!(full.timesteps().len(), 20);
        assert_eq!(full.timesteps()[0], 950);
        assert!(full.pure_noise_start());
        assert_eq!(Ddim::new(4, 1.0, Spacing::Trailing).timesteps(), &[999, 749, 499, 249]);
    }

    #[test]
    fn alphas_cumprod_match_diffusers() {
        let d = Ddim::new(20, 0.99, Spacing::Leading);
        let got: Vec<f32> = [0, 1, 500, 900, 999].iter().map(|&t| d.alpha_cumprod(t)).collect();
        close(&got, &[9.991499782e-01, 9.982960224e-01, 2.763324678e-01, 1.415113267e-02, 4.660095088e-03], 1e-5);
    }

    #[test]
    fn step_and_add_noise_match_diffusers() {
        let d = Ddim::new(20, 0.99, Spacing::Leading);
        let x = [0.5f32, -1.25, 2.0];
        let eps = [0.1f32, -0.3, 0.7];
        for (t, want) in [
            (900, [6.129949689e-01, -1.518200040e+00, 2.366255760e+00]),
            (50, [4.927129745e-01, -1.221965790e+00, 1.911952496e+00]),
            (0, [4.972958863e-01, -1.241781354e+00, 1.980433226e+00]), // ᾱ_prev = 1
        ] {
            let mut s = x;
            d.step(&eps, t, &mut s);
            close(&s, &want, 1e-5);
        }
        let mut noisy = [0.0f32; 3];
        d.add_noise(&x, &eps, 900, &mut noisy);
        close(&noisy, &[1.587691903e-01, -4.465679526e-01, 9.329465032e-01], 1e-5);
    }
}

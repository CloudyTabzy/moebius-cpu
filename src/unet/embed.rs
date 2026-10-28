//! Time and conditioning embeddings (M5).
//!
//! * Time: diffusers `Timesteps(320, flip_sin_to_cos=True, freq_shift=0)` →
//!   `TimestepEmbedding` (Linear 320→1280, SiLU, Linear 1280→1280). Each
//!   resnet then applies SiLU + its own `time_emb_proj`; all of that is a
//!   function of `t` alone and is computed once per step for every resnet.
//! * Conditioning: `Embedding(20, 3072)` rows 10..20 (uncond) and 0..10
//!   (cond) → `encoder_hid_proj` Linear 3072→768. Fixed ids, so the two
//!   `[10, 768]` hidden-state matrices are constants built once at load.

use super::load::Loader;
use crate::Result;
use crate::config::{
    COND_TOKENS, CROSS_ATTENTION_DIM, ENCODER_HID_DIM, NUM_EMBEDDINGS, TIME_EMBED_DIM, TIME_PROJ_DIM,
};
use crate::engine::conv::Conv1x1;
use crate::engine::ops::silu_inplace;
use crate::engine::{Scratch, Tensor};

/// `[cos | sin]` of `t · exp(−ln(10000)·i/160)`, `i in 0..160`, in f32 like torch.
pub fn timestep_sinusoid(t: usize, out: &mut [f32; TIME_PROJ_DIM]) {
    let half = TIME_PROJ_DIM / 2;
    let log_period = -(10000.0f64.ln() as f32);
    for i in 0..half {
        let freq = (log_period * i as f32 / half as f32).exp();
        let arg = t as f32 * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
}

/// `time_embedding`: Linear(320→1280), SiLU, Linear(1280→1280).
pub struct TimeEmbedding {
    linear_1: Conv1x1,
    linear_2: Conv1x1,
}

impl TimeEmbedding {
    /// `l` is the `time_embedding` module.
    pub fn load(l: &Loader) -> Result<Self> {
        Ok(Self {
            linear_1: l.conv1x1("linear_1", TIME_PROJ_DIM, TIME_EMBED_DIM, true)?,
            linear_2: l.conv1x1("linear_2", TIME_EMBED_DIM, TIME_EMBED_DIM, true)?,
        })
    }

    /// The raw embedding for timestep `t`, tiled over `batch` items: `[batch, 1, 1, 1280]`.
    pub fn forward(&self, t: usize, batch: usize, s: &mut Scratch) -> Tensor {
        let mut sin = [0.0f32; TIME_PROJ_DIM];
        timestep_sinusoid(t, &mut sin);
        let x = Tensor::from_vec(sin.to_vec(), 1, 1, 1, TIME_PROJ_DIM);
        let (mut a, mut b) = (Tensor::default(), Tensor::default());
        self.linear_1.forward(&x, &mut a, None, s);
        silu_inplace(&mut a.data);
        self.linear_2.forward(&a, &mut b, None, s);
        Tensor::from_vec(b.data.repeat(batch), batch, 1, 1, TIME_EMBED_DIM)
    }
}

/// Constant cross-attention hidden states `[uncond, cond]`, each `[10 × 768]` row-major.
///
/// `root` is the loader at the top of the file (`embedding_layer`), `diff` the one at `diff_model`.
pub fn conditioning(root: &Loader, diff: &Loader, s: &mut Scratch) -> Result<[Vec<f32>; 2]> {
    let table = root.sub("embedding_layer").f32s("weight", &[NUM_EMBEDDINGS, ENCODER_HID_DIM])?;
    let proj = diff.conv1x1("encoder_hid_proj", ENCODER_HID_DIM, CROSS_ATTENTION_DIM, true)?;
    let project = |first_row: usize, s: &mut Scratch| {
        let rows = table[first_row * ENCODER_HID_DIM..(first_row + COND_TOKENS) * ENCODER_HID_DIM].to_vec();
        let x = Tensor::from_vec(rows, 1, 1, COND_TOKENS, ENCODER_HID_DIM);
        let mut out = Tensor::default();
        proj.forward(&x, &mut out, None, s);
        out.data
    };
    Ok([project(COND_TOKENS, s), project(0, s)])
}

//! Time and conditioning embeddings (M5).
//!
//! * Time: diffusers `Timesteps(320, flip_sin_to_cos=True, freq_shift=0)` →
//!   `TimestepEmbedding` (Linear 320→1280, SiLU, Linear 1280→1280). Each
//!   resnet then applies SiLU + its own `time_emb_proj`; all of that is a
//!   function of `t` alone and is computed once per step for every resnet.
//! * Conditioning: `Embedding(20, 3072)` rows 10..20 (uncond) and 0..10
//!   (cond) → `text_proj` Linear 3072→768. Fixed ids, so the two `[10, 768]`
//!   hidden-state matrices are constants built once at load.

pub fn timestep_sinusoid(_t: usize, _out: &mut [f32; crate::config::TIME_PROJ_DIM]) {
    todo!("M5: [cos | sin] halves, exponent = -ln(10000)·i/160")
}

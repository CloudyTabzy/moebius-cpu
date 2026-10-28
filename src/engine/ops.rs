//! Elementwise ops that are not fused into a neighbouring kernel.

use rayon::prelude::*;

use super::math::silu;
use super::simd::multiversion;

const CHUNK: usize = 1 << 15;

/// `x = SiLU(x)` (MixFFN inverted-conv activation, time-embedding MLP).
pub fn silu_inplace(x: &mut [f32]) {
    x.par_chunks_mut(CHUNK).for_each(silu_chunk);
}

/// `a += b`.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "add length mismatch");
    a.par_chunks_mut(CHUNK).zip(b.par_chunks(CHUNK)).for_each(|(a, b)| add_chunk(a, b));
}

multiversion! {
    fn silu_chunk(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = silu(*v);
        }
    }
}

multiversion! {
    fn add_chunk(a: &mut [f32], b: &[f32]) {
        for (a, &b) in a.iter_mut().zip(b) {
            *a += b;
        }
    }
}

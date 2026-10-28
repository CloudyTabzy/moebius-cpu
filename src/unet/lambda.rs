//! LλMI blocks: multi-query lambda self- and cross-layers (M3).
//!
//! Notation per layer: C channels, h = 8 heads, k = C/8 (query/key width),
//! v = C/8 (value width), u = 1, N = H·W tokens. BatchNorms (`norm_q`,
//! `norm_v`) are eval-mode and folded into `to_q` / `to_v` by the converter.
//!
//! # Self-λ (`MultiQuerySelfLambda`, local, r = 15)
//! ```text
//! Q[h,k,n] = to_q(x)          K[k,n] = softmax_n(to_k(x))     V[v,n] = to_v(x)
//! λc[k,v]  = Σ_n K[k,n] V[v,n]                       content lambda (global)
//! Yc[h,v,n] = Σ_k Q[h,k,n] λc[k,v]
//! λp[k,v,n] = Σ_δ W[k,δ] V[v,n+δ] + b[k]              δ ∈ 15×15 window
//! Yp[h,v,n] = Σ_k Q[h,k,n] λp[k,v,n]
//! ```
//! Upstream materialises `λp` with a `Conv3d(1, k, (1,15,15))` — 14% of the
//! UNet's FLOPs and a `[k·v·N]` intermediate. Reassociating the sums:
//! ```text
//! A[h,δ,n]  = Σ_k Q[h,k,n] W[k,δ]                      small GEMM, 225 cols
//! Yp[h,v,n] = Σ_δ A[h,δ,n] V[v,n+δ]  +  Σ_k Q[h,k,n] b[k]
//! ```
//! Cost drops from k·v·225·N to (k+v)·h·225·N: 2.5× / 5× / 10× fewer FLOPs
//! at the 64² / 32² / 16² levels (88.5 → ~21 GFLOP per CFG step). The second
//! line is a per-pixel 225-tap filter over V rows — an AXPY over the
//! contiguous NHWC channel vector, vectorised over v.
//! Output `Y = Yc + Yp`, heads×v re-flattened to C (no output projection).
//!
//! # Cross-λ (`MultiQueryCrossLambda`, global, m = 10 tokens)
//! `to_k` / `to_v` read the conditioning hidden states, which are fixed
//! (embedding ids 0..20 → text_proj). So per layer and per CFG branch,
//! `softmax(K)`, `V` and `λc = K Vᵀ` are **load-time constants**. The
//! position table `E = rel_pos_emb[n, m, k]` is indexed by an identity
//! meshgrid, so no gather is needed either. Per step only:
//! ```text
//! Q = to_q(x)                                   1x1 GEMM
//! Yc[h,v,n] = Σ_k Q[h,k,n] λc[k,v]               GEMM against a constant
//! B[h,m,n]  = Σ_k Q[h,k,n] E[n,m,k]              per-pixel h×k · k×m
//! Yp[h,v,n] = Σ_m B[h,m,n] V[m,v]                per-pixel h×m · m×v
//! ```
//! Precomputing `λp = E·V` instead would cost ~0.8 GB (15 layers × 2 CFG
//! branches × N·k·v f32) for no meaningful time win; the two-stage form keeps memory at the original
//! 14.3 M-parameter table. `E` is tied to N = 64²/32²/16², hence the hard
//! 512² input size.

pub struct SelfLambda;
pub struct CrossLambda;

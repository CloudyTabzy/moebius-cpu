//! Fixed architecture of the shipped Moebius student network.
//!
//! Values come from `config/model_cfg/moebius.yaml` plus the defaults of
//! `UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8`
//! (`model_lib/nets/unet_lambda_prune_lite.py`). The network is hardcoded
//! rather than parsed: the weights only fit this one shape.

/// Model input side in pixels. Not a default — a hard limit: the cross-λ
/// `rel_pos_emb` is a learned `(n², m, k, 1)` table with `n` = 64/32/16, and
/// the λ wrappers and MixFFN recover H = W = sqrt(tokens). Square 512 only.
pub const IMAGE_SIZE: usize = 512;
pub const VAE_DOWNSAMPLE: usize = 8;
pub const LATENT_SIZE: usize = IMAGE_SIZE / VAE_DOWNSAMPLE;
pub const LATENT_CHANNELS: usize = 4;

/// noisy latent (4) + mask (1) + masked-image latent (4).
pub const UNET_IN_CHANNELS: usize = 9;
pub const UNET_OUT_CHANNELS: usize = 4;
/// One entry per resolution level: 64², 32², 16² latents. No mid block.
pub const BLOCK_OUT_CHANNELS: [usize; 3] = [320, 640, 1280];
/// Down blocks hold 2 (resnet, transformer) pairs; up blocks hold 3.
pub const LAYERS_PER_BLOCK: usize = 2;
/// `attention_head_dim: 8` in moebius.yaml is the head *count* (diffusers
/// naming quirk); per-head width is channels / 8 = 40 / 80 / 160.
pub const NUM_HEADS: usize = 8;
pub const MIX_MLP_RATIO: f32 = 2.5;
pub const GROUP_NORM_GROUPS: usize = 32;

/// Timestep embedding: sinusoidal(320, flip_sin_to_cos, shift 0) → 1280.
pub const TIME_PROJ_DIM: usize = BLOCK_OUT_CHANNELS[0];
pub const TIME_EMBED_DIM: usize = 4 * BLOCK_OUT_CHANNELS[0];

/// Conditioning is not text: `Embedding(20, 3072)` → `text_proj` 3072 → 768.
/// Ids 0..10 are the conditional prompt, 10..20 the unconditional one.
pub const NUM_EMBEDDINGS: usize = 20;
pub const COND_TOKENS: usize = NUM_EMBEDDINGS / 2;
pub const ENCODER_HID_DIM: usize = 3072;
pub const CROSS_ATTENTION_DIM: usize = 768;

/// Self-λ local position kernel, `Conv3d(1, k, (1, 15, 15))`.
pub const SELF_LAMBDA_KERNEL: usize = 15;

/// UNet resnets and `conv_norm_out` (`norm_eps`).
pub const RESNET_EPS: f32 = 1e-5;
/// `MixTransformer2DModel.norm` hardcodes 1e-6.
pub const TRANSFORMER_GROUP_NORM_EPS: f32 = 1e-6;
/// `BasicTransformerBlock` LayerNorms.
pub const LAYER_NORM_EPS: f32 = 1e-5;
/// diffusers `AutoencoderKL` GroupNorms.
pub const VAE_EPS: f32 = 1e-6;

/// PixelHacker VAE (`sdvae_f8d4`, SDXL-VAE weights): block_out_channels
/// (128, 256, 512, 512), 2 layers. Shipped in fp16, so f16 storage is exact.
pub const VAE_BLOCK_OUT_CHANNELS: [usize; 4] = [128, 256, 512, 512];
/// From `hustvl/PixelHacker/vae/config.json`: the SDXL value, not SD-1.x's
/// 0.18215. The converter stores the config's value in the safetensors
/// metadata and the loader checks it against this constant.
pub const VAE_SCALING_FACTOR: f32 = 0.13025;

//! Moebius student UNet (M3–M5).
//!
//! ```text
//! conv_in   DWConv2d 9 → 320
//! down 0    64², 320  : 2 × (DWResnet, MixTransformer) + Downsample2D
//! down 1    32², 640  : 2 × (DWResnet, MixTransformer) + Downsample2D
//! down 2    16², 1280 : 2 × (DWResnet, MixTransformer)
//! (no mid block)
//! up 0      16², 1280 : 3 × (concat skip, DWResnet, MixTransformer) + Upsample2D
//! up 1      32², 640  : 3 × (…) + Upsample2D
//! up 2      64², 320  : 3 × (…)
//! out       GroupNorm(32)+SiLU → DWConv2d 320 → 4
//! ```
//!
//! Per step at batch 2 (CFG): 632 GFLOP — 64% 1x1 GEMM, 20% 3x3 dense
//! (Upsample2D convs), 14% λ position conv (→ ~3% after the fold in
//! `lambda.rs`), 1% stride-2, 0.5% depthwise (`tools/count_flops.py`).
//!
//! Batch layout: item 0 = unconditional, item 1 = conditional, as upstream.
//! Skip stack: the `conv_in` output, then every (resnet, transformer) output
//! of the down blocks and every downsampler output (9 entries); each up
//! resnet pops one and concatenates it after the hidden state (`[h ‖ skip]`).
//!
//! Batch layout: item 0 = unconditional, item 1 = conditional, as upstream.
//! The two items are identical until the first cross-λ, so `conv_in`, the
//! first resnet and the first self-λ could run once at batch 1 and be copied
//! (PLAN §6.5); for now the whole forward runs at batch 2. When the Fast
//! preset turns CFG off, the whole forward runs at batch 1.

pub mod blocks;
pub mod embed;
pub mod lambda;
pub mod load;
pub mod transformer;

use self::blocks::{Downsample, DwConv2d, DwResnet, Upsample};
use self::embed::TimeEmbedding;
use self::load::Loader;
use self::transformer::MixTransformer;
use crate::Result;
use crate::config::{BLOCK_OUT_CHANNELS, LAYERS_PER_BLOCK, RESNET_EPS, UNET_IN_CHANNELS, UNET_OUT_CHANNELS};
use crate::engine::norm::GroupNorm;
use crate::engine::{Scratch, Tensor};
use crate::weights::Weights;

/// Which conditioning rows to run: `[uncond, cond]` for CFG, `[cond]` without.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guidance {
    Cfg,
    CondOnly,
}

impl Guidance {
    /// Conditioning branch of each batch item (0 = uncond, 1 = cond).
    fn branches(self) -> &'static [usize] {
        match self {
            Guidance::Cfg => &[0, 1],
            Guidance::CondOnly => &[1],
        }
    }
}

struct DownBlock {
    resnets: Vec<DwResnet>,
    attns: Vec<MixTransformer>,
    downsample: Option<Downsample>,
}

struct UpBlock {
    resnets: Vec<DwResnet>,
    attns: Vec<MixTransformer>,
    upsample: Option<Upsample>,
}

pub struct UNet {
    time: TimeEmbedding,
    conv_in: DwConv2d,
    down: Vec<DownBlock>,
    up: Vec<UpBlock>,
    norm_out: GroupNorm,
    conv_out: DwConv2d,
}

impl UNet {
    /// Loads weights and precomputes everything input-independent:
    /// conditioning tables, cross-λ constants (`lambda::CrossLambda`).
    pub fn load(weights: &Weights) -> Result<Self> {
        let root = Loader::new(weights, "");
        let d = root.sub("diff_model");
        let ehs = embed::conditioning(&root, &d, &mut Scratch::default())?;
        let ch = BLOCK_OUT_CHANNELS;

        let mut down = Vec::new();
        let mut c_prev = ch[0];
        for (i, &c) in ch.iter().enumerate() {
            let l = d.sub(&format!("down_blocks.{i}"));
            let (mut resnets, mut attns) = (Vec::new(), Vec::new());
            for j in 0..LAYERS_PER_BLOCK {
                let c_in = if j == 0 { c_prev } else { c };
                resnets.push(DwResnet::load(&l.sub(&format!("resnets.{j}")), c_in, c)?);
                attns.push(MixTransformer::load(&l.sub(&format!("attentions.{j}")), c, &ehs)?);
            }
            let downsample =
                if i + 1 < ch.len() { Some(Downsample::load(&l.sub("downsamplers.0"), c)?) } else { None };
            down.push(DownBlock { resnets, attns, downsample });
            c_prev = c;
        }

        // Skip-stack channels in push order; up resnets pop them from the end.
        let mut skips = vec![ch[0]];
        for (i, &c) in ch.iter().enumerate() {
            skips.extend([c; LAYERS_PER_BLOCK]);
            if i + 1 < ch.len() {
                skips.push(c);
            }
        }
        let mut up = Vec::new();
        let mut h_c = ch[ch.len() - 1];
        for (i, &c) in ch.iter().rev().enumerate() {
            let l = d.sub(&format!("up_blocks.{i}"));
            let (mut resnets, mut attns) = (Vec::new(), Vec::new());
            for j in 0..=LAYERS_PER_BLOCK {
                let skip_c = skips.pop().expect("skip stack underflow");
                resnets.push(DwResnet::load(&l.sub(&format!("resnets.{j}")), h_c + skip_c, c)?);
                attns.push(MixTransformer::load(&l.sub(&format!("attentions.{j}")), c, &ehs)?);
                h_c = c;
            }
            let upsample = if i + 1 < ch.len() { Some(Upsample::load(&l.sub("upsamplers.0"), c)?) } else { None };
            up.push(UpBlock { resnets, attns, upsample });
        }
        assert!(skips.is_empty(), "skip stack not consumed");

        Ok(Self {
            time: TimeEmbedding::load(&d.sub("time_embedding"))?,
            conv_in: DwConv2d::load(&d.sub("conv_in"), UNET_IN_CHANNELS, ch[0])?,
            down,
            up,
            norm_out: d.group_norm("conv_norm_out", ch[0], RESNET_EPS)?,
            conv_out: DwConv2d::load(&d.sub("conv_out"), ch[0], UNET_OUT_CHANNELS)?,
        })
    }

    /// `x`: `[1, 64, 64, 9]` (noisy latent ‖ mask ‖ masked latent).
    /// Returns ε `[b, 64, 64, 4]` with b = 2 for `Cfg`, 1 for `CondOnly`.
    pub fn forward(&self, x: &Tensor, timestep: usize, guidance: Guidance, s: &mut Scratch) -> Tensor {
        self.forward_traced(x, timestep, guidance, s, &mut |_, _| {})
    }

    /// Like `forward`, calling `trace(name, tensor)` after every block with the
    /// names of the golden dump (`unet.conv_in`, `unet.down.0.res.1`, ...).
    pub fn forward_traced(
        &self,
        x: &Tensor,
        timestep: usize,
        guidance: Guidance,
        s: &mut Scratch,
        trace: &mut dyn FnMut(&str, &Tensor),
    ) -> Tensor {
        assert_eq!(x.n, 1, "UNet::forward takes one latent; CFG tiling happens inside");
        assert_eq!(x.c, UNET_IN_CHANNELS, "UNet input channels");
        let branches = guidance.branches();
        let b = branches.len();

        let temb = self.time.forward(timestep, b, s);
        trace("unet.temb", &temb);

        // Buffers come from the scratch pool sized by role (`tensor_for`) and go
        // back as soon as they are dead; see `Scratch::tensor_for`.
        let px = b * x.h * x.w;
        let mut input = s.tensor_for(px * x.c);
        input.set_shape(b, x.h, x.w, x.c);
        for item in input.data.chunks_exact_mut(x.data.len()) {
            item.copy_from_slice(&x.data);
        }
        let mut h = s.tensor_for(px * BLOCK_OUT_CHANNELS[0]);
        self.conv_in.forward(&input, &mut h, s);
        s.recycle(input);
        trace("unet.conv_in", &h);

        // Every skip is moved onto the stack and read from there (the next block's
        // input is the stack top), so the only copy is the down -> up hand-over.
        let mut skips = vec![h];
        for (i, blk) in self.down.iter().enumerate() {
            for (j, (res, attn)) in blk.resnets.iter().zip(&blk.attns).enumerate() {
                let top = skips.last().expect("skip stack");
                let mut a = s.tensor_for(top.pixels() * res.c_out);
                res.forward(top, &temb, &mut a, s);
                trace(&format!("unet.down.{i}.res.{j}"), &a);
                let mut h = s.tensor_for(a.data.len());
                attn.forward(&a, branches, &mut h, s);
                trace(&format!("unet.down.{i}.attn.{j}"), &h);
                s.recycle(a);
                skips.push(h);
            }
            if let Some(ds) = &blk.downsample {
                let top = skips.last().expect("skip stack");
                let mut h = s.tensor_for(top.data.len() / 4);
                ds.forward(top, &mut h, s);
                skips.push(h);
            }
            trace(&format!("unet.down.{i}.out"), skips.last().expect("skip stack"));
        }

        // The last down output is both the hidden state and the first skip.
        let last = skips.last().expect("skip stack");
        let mut h = s.tensor_for(last.data.len());
        h.set_shape(last.n, last.h, last.w, last.c);
        h.data.copy_from_slice(&last.data);
        for (i, blk) in self.up.iter().enumerate() {
            for (j, (res, attn)) in blk.resnets.iter().zip(&blk.attns).enumerate() {
                let skip = skips.pop().expect("skip stack underflow");
                let mut cat = s.tensor_for(h.pixels() * res.c_in);
                Tensor::concat_channels_into(&h, &skip, &mut cat);
                s.recycle(skip);
                s.recycle(h);
                let mut a = s.tensor_for(cat.pixels() * res.c_out);
                res.forward(&cat, &temb, &mut a, s);
                s.recycle(cat);
                trace(&format!("unet.up.{i}.res.{j}"), &a);
                h = s.tensor_for(a.data.len());
                attn.forward(&a, branches, &mut h, s);
                s.recycle(a);
                trace(&format!("unet.up.{i}.attn.{j}"), &h);
            }
            if let Some(us) = &blk.upsample {
                let mut a = s.tensor_for(h.data.len() * 4);
                us.forward(&h, &mut a, s);
                s.recycle(h);
                h = a;
            }
            trace(&format!("unet.up.{i}.out"), &h);
        }
        assert!(skips.is_empty(), "skip stack not consumed");

        let mut a = s.tensor_for(h.data.len());
        self.norm_out.forward(&h, None, true, &mut a);
        let mut out = s.tensor_for(h.pixels() * UNET_OUT_CHANNELS);
        self.conv_out.forward(&a, &mut out, s);
        trace("unet.out", &out);
        s.recycle(a);
        s.recycle(h);
        s.recycle(temb);
        out
    }
}

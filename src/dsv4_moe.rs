//! Host-side weight prep for the DSV4 routed-expert MoE (Phase 2 lane 2B;
//! DEEPSEEK_V4_PORT.md §12.A.6 + §B.9 + §F.3).
//!
//! `dsv4_load::load_layer` yields 256 per-expert NVFP4 tensors (`w1/w2/w3`,
//! §F.3-proven lossless repack). The engine's tensor-core expert GEMMs
//! (`gemm_moe_mma_fp4` N=1, `gemm_moe_grouped_mma_fp4` N=2..16/prefill,
//! gpu_batch.cu:3699/:3832) consume ONE stacked MMA-repacked weight per GEMM
//! with expert `e`'s 16-row tiles at `e·(M>>4)` and a per-tile reciprocal
//! global scale at `gs + e·(M>>4)`. This module builds exactly that layout:
//!
//!   gate_up [ne·2I, H]: per expert, `fuse_nvfp4([w1, w3])` (gate rows 0..I,
//!     then up rows I..2I — the `moe_silu_bf16_b`/`dsv4_swiglu_clamp`
//!     convention), then `repack_nvfp4_mma`.
//!   down    [ne·H, I]: per expert, `repack_nvfp4_mma(w2)`; gs = 1/global
//!     repeated per 16-row tile (`fuse_nvfp4` semantics, quant.rs:667).
//!
//! Per-expert fuse+repack concatenated over ascending `e` is byte-identical to
//! fusing/repacking the full stack in one call (every expert spans a whole
//! number of 16-row tiles, and a tile's bytes depend only on its own (mt, kb))
//! — but the per-expert loop keeps the host peak at ~one expert instead of
//! ~2× the full stack. Geometry (asserted, not patched): M % 16 == 0,
//! K % 32 == 0 — 2I = 4096, H = 4096, I = 2048 all satisfy it.

use anyhow::{bail, Result};

use crate::dsv4_load::{Dsv4Config, Dsv4Layer};
use crate::quant::{fuse_nvfp4, repack_nvfp4_mma};

/// MMA-ready stacked routed experts for one DSV4 layer (see module doc).
/// `*_wt`/`*_st` are the MMA tile-order weight/scale bytes; `*_gs` is the
/// per-16-row-tile reciprocal global scale (`1.0 / global_scale`, the
/// multiply convention the epilogue expects).
pub struct Dsv4MoeHost {
    pub gu_wt: Vec<u8>,
    pub gu_st: Vec<u8>,
    pub gu_gs: Vec<f32>, // [ne·2I/16]
    pub dn_wt: Vec<u8>,
    pub dn_st: Vec<u8>,
    pub dn_gs: Vec<f32>, // [ne·H/16]
    pub ne: usize,
    pub h: usize,
    pub inter: usize,
}

/// Stack + fuse + MMA-repack one layer's 256 routed experts.
pub fn pack_moe_layer(layer: &Dsv4Layer, cfg: &Dsv4Config) -> Result<Dsv4MoeHost> {
    let (ne, h, inter) = (cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim);
    if layer.experts_w1.len() != ne || layer.experts_w2.len() != ne || layer.experts_w3.len() != ne {
        bail!(
            "expert bank incomplete: w1/w2/w3 lens {}/{}/{}, expected {ne}",
            layer.experts_w1.len(),
            layer.experts_w2.len(),
            layer.experts_w3.len()
        );
    }
    let gu_tiles = (2 * inter / 16) * (h / 16); // per-expert gate_up tiles
    let dn_tiles = (h / 16) * (inter / 16); //   per-expert down tiles
    let mut out = Dsv4MoeHost {
        gu_wt: Vec::with_capacity(ne * gu_tiles * 128),
        gu_st: Vec::with_capacity(ne * gu_tiles * 16),
        gu_gs: Vec::with_capacity(ne * 2 * inter / 16),
        dn_wt: Vec::with_capacity(ne * dn_tiles * 128),
        dn_st: Vec::with_capacity(ne * dn_tiles * 16),
        dn_gs: Vec::with_capacity(ne * h / 16),
        ne,
        h,
        inter,
    };
    for e in 0..ne {
        let (w1, w2, w3) = (&layer.experts_w1[e], &layer.experts_w2[e], &layer.experts_w3[e]);
        if w1.m != inter || w1.k != h || w3.m != inter || w3.k != h {
            bail!("expert {e} w1/w3 shape: got [{}x{}]/[{}x{}], expected [{inter}x{h}]",
                  w1.m, w1.k, w3.m, w3.k);
        }
        if w2.m != h || w2.k != inter {
            bail!("expert {e} w2 shape: got [{}x{}], expected [{h}x{inter}]", w2.m, w2.k);
        }
        // gate_up: fuse w1 (gate rows 0..I) then w3 (up rows I..2I), MMA-repack.
        let (qw, sc, gsv) = fuse_nvfp4(
            &[
                (&w1.qweight, &w1.scales, 1.0 / w1.global_scale, w1.m),
                (&w3.qweight, &w3.scales, 1.0 / w3.global_scale, w3.m),
            ],
            h,
        );
        let (wt, st) = repack_nvfp4_mma(&qw, &sc, 2 * inter, h);
        out.gu_wt.extend_from_slice(&wt);
        out.gu_st.extend_from_slice(&st);
        out.gu_gs.extend_from_slice(&gsv);
        // down: single tensor per expert; gs repeated per 16-row tile.
        let (wt, st) = repack_nvfp4_mma(&w2.qweight, &w2.scales, h, inter);
        out.dn_wt.extend_from_slice(&wt);
        out.dn_st.extend_from_slice(&st);
        out.dn_gs
            .extend(std::iter::repeat(1.0 / w2.global_scale).take(h / 16));
    }
    Ok(out)
}

/// Slice `host`'s full-`ne` stacked expert bank to global experts `[e_base, e_base+e_span)` as a
/// NEW `Dsv4MoeHost` (pure host op — no device). The per-expert byte counts mirror the kernel's
/// tile geometry (same as `Dsv4MoeGpu::upload_sharded`'s slice, gpu.rs:9330). Used by the offline
/// converter to write PER-RANK shards: each rank's artifact carries only its 128-expert slice, so
/// each node reads + caches + loads only ~84 GB (not the whole 156 GB).
pub fn slice_moe_host(host: &Dsv4MoeHost, e_base: usize, e_span: usize) -> Dsv4MoeHost {
    assert!(e_base + e_span <= host.ne, "moe slice [{e_base},{e_base}+{e_span}) > ne {}", host.ne);
    let (ne, h, inter) = (host.ne, host.h, host.inter);
    let gu_tiles = (2 * inter / 16) * (h / 16);
    let dn_tiles = (h / 16) * (inter / 16);
    Dsv4MoeHost {
        gu_wt: host.gu_wt[e_base * gu_tiles * 128..(e_base + e_span) * gu_tiles * 128].to_vec(),
        gu_st: host.gu_st[e_base * gu_tiles * 16..(e_base + e_span) * gu_tiles * 16].to_vec(),
        gu_gs: host.gu_gs[e_base * (2 * inter / 16)..(e_base + e_span) * (2 * inter / 16)].to_vec(),
        dn_wt: host.dn_wt[e_base * dn_tiles * 128..(e_base + e_span) * dn_tiles * 128].to_vec(),
        dn_st: host.dn_st[e_base * dn_tiles * 16..(e_base + e_span) * dn_tiles * 16].to_vec(),
        dn_gs: host.dn_gs[e_base * (h / 16)..(e_base + e_span) * (h / 16)].to_vec(),
        ne, h, inter,
    }
}

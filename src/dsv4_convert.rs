//! DSV4 offline converter — turns the bundle into ONE engine-native artifact so serving
//! loads become read+upload (target <90 s/node vs the ~8–11 min streaming repack).
//!
//! # Contract
//! The streaming path (`Dsv4AttnRuntime::upload_layer` in `dsv4_attn.rs`) reads the bundle and,
//! per layer, computes a set of HOST byte-vec precursors (MMA-repacked FP8 `wt`/`sb`, the fused
//! `Dsv4MoeHost`, the cast-rule f32/bf16/i32 tensors, the `CompLoad`/`IndexerLoad` weights) which it
//! then `htod_sync_copy`s into a `Dsv4GpuLayer`. This module reproduces **exactly those host vecs**
//! (calling the same pub primitives: `read_raw_fp8`, `repack_fp8_mma`, `pack_moe_layer`,
//! `load_layer`'s cast rules) and writes them to a safetensors artifact. `Dsv4GpuModel::load_converted`
//! reads them back and `htod`s directly — no cast, no repack, no fuse at load time.
//!
//! Because the bytes are the same bytes, the A/B gate is bitwise by construction (the test
//! `tests/dsv4_convert_test.rs` confirms: load both ways → bit-identical logits, and TP expert slice
//! byte-for-byte). The prep here is a faithful transcription of `upload_layer`'s host side; if that
//! function changes, this must change with it (the test is the tripwire).
//!
//! # Scope (this lane)
//! 43 trunk layers + trunk top. DSpark `mtp.*` stages use the same machinery and are a Phase-5
//! follow-up (explicitly out of scope here).

use anyhow::{anyhow, Context, Result};
use half::bf16;
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::{dsv4_cpu, dsv4_load, dsv4_moe, quant};

/// One artifact tensor: name, dtype, logical shape, raw bytes (the exact bytes `load_converted`
/// `htod`s). For the MMA-repacked FP8/NVFP4 blobs the bytes are in MMA tile order (NOT row-major);
/// the logical shape records `(m, k)` for reconstruction and the manifest names the layout.
#[derive(Clone)]
pub struct Art {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Art {
    fn f32(name: &str, v: &[f32]) -> Self {
        let mut d = Vec::with_capacity(v.len() * 4);
        for x in v { d.extend_from_slice(&x.to_le_bytes()); }
        Art { name: name.to_string(), dtype: Dtype::F32, shape: vec![v.len()], data: d }
    }
    fn bf16(name: &str, v: &[bf16]) -> Self {
        let mut d = Vec::with_capacity(v.len() * 2);
        for x in v { d.extend_from_slice(&x.to_bits().to_le_bytes()); }
        Art { name: name.to_string(), dtype: Dtype::BF16, shape: vec![v.len()], data: d }
    }
    fn i32(name: &str, v: &[i32]) -> Self {
        let mut d = Vec::with_capacity(v.len() * 4);
        for x in v { d.extend_from_slice(&x.to_le_bytes()); }
        Art { name: name.to_string(), dtype: Dtype::I32, shape: vec![v.len()], data: d }
    }
    /// MMA-repacked (or raw-block) u8 blob with a 2-D logical shape `[m, k]` (load_converted reads
    /// the bytes verbatim; the shape carries `m`,`k` for `Fp8Weight`/`Dsv4MoeHost` reconstruction).
    /// NOTE: only valid where bytes == m*k (FP8 weights — 1 byte/elem). NVFP4-packed blobs (moe
    /// gu/dn wt+st, which are m*k/2 + per-tile scales) use [`Art::u8raw`] instead — TensorView
    /// validates size == prod(shape), and the MMA tile order is not row-major anyway.
    fn u8mk(name: &str, bytes: &[u8], m: usize, k: usize) -> Self {
        Art { name: name.to_string(), dtype: Dtype::I8, shape: vec![m, k], data: bytes.to_vec() }
    }
    /// 1-D raw u8 blob `[len]` — for NVFP4-packed moe weights whose byte count ≠ logical m*k.
    /// `Dsv4MoeGpu::upload` consumes them as opaque Vec<u8>; ne/h/inter come from cfg.
    fn u8raw(name: &str, bytes: &[u8]) -> Self {
        Art { name: name.to_string(), dtype: Dtype::I8, shape: vec![bytes.len()], data: bytes.to_vec() }
    }
    /// u8 scale blob `[m/128, k/128]`.
    fn u8sb(name: &str, sb: &[u8], m: usize, k: usize) -> Self {
        Art { name: name.to_string(), dtype: Dtype::I8, shape: vec![m / 128, k / 128], data: sb.to_vec() }
    }
}

// -------------------------------------------------------------------------------------------------
// Per-layer prep — a faithful mirror of `Dsv4AttnRuntime::upload_layer`'s HOST side.
// -------------------------------------------------------------------------------------------------

/// Reproduce `upload_layer`'s host prep for trunk layer `layer_id`, returning the REPLICATED
/// artifact tensors (everything except the moe) AND the full `Dsv4MoeHost`. The per-rank writer
/// slices the host per rank; the full artifact uses it as-is. Every `htod_sync_copy` in
/// `upload_layer` becomes an `Art::` here; the bytes are identical.
pub fn prepare_layer_with_moe(
    bundle: &Path,
    cfg: &dsv4_load::Dsv4Config,
    layer_id: usize,
) -> Result<(Vec<Art>, crate::dsv4_moe::Dsv4MoeHost)> {
    let p = |s: &str| format!("layers.{layer_id}.{s}");
    // FP8 weights → MMA-repacked `wt` + raw `sb` (mirrors `upload_layer`'s `fp8` closure).
    let mut out = Vec::new();
    let mut fp8_push = |out: &mut Vec<Art>, name: &str, art_wt: &str, art_sb: &str| -> Result<(usize, usize)> {
        let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, name)
            .with_context(|| format!("read_raw_fp8 {name}"))?;
        let (m, k) = (shape[0], shape[1]);
        let wt = quant::repack_fp8_mma(&codes, m, k);
        out.push(Art::u8mk(art_wt, &wt, m, k));
        out.push(Art::u8sb(art_sb, &sb, m, k));
        Ok((m, k))
    };
    fp8_push(&mut out, &p("attn.wq_a.weight"), "wq_a.wt", "wq_a.sb")?;
    fp8_push(&mut out, &p("attn.wq_b.weight"), "wq_b.wt", "wq_b.sb")?;
    fp8_push(&mut out, &p("attn.wkv.weight"), "wkv.wt", "wkv.sb")?;
    fp8_push(&mut out, &p("attn.wo_b.weight"), "wo_b.wt", "wo_b.sb")?;
    // Shared expert w2.
    fp8_push(&mut out, &p("ffn.shared_experts.w2.weight"), "sh_w2.wt", "sh_w2.sb")?;
    // Shared expert fused gate_up [w1; w3] (per-16-row-tile repack makes concat == fused).
    let (sh_w1_shape, sh_w1_codes, sh_w1_sb) =
        dsv4_load::read_raw_fp8(bundle, &p("ffn.shared_experts.w1.weight"))?;
    let (_, sh_w3_codes, sh_w3_sb) =
        dsv4_load::read_raw_fp8(bundle, &p("ffn.shared_experts.w3.weight"))?;
    let (m1, k) = (sh_w1_shape[0], sh_w1_shape[1]);
    anyhow::ensure!(sh_w3_codes.len() == sh_w1_codes.len() && k == sh_w1_shape[1] && (m1 + m1) % 128 == 0,
        "shared gu geometry");
    let mut codes = sh_w1_codes;
    codes.extend_from_slice(&sh_w3_codes);
    let mut sb = sh_w1_sb;
    sb.extend_from_slice(&sh_w3_sb);
    let gu_wt = quant::repack_fp8_mma(&codes, m1 + m1, k);
    out.push(Art::u8mk("sh_gu.wt", &gu_wt, m1 + m1, k));
    out.push(Art::u8sb("sh_gu.sb", &sb, m1 + m1, k));

    // Strict-loaded layer for everything else (wo_a bf16 dequant, norms, hc, gate, experts).
    let layer = dsv4_load::load_layer(bundle, cfg, layer_id).context("load_layer")?;
    let mut map = layer.tensors;
    let wo_a: Vec<bf16> = match map.remove("attn.wo_a.weight") {
        Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
            anyhow::ensure!(shape == vec![cfg.o_groups * cfg.o_lora_rank, cfg.dim], "wo_a shape {shape:?}");
            data
        }
        other => return Err(anyhow!("attn.wo_a.weight: expected BF16 dequant, got {:?}", other.map(|t| t.shape().to_vec()))),
    };
    out.push(Art::bf16("wo_a", &wo_a));
    let gate_w_bf: Vec<bf16> = match map.remove("ffn.gate.weight") {
        Some(dsv4_load::HostTensor::BF16 { data, .. }) => data,
        other => return Err(anyhow!("ffn.gate.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
    };
    let gate_w: Vec<f32> = gate_w_bf.iter().map(|v| v.to_f32()).collect();
    out.push(Art::f32("gate_w", &gate_w));
    if cfg.is_hash_layer(layer_id) {
        let t = dsv4_cpu::take_i32(&mut map, "ffn.gate.tid2eid", cfg.vocab_size * cfg.n_activated_experts)?;
        out.push(Art::i32("tid2eid", &t));
    } else {
        let b = dsv4_cpu::take_f32(&mut map, "ffn.gate.bias", cfg.n_routed_experts)?;
        out.push(Art::f32("gate_bias", &b));
    }
    out.push(Art::f32("q_norm", &dsv4_cpu::take_f32(&mut map, "attn.q_norm.weight", cfg.q_lora_rank)?));
    out.push(Art::f32("kv_norm", &dsv4_cpu::take_f32(&mut map, "attn.kv_norm.weight", cfg.head_dim)?));
    out.push(Art::f32("attn_norm", &dsv4_cpu::take_f32(&mut map, "attn_norm.weight", cfg.dim)?));
    out.push(Art::f32("ffn_norm", &dsv4_cpu::take_f32(&mut map, "ffn_norm.weight", cfg.dim)?));
    out.push(Art::f32("attn_sink", &dsv4_cpu::take_f32(&mut map, "attn.attn_sink", cfg.n_heads)?));
    out.push(Art::f32("hc_attn_fn", &dsv4_cpu::take_f32(&mut map, "hc_attn_fn", 24 * cfg.hc_mult * cfg.dim)?));
    out.push(Art::f32("hc_attn_base", &dsv4_cpu::take_f32(&mut map, "hc_attn_base", 24)?));
    out.push(Art::f32("hc_attn_scale", &dsv4_cpu::take_f32(&mut map, "hc_attn_scale", 3)?));
    out.push(Art::f32("hc_ffn_fn", &dsv4_cpu::take_f32(&mut map, "hc_ffn_fn", 24 * cfg.hc_mult * cfg.dim)?));
    out.push(Art::f32("hc_ffn_base", &dsv4_cpu::take_f32(&mut map, "hc_ffn_base", 24)?));
    out.push(Art::f32("hc_ffn_scale", &dsv4_cpu::take_f32(&mut map, "hc_ffn_scale", 3)?));

    // Attention compressor (CSA/HCA) + indexer (CSA) — same keys/geometry as upload_layer.
    let kind = cfg.layer_kind(layer_id);
    let (hd, ihd, inh) = (cfg.head_dim, cfg.index_head_dim, cfg.index_n_heads);
    match kind {
        dsv4_load::LayerKind::Csa => {
            out.push(Art::f32("comp.wkv", &dsv4_cpu::take_f32(&mut map, "attn.compressor.wkv.weight", 2 * hd * cfg.dim)?));
            out.push(Art::f32("comp.wgate", &dsv4_cpu::take_f32(&mut map, "attn.compressor.wgate.weight", 2 * hd * cfg.dim)?));
            out.push(Art::f32("comp.norm", &dsv4_cpu::take_f32(&mut map, "attn.compressor.norm.weight", hd)?));
            out.push(Art::f32("comp.ape", &dsv4_cpu::take_f32(&mut map, "attn.compressor.ape", 4 * 2 * hd)?));
            // Indexer wq_b (MMA-repacked) + its compressor + weights_proj.
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &p("attn.indexer.wq_b.weight"))?;
            anyhow::ensure!(shape[0] == inh * ihd, "indexer wq_b shape {shape:?}");
            let wt = quant::repack_fp8_mma(&codes, inh * ihd, cfg.q_lora_rank);
            out.push(Art::u8mk("idx.wq_b.wt", &wt, inh * ihd, cfg.q_lora_rank));
            out.push(Art::u8sb("idx.wq_b.sb", &sb, inh * ihd, cfg.q_lora_rank));
            out.push(Art::f32("idx.comp.wkv", &dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.wkv.weight", 2 * ihd * cfg.dim)?));
            out.push(Art::f32("idx.comp.wgate", &dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.wgate.weight", 2 * ihd * cfg.dim)?));
            out.push(Art::f32("idx.comp.norm", &dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.norm.weight", ihd)?));
            out.push(Art::f32("idx.comp.ape", &dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.ape", 4 * 2 * ihd)?));
            let wp = dsv4_cpu::take_bf16_as_f32(&mut map, "attn.indexer.weights_proj.weight", inh * cfg.dim)?;
            out.push(Art::f32("idx.weights_proj", &wp));
        }
        dsv4_load::LayerKind::Hca => {
            out.push(Art::f32("comp.wkv", &dsv4_cpu::take_f32(&mut map, "attn.compressor.wkv.weight", hd * cfg.dim)?));
            out.push(Art::f32("comp.wgate", &dsv4_cpu::take_f32(&mut map, "attn.compressor.wgate.weight", hd * cfg.dim)?));
            out.push(Art::f32("comp.norm", &dsv4_cpu::take_f32(&mut map, "attn.compressor.norm.weight", hd)?));
            out.push(Art::f32("comp.ape", &dsv4_cpu::take_f32(&mut map, "attn.compressor.ape", 128 * hd)?));
        }
        dsv4_load::LayerKind::Swa => {}
    }

    // MoE: pack_moe_layer → Dsv4MoeHost (the exact bytes Dsv4MoeGpu::upload consumes).
    let host = dsv4_moe::pack_moe_layer(
        &dsv4_load::Dsv4Layer {
            tensors: std::mem::take(&mut map),
            experts_w1: layer.experts_w1,
            experts_w2: layer.experts_w2,
            experts_w3: layer.experts_w3,
        },
        cfg,
    )
    .context("pack_moe_layer")?;
    Ok((out, host))
}

/// The 6 moe artifact tensors from a (possibly per-rank-sliced) `Dsv4MoeHost`.
pub fn moe_arts(host: &crate::dsv4_moe::Dsv4MoeHost) -> Vec<Art> {
    vec![
        Art::u8raw("moe.gu_wt", &host.gu_wt),
        Art::u8raw("moe.gu_st", &host.gu_st),
        Art::f32("moe.gu_gs", &host.gu_gs),
        Art::u8raw("moe.dn_wt", &host.dn_wt),
        Art::u8raw("moe.dn_st", &host.dn_st),
        Art::f32("moe.dn_gs", &host.dn_gs),
    ]
}

/// Reproduce `upload_layer`'s host prep for trunk layer `layer_id`, returning the artifact tensors
/// (full 256-expert moe). Mirrors `prepare_layer_with_moe` + the full moe arts.
pub fn prepare_layer(
    bundle: &Path,
    cfg: &dsv4_load::Dsv4Config,
    layer_id: usize,
) -> Result<Vec<Art>> {
    let (mut out, host) = prepare_layer_with_moe(bundle, cfg, layer_id)?;
    out.extend(moe_arts(&host));
    Ok(out)
}

/// Reproduce `Dsv4AttnRuntime::upload_mtp_stage`'s HOST side for DSpark stage `stage`
/// (0..n_mtp_layers). Returns the block arts (same names as a trunk SWA layer — wq_a/wq_b/wkv/
/// wo_b/sh_*/wo_a/norms/sink/hc/gate_bias/moe) + the stage extras (main_proj/main_norm for
/// stage 0; norm/hc_head/markov_w1/markov_w2/confidence for stage 2). No comp/indexer/tid2eid
/// (DSpark stages are SWA-kind, bias-routed). Caller prefixes names with `s{stage}.` so the 3
/// stages fit in one safetensors. DSpark stages are REPLICATED (full 256 experts, both ranks).
pub fn prepare_mtp_stage(
    bundle: &Path,
    cfg: &dsv4_load::Dsv4Config,
    stage: usize,
) -> Result<Vec<Art>> {
    let p = |s: &str| format!("mtp.{stage}.{s}");
    let mut out = Vec::new();
    let mut fp8_push = |out: &mut Vec<Art>, name: &str, art_wt: &str, art_sb: &str| -> Result<(usize, usize)> {
        let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, name)
            .with_context(|| format!("read_raw_fp8 {name}"))?;
        let (m, k) = (shape[0], shape[1]);
        let wt = quant::repack_fp8_mma(&codes, m, k);
        out.push(Art::u8mk(art_wt, &wt, m, k));
        out.push(Art::u8sb(art_sb, &sb, m, k));
        Ok((m, k))
    };
    fp8_push(&mut out, &p("attn.wq_a.weight"), "wq_a.wt", "wq_a.sb")?;
    fp8_push(&mut out, &p("attn.wq_b.weight"), "wq_b.wt", "wq_b.sb")?;
    fp8_push(&mut out, &p("attn.wkv.weight"), "wkv.wt", "wkv.sb")?;
    fp8_push(&mut out, &p("attn.wo_b.weight"), "wo_b.wt", "wo_b.sb")?;
    fp8_push(&mut out, &p("ffn.shared_experts.w2.weight"), "sh_w2.wt", "sh_w2.sb")?;
    // Shared fused gate_up [w1; w3].
    let (sh_w1_shape, sh_w1_codes, sh_w1_sb) =
        dsv4_load::read_raw_fp8(bundle, &p("ffn.shared_experts.w1.weight"))?;
    let (_, sh_w3_codes, sh_w3_sb) =
        dsv4_load::read_raw_fp8(bundle, &p("ffn.shared_experts.w3.weight"))?;
    let (m1, k) = (sh_w1_shape[0], sh_w1_shape[1]);
    anyhow::ensure!((m1 + m1) % 128 == 0, "dspark shared gu geometry");
    let mut codes = sh_w1_codes; codes.extend_from_slice(&sh_w3_codes);
    let mut sb = sh_w1_sb; sb.extend_from_slice(&sh_w3_sb);
    let gu_wt = quant::repack_fp8_mma(&codes, m1 + m1, k);
    out.push(Art::u8mk("sh_gu.wt", &gu_wt, m1 + m1, k));
    out.push(Art::u8sb("sh_gu.sb", &sb, m1 + m1, k));

    // Strict load (mtp.{stage}.* — embed/head tied, skipped).
    let layer = dsv4_load::load_mtp_stage(bundle, cfg, stage).context("load_mtp_stage")?;
    let mut map = layer.tensors;
    let wo_a: Vec<bf16> = match map.remove("attn.wo_a.weight") {
        Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
            anyhow::ensure!(shape == vec![cfg.o_groups * cfg.o_lora_rank, cfg.dim], "dspark wo_a shape {shape:?}");
            data
        }
        other => return Err(anyhow!("dspark attn.wo_a.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
    };
    out.push(Art::bf16("wo_a", &wo_a));
    let gate_w: Vec<f32> = match map.remove("ffn.gate.weight") {
        Some(dsv4_load::HostTensor::BF16 { data, .. }) => data.iter().map(|v| v.to_f32()).collect(),
        other => return Err(anyhow!("dspark ffn.gate.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
    };
    out.push(Art::f32("gate_w", &gate_w));
    // DSpark stages always have gate.bias (never tid2eid — never hash).
    out.push(Art::f32("gate_bias", &dsv4_cpu::take_f32(&mut map, "ffn.gate.bias", cfg.n_routed_experts)?));
    out.push(Art::f32("q_norm", &dsv4_cpu::take_f32(&mut map, "attn.q_norm.weight", cfg.q_lora_rank)?));
    out.push(Art::f32("kv_norm", &dsv4_cpu::take_f32(&mut map, "attn.kv_norm.weight", cfg.head_dim)?));
    out.push(Art::f32("attn_norm", &dsv4_cpu::take_f32(&mut map, "attn_norm.weight", cfg.dim)?));
    out.push(Art::f32("ffn_norm", &dsv4_cpu::take_f32(&mut map, "ffn_norm.weight", cfg.dim)?));
    out.push(Art::f32("attn_sink", &dsv4_cpu::take_f32(&mut map, "attn.attn_sink", cfg.n_heads)?));
    out.push(Art::f32("hc_attn_fn", &dsv4_cpu::take_f32(&mut map, "hc_attn_fn", 24 * cfg.hc_mult * cfg.dim)?));
    out.push(Art::f32("hc_attn_base", &dsv4_cpu::take_f32(&mut map, "hc_attn_base", 24)?));
    out.push(Art::f32("hc_attn_scale", &dsv4_cpu::take_f32(&mut map, "hc_attn_scale", 3)?));
    out.push(Art::f32("hc_ffn_fn", &dsv4_cpu::take_f32(&mut map, "hc_ffn_fn", 24 * cfg.hc_mult * cfg.dim)?));
    out.push(Art::f32("hc_ffn_base", &dsv4_cpu::take_f32(&mut map, "hc_ffn_base", 24)?));
    out.push(Art::f32("hc_ffn_scale", &dsv4_cpu::take_f32(&mut map, "hc_ffn_scale", 3)?));

    // Stage extras (pull BEFORE pack_moe_layer takes the map).
    let mut extras = Vec::new();
    if stage == 0 {
        // main_proj: raw FP8 [dim, 3*dim] → MMA-repacked wt + sb.
        let (mp_shape, mp_codes, mp_sb) = dsv4_load::read_raw_fp8(bundle, &p("main_proj.weight"))
            .context("read_raw_fp8 mtp.0.main_proj.weight")?;
        let (mp_m, mp_k) = (mp_shape[0], mp_shape[1]);
        let mp_wt = quant::repack_fp8_mma(&mp_codes, mp_m, mp_k);
        extras.push(Art::u8mk("main_proj.wt", &mp_wt, mp_m, mp_k));
        extras.push(Art::u8sb("main_proj.sb", &mp_sb, mp_m, mp_k));
        extras.push(Art::f32("main_norm", &dsv4_cpu::take_f32(&mut map, "main_norm.weight", cfg.dim)?));
    }
    if stage == cfg.n_mtp_layers - 1 {
        let rank = cfg.dspark_markov_rank;
        extras.push(Art::f32("norm", &dsv4_cpu::take_f32(&mut map, "norm.weight", cfg.dim)?));
        extras.push(Art::f32("hc_head_fn", &dsv4_cpu::take_f32(&mut map, "hc_head_fn", cfg.hc_mult * cfg.hc_mult * cfg.dim)?));
        extras.push(Art::f32("hc_head_base", &dsv4_cpu::take_f32(&mut map, "hc_head_base", cfg.hc_mult)?));
        extras.push(Art::f32("hc_head_scale", &dsv4_cpu::take_f32(&mut map, "hc_head_scale", 1)?));
        // markov_w1: bf16 embedding [vocab, rank].
        let mw1 = match map.remove("markov_head.markov_w1.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, .. }) => data,
            other => return Err(anyhow!("markov_w1: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        anyhow::ensure!(mw1.len() == cfg.vocab_size * rank, "markov_w1 len");
        extras.push(Art::bf16("markov_w1", &mw1));
        // markov_w2: bf16-exact f32 [vocab, rank] (HEAD_F32_KEYS cast rule).
        let mw2 = dsv4_cpu::take_f32(&mut map, "markov_head.markov_w2.weight", cfg.vocab_size * rank)?;
        extras.push(Art::f32("markov_w2", &mw2));
        extras.push(Art::f32("confidence", &dsv4_cpu::take_f32(&mut map, "confidence_head.proj.weight", cfg.dim + rank)?));
    }

    // MoE (full 256 experts — DSpark stages are replicated, not TP-sharded).
    let host = dsv4_moe::pack_moe_layer(
        &dsv4_load::Dsv4Layer {
            tensors: std::mem::take(&mut map),
            experts_w1: layer.experts_w1,
            experts_w2: layer.experts_w2,
            experts_w3: layer.experts_w3,
        }, cfg,
    ).context("pack_moe_layer (dspark stage)")?;
    out.extend(moe_arts(&host));
    out.extend(extras);
    Ok(out)
}

/// Write the 3 DSpark stages to `{rank_dir}/dspark_stage{0,1,2}.safetensors` (one file per stage,
/// NOT one combined file — ArtFile::open reads the whole file into RAM, and a combined 11.6 GB
/// buf + the 84 GB trunk (unified memory) tips the budget at long context; per-stage files cap
/// the host peak at ~3.8 GB). REPLICATED — the converter writes the SAME files to every rank dir;
/// `build_manifest` ships `{rank}/dspark_stage*.safetensors` to the node.
pub fn write_dspark_artifact(
    bundle: &Path,
    cfg: &dsv4_load::Dsv4Config,
    rank_dir: &Path,
) -> Result<()> {
    std::fs::create_dir_all(rank_dir)?;
    for stage in 0..cfg.n_mtp_layers {
        let path = rank_dir.join(format!("dspark_stage{stage}.safetensors"));
        if path.exists() {
            eprintln!("[convert] dspark_stage{stage}.safetensors cached ({})", rank_dir.display());
            continue;
        }
        let arts = prepare_mtp_stage(bundle, cfg, stage)
            .with_context(|| format!("prepare_mtp_stage {stage}"))?;
        write_arts_atomic(path.clone(), &arts)?;
        eprintln!("[convert] wrote {} ({} tensors, stage {stage})", path.display(), arts.len());
    }
    Ok(())
}

/// Trunk top: embed (bf16), norm (f32), head (bf16 — the bf16-exact values `gemm_binv_f32_b`
/// reads; extract_trunk_top downcasts the loader's f32 head to bf16, so store the bf16 form
/// directly), hc_head_* (f32).
pub fn prepare_trunk_top(bundle: &Path, cfg: &dsv4_load::Dsv4Config) -> Result<Vec<Art>> {
    let mut map = dsv4_load::load_trunk_top(bundle, cfg).context("load_trunk_top")?;
    let (vocab, dim, hc) = (cfg.vocab_size, cfg.dim, cfg.hc_mult);
    let embed: Vec<bf16> = match map.remove("embed.weight") {
        Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
            anyhow::ensure!(shape == vec![vocab, dim], "embed shape {shape:?}");
            data
        }
        other => return Err(anyhow!("embed.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
    };
    let norm = dsv4_cpu::take_f32(&mut map, "norm.weight", dim)?;
    let head_f32 = dsv4_cpu::take_f32(&mut map, "head.weight", vocab * dim)?;
    let head_bf16: Vec<bf16> = head_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let hc_head_fn = dsv4_cpu::take_f32(&mut map, "hc_head_fn", hc * hc * dim)?;
    let hc_head_base = dsv4_cpu::take_f32(&mut map, "hc_head_base", hc)?;
    let hc_head_scale = dsv4_cpu::take_f32(&mut map, "hc_head_scale", 1)?;
    Ok(vec![
        Art::bf16("embed", &embed),
        Art::f32("norm", &norm),
        Art::bf16("head", &head_bf16),
        Art::f32("hc_head_fn", &hc_head_fn),
        Art::f32("hc_head_base", &hc_head_base),
        Art::f32("hc_head_scale", &hc_head_scale),
    ])
}

// -------------------------------------------------------------------------------------------------
// Artifact write — safetensors (one file per layer + trunk_top) + manifest.json.
// -------------------------------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub model_type: String,
    pub layout_version: u32,
    pub n_layers: usize,
    pub source_bundle: String,
    pub cfg: serde_json::Value,
    /// name → file (each tensor's home safetensors file).
    pub files: BTreeMap<String, String>,
    /// Per-rank shard fields (only set in `rank{R}/manifest.json` by `write_artifact_sharded`).
    /// The artifact at this dir carries only global experts `[e_base, e_base+e_span)`. The fast
    /// reader uploads the moe with exactly this band (no re-slice) — each node loads only its part.
    #[serde(default)]
    pub rank: Option<usize>,
    #[serde(default)]
    pub world: Option<usize>,
    #[serde(default)]
    pub e_base: Option<usize>,
    #[serde(default)]
    pub e_span: Option<usize>,
}

/// Build the full artifact at `out_dir`: writes `layer{N}.safetensors` for N in 0..n_layers,
/// `trunk_top.safetensors`, copies `inference/config.json`, and writes `manifest.json`.
pub fn write_artifact(
    bundle: &Path,
    cfg: &dsv4_load::Dsv4Config,
    out_dir: &Path,
    n_layers: usize,
) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    for layer_id in 0..n_layers {
        let fname = format!("layer{layer_id}.safetensors");
        let final_path = out_dir.join(&fname);
        if final_path.exists() {
            // Resumable: a finished layer file is never rewritten (atomic .tmp rename guarantees
            // no half-written final file survives an interruption). Re-link its tensors to the manifest.
            eprintln!("[convert] layer {layer_id:2} cached ({fname})");
        } else {
            let arts = prepare_layer(bundle, cfg, layer_id)
                .with_context(|| format!("prepare_layer {layer_id}"))?;
            write_arts_atomic(out_dir.join(&fname), &arts)?;
            eprintln!("[convert] layer {layer_id:2} → {fname} ({} tensors)", arts.len());
        }
        // Record the manifest mapping from the (now-existing) file's tensor names.
        let art_names = artifact_tensor_names(&final_path)?;
        for n in art_names { files.insert(format!("layers.{layer_id}.{n}"), fname.clone()); }
    }
    let top_path = out_dir.join("trunk_top.safetensors");
    if top_path.exists() {
        eprintln!("[convert] trunk_top cached");
    } else {
        let top = prepare_trunk_top(bundle, cfg).context("prepare_trunk_top")?;
        write_arts_atomic(top_path.clone(), &top)?;
    }
    for n in artifact_tensor_names(&top_path)? { files.insert(format!("trunk_top.{n}"), "trunk_top.safetensors".to_string()); }
    // Carry inference/config.json so load_config works on the assembled artifact dir as-is.
    let cfg_src = bundle.join("inference").join("config.json");
    let cfg_dst = out_dir.join("inference").join("config.json");
    std::fs::create_dir_all(cfg_dst.parent().unwrap())?;
    std::fs::copy(&cfg_src, &cfg_dst).context("copy inference/config.json")?;
    let cfg_json = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(&cfg_src)?)?;
    let manifest = Manifest {
        model_type: "deepseek_v4".to_string(),
        layout_version: 1,
        n_layers,
        source_bundle: bundle.display().to_string(),
        cfg: cfg_json,
        files,
        rank: None, world: None, e_base: None, e_span: None,
    };
    let mf_path = out_dir.join("manifest.json");
    std::fs::write(&mf_path, serde_json::to_string_pretty(&manifest)?).context("write manifest.json")?;
    eprintln!("[convert] wrote {} + manifest.json ({} layers + trunk top)", out_dir.display(), n_layers);
    Ok(())
}

/// Build PER-RANK self-contained artifacts (the load-speed lane's TP=2 design): for each rank r,
/// `out_dir/rank{r}/` holds the REPLICATED parts (attn/mHC/router/shared/compressor/indexer/norms)
/// + THAT RANK's `[r·ne/world, (r+1)·ne/world)` expert slice. The head ships only `rank1/` to the
/// node once; the node caches only its ~84 GB shard; each node reads + loads only its part.
/// Resumable per (rank, layer) file. `world` must divide `cfg.n_routed_experts`.
pub fn write_artifact_sharded(
    bundle: &Path,
    cfg: &dsv4_load::Dsv4Config,
    out_dir: &Path,
    n_layers: usize,
    world: usize,
) -> Result<()> {
    let e_span = cfg.n_routed_experts / world;
    anyhow::ensure!(e_span * world == cfg.n_routed_experts, "ne {} / world {world} not integral", cfg.n_routed_experts);
    std::fs::create_dir_all(out_dir)?;
    // Trunk top is replicated — prepare once, write to every rank dir.
    let top = prepare_trunk_top(bundle, cfg).context("prepare_trunk_top")?;
    let cfg_src = bundle.join("inference").join("config.json");
    let cfg_json = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(&cfg_src)?)?;
    // Root inference/config.json too (the cluster ships it to the node for load_config).
    let root_cfg = out_dir.join("inference").join("config.json");
    std::fs::create_dir_all(root_cfg.parent().unwrap())?;
    std::fs::copy(&cfg_src, &root_cfg)?;
    // Tokenizer + generation configs at the root (the head's broadcast_prompt reads tokenizer.json;
    // dsv4_tp_serve reads generation_config for eos). Plus the HF root config.json — is_dsv4_bundle
    // detects the model_type there. Not sharded — small, shared.
    for f in &["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"] {
        let p = bundle.join(f);
        if p.exists() { let _ = std::fs::copy(&p, out_dir.join(f)); }
    }
    for r in 0..world {
        let rank_dir = out_dir.join(format!("rank{r}"));
        std::fs::create_dir_all(&rank_dir)?;
        // inference/config.json (so load_config works on the rank dir as-is).
        let cfg_dst = rank_dir.join("inference").join("config.json");
        std::fs::create_dir_all(cfg_dst.parent().unwrap())?;
        std::fs::copy(&cfg_src, &cfg_dst)?;
        // trunk_top (replicated).
        let top_path = rank_dir.join("trunk_top.safetensors");
        if !top_path.exists() { write_arts_atomic(top_path.clone(), &top)?; }
        let mut files: BTreeMap<String, String> = BTreeMap::new();
        for n in artifact_tensor_names(&top_path)? { files.insert(format!("trunk_top.{n}"), "trunk_top.safetensors".to_string()); }
        let mf = Manifest {
            model_type: "deepseek_v4".to_string(), layout_version: 1, n_layers,
            source_bundle: bundle.display().to_string(), cfg: cfg_json.clone(), files,
            rank: Some(r), world: Some(world), e_base: Some(r * e_span), e_span: Some(e_span),
        };
        std::fs::write(rank_dir.join("manifest.json"), serde_json::to_string_pretty(&mf)?)?;
    }
    // Per-layer: prepare replicated + full moe ONCE, then write each rank's (replicated + sliced moe).
    for layer_id in 0..n_layers {
        let (replicated, full_moe) = prepare_layer_with_moe(bundle, cfg, layer_id)
            .with_context(|| format!("prepare_layer {layer_id}"))?;
        for r in 0..world {
            let rank_dir = out_dir.join(format!("rank{r}"));
            let fname = format!("layer{layer_id}.safetensors");
            let final_path = rank_dir.join(&fname);
            if final_path.exists() {
                eprintln!("[convert] rank{r} layer {layer_id:2} cached");
            } else {
                let sliced = crate::dsv4_moe::slice_moe_host(&full_moe, r * e_span, e_span);
                let mut arts = replicated.clone();
                arts.extend(moe_arts(&sliced));
                write_arts_atomic(final_path.clone(), &arts)?;
                eprintln!("[convert] rank{r} layer {layer_id:2} → {fname} ({} tensors, e[{}..{}))",
                    arts.len(), r * e_span, (r + 1) * e_span);
            }
            // Re-link this layer's tensors into the rank's manifest.
            let mut mf: Manifest = serde_json::from_str(&std::fs::read_to_string(rank_dir.join("manifest.json"))?)?;
            for n in artifact_tensor_names(&final_path)? { mf.files.insert(format!("layers.{layer_id}.{n}"), fname.clone()); }
            std::fs::write(rank_dir.join("manifest.json"), serde_json::to_string_pretty(&mf)?)?;
        }
    }
    eprintln!("[convert] wrote {world} per-rank shards at {}/rank{{0..{world}}}/ ({} layers each, {e_span} experts/rank)",
        out_dir.display(), n_layers);
    Ok(())
}

fn write_arts(path: std::path::PathBuf, arts: &[Art]) -> Result<()> {
    let views: Vec<(String, TensorView)> = arts
        .iter()
        .map(|a| (a.name.clone(), TensorView::new(a.dtype, a.shape.clone(), &a.data).expect("TensorView")))
        .collect();
    safetensors::serialize_to_file(views, None, &path).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Atomic write: serialize to `<path>.tmp`, then rename over `<path>`. An interruption leaves at
/// worst a stale `.tmp` (never a half-written final file), so the resumable converter can skip any
/// layer whose final file already exists.
fn write_arts_atomic(path: std::path::PathBuf, arts: &[Art]) -> Result<()> {
    let tmp = path.with_extension("safetensors.tmp");
    write_arts(tmp.clone(), arts)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Open an existing artifact file and list its tensor names (for manifest reconstruction on resume).
fn artifact_tensor_names(path: &Path) -> Result<Vec<String>> {
    let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&buf).with_context(|| format!("parse {}", path.display()))?;
    Ok(st.tensors().into_iter().map(|(n, _)| n).collect())
}

// -------------------------------------------------------------------------------------------------
// Artifact read — the fast reader's input. `load_converted` (dsv4_model.rs) calls these, then
// `htod_sync_copy`s each Vec directly — no cast, no repack, no fuse (the whole point).
// -------------------------------------------------------------------------------------------------

/// One artifact file held as a single in-memory buffer (read once). `SafeTensors` is re-parsed
/// per-access (header parse is cheap; the data slices into the buffer — zero tensor copies). The
/// bulk u8 blobs (moe gu/dn, fp8 wt/sb) `htod` straight from the slice; only the small f32/bf16/i32
/// tensors materialize a typed Vec. This is what makes `load_converted` fast (no per-tensor .to_vec()
/// of multi-GB moe weights).
pub struct ArtFile {
    buf: Vec<u8>,
}

impl ArtFile {
    /// Read the whole file into RAM (one copy). Per-layer peak ≈ the file size (~3.5 GB); the
    /// buffer drops before the next layer. `posix_fadvise(DONTNEED)` after the read drops the
    /// page-cache pages so 43 × 3.5 GB of reads don't accumulate — critical on GB10's unified
    /// memory (CPU+GPU share 124 GB; the cache + GPU weights would OOM-kill the load otherwise).
    pub fn open(path: &Path) -> Result<Self> {
        use std::os::unix::io::AsRawFd;
        let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).with_context(|| format!("read {}", path.display()))?;
        // Drop the just-read pages from the page cache (the Vec holds our working copy).
        // POSIX_FADV_DONTNEED = 4. Errors are non-fatal (best-effort cache hint).
        unsafe { let _ = libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED); }
        Ok(Self { buf })
    }
    fn st(&self) -> Result<safetensors::SafeTensors> {
        Ok(safetensors::SafeTensors::deserialize(&self.buf).with_context(|| "parse artifact safetensors")?)
    }
    /// Zero-copy slice into the buffer for a u8 tensor (the bulk moe/fp8 blobs).
    pub fn u8_slice(&self, name: &str) -> Result<&[u8]> {
        Ok(self.st()?.tensor(name).with_context(|| format!("tensor {name:?}"))?.data())
    }
    pub fn f32_of(&self, name: &str) -> Result<Vec<f32>> {
        let d = self.u8_slice(name)?;
        Ok(d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    pub fn bf16_of(&self, name: &str) -> Result<Vec<bf16>> {
        let d = self.u8_slice(name)?;
        Ok(d.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect())
    }
    pub fn i32_of(&self, name: &str) -> Result<Vec<i32>> {
        let d = self.u8_slice(name)?;
        Ok(d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    /// (m, k) from the 2-D shape of a u8mk tensor.
    pub fn mk_of(&self, name: &str) -> Result<(usize, usize)> {
        let sh = self.st()?.tensor(name)?.shape().to_vec();
        anyhow::ensure!(sh.len() == 2, "{name}: expected 2-D shape, got {sh:?}");
        Ok((sh[0], sh[1]))
    }
}

/// Read + parse manifest.json.
pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let txt = std::fs::read_to_string(dir.join("manifest.json")).context("read manifest.json")?;
    Ok(serde_json::from_str(&txt).context("parse manifest.json")?)
}

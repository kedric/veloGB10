//! DSV4 (DeepSeek-V4-Flash-DSpark) loader v0 — LOADING CONTRACT (DEEPSEEK_V4_PORT.md §A/§D/§F).
//!
//! This module is the single source of truth for how the 48-shard bundle at
//! `/mnt/models/DeepSeek-V4-Flash-DSpark` becomes host-side Rust tensors for the G1
//! CPU-model fidelity gate. Semantics mirror `scripts/dsv4_ref.py`
//! (`load_weight_map` / `stream_tensors` / `cast_rule` / `build_state_dict` /
//! `dequant_wo_a`) — when in doubt, the Python oracle runner is the reference.
//!
//! ## The contract ( Lane A implements exactly this )
//!
//! 1. **Config** (`load_config`): parse `<bundle>/inference/config.json` — NOT the HF
//!    config, NOT ModelArgs demo defaults (§D traps: 43 layers, 256 experts top-6,
//!    46-entry compress_ratios, n_hash_layers=3, n_mtp_layers=3, dspark_block_size=5,
//!    route_scale=1.5, swiglu_limit=10, max_seq_len=1048576, rope_factor=16,
//!    compress_rope_theta=160000). `norm_eps`/`hc_eps` are NOT in the json — the
//!    reference uses 1e-6 for both (hardcode with a comment).
//! 2. **Streaming** (`load_layer` / `load_mtp_stage`): read
//!    `model.safetensors.index.json` `weight_map`, collect keys for the layer prefix
//!    (`layers.{N}.` / `mtp.{S}.`), group by shard so each shard file opens once
//!    (`safetensors` crate), apply cast rules below. Loader v0 is PER-LAYER — no
//!    whole-model residency (that is the later shard-at-load GPU path, HY3_HANDOFF §4).
//! 3. **Cast rules** (mirror `cast_rule` exactly):
//!    - `attn.wo_a.weight` + `attn.wo_a.scale`: consumed TOGETHER via
//!      [`dequant_wo_a_bf16`] (§F.2 — loading raw drops the scales and corrupts O);
//!      the scale tensor is then dropped.
//!    - Other FP8 tensors (`*.weight` with a sibling `*.scale`, F8_E4M3 + F8_E8M0
//!      128×128 blocks): delivered as **exact f32 dequant** via [`dequant_fp8_exact`]
//!      (e4m3 values and pow2 scales are exact in f32 — no re-quantization anywhere).
//!    - `ffn.experts.{E}.w{1,2,3}.weight/.scale` (I8-packed FP4-E2M1, E8M0 per-32-K):
//!      NOT in the tensor map — repacked per expert via [`repack_expert_fp4_to_nvfp4`]
//!      (§F.3, proven bit-exact) into `experts_w{1,2,3}[E]`.
//!    - Every `*norm.weight` (q/kv/attn/ffn/compressor/main/trunk RMSNorm), compressor
//!      `wkv`/`wgate` (attention AND indexer compressor), `head.weight`,
//!      `markov_head.markov_w2.weight`, `confidence_head.proj.weight`: bf16 → **f32**
//!      (upcast is exact; the reference uses these as fp32).
//!    - `ffn.gate.tid2eid` (layers 0–2 only): I64 → **I32** (easy to overlook; missing
//!      it passes smoke tests and fails on quality — §9.6).
//!    - Everything else keeps checkpoint dtype: `embed.weight`, `ffn.gate.weight`,
//!      `attn.indexer.weights_proj.weight`, `markov_head.markov_w1.weight` → BF16;
//!      `attn.attn_sink`, all `hc_*`, `ffn.gate.bias`, `*.ape` → F32 (already f32).
//!    - Skip `mtp.*embed*`, `mtp.*head.weight` (tied to trunk).
//! 4. **Keys in `Dsv4Layer.tensors`** are the checkpoint keys with the layer prefix
//!    stripped (e.g. `attn.wq_b.weight`, `ffn.gate.bias`), matching
//!    `build_state_dict`'s naming exactly.
//! 5. **A/B bit-exactness gate (§F.3, G1)**: for every repacked expert,
//!    `dequant_nvfp4_f32(repack(w, s)) == e8m0_dequant_fp4_exact(w, s)` bitwise.
//!
//! ## npy exchange format (with Lane B's replay + Lane C's differ)
//!
//! The oracle npz is converted to one `.npy` per array (v1.0, little-endian, C-order;
//! dtypes `<f4` and `<i8` only — npz stores f32 and int64) by `scripts/dsv4_diff.py
//! export`; the Rust replay reads inputs and writes outputs as `.npy` under the same
//! key names. `meta.sample_rows` (present in large profiles) is handled differ-side.

use anyhow::{anyhow, bail, Context, Result};
use half::bf16;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use crate::quant::{e2m1_to_f32, e4m3_to_f32, f32_to_e4m3, Nvfp4Tensor};

/// FP8/FP4 block geometry (§B global, §C — hardcoded in the reference, §D "hardcoded
/// but looks configurable").
pub const FP8_GROUP: usize = 128;
pub const KV_SIM_GROUP: usize = 64;
pub const FP4_GROUP: usize = 32;

#[derive(Debug, Clone)]
pub struct Dsv4Config {
    pub vocab_size: usize,           // 129280
    pub dim: usize,                  // 4096
    pub moe_inter_dim: usize,        // 2048
    pub n_layers: usize,             // 43 trunk layers (all MoE)
    pub n_hash_layers: usize,        // 3 — layers 0..2 route via tid2eid
    pub n_mtp_layers: usize,         // 3 — mtp.0..2 DSpark blocks
    pub dspark_block_size: usize,    // 5
    pub dspark_noise_token_id: u32,  // 128799
    pub dspark_target_layer_ids: Vec<usize>, // [40, 41, 42]
    pub dspark_markov_rank: usize,   // 256
    pub n_heads: usize,              // 64 Q heads, single 512-dim KV latent (MQA)
    pub n_routed_experts: usize,     // 256
    pub n_shared_experts: usize,     // 1
    pub n_activated_experts: usize,  // 6
    pub route_scale: f32,            // 1.5
    pub swiglu_limit: f32,           // 10.0 — asymmetric clamps (up ±10, gate ≤10)
    pub q_lora_rank: usize,          // 1024
    pub head_dim: usize,             // 512
    pub rope_head_dim: usize,        // 64 (only last 64 dims RoPE'd)
    pub o_groups: usize,             // 8
    pub o_lora_rank: usize,          // 1024
    pub window_size: usize,          // 128 (SWA ring)
    pub original_seq_len: usize,     // 65536 (YaRN boundary; SWA layers force-disable)
    pub rope_theta: f32,             // 10000 (SWA)
    pub rope_factor: f32,            // 16
    pub beta_fast: u32,              // 32
    pub beta_slow: u32,              // 1
    pub index_n_heads: usize,        // 64
    pub index_head_dim: usize,       // 128
    pub index_topk: usize,           // 512
    pub hc_mult: usize,              // 4 residual streams
    pub hc_sinkhorn_iters: usize,    // 20
    pub compress_rope_theta: f32,    // 160000 (CSA/HCA YaRN base)
    pub compress_ratios: Vec<i32>,   // 46 entries: 0,0 then alternating 4/128, then 0,0,0
    // Not in the json — reference-hardcoded (model.py:62, :81; §A eps note, §B.8):
    pub norm_eps: f32,               // 1e-6 — ALL RMSNorms incl. hc_pre's RMS
    pub hc_eps: f32,                 // 1e-6 — Sinkhorn/sigmoid eps
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Swa, // compress_ratios 0 — 128-window ring only
    Csa, // 4 — window + indexer top-512 blocks (overlapping compressor)
    Hca, // 128 — window + ALL compressed blocks (non-overlapping compressor)
}

impl Dsv4Config {
    /// Trunk layer index 0..42 → kind (compress_ratios[idx]; 43..45 are DSpark stages).
    pub fn layer_kind(&self, idx: usize) -> LayerKind {
        match self.compress_ratios.get(idx) {
            Some(0) => LayerKind::Swa,
            Some(4) => LayerKind::Csa,
            Some(128) => LayerKind::Hca,
            Some(r) => panic!("compress_ratios[{idx}] = {r}: not one of 0/4/128 (§D)"),
            None => panic!("layer index {idx} out of range ({} compress_ratios entries)",
                           self.compress_ratios.len()),
        }
    }
    /// Layers 0..n_hash_layers-1 route by token-id lookup (gate GEMM still runs).
    pub fn is_hash_layer(&self, layer: usize) -> bool {
        layer < self.n_hash_layers
    }
}

/// Host tensor in a G1-consumable form (see module docs for which variant each
/// checkpoint tensor lands in). All shapes row-major, checkpoint layout.
#[derive(Debug, Clone)]
pub enum HostTensor {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    BF16 { shape: Vec<usize>, data: Vec<bf16> },
    I32 { shape: Vec<usize>, data: Vec<i32> },
    /// Routed-expert matrices only (w1/w2/w3), §F.3-repacked.
    Nvfp4(Nvfp4Tensor),
}

impl HostTensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            HostTensor::F32 { shape, .. } => shape,
            HostTensor::BF16 { shape, .. } => shape,
            HostTensor::I32 { shape, .. } => shape,
            // Nvfp4Tensor stores m/k as scalars, not a slice; the shape accessor is a
            // metadata path (replay/test setup), so we leak a 2-usize box per call.
            HostTensor::Nvfp4(t) => Box::leak(Box::new([t.m, t.k])),
        }
    }
}

/// One strict-loaded layer (trunk `layers.{N}.*` or DSpark `mtp.{S}.*`).
#[derive(Debug)]
pub struct Dsv4Layer {
    /// Stripped-key map, cast rules applied (module doc §4). Empty for experts.
    pub tensors: HashMap<String, HostTensor>,
    /// 256 entries, ascending expert id; NVFP4-repacked (§F.3). Empty Vec if absent.
    pub experts_w1: Vec<Nvfp4Tensor>,
    pub experts_w2: Vec<Nvfp4Tensor>,
    pub experts_w3: Vec<Nvfp4Tensor>,
}

// ---------------------------------------------------------------------------
// Raw shard streaming (mirrors dsv4_ref.py load_weight_map / stream_tensors)
// ---------------------------------------------------------------------------

/// Checkpoint dtypes this loader understands (subset of `safetensors::Dtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    BF16,
    F32,
    I8,
    I64,
    F8E4M3,
    F8E8M0,
}

impl StDtype {
    fn from_st(d: safetensors::Dtype) -> Result<Self> {
        use safetensors::Dtype as D;
        Ok(match d {
            D::BF16 => StDtype::BF16,
            D::F32 => StDtype::F32,
            D::I8 => StDtype::I8,
            D::I64 => StDtype::I64,
            D::F8_E4M3 => StDtype::F8E4M3,
            D::F8_E8M0 => StDtype::F8E8M0,
            other => bail!("unsupported checkpoint dtype {other:?}"),
        })
    }
}

/// One raw checkpoint tensor, bytes copied out of its shard (shard file is dropped).
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl RawTensor {
    pub fn as_bf16(&self) -> Result<Vec<bf16>> {
        if self.dtype != StDtype::BF16 {
            bail!("expected BF16, got {:?}", self.dtype);
        }
        Ok(self.data.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect())
    }
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != StDtype::F32 {
            bail!("expected F32, got {:?}", self.dtype);
        }
        Ok(self.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    /// bf16 → f32 upcast (exact — bf16 is a truncation of f32's bit pattern).
    pub fn bf16_to_f32(&self) -> Result<Vec<f32>> {
        Ok(self.as_bf16()?.iter().map(|x| x.to_f32()).collect())
    }
}

/// `dsv4_ref.py::load_weight_map`: full checkpoint key → shard filename.
pub fn load_weight_map(bundle_dir: &Path) -> Result<HashMap<String, String>> {
    let p = bundle_dir.join("model.safetensors.index.json");
    let txt = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&txt).context("parse index.json")?;
    let wm = v.get("weight_map").and_then(|w| w.as_object())
        .ok_or_else(|| anyhow!("index.json has no weight_map object"))?;
    Ok(wm.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
}

/// `dsv4_ref.py::stream_tensors`: load exactly `names`, grouping by shard so each
/// shard file is opened (read + parsed) exactly once.
pub fn stream_raw_tensors(bundle_dir: &Path, names: &[String]) -> Result<HashMap<String, RawTensor>> {
    let wm = load_weight_map(bundle_dir)?;
    let mut by_shard: BTreeMap<&str, Vec<&String>> = BTreeMap::new();
    for n in names {
        let shard = wm.get(n.as_str()).ok_or_else(|| anyhow!("{n}: not in weight_map"))?;
        by_shard.entry(shard.as_str()).or_default().push(n);
    }
    let mut out = HashMap::with_capacity(names.len());
    for (shard, keys) in by_shard {
        let path = bundle_dir.join(shard);
        let buf = std::fs::read(&path).with_context(|| format!("read shard {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&buf)
            .with_context(|| format!("parse shard {shard}"))?;
        for k in keys {
            let view = st.tensor(k).with_context(|| format!("{k} missing in {shard}"))?;
            out.insert(k.clone(), RawTensor {
                dtype: StDtype::from_st(view.dtype()).with_context(|| k.clone())?,
                shape: view.shape().to_vec(),
                data: view.data().to_vec(),
            });
        }
    }
    Ok(out)
}

/// Raw FP8 tensor + its E8M0 block scales, for engine-path consumers (kernel tests,
/// GPU-side loaders): the on-disk encoding, NOT dequantized. `name` is a full
/// checkpoint key ending in `.weight`; returns `(shape [out, k], e4m3 data,
/// ue8m0 scales [out/128, k/128])`.
pub fn read_raw_fp8(bundle_dir: &Path, name: &str) -> Result<(Vec<usize>, Vec<u8>, Vec<u8>)> {
    let scale_key = format!(
        "{}.scale",
        name.strip_suffix(".weight").with_context(|| format!("fp8 key {name} must end in .weight"))?
    );
    let raw = stream_raw_tensors(bundle_dir, &[name.to_string(), scale_key.clone()])?;
    let w = raw.get(name).with_context(|| format!("{name} missing"))?;
    let s = raw.get(&scale_key).with_context(|| format!("{scale_key} missing"))?;
    anyhow::ensure!(w.dtype == StDtype::F8E4M3, "{name}: expected F8_E4M3, got {:?}", w.dtype);
    anyhow::ensure!(s.dtype == StDtype::F8E8M0, "{scale_key}: expected F8_E8M0, got {:?}", s.dtype);
    anyhow::ensure!(w.shape.len() == 2 && w.shape[0] % 128 == 0 && w.shape[1] % 128 == 0,
        "{name}: shape {:?} not 128-aligned", w.shape);
    anyhow::ensure!(s.shape == vec![w.shape[0] / 128, w.shape[1] / 128],
        "{scale_key}: shape {:?} != {:?}", s.shape, [w.shape[0] / 128, w.shape[1] / 128]);
    Ok((w.shape.clone(), w.data.clone(), s.data.clone()))
}

/// Parse `<bundle>/inference/config.json` per the module contract (§D traps).
pub fn load_config(bundle_dir: &Path) -> Result<Dsv4Config> {
    let p = bundle_dir.join("inference").join("config.json");
    let txt = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&txt).context("parse config.json")?;

    let u = |k: &str| -> Result<usize> {
        v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
            .ok_or_else(|| anyhow!("config.json: missing/invalid `{k}`"))
    };
    let f = |k: &str| -> Result<f32> {
        v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32)
            .ok_or_else(|| anyhow!("config.json: missing/invalid `{k}`"))
    };
    let ratios: Vec<i32> = v.get("compress_ratios").and_then(|x| x.as_array())
        .ok_or_else(|| anyhow!("config.json: missing `compress_ratios`"))?
        .iter().map(|x| x.as_i64().map(|v| v as i32)
            .ok_or_else(|| anyhow!("compress_ratios: non-integer entry"))).collect::<Result<_>>()?;
    let target_ids: Vec<usize> = v.get("dspark_target_layer_ids").and_then(|x| x.as_array())
        .ok_or_else(|| anyhow!("config.json: missing `dspark_target_layer_ids`"))?
        .iter().map(|x| x.as_u64().map(|v| v as usize)
            .ok_or_else(|| anyhow!("dspark_target_layer_ids: non-integer entry"))).collect::<Result<_>>()?;

    let (n_layers, n_mtp_layers) = (u("n_layers")?, u("n_mtp_layers")?);
    if ratios.len() != n_layers + n_mtp_layers {
        bail!("compress_ratios has {} entries, expected n_layers+n_mtp_layers = {}",
              ratios.len(), n_layers + n_mtp_layers);
    }
    for (i, r) in ratios.iter().enumerate() {
        if !matches!(r, 0 | 4 | 128) {
            bail!("compress_ratios[{i}] = {r}: expected 0 (SWA), 4 (CSA) or 128 (HCA)");
        }
    }
    // NB: inference/config.json carries NO max_seq_len — ModelArgs' demo default is 4096
    // (§D trap) and the serving value 1048576 lives in the HF config's
    // max_position_embeddings. Loader v0 is per-layer and deliberately does not model it.

    Ok(Dsv4Config {
        vocab_size: u("vocab_size")?,
        dim: u("dim")?,
        moe_inter_dim: u("moe_inter_dim")?,
        n_layers,
        n_hash_layers: u("n_hash_layers")?,
        n_mtp_layers,
        dspark_block_size: u("dspark_block_size")?,
        dspark_noise_token_id: u("dspark_noise_token_id")? as u32,
        dspark_target_layer_ids: target_ids,
        dspark_markov_rank: u("dspark_markov_rank")?,
        n_heads: u("n_heads")?,
        n_routed_experts: u("n_routed_experts")?,
        n_shared_experts: u("n_shared_experts")?,
        n_activated_experts: u("n_activated_experts")?,
        route_scale: f("route_scale")?,
        swiglu_limit: f("swiglu_limit")?,
        q_lora_rank: u("q_lora_rank")?,
        head_dim: u("head_dim")?,
        rope_head_dim: u("rope_head_dim")?,
        o_groups: u("o_groups")?,
        o_lora_rank: u("o_lora_rank")?,
        window_size: u("window_size")?,
        original_seq_len: u("original_seq_len")?,
        rope_theta: f("rope_theta")?,
        rope_factor: f("rope_factor")?,
        beta_fast: u("beta_fast")? as u32,
        beta_slow: u("beta_slow")? as u32,
        index_n_heads: u("index_n_heads")?,
        index_head_dim: u("index_head_dim")?,
        index_topk: u("index_topk")?,
        hc_mult: u("hc_mult")?,
        hc_sinkhorn_iters: u("hc_sinkhorn_iters")?,
        compress_rope_theta: f("compress_rope_theta")?,
        compress_ratios: ratios,
        // NOT in the json: ModelArgs defaults (model.py:62, :81) are the reference values.
        norm_eps: 1e-6,
        hc_eps: 1e-6,
    })
}

// ---------------------------------------------------------------------------
// Strict key sets (checkpoint-verified against §A and dsv4_ref.py selfload)
// ---------------------------------------------------------------------------

/// Keys every MoE block owns (trunk layer AND DSpark stage), prefix stripped.
/// Scales are listed because the strict gate compares RAW checkpoint keys.
const COMMON_KEYS: &[&str] = &[
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.scale",
    "attn.wkv.weight",
    "attn.wo_a.scale",
    "attn.wo_a.weight",
    "attn.wo_b.scale",
    "attn.wo_b.weight",
    "attn.wq_a.scale",
    "attn.wq_a.weight",
    "attn.wq_b.scale",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
];

/// Attention compressor (CSA + HCA layers; wkv/wgate shapes differ, keys identical).
const COMPRESSOR_KEYS: &[&str] = &[
    "attn.compressor.ape",
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
];

/// Block indexer (CSA only).
const INDEXER_KEYS: &[&str] = &[
    "attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight",
    "attn.indexer.compressor.wgate.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wq_b.scale",
    "attn.indexer.wq_b.weight",
];

const MTP0_KEYS: &[&str] = &["main_norm.weight", "main_proj.scale", "main_proj.weight"];

const MTP2_KEYS: &[&str] = &[
    "confidence_head.proj.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "norm.weight",
];

fn push_expert_keys(set: &mut BTreeSet<String>, n_experts: usize) {
    for e in 0..n_experts {
        for w in 1..=3u8 {
            set.insert(format!("ffn.experts.{e}.w{w}.weight"));
            set.insert(format!("ffn.experts.{e}.w{w}.scale"));
        }
    }
}

/// Expected RAW checkpoint keys (prefix stripped) for trunk layer `layer` (0..42).
fn expected_trunk_keys(cfg: &Dsv4Config, layer: usize) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = COMMON_KEYS.iter().map(|s| s.to_string()).collect();
    // Routing side (§A.2): layers 0..2 hash-route (tid2eid, no bias); ≥3 bias-select.
    if cfg.is_hash_layer(layer) {
        set.insert("ffn.gate.tid2eid".to_string());
    } else {
        set.insert("ffn.gate.bias".to_string());
    }
    match cfg.layer_kind(layer) {
        LayerKind::Swa => {}
        LayerKind::Csa => {
            set.extend(COMPRESSOR_KEYS.iter().map(|s| s.to_string()));
            set.extend(INDEXER_KEYS.iter().map(|s| s.to_string()));
        }
        LayerKind::Hca => set.extend(COMPRESSOR_KEYS.iter().map(|s| s.to_string())),
    }
    push_expert_keys(&mut set, cfg.n_routed_experts);
    set
}

/// Expected RAW checkpoint keys (prefix stripped) for DSpark stage `stage` (0..2).
/// Every stage is a full SWA MoE block with gate.bias (§A.3); embed/head are tied
/// references that do not exist as checkpoint keys.
fn expected_mtp_keys(cfg: &Dsv4Config, stage: usize) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = COMMON_KEYS.iter().map(|s| s.to_string()).collect();
    set.insert("ffn.gate.bias".to_string());
    if stage == 0 {
        set.extend(MTP0_KEYS.iter().map(|s| s.to_string()));
    }
    if stage == cfg.n_mtp_layers - 1 {
        set.extend(MTP2_KEYS.iter().map(|s| s.to_string()));
    }
    push_expert_keys(&mut set, cfg.n_routed_experts);
    set
}

/// The strict gate: zero missing AND zero unexpected keys (dsv4_ref.py selfload's
/// `load_state_dict(strict=True)` semantics, mirrored on raw checkpoint keys).
fn strict_check(tag: &str, expected: &BTreeSet<String>, actual: &BTreeSet<String>) -> Result<()> {
    let missing: Vec<_> = expected.difference(actual).take(8).cloned().collect();
    let unexpected: Vec<_> = actual.difference(expected).take(8).cloned().collect();
    if !missing.is_empty() || !unexpected.is_empty() {
        bail!("{tag}: strict load failed — missing={missing:?} unexpected={unexpected:?} \
               ({} expected, {} actual)", expected.len(), actual.len());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cast rules (mirror dsv4_ref.py cast_rule / build_state_dict exactly)
// ---------------------------------------------------------------------------

/// bf16→f32 Linear weights of the attention/indexer compressors (cast_rule's list).
const COMPRESSOR_F32_KEYS: &[&str] = &[
    "attn.compressor.wkv.weight",
    "attn.compressor.wgate.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.compressor.wgate.weight",
];

/// bf16→f32 head-side weights (cast_rule's list).
const HEAD_F32_KEYS: &[&str] = &[
    "head.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
];

fn is_expert_key(key: &str) -> bool {
    // ffn.experts.{E}.w{1,2,3}.{weight,scale}
    let Some(rest) = key.strip_prefix("ffn.experts.") else { return false };
    let mut it = rest.split('.');
    let (e, w, kind) = (it.next(), it.next(), it.next());
    matches!(kind, Some("weight" | "scale"))
        && matches!(w, Some("w1" | "w2" | "w3"))
        && e.map_or(false, |e| !e.is_empty() && e.bytes().all(|b| b.is_ascii_digit()))
        && it.next().is_none()
}

/// Apply the cast rules to one layer's raw tensors (stripped keys), returning the
/// tensor map + repacked experts. The key set has already passed `strict_check`.
fn process_layer(raw: HashMap<String, RawTensor>, cfg: &Dsv4Config, tag: &str) -> Result<Dsv4Layer> {
    let mut tensors: HashMap<String, HostTensor> = HashMap::new();
    let mut experts_w: [Vec<Nvfp4Tensor>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // Experts first: repack per expert in ascending id order (Vec index == expert id).
    // Absent only for the trunk top level, which owns no experts (strict-checked).
    let has_experts = raw.contains_key("ffn.experts.0.w1.weight");
    for (wi, experts) in experts_w.iter_mut().enumerate() {
        if !has_experts {
            break;
        }
        for e in 0..cfg.n_routed_experts {
            let wk = format!("ffn.experts.{e}.w{}.weight", wi + 1);
            let sk = format!("ffn.experts.{e}.w{}.scale", wi + 1);
            let w = raw.get(&wk).ok_or_else(|| anyhow!("{tag}: {wk} missing"))?;
            let s = raw.get(&sk).ok_or_else(|| anyhow!("{tag}: {sk} missing"))?;
            if w.dtype != StDtype::I8 || s.dtype != StDtype::F8E8M0 {
                bail!("{tag}: {wk} dtypes {:?}/{:?}, expected I8/F8_E8M0", w.dtype, s.dtype);
            }
            if w.shape.len() != 2 {
                bail!("{tag}: {wk} shape {:?}, expected 2-D packed", w.shape);
            }
            let (out, k) = (w.shape[0], w.shape[1] * 2); // I8 packs 2 nibbles along K
            experts.push(repack_expert_fp4_to_nvfp4(&w.data, &s.data, out, k));
        }
    }

    for (key, t) in &raw {
        if is_expert_key(key) {
            continue; // handled above (weight+scale → experts_w{1,2,3})
        }
        if let Some(base) = key.strip_suffix(".scale") {
            // Consumed by the sibling FP8 weight's dequant (incl. wo_a.scale). Verify
            // the pair exists and is an FP8 pair — a bare scale is a checkpoint bug.
            let w = raw.get(&format!("{base}.weight"))
                .ok_or_else(|| anyhow!("{tag}: {key} has no sibling weight"))?;
            if t.dtype != StDtype::F8E8M0 || w.dtype != StDtype::F8E4M3 {
                bail!("{tag}: {base} FP8 pair dtypes {:?}/{:?}", w.dtype, t.dtype);
            }
            continue;
        }
        let host = if key == "attn.wo_a.weight" {
            // §F.2: FP8 + E8M0 → bf16 (the scale is folded; loading raw corrupts O).
            if t.dtype != StDtype::F8E4M3 {
                bail!("{tag}: attn.wo_a.weight dtype {:?}, expected F8_E4M3", t.dtype);
            }
            let s = raw.get("attn.wo_a.scale").unwrap(); // strict-checked
            let (out, k) = fp8_pair_dims(&t, &s, tag, "attn.wo_a.weight")?;
            HostTensor::BF16 { shape: t.shape.clone(),
                               data: dequant_wo_a_bf16(&t.data, &s.data, out, k) }
        } else if t.dtype == StDtype::F8E4M3 {
            // Any other FP8 weight: exact f32 dequant with its 128×128 block scales.
            let s = raw.get(&scale_key(key)).ok_or_else(|| anyhow!("{tag}: {key} missing scale"))?;
            let (out, k) = fp8_pair_dims(&t, &s, tag, key)?;
            HostTensor::F32 { shape: t.shape.clone(),
                              data: dequant_fp8_exact(&t.data, &s.data, out, k) }
        } else if key == "ffn.gate.tid2eid" {
            if t.dtype != StDtype::I64 {
                bail!("{tag}: ffn.gate.tid2eid dtype {:?}, expected I64", t.dtype);
            }
            let data: Result<Vec<i32>> = t.data.chunks_exact(8)
                .map(|c| i32::try_from(i64::from_le_bytes(c.try_into().unwrap()))
                    .map_err(|_| anyhow!("{tag}: tid2eid value out of i32 range")))
                .collect();
            HostTensor::I32 { shape: t.shape.clone(), data: data? }
        } else if key.ends_with("norm.weight")
            || COMPRESSOR_F32_KEYS.contains(&key.as_str())
            || HEAD_F32_KEYS.contains(&key.as_str())
        {
            // RMSNorms / compressor projections / head-side weights: bf16 → f32.
            HostTensor::F32 { shape: t.shape.clone(), data: t.bf16_to_f32()? }
        } else {
            // Everything else keeps checkpoint dtype (§F.5): embed/gate/weights_proj/
            // markov_w1 stay BF16; attn_sink, hc_*, gate.bias, *.ape are already F32.
            match t.dtype {
                StDtype::BF16 => HostTensor::BF16 { shape: t.shape.clone(), data: t.as_bf16()? },
                StDtype::F32 => HostTensor::F32 { shape: t.shape.clone(), data: t.as_f32()? },
                other => bail!("{tag}: {key}: no cast rule for dtype {other:?}"),
            }
        };
        tensors.insert(key.clone(), host);
    }

    let [experts_w1, experts_w2, experts_w3] = experts_w;
    Ok(Dsv4Layer { tensors, experts_w1, experts_w2, experts_w3 })
}

/// `X.weight` → `X.scale`.
fn scale_key(weight_key: &str) -> String {
    format!("{}.scale", weight_key.strip_suffix(".weight").unwrap_or(weight_key))
}

/// Validate FP8 pair geometry, returning (out, k): weight [out,k] F8_E4M3 with
/// out,k % 128 == 0 and scale [out/128, k/128] (§B block geometry).
fn fp8_pair_dims(w: &RawTensor, s: &RawTensor, tag: &str, key: &str) -> Result<(usize, usize)> {
    if w.shape.len() != 2 || s.shape.len() != 2 {
        bail!("{tag}: {key} shapes {:?}/{:?}, expected 2-D pair", w.shape, s.shape);
    }
    let (out, k) = (w.shape[0], w.shape[1]);
    if out % FP8_GROUP != 0 || k % FP8_GROUP != 0 {
        bail!("{tag}: {key} [{out},{k}] not divisible by {FP8_GROUP}");
    }
    if s.shape != [out / FP8_GROUP, k / FP8_GROUP] {
        bail!("{tag}: {key}.scale shape {:?}, expected [{},{}]", s.shape, out / FP8_GROUP, k / FP8_GROUP);
    }
    Ok((out, k))
}

/// Load all keys under `prefix` and run the strict gate + cast rules.
fn load_block(bundle_dir: &Path, cfg: &Dsv4Config, prefix: &str,
              expected: BTreeSet<String>, skip: impl Fn(&str) -> bool, tag: &str) -> Result<Dsv4Layer> {
    let wm = load_weight_map(bundle_dir)?;
    let names: Vec<String> = wm.keys().filter(|k| k.starts_with(prefix)).cloned().collect();
    let raw = stream_raw_tensors(bundle_dir, &names)?;
    let mut stripped: HashMap<String, RawTensor> = HashMap::with_capacity(raw.len());
    for (k, t) in raw {
        let key = k.strip_prefix(prefix).unwrap().to_string();
        if skip(&key) {
            continue; // mtp.*embed* / mtp.*head.weight — tied to trunk (§F.5)
        }
        stripped.insert(key, t);
    }
    let actual: BTreeSet<String> = stripped.keys().cloned().collect();
    strict_check(tag, &expected, &actual)?;
    process_layer(stripped, cfg, tag)
}

/// Strict-load trunk layer `layer` (0..42): stream shards, stack experts, cast rules.
/// Every key the reference module owns must be present (missing = error), and no
/// unexpected keys may remain (unexpected = error) — the §7 "strict-load" gate.
pub fn load_layer(bundle_dir: &Path, cfg: &Dsv4Config, layer: usize) -> Result<Dsv4Layer> {
    if layer >= cfg.n_layers {
        bail!("layer {layer} out of range (n_layers = {})", cfg.n_layers);
    }
    let expected = expected_trunk_keys(cfg, layer);
    load_block(bundle_dir, cfg, &format!("layers.{layer}."), expected, |_| false,
               &format!("layers.{layer}"))
}

/// Strict-load DSpark stage `stage` (0..2; `mtp.{S}.*`; embed/head skipped as tied).
pub fn load_mtp_stage(bundle_dir: &Path, cfg: &Dsv4Config, stage: usize) -> Result<Dsv4Layer> {
    if stage >= cfg.n_mtp_layers {
        bail!("mtp stage {stage} out of range (n_mtp_layers = {})", cfg.n_mtp_layers);
    }
    let expected = expected_mtp_keys(cfg, stage);
    load_block(bundle_dir, cfg, &format!("mtp.{stage}."), expected,
               |key| key.contains("embed") || key == "head.weight",
               &format!("mtp.{stage}"))
}

/// Trunk top level: `embed.weight` (BF16), `norm.weight` (F32), `head.weight` (F32),
/// `hc_head_fn` / `hc_head_base` / `hc_head_scale` (F32). Keys unstripped.
pub fn load_trunk_top(bundle_dir: &Path, cfg: &Dsv4Config) -> Result<HashMap<String, HostTensor>> {
    let expected: BTreeSet<String> = ["embed.weight", "norm.weight", "head.weight",
                                      "hc_head_fn", "hc_head_base", "hc_head_scale"]
        .iter().map(|s| s.to_string()).collect();
    let names: Vec<String> = expected.iter().cloned().collect();
    let raw = stream_raw_tensors(bundle_dir, &names)?;
    let actual: BTreeSet<String> = raw.keys().cloned().collect();
    strict_check("trunk_top", &expected, &actual)?;
    let layer = process_layer(raw, cfg, "trunk_top")?;
    Ok(layer.tensors)
}

// ---------------------------------------------------------------------------
// Dequant / repack primitives (the §F exactness claims live here)
// ---------------------------------------------------------------------------

/// UE8M0 scale byte → f32: `2^(b-127)`; 0xFF is NaN in the spec — error if seen.
pub fn e8m0_to_f32(b: u8) -> f32 {
    assert!(b != 0xFF, "E8M0 0xFF (NaN) in a scale tensor — checkpoint corruption");
    2f32.powi(b as i32 - 127)
}

/// FP8-E4M3 `[out,k]` + UE8M0 `[out/128, k/128]` → exact f32 (each product exact).
pub fn dequant_fp8_exact(data: &[u8], scales_e8m0: &[u8], out: usize, k: usize) -> Vec<f32> {
    assert_eq!(data.len(), out * k, "fp8 data size mismatch");
    assert_eq!(scales_e8m0.len(), (out / FP8_GROUP) * (k / FP8_GROUP), "fp8 scale size mismatch");
    assert!(out % FP8_GROUP == 0 && k % FP8_GROUP == 0);
    let mut v = vec![0.0f32; out * k];
    let ks = k / FP8_GROUP;
    for rb in 0..out / FP8_GROUP {
        for cb in 0..ks {
            let s = e8m0_to_f32(scales_e8m0[rb * ks + cb]);
            for i in 0..FP8_GROUP {
                let row = (rb * FP8_GROUP + i) * k + cb * FP8_GROUP;
                for j in 0..FP8_GROUP {
                    v[row + j] = e4m3_to_f32(data[row + j]) * s;
                }
            }
        }
    }
    v
}

/// §F.2 (`convert.py:122-127` exactly): FP8 `[out,k]` + UE8M0 `[out/128, k/128]` →
/// **bf16** (fp32 multiply per 128×128 block, then round-to-nearest bf16).
pub fn dequant_wo_a_bf16(data: &[u8], scales_e8m0: &[u8], out: usize, k: usize) -> Vec<bf16> {
    assert_eq!(data.len(), out * k, "wo_a data size mismatch");
    assert_eq!(scales_e8m0.len(), (out / FP8_GROUP) * (k / FP8_GROUP), "wo_a scale size mismatch");
    assert!(out % FP8_GROUP == 0 && k % FP8_GROUP == 0);
    let mut v = vec![bf16::ZERO; out * k];
    let ks = k / FP8_GROUP;
    for rb in 0..out / FP8_GROUP {
        for cb in 0..ks {
            let s = e8m0_to_f32(scales_e8m0[rb * ks + cb]);
            for i in 0..FP8_GROUP {
                let row = (rb * FP8_GROUP + i) * k + cb * FP8_GROUP;
                for j in 0..FP8_GROUP {
                    // one f32 multiply, one rounding to bf16 — matches convert.py.
                    v[row + j] = bf16::from_f32(e4m3_to_f32(data[row + j]) * s);
                }
            }
        }
    }
    v
}

/// §F.3: on-disk FP4-E2M1 (I8-packed, low nibble = even K) + UE8M0 per-32-K → our
/// NVFP4 (E2M1 nibbles copied unchanged; per-16 E4M3 subgroup scales; f32 global
/// `2^(8-kmax)` with the reciprocal convention of `quant.rs::Nvfp4Tensor`).
/// Construction is verified bit-exact by [`e8m0_dequant_fp4_exact`] A/B.
pub fn repack_expert_fp4_to_nvfp4(packed: &[u8], scales_e8m0: &[u8], out: usize, k: usize) -> Nvfp4Tensor {
    assert_eq!(packed.len(), out * (k / 2), "packed fp4 size mismatch");
    assert_eq!(scales_e8m0.len(), out * (k / FP4_GROUP), "fp4 scale size mismatch");
    assert!(k % FP4_GROUP == 0);

    // kmax = max E8M0 exponent (biased byte) over the whole tensor. §F.3 measured a
    // worst-case span of 6 octaves across all 45 sampled tensors, so the per-16 E4M3
    // codes 2^(b+8-kmax) land in E4M3's exact-pow2 window [2^-9, 2^8] with margin.
    let kmax = *scales_e8m0.iter().max().unwrap() as i32;
    assert!(!scales_e8m0.contains(&0xFF), "E8M0 0xFF (NaN) in expert scales");
    assert!(kmax >= 8, "expert scales pathologically small (kmax byte {kmax} < 8): \
                        NVFP4 global 2^(8-(kmax-127)) not representable in f32");
    // Reciprocal convention (quant.rs): dequant DIVIDES by global_scale.
    let global_scale = 2f32.powi(8 - (kmax - 127));

    let nblk = k / 16; // NVFP4 subgroup size along K (format-fixed, quant.rs BLOCK)
    let mut scales = vec![0u8; out * nblk];
    for r in 0..out {
        for g in 0..k / FP4_GROUP {
            let b = scales_e8m0[r * (k / FP4_GROUP) + g] as i32;
            let p = b + 8 - kmax; // E4M3 scale exponent for both 16-K subgroups of this 32-group
            assert!((-9..=8).contains(&p),
                    "E8M0 span > 17 octaves (byte {b}, kmax {kmax}) — not E4M3-exact");
            let code = f32_to_e4m3(2f32.powi(p)); // exact pow2 → exact code
            scales[r * nblk + 2 * g] = code;
            scales[r * nblk + 2 * g + 1] = code;
        }
    }
    // Nibbles copied unchanged — the on-disk packing (low nibble = even K) IS ours.
    Nvfp4Tensor { qweight: packed.to_vec(), scales, global_scale, m: out, k }
}

/// Reference-side exact dequant of the on-disk FP4 encoding (E2M1 × 2^(e8m0-127)),
/// for the §F.3 A/B gate.
pub fn e8m0_dequant_fp4_exact(packed: &[u8], scales_e8m0: &[u8], out: usize, k: usize) -> Vec<f32> {
    assert_eq!(packed.len(), out * (k / 2), "packed fp4 size mismatch");
    assert_eq!(scales_e8m0.len(), out * (k / FP4_GROUP), "fp4 scale size mismatch");
    let mut v = vec![0.0f32; out * k];
    for r in 0..out {
        for g in 0..k / FP4_GROUP {
            let s = e8m0_to_f32(scales_e8m0[r * (k / FP4_GROUP) + g]);
            for j in 0..FP4_GROUP {
                let col = g * FP4_GROUP + j;
                let byte = packed[r * (k / 2) + col / 2];
                let code = if col % 2 == 0 { byte & 0x0F } else { byte >> 4 }; // low nibble = even K
                v[r * k + col] = e2m1_to_f32(code) * s;
            }
        }
    }
    v
}

/// NVFP4 → exact f32 (E2M1 × E4M3 × (1/global) — all factors exact in f32).
pub fn dequant_nvfp4_f32(t: &Nvfp4Tensor) -> Vec<f32> {
    let nblk = t.k / 16; // quant.rs BLOCK
    let s_tensor = 1.0 / t.global_scale; // reciprocal convention — exact pow2 here
    let mut v = vec![0.0f32; t.m * t.k];
    for row in 0..t.m {
        for b in 0..nblk {
            let s = e4m3_to_f32(t.scales[row * nblk + b]) * s_tensor;
            for i in 0..16 {
                let idx = row * t.k + b * 16 + i;
                let byte = t.qweight[idx / 2];
                let code = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                v[idx] = e2m1_to_f32(code) * s;
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// npy I/O (v1.0, little-endian, C-order; `<f4` / `<i8` only)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum NpyData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// Extract a `'key': value` field from a numpy v1.0 header dict literal.
fn npy_header_field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let pat = format!("'{key}':");
    let start = header.find(&pat).ok_or_else(|| anyhow!("npy header: no `{key}`"))? + pat.len();
    let rest = header[start..].trim_start();
    let end = rest.find([',', '}']).ok_or_else(|| anyhow!("npy header: malformed `{key}`"))?;
    Ok(rest[..end].trim())
}

pub fn read_npy(path: &Path) -> Result<(Vec<usize>, NpyData)> {
    let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_npy(&buf).with_context(|| format!("parse {}", path.display()))
}

/// Parse a numpy v1.0 little-endian buffer (the `.npy` magic + header + data). Shared by
/// [`read_npy`] (file) and [`read_npz_key`] (zip entry). `<f4` / `<i8` only, C-order.
pub fn parse_npy(buf: &[u8]) -> Result<(Vec<usize>, NpyData)> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        bail!("not a npy buffer (magic mismatch)");
    }
    let (major, _minor) = (buf[6], buf[7]);
    if major != 1 {
        bail!("npy version {major}.x unsupported (v1.0 only)");
    }
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(buf.get(10..10 + hlen).ok_or_else(|| anyhow!("truncated header"))?)
        .context("npy header not utf8")?;
    let descr = npy_header_field(header, "descr")?.trim_matches(['\'', '"']);
    if npy_header_field(header, "fortran_order")? != "False" {
        bail!("npy fortran_order=True unsupported");
    }
    let shape_str = {
        // The shape tuple contains commas — read to its closing paren, not the next field.
        let pat = "'shape':";
        let start = header.find(pat).ok_or_else(|| anyhow!("npy header: no `shape`"))? + pat.len();
        let rest = &header[start..];
        let open = rest.find('(').ok_or_else(|| anyhow!("npy header: malformed shape"))?;
        let close = rest[open..].find(')').ok_or_else(|| anyhow!("npy header: malformed shape"))?;
        &rest[open + 1..open + close]
    };
    let shape: Vec<usize> = shape_str.split(',')
        .filter_map(|s| { let s = s.trim(); if s.is_empty() { None } else { Some(s.parse::<usize>()) } })
        .collect::<Result<_, _>>().context("npy header: bad shape")?;
    let n: usize = shape.iter().product();
    let data = &buf[10 + hlen..];
    let out = match descr {
        "<f4" => {
            if data.len() != n * 4 { bail!("npy data {} bytes, expected {}", data.len(), n * 4); }
            NpyData::F32(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        "<i8" => {
            if data.len() != n * 8 { bail!("npy data {} bytes, expected {}", data.len(), n * 8); }
            NpyData::I64(data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
        }
        other => bail!("npy descr `{other}` unsupported (`<f4`/`<i8` only)"),
    };
    Ok((shape, out))
}

/// Read one array from an `.npz` (a zip of `.npy` entries) by key. `key` selects the
/// `"{key}.npy"` archive member. Used by `--probe-dsv4` to read the oracle pieces directly
/// (no `dsv4_diff.py export` step needed). `<f4` / `<i8` only, C-order.
pub fn read_npz_key(npz_path: &Path, key: &str) -> Result<(Vec<usize>, NpyData)> {
    let f = std::fs::File::open(npz_path).with_context(|| format!("open {}", npz_path.display()))?;
    let mut zip = zip::ZipArchive::new(f).with_context(|| format!("read zip {}", npz_path.display()))?;
    // npz members are "{key}.npy"; some writers use "arr_0" etc., but numpy.savez uses the kwarg name.
    let member = format!("{key}.npy");
    let mut entry = zip
        .by_name(&member)
        .with_context(|| format!("npz {}: no member {member:?}", npz_path.display()))?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    use std::io::Read;
    entry.read_to_end(&mut buf).context("read npz member")?;
    parse_npy(&buf).with_context(|| format!("parse npz::{key}"))
}

/// Write one npy v1.0 file: 10-byte preamble + padded dict + little-endian data.
fn write_npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> Result<()> {
    let shape_str = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Pad with spaces so (preamble + header) is a multiple of 64; header ends in \n.
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    let header = format!("{dict}{}{}", " ".repeat(pad), "\n");
    let mut f = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    use std::io::Write;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1, 0])?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(data)?;
    Ok(())
}

pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    let n: usize = shape.iter().product();
    if data.len() != n { bail!("write_npy_f32: {} values for shape {:?}", data.len(), shape); }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data { bytes.extend_from_slice(&x.to_le_bytes()); }
    write_npy(path, "<f4", shape, &bytes)
}

pub fn write_npy_i64(path: &Path, shape: &[usize], data: &[i64]) -> Result<()> {
    let n: usize = shape.iter().product();
    if data.len() != n { bail!("write_npy_i64: {} values for shape {:?}", data.len(), shape); }
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for x in data { bytes.extend_from_slice(&x.to_le_bytes()); }
    write_npy(path, "<i8", shape, &bytes)
}

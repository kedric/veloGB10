//! Visual tower of the Qwen3.5-27B vision model: weight structures + strict loader.
//!
//! The 333 `model.visual.*` tensors (all BF16) form: patch_embed (Conv3d), pos_embed (learned
//! bilinear table), 27 ViT blocks, and the merger. Shapes are pinned in PLAN/W2_PREPROC_SPEC.md;
//! the weights are proven correct by the V2 cross-chain per-block rel-L2 oracle. This module loads
//! them strictly: a missing or unexpected `model.visual.*` tensor is an ERROR, never a warning.

use anyhow::{anyhow, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub const PATCH: usize = 16;
pub const TEMPORAL: usize = 2;
pub const MERGE: usize = 2;
pub const IN_CH: usize = 3;
pub const HIDDEN: usize = 1152;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / HEADS; // 72
pub const INTER: usize = 4304;
pub const NUM_BLOCKS: usize = 27;
pub const NUM_POS: usize = 2304;
/// Merger output width of the Qwen3.5-class towers (= text hidden 5120). qwen4_exp's tower is the
/// same network with a 2560-wide merger output: the width is read from the checkpoint
/// (`VisualTower::out_hidden`), this constant only names the historical default.
pub const OUT_HIDDEN: usize = 5120;
pub const MERGE_INTER: usize = HIDDEN * MERGE * MERGE; // 4608

/// Per-ViT-block weights (12 tensors). Stored as f32 (weights are BF16 on disk).
#[derive(Clone, Debug)]
pub struct VisualBlock {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub qkv_w: Vec<f32>,   // [3*HIDDEN, HIDDEN] = [3456, 1152]
    pub qkv_b: Vec<f32>,   // [3*HIDDEN]
    pub proj_w: Vec<f32>,  // [HIDDEN, HIDDEN]
    pub proj_b: Vec<f32>,  // [HIDDEN]
    pub fc1_w: Vec<f32>,   // [INTER, HIDDEN] = [4304, 1152]
    pub fc1_b: Vec<f32>,   // [INTER]
    pub fc2_w: Vec<f32>,   // [HIDDEN, INTER]
    pub fc2_b: Vec<f32>,   // [HIDDEN]
}

/// The complete vision tower weights.
#[derive(Clone, Debug)]
pub struct VisualTower {
    pub patch_embed_w: Vec<f32>, // [HIDDEN, IN_CH, TEMPORAL, PATCH, PATCH] = [1152, 3,2,16,16]
    pub patch_embed_b: Vec<f32>, // [HIDDEN]
    pub pos_embed_w: Vec<f32>,   // [NUM_POS, HIDDEN] = [2304, 1152]
    pub blocks: Vec<VisualBlock>,
    pub merger_norm_w: Vec<f32>, // [HIDDEN]
    pub merger_norm_b: Vec<f32>, // [HIDDEN]
    pub merger_fc1_w: Vec<f32>,  // [MERGE_INTER, MERGE_INTER] = [4608, 4608]
    pub merger_fc1_b: Vec<f32>,  // [MERGE_INTER]
    pub merger_fc2_w: Vec<f32>,  // [out_hidden, MERGE_INTER] = [5120|2560, 4608]
    pub merger_fc2_b: Vec<f32>,  // [out_hidden]
    /// Merger output width = the language model's hidden size (5120 Qwen3.5 27B, 2560 qwen4_exp).
    pub out_hidden: usize,
}

struct Map<'a> {
    m: HashMap<String, (&'a str, &'a [u8])>,
}

impl<'a> Map<'a> {
    fn build(all_raw: &'a [Vec<u8>]) -> Result<Self> {
        let mut m = HashMap::new();
        for raw in all_raw {
            let st = SafeTensors::deserialize(raw)?;
            use safetensors::Dtype;
            for (name, view) in st.tensors() {
                let dt = match view.dtype() {
                    Dtype::BF16 => "BF16",
                    Dtype::F16 => "F16",
                    Dtype::F32 => "F32",
                    _ => "OTHER",
                };
                m.insert(name.to_string(), (dt, view.data()));
            }
        }
        Ok(Map { m })
    }

    /// Element count of a tensor (by its byte length and dtype).
    fn len(&self, name: &str) -> Result<usize> {
        let (dt, data) = self.m.get(name).ok_or_else(|| anyhow!("missing tensor: {}", name))?;
        Ok(match *dt { "F32" => data.len() / 4, _ => data.len() / 2 })
    }

    fn get(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let (dt, data) = self
            .m
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor: {}", name))?;
        let v = match *dt {
            "BF16" | "F16" => {
                let m = data.len() / 2;
                let mut out = Vec::with_capacity(m);
                for i in 0..m {
                    let b = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                    out.push(f32::from_bits((b as u32) << 16));
                }
                out
            }
            "F32" => {
                let m = data.len() / 4;
                let mut out = Vec::with_capacity(m);
                for i in 0..m {
                    out.push(f32::from_le_bytes([
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ]));
                }
                out
            }
            other => return Err(anyhow!("unsupported dtype {} for {}", other, name)),
        };
        assert_eq!(v.len(), n, "shape mismatch {}: got {} expect {}", name, v.len(), n);
        Ok(v)
    }
}

impl VisualTower {
    /// Strict-load the 333 `model.visual.*` tensors from a model directory. Errors on any missing
    /// or unexpected `model.visual.*` tensor; every loaded shape is asserted.
    pub fn load(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);
        if !dir.is_dir() {
            return Err(anyhow!("not a directory: {}", model_dir));
        }
        // gather safetensors shards
        let mut shards: Vec<String> = vec![];
        let index = dir.join("model.safetensors.index.json");
        if index.exists() {
            let raw = std::fs::read_to_string(&index)?;
            let j: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(wm) = j["weight_map"].as_object() {
                // Only the shards that hold `model.visual.*`. Reading every shard into host memory
                // to find them is what pushed the box into the kernel's OOM path on a 97 GB
                // artifact (2026-08-28): the tower is ~1.3 GB and lives in one 4 GB shard.
                let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                for (k, v) in wm {
                    if !k.starts_with("model.visual.") { continue; }
                    if let Some(s) = v.as_str() {
                        set.insert(s.to_string());
                    }
                }
                for s in set {
                    shards.push(dir.join(s).to_string_lossy().to_string());
                }
            }
        } else {
            for entry in std::fs::read_dir(dir)? {
                let e = entry?;
                let nm = e.file_name().to_string_lossy().to_string();
                if nm.ends_with(".safetensors") {
                    shards.push(e.path().to_string_lossy().to_string());
                }
            }
            shards.sort();
        }
        if shards.is_empty() {
            return Err(anyhow!("no safetensors found in {}", model_dir));
        }
        let mut all_raw: Vec<Vec<u8>> = Vec::new();
        for s in &shards {
            all_raw.push(std::fs::read(s)?);
        }
        let map = Map::build(&all_raw)?;

        // Load all 333 visual tensors strictly.
        let mut block_names: Vec<String> = Vec::new();
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            let bp = format!("blocks.{i}.");
            let p = format!("model.visual.{bp}");
            let b = VisualBlock {
                norm1_w: map.get(&format!("{p}norm1.weight"), HIDDEN)?,
                norm1_b: map.get(&format!("{p}norm1.bias"), HIDDEN)?,
                norm2_w: map.get(&format!("{p}norm2.weight"), HIDDEN)?,
                norm2_b: map.get(&format!("{p}norm2.bias"), HIDDEN)?,
                qkv_w: map.get(&format!("{p}attn.qkv.weight"), 3 * HIDDEN * HIDDEN)?,
                qkv_b: map.get(&format!("{p}attn.qkv.bias"), 3 * HIDDEN)?,
                proj_w: map.get(&format!("{p}attn.proj.weight"), HIDDEN * HIDDEN)?,
                proj_b: map.get(&format!("{p}attn.proj.bias"), HIDDEN)?,
                fc1_w: map.get(&format!("{p}mlp.linear_fc1.weight"), INTER * HIDDEN)?,
                fc1_b: map.get(&format!("{p}mlp.linear_fc1.bias"), INTER)?,
                fc2_w: map.get(&format!("{p}mlp.linear_fc2.weight"), HIDDEN * INTER)?,
                fc2_b: map.get(&format!("{p}mlp.linear_fc2.bias"), HIDDEN)?,
            };
            block_names.extend(block_names_12(&bp));
            blocks.push(b);
        }
        // aggregate the consumed names for the strict "unexpected" check
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        consumed.insert("model.visual.patch_embed.proj.weight".into());
        consumed.insert("model.visual.patch_embed.proj.bias".into());
        consumed.insert("model.visual.pos_embed.weight".into());
        consumed.insert("model.visual.merger.norm.weight".into());
        consumed.insert("model.visual.merger.norm.bias".into());
        consumed.insert("model.visual.merger.linear_fc1.weight".into());
        consumed.insert("model.visual.merger.linear_fc1.bias".into());
        consumed.insert("model.visual.merger.linear_fc2.weight".into());
        consumed.insert("model.visual.merger.linear_fc2.bias".into());
        for b in &block_names {
            consumed.insert(b.clone());
        }

        let out_hidden = map.len("model.visual.merger.linear_fc2.bias")?;
        let tower = VisualTower {
            patch_embed_w: map.get("model.visual.patch_embed.proj.weight", HIDDEN * IN_CH * TEMPORAL * PATCH * PATCH)?,
            patch_embed_b: map.get("model.visual.patch_embed.proj.bias", HIDDEN)?,
            pos_embed_w: map.get("model.visual.pos_embed.weight", NUM_POS * HIDDEN)?,
            blocks,
            merger_norm_w: map.get("model.visual.merger.norm.weight", HIDDEN)?,
            merger_norm_b: map.get("model.visual.merger.norm.bias", HIDDEN)?,
            merger_fc1_w: map.get("model.visual.merger.linear_fc1.weight", MERGE_INTER * MERGE_INTER)?,
            merger_fc1_b: map.get("model.visual.merger.linear_fc1.bias", MERGE_INTER)?,
            merger_fc2_w: map.get("model.visual.merger.linear_fc2.weight", out_hidden * MERGE_INTER)?,
            merger_fc2_b: map.get("model.visual.merger.linear_fc2.bias", out_hidden)?,
            out_hidden,
        };

        // Strict "unexpected tensor" check: every model.visual.* key must have been consumed.
        for name in map.m.keys() {
            if name.starts_with("model.visual.") && !consumed.contains(name.as_str()) {
                return Err(anyhow!("unexpected visual tensor not consumed: {}", name));
            }
        }
        Ok(tower)
    }

    /// Number of visual tensors consumed (should be 333).
    pub fn tensor_count(&self) -> usize {
        // 1 (patch w) + 1 (patch b) + 1 (pos) + 27*12 + 6 (merger)
        3 + NUM_BLOCKS * 12 + 6
    }
}

fn block_names_12(bp: &str) -> Vec<String> {
    [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
        "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
    ]
    .iter()
    .map(|s| format!("model.visual.{bp}{s}"))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_load_333() {
        let dir = std::env::var("GB10_TEST_MODEL_DIR")
            .unwrap_or_else(|_| "models/3.8-27b-nvfp4-full-all".to_string());
        let t = VisualTower::load(&dir).unwrap();
        assert_eq!(t.tensor_count(), 333, "visual tensor count");
        assert_eq!(t.blocks.len(), 27);
        assert_eq!(t.patch_embed_w.len(), 1152 * 3 * 2 * 16 * 16);
        assert_eq!(t.pos_embed_w.len(), 2304 * 1152);
        assert_eq!(t.merger_fc2_w.len(), t.out_hidden * 4608);
    }
}

//! CPU vision encoder — the validated visual tower forward (reference path).
//!
//! This is the exact math validated by the V2 cross-chain per-block rel-L2 oracle
//! (PLAN/W2_PREPROC_SPEC.md §10, GREEN: patch_embed+pos 3.9e-7, 27 blocks 3e-6..1.6e-5, merger
//! 1.47e-5). It consumes `pixel_values` (flatten [N, 1536], N = gh*gw, in the
//! `vision_preproc::flatten` order) and grid `(gh, gw)`, and produces the merged image
//! embeddings `[N/4, 5120]` to splice at the language embed layer.
//!
//! This is the CPU reference the GPU path must match, and is the compute engine for a first
//! (correctity-first) vision serving path. The GPU port (V2) replaces the heavy matmuls.

use base64::Engine;
use crate::vision_tower::{VisualTower, HIDDEN, HEAD_DIM, INTER, MERGE, PATCH, IN_CH, TEMPORAL};

/// Vision rotary: inv_freq length = (head_dim//2)/2 = 18 (mirrors Qwen3VLVisionRotaryEmbedding).
fn vision_inv_freq() -> Vec<f64> {
    let dim = (HEAD_DIM / 2) as f64; // 36
    (0..HEAD_DIM / 4).map(|i| 10000.0f64.powf(-(2.0 * i as f64) / dim)).collect()
}

fn layernorm(x: &[f32], w: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (ri, row) in x.chunks_exact(n).enumerate() {
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for (i, v) in row.iter().enumerate() {
            out[ri * n + i] = (v - mean) * inv * w[i] + b[i];
        }
    }
    out
}

fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu(x: f32) -> f32 {
    // nn.GELU() default erf form (the merger).
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

/// Generic GEMM `[N, inn] @ [inn, outn]^T + bias`, weights row-major `[outn, inn]`.
fn gemm_vec(x: &[f32], wt: &[f32], b: &[f32], out: &mut [f32], inn: usize, outn: usize, n: usize) {
    for i in 0..n {
        let xr = &x[i * inn..(i + 1) * inn];
        for o in 0..outn {
            let w = &wt[o * inn..(o + 1) * inn];
            let mut acc = b[o];
            for k in 0..inn {
                acc += xr[k] * w[k];
            }
            out[i * outn + o] = acc;
        }
    }
}

pub fn pos_embed_bilinear(w: &[f32], gh: usize, gw: usize, num_side: usize) -> Vec<f32> {
    let n = gh * gw;
    let mut pe = vec![0.0f32; n * HIDDEN];
    let mut hfl = vec![0usize; gh];
    let mut hfr = vec![0.0f32; gh];
    let mut wfl = vec![0usize; gw];
    let mut wfr = vec![0.0f32; gw];
    for i in 0..gh {
        let g = (num_side - 1) as f32 * i as f32 / (gh - 1).max(1) as f32;
        let f = g.floor();
        hfl[i] = f as usize;
        hfr[i] = g - f;
    }
    for i in 0..gw {
        let g = (num_side - 1) as f32 * i as f32 / (gw - 1).max(1) as f32;
        let f = g.floor();
        wfl[i] = f as usize;
        wfr[i] = g - f;
    }
    let hcel = |i: usize| (hfl[i] + 1).min(num_side - 1);
    let wcel = |i: usize| (wfl[i] + 1).min(num_side - 1);
    let mut reorder = Vec::with_capacity(n);
    for hi in 0..gh / MERGE {
        for wi in 0..gw / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    reorder.push((hi * MERGE + mh) * gw + (wi * MERGE + mw));
                }
            }
        }
    }
    for (tok, &pat) in reorder.iter().enumerate() {
        let (py, px) = (pat / gw, pat % gw);
        let (hy, hf) = (hfl[py], hfr[py]);
        let (wy, wf) = (wfl[px], wfr[px]);
        let (hcy, wcx) = (hcel(py), wcel(px));
        let corners = [
            (hy * num_side + wy, (1.0 - hf) * (1.0 - wf)),
            (hy * num_side + wcx, (1.0 - hf) * wf),
            (hcy * num_side + wy, hf * (1.0 - wf)),
            (hcy * num_side + wcx, hf * wf),
        ];
        for (gi, wt) in corners {
            for d in 0..HIDDEN {
                pe[tok * HIDDEN + d] += w[gi * HIDDEN + d] * wt;
            }
        }
    }
    pe
}

fn vision_position_ids(gh: usize, gw: usize) -> Vec<[usize; 2]> {
    let mut pos = Vec::with_capacity(gh * gw);
    for hi in 0..gh / MERGE {
        for wi in 0..gw / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    pos.push([hi * MERGE + mh, wi * MERGE + mw]);
                }
            }
        }
    }
    pos
}

pub fn vision_cos_sin(gh: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
    let n = gh * gw;
    let hd = HEAD_DIM;
    let inv = vision_inv_freq();
    let nu = inv.len(); // 18
    let pos = vision_position_ids(gh, gw);
    let mut cos = vec![0.0f32; n * hd];
    let mut sin = vec![0.0f32; n * hd];
    for (tok, [py, px]) in pos.iter().enumerate() {
        // ang = [py*inv (18) | px*inv (18)] -> 36
        let mut ang = vec![0.0f64; hd / 2];
        for j in 0..nu {
            ang[j] = *py as f64 * inv[j];
            ang[nu + j] = *px as f64 * inv[j];
        }
        // emb = [ang(36) | ang(36)] -> 72
        for d in 0..hd / 2 {
            let a = ang[d];
            cos[tok * hd + d] = a.cos() as f32;
            sin[tok * hd + d] = a.sin() as f32;
            cos[tok * hd + hd / 2 + d] = a.cos() as f32;
            sin[tok * hd + hd / 2 + d] = a.sin() as f32;
        }
    }
    (cos, sin)
}

fn rotate_half_hd(q: &[f32], hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hd];
    for i in 0..hd / 2 {
        out[i] = -q[hd / 2 + i];
        out[hd / 2 + i] = q[i];
    }
    out
}

impl VisualTower {
    /// CPU forward. `pixel_values`: [N, 1536] f32 flatten; `(gh, gw)` patch grid.
    /// Returns merged embeddings [N/4, OUT_HIDDEN].
    pub fn forward_cpu(&self, pixel_values: &[f32], gh: usize, gw: usize) -> Vec<f32> {
        let n = gh * gw;
        assert_eq!(pixel_values.len(), n * IN_CH * TEMPORAL * PATCH * PATCH);
        let wpv = IN_CH * TEMPORAL * PATCH * PATCH; // 1536
        let hd = HEAD_DIM;

        // 1. patch_embed [N,1536] @ [1536,1152]^T + b
        let mut h = vec![0.0f32; n * HIDDEN];
        for i in 0..n {
            let row = &pixel_values[i * wpv..(i + 1) * wpv];
            for o in 0..HIDDEN {
                let w = &self.patch_embed_w[o * wpv..(o + 1) * wpv];
                let mut acc = self.patch_embed_b[o];
                for k in 0..wpv {
                    acc += row[k] * w[k];
                }
                h[i * HIDDEN + o] = acc;
            }
        }
        // 2. pos-embed bilinear
        let num_side = ((crate::vision_tower::NUM_POS) as f64).sqrt() as usize;
        let pe = pos_embed_bilinear(&self.pos_embed_w, gh, gw, num_side);
        for i in 0..n {
            for d in 0..HIDDEN {
                h[i * HIDDEN + d] += pe[i * HIDDEN + d];
            }
        }
        // 3. rotary cos/sin
        let (cos, sin) = vision_cos_sin(gh, gw);

        // 4. blocks
        for blk in &self.blocks {
            let norm1 = layernorm(&h, &blk.norm1_w, &blk.norm1_b, HIDDEN);
            let attn_out = block_attn(blk, &norm1, &cos, &sin, n, hd);
            for i in 0..h.len() {
                h[i] += attn_out[i];
            }
            let norm2 = layernorm(&h, &blk.norm2_w, &blk.norm2_b, HIDDEN);
            let mut fc1 = vec![0.0f32; n * INTER];
            gemm_vec(&norm2, &blk.fc1_w, &blk.fc1_b, &mut fc1, HIDDEN, INTER, n);
            for v in fc1.iter_mut() {
                *v = gelu_tanh(*v);
            }
            let mut fc2 = vec![0.0f32; n * HIDDEN];
            gemm_vec(&fc1, &blk.fc2_w, &blk.fc2_b, &mut fc2, INTER, HIDDEN, n);
            for i in 0..h.len() {
                h[i] += fc2[i];
            }
        }

        // 5. merger: layernorm -> view [N,1152] -> [N/4, 4608] -> fc1 -> gelu -> fc2 -> [N/4,5120]
        let ln = layernorm(&h, &self.merger_norm_w, &self.merger_norm_b, HIDDEN);
        let tn = n / (MERGE * MERGE);
        let mut m = vec![0.0f32; tn * 4608];
        for r in 0..tn {
            for g in 0..MERGE * MERGE {
                let src = (r * 4 + g) * HIDDEN;
                for d in 0..HIDDEN {
                    m[r * 4608 + g * HIDDEN + d] = ln[src + d];
                }
            }
        }
        let mut fc1 = vec![0.0f32; tn * 4608];
        gemm_vec(&m, &self.merger_fc1_w, &self.merger_fc1_b, &mut fc1, 4608, 4608, tn);
        for v in fc1.iter_mut() {
            *v = gelu(*v);
        }
        let out_hidden = self.out_hidden;
        let mut out = vec![0.0f32; tn * out_hidden];
        gemm_vec(&fc1, &self.merger_fc2_w, &self.merger_fc2_b, &mut out, 4608, out_hidden, tn);
        out
    }
}

/// Vision self-attention block (norm1 applied). qkv -> reshape [N,3,16,72] -> rope -> full
/// attention (N tokens, not causal) -> proj.
fn block_attn(blk: &crate::vision_tower::VisualBlock, x: &[f32], cos: &[f32], sin: &[f32], n: usize, hd: usize) -> Vec<f32> {
    let heads = crate::vision_tower::HEADS; // 16
    let mut qkv = vec![0.0f32; n * 3 * HIDDEN];
    gemm_vec(x, &blk.qkv_w, &blk.qkv_b, &mut qkv, HIDDEN, 3 * HIDDEN, n);
    let mut q = vec![0.0f32; n * heads * hd];
    let mut k = vec![0.0f32; n * heads * hd];
    let mut v = vec![0.0f32; n * heads * hd];
    for i in 0..n {
        for hh in 0..heads {
            for d in 0..hd {
                let base = i * 3 * heads * hd + hh * hd + d;
                let off = i * heads * hd + hh * hd + d;
                q[off] = qkv[base];
                k[off] = qkv[base + heads * hd];
                v[off] = qkv[base + 2 * heads * hd];
            }
        }
    }
    apply_rope_vision(&mut q, cos, sin, n, heads, hd);
    apply_rope_vision(&mut k, cos, sin, n, heads, hd);
    let scale = (hd as f32).powf(-0.5);
    let mut out = vec![0.0f32; n * heads * hd];
    for i in 0..n {
        for hh in 0..heads {
            let qi = i * heads * hd + hh * hd;
            let mut scores = vec![0.0f32; n];
            for j in 0..n {
                let kj = j * heads * hd + hh * hd;
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += q[qi + d] * k[kj + d];
                }
                scores[j] = acc * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut ssum = 0.0f32;
            for s in scores.iter_mut() {
                *s = ((*s - mx) as f64).exp() as f32;
                ssum += *s;
            }
            for j in 0..n {
                let kj = j * heads * hd + hh * hd;
                let wj = scores[j] / ssum;
                for d in 0..hd {
                    out[qi + d] += wj * v[kj + d];
                }
            }
        }
    }
    let mut proj = vec![0.0f32; n * HIDDEN];
    gemm_vec(&out, &blk.proj_w, &blk.proj_b, &mut proj, HIDDEN, HIDDEN, n);
    proj
}

fn apply_rope_vision(qk: &mut [f32], cos: &[f32], sin: &[f32], n: usize, heads: usize, hd: usize) {
    for i in 0..n {
        for hh in 0..heads {
            let off = i * heads * hd + hh * hd;
            let val: Vec<f32> = qk[off..off + hd].to_vec();
            let rh = rotate_half_hd(&val, hd);
            for d in 0..hd {
                let c = cos[i * hd + d];
                let s = sin[i * hd + d];
                qk[off + d] = val[d] * c + rh[d] * s;
            }
        }
    }
}

/// A decoded + processed vision input ready to splice: merged image embeddings + grid.
pub struct VisionOutput {
    /// Merged embeddings [num_tokens, OUT_HIDDEN] (the language embed-splice rows).
    pub merged: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Number of image tokens = grid_h*grid_w / merge^2.
    pub num_tokens: usize,
}

/// Decode a `data:` image URL (or a bare base64), preprocess it, and run the vision tower.
/// Returns the merged embeddings to splice at the language embed layer.
pub fn process_data_url(tower: &VisualTower, data_url: &str) -> anyhow::Result<VisionOutput> {
    let pre = preprocess_data_url(data_url)?;
    let merged = tower.forward_cpu(&pre.pixel_values, pre.grid_h, pre.grid_w);
    let tn = (pre.grid_h * pre.grid_w) / (crate::vision_tower::MERGE * crate::vision_tower::MERGE);
    Ok(VisionOutput { merged, grid_h: pre.grid_h, grid_w: pre.grid_w, num_tokens: tn })
}

/// Decode a `data:` image URL (or bare base64) and run the vision tower on the GPU.
/// Same decode + preprocess as `process_data_url`; only the forward is `GpuVisualTower` instead of
/// `forward_cpu`. The GPU tower must already be built (`GpuVisualTower::new`).
pub fn process_data_url_gpu(gvt: &mut crate::vision_gpu::GpuVisualTower, data_url: &str) -> anyhow::Result<VisionOutput> {
    let pre = preprocess_data_url(data_url)?;
    let (merged, _states) = gvt.forward(&pre.pixel_values, pre.grid_h, pre.grid_w, false)?;
    let tn = (pre.grid_h * pre.grid_w) / (crate::vision_tower::MERGE * crate::vision_tower::MERGE);
    Ok(VisionOutput { merged, grid_h: pre.grid_h, grid_w: pre.grid_w, num_tokens: tn })
}

/// Decode + preprocess a `data:` URL into the flatten `pixel_values` + grid (the CPU and GPU paths
/// share this; only the tower forward differs).
fn preprocess_data_url(data_url: &str) -> anyhow::Result<crate::vision_preproc::PreprocessedImage> {
    use anyhow::anyhow;
    let b64 = data_url
        .split("base64,")
        .nth(1)
        .ok_or_else(|| anyhow!("not a base64 data url: {}", &data_url[..data_url.len().min(32)]))?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim())?;
    let img = image::load_from_memory(&bytes)?.to_rgb8();
    let (w, h) = img.dimensions();
    let cfg = crate::vision_preproc::QWEN27B_PREPROC;
    Ok(crate::vision_preproc::preprocess_image(h as usize, w as usize, img.as_raw(), &cfg))
}

/// A region of the token stream occupied by one image's merged embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Token index (in the EXPANDED stream) where this image's embeddings begin.
    pub start: usize,
    pub num_tokens: usize,
}

/// Expand each `image_pad` (248056) placeholder in `tokens` into `n = grid_h*grid_w/merge^2`
/// copies — the number of merged image tokens the vision tower emits — and record the span of each.
/// This is the CPU-side prerequisite for the model-path splice: the expanded token stream has the
/// same length as the merged-embedding rows, and each span tells the prefill where to overwrite.
///
/// `image_token_count` must be in the same order as the images in the rendered prompt.
pub fn expand_image_pads(tokens: &[u32], image_token_count: &[usize]) -> (Vec<u32>, Vec<ImageSpan>) {
    const IMAGE_PAD: u32 = 248056;
    let mut out = Vec::with_capacity(tokens.len());
    let mut spans = Vec::new();
    let mut img = 0usize;
    for &t in tokens {
        if t == IMAGE_PAD && img < image_token_count.len() {
            let n = image_token_count[img];
            let start = out.len();
            for _ in 0..n {
                out.push(IMAGE_PAD);
            }
            spans.push(ImageSpan { start, num_tokens: n });
            img += 1;
        } else {
            out.push(t);
        }
    }
    (out, spans)
}

/// The server-side vision-prep result: the expanded token stream (image_pad spans expanded to the
/// merged-token count) plus the concatenated merged embeddings and each image's span. This is what
/// gets threaded to the model prefill, which overwrites the span rows with `image_embeds`.
pub struct PreparedVision {
    pub expanded_tokens: Vec<u32>,
    /// Concatenated merged embeddings, in image order: len = sum(num_tokens) * OUT_HIDDEN.
    pub image_embeds: Vec<f32>,
    pub spans: Vec<ImageSpan>,
}

/// For a request whose messages contain images, decode+preprocess+run the tower for each image (in
/// prompt order) and expand the `image_pad` span accordingly. `image_urls` = the captured
/// `ChatMessage.images[*].url` in prompt order; `prompt_tokens` = the raw (un-expanded) prompt.
pub fn prepare_vision_request(
    tower: &VisualTower,
    image_urls: &[String],
    prompt_tokens: &[u32],
) -> anyhow::Result<PreparedVision> {
    use anyhow::Result as R;
    if image_urls.is_empty() {
        return Ok(PreparedVision { expanded_tokens: prompt_tokens.to_vec(), image_embeds: vec![], spans: vec![] });
    }
    let mut embeds: Vec<f32> = Vec::new();
    let mut counts = Vec::with_capacity(image_urls.len());
    for url in image_urls {
        let o = process_data_url(tower, url)?;
        counts.push(o.num_tokens);
        embeds.extend_from_slice(&o.merged);
    }
    let (expanded_tokens, spans) = expand_image_pads(prompt_tokens, &counts);
    Ok(PreparedVision { expanded_tokens, image_embeds: embeds, spans })
}

/// GPU twin of `prepare_vision_request`: decode+preprocess on the host, run the GPU tower, expand
/// the `image_pad` span. `image_embeds` comes back as the merged f32 embeddings (the scheduler
/// uploads them to bf16 for the prefill splice, exactly as the CPU path does).
pub fn prepare_vision_request_gpu(
    gvt: &mut crate::vision_gpu::GpuVisualTower,
    image_urls: &[String],
    prompt_tokens: &[u32],
) -> anyhow::Result<PreparedVision> {
    use anyhow::Result as R;
    if image_urls.is_empty() {
        return Ok(PreparedVision { expanded_tokens: prompt_tokens.to_vec(), image_embeds: vec![], spans: vec![] });
    }
    let mut embeds: Vec<f32> = Vec::new();
    let mut counts = Vec::with_capacity(image_urls.len());
    for url in image_urls {
        let o = process_data_url_gpu(gvt, url)?;
        counts.push(o.num_tokens);
        embeds.extend_from_slice(&o.merged);
    }
    let (expanded_tokens, spans) = expand_image_pads(prompt_tokens, &counts);
    Ok(PreparedVision { expanded_tokens, image_embeds: embeds, spans })
}

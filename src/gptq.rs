//! `--gptq`: GPTQ (optionally micro-rotated, "MR-GPTQ") quantization of qwen3_5 dense and
//! qwen4_exp MoE checkpoints to the engine's NVFP4 artifact, ONE LAYER AT A TIME on one GB10.
//!
//! The memory trick: the already-quantized `--base` artifact is loaded as the model (embedding,
//! norms, PLE table on the SSD, MTP head…), and for each layer l its linear weights are swapped
//! for the bf16 originals read straight from the source shards (~2.5 GB). The calibration forward
//! is the engine's own prefill (`prefill_batch_range(lo=l, hi=l+1)`), so the activations GPTQ sees
//! are the ones the serving kernels compute; `gemm_act` / `moe_batch` taps accumulate the input
//! Hessians (per routed expert for the MoE). GPTQ then runs on the GPU (cuSOLVER Cholesky, a
//! row-parallel block sweep with NVFP4 group scales, cuBLAS propagation), the layer is swapped to
//! its quantized weights and re-run to produce the next layer's inputs (sequential GPTQ), and the
//! quantized records stream to the output shards. Peak footprint ≈ base artifact + one bf16 layer
//! + the Hessians (512 experts × 2560² f32 = 13 GB) + the calibration hidden states.
//!
//! `--rotate` applies the 16-point Hadamard micro-rotation (W' = W·R, H' = R·H·R, R = H16/4) before
//! quantizing; such an artifact needs the engine to rotate activations (`transform: hadamard16` in
//! its config.json) — see `GpuModel::rotated_ptrs`.
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use anyhow::{anyhow, Context, Result};
use half::bf16;
use cudarc::driver::DevicePtr;
use crate::gpu::{GpuModel, GptqTap, GptqHess, W, B, S, Pool, AttnIn, GdnIn, Ffn};
use crate::quant::{self, Group};

// ---------------------------------------------------------------- cuSOLVER (dense Cholesky)
#[link(name = "cusolver")]
extern "C" {
    fn cusolverDnCreate(handle: *mut *mut std::ffi::c_void) -> i32;
    fn cusolverDnDestroy(handle: *mut std::ffi::c_void) -> i32;
    fn cusolverDnSpotrf_bufferSize(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, lwork: *mut i32) -> i32;
    fn cusolverDnSpotrf(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, work: *mut f32, lwork: i32, info: *mut i32) -> i32;
    fn cusolverDnSpotri_bufferSize(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, lwork: *mut i32) -> i32;
    fn cusolverDnSpotri(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, work: *mut f32, lwork: i32, info: *mut i32) -> i32;
}
const CUBLAS_FILL_MODE_LOWER: i32 = 0;

struct Cusolver { h: *mut std::ffi::c_void, work: Option<(cudarc::driver::CudaSlice<f32>, usize)>, info: cudarc::driver::CudaSlice<i32> }
impl Cusolver {
    fn new(gpu: &GpuModel) -> Result<Self> {
        let mut h: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { cusolverDnCreate(&mut h) };
        if rc != 0 { return Err(anyhow!("cusolverDnCreate failed ({rc})")); }
        Ok(Self { h, work: None, info: gpu.gptq_dev().alloc_zeros::<i32>(1)? })
    }
    /// In place on the device buffer `a` (f32 [n, n], symmetric PD): a ← upper Cholesky factor U
    /// of a⁻¹ in ROW-MAJOR terms (a⁻¹ = UᵀU), i.e. cuSOLVER's lower factor of the lower-mode inverse.
    fn chol_inv_chol(&mut self, gpu: &GpuModel, a: &S, n: usize) -> Result<()> {
        let dev = gpu.gptq_dev().clone();
        gpu.gptq_sync();
        let mut lw1 = 0i32; let mut lw2 = 0i32;
        let ap = *a.device_ptr() as *mut f32;
        unsafe {
            cusolverDnSpotrf_bufferSize(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, &mut lw1);
            cusolverDnSpotri_bufferSize(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, &mut lw2);
        }
        let lw = lw1.max(lw2).max(1) as usize;
        if self.work.as_ref().map_or(true, |(_, cap)| *cap < lw) { self.work = Some((dev.alloc_zeros::<f32>(lw)?, lw)); }
        let wp = *self.work.as_ref().unwrap().0.device_ptr() as *mut f32;
        let ip = *self.info.device_ptr() as *mut i32;
        let check = |tag: &str, rc: i32, info: &cudarc::driver::CudaSlice<i32>| -> Result<()> {
            if rc != 0 { return Err(anyhow!("cusolver {tag} returned {rc}")); }
            dev.synchronize()?;
            let v = dev.dtoh_sync_copy(info)?;
            if v[0] != 0 { return Err(anyhow!("cusolver {tag}: info = {} (not positive definite?)", v[0])); }
            Ok(())
        };
        unsafe {
            let rc = cusolverDnSpotrf(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, wp, lw as i32, ip); check("potrf", rc, &self.info)?;
            let rc = cusolverDnSpotri(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, wp, lw as i32, ip); check("potri", rc, &self.info)?;
            let rc = cusolverDnSpotrf(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, wp, lw as i32, ip); check("potrf(inv)", rc, &self.info)?;
        }
        Ok(())
    }
}
impl Drop for Cusolver { fn drop(&mut self) { unsafe { cusolverDnDestroy(self.h); } } }

// ---------------------------------------------------------------- safetensors range reader
#[derive(Clone)]
pub struct TensorMeta { pub dtype: String, pub shape: Vec<usize>, pub off: (u64, u64), pub file: PathBuf, pub data_start: u64 }

pub struct ShardReader { pub metas: BTreeMap<String, TensorMeta> }
impl ShardReader {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        let idx = dir.join("model.safetensors.index.json");
        if idx.exists() {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx)?)?;
            let mut set = std::collections::BTreeSet::new();
            for (_, v) in j["weight_map"].as_object().unwrap() { set.insert(v.as_str().unwrap().to_string()); }
            files.extend(set.into_iter().map(|f| dir.join(f)));
        } else { files.push(dir.join("model.safetensors")); }
        let mut metas = BTreeMap::new();
        for f in &files {
            let mut fh = std::fs::File::open(f).with_context(|| format!("open {}", f.display()))?;
            let mut lenb = [0u8; 8]; fh.read_exact(&mut lenb)?;
            let hlen = u64::from_le_bytes(lenb);
            let mut hb = vec![0u8; hlen as usize]; fh.read_exact(&mut hb)?;
            let hj: serde_json::Value = serde_json::from_slice(&hb)?;
            for (name, v) in hj.as_object().unwrap() {
                if name == "__metadata__" { continue; }
                let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                let o = v["data_offsets"].as_array().unwrap();
                metas.insert(name.clone(), TensorMeta { dtype: v["dtype"].as_str().unwrap().to_string(), shape,
                    off: (o[0].as_u64().unwrap(), o[1].as_u64().unwrap()), file: f.clone(), data_start: 8 + hlen });
            }
        }
        Ok(Self { metas })
    }
    pub fn read_bytes(&self, name: &str) -> Result<(TensorMeta, Vec<u8>)> {
        let m = self.metas.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))?.clone();
        let mut fh = std::fs::File::open(&m.file)?;
        fh.seek(SeekFrom::Start(m.data_start + m.off.0))?;
        let mut buf = vec![0u8; (m.off.1 - m.off.0) as usize];
        fh.read_exact(&mut buf)?;
        Ok((m, buf))
    }
    pub fn read_bf16(&self, name: &str) -> Result<(Vec<usize>, Vec<bf16>)> {
        let (m, b) = self.read_bytes(name)?;
        anyhow::ensure!(m.dtype == "BF16", "{name}: expected BF16, got {}", m.dtype);
        Ok((m.shape, bytemuck::cast_slice::<u8, bf16>(&b).to_vec()))
    }
}

// ---------------------------------------------------------------- output artifact writer
struct Out { name: String, dtype: safetensors::Dtype, shape: Vec<usize>, data: Vec<u8> }
struct Writer { dir: PathBuf, outs: Vec<Out>, bytes: usize, shard_idx: usize, weight_map: serde_json::Map<String, serde_json::Value>, total: u64, shard_bytes: usize }
impl Writer {
    fn new(dir: &Path, shard_bytes: usize) -> Self { Self { dir: dir.to_path_buf(), outs: Vec::new(), bytes: 0, shard_idx: 0, weight_map: Default::default(), total: 0, shard_bytes } }
    // A shard boundary may only fall BETWEEN tensor families: the loader pairs an NVFP4 triple
    // (weight_packed / weight_scale / weight_global_scale) within one shard.
    fn push(&mut self, o: Out) {
        // Verbatim copies of a packed family arrive in name order (weight_global_scale, weight_packed,
        // weight_scale): hold the shard boundary until the family's last member.
        // (input_global_scale sorts first: ".input_global_scale" < ".weight_*" — it holds too)
        let hold = o.name.ends_with(".weight_global_scale") || o.name.ends_with(".weight_packed") || o.name.ends_with(".input_global_scale");
        self.push_raw(o);
        if !hold && self.bytes >= self.shard_bytes { self.flush(); }
    }
    fn push_raw(&mut self, o: Out) { self.bytes += o.data.len(); self.total += o.data.len() as u64; self.outs.push(o); }
    fn push_fp8(&mut self, stem: &str, q: quant::Fp8Tensor) {
        let sc: Vec<u8> = q.row_scale.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.push_raw(Out { name: format!("{stem}.weight"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![q.m, q.k], data: q.qweight });
        self.push_raw(Out { name: format!("{stem}.weight_scale"), dtype: safetensors::Dtype::F32, shape: vec![q.m], data: sc });
        if self.bytes >= self.shard_bytes { self.flush(); }
    }
    fn push_nvfp4(&mut self, stem: &str, qw: Vec<u8>, sc: Vec<u8>, m: usize, k: usize, gs: f32, igs: Option<f32>) {
        self.push_raw(Out { name: format!("{stem}.weight_packed"), dtype: safetensors::Dtype::U8, shape: vec![m, k / 2], data: qw });
        self.push_raw(Out { name: format!("{stem}.weight_scale"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![m, k / 16], data: sc });
        self.push_raw(Out { name: format!("{stem}.weight_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: gs.to_le_bytes().to_vec() });
        if let Some(g) = igs {
            self.push_raw(Out { name: format!("{stem}.input_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: g.to_le_bytes().to_vec() });
        }
        if self.bytes >= self.shard_bytes { self.flush(); }
    }
    fn flush(&mut self) {
        if self.outs.is_empty() { return; }
        self.shard_idx += 1;
        let fname = format!("model-{:05}.safetensors", self.shard_idx);
        let views: Vec<(String, safetensors::tensor::TensorView)> = self.outs.iter()
            .map(|o| (o.name.clone(), safetensors::tensor::TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        let meta: std::collections::HashMap<String, String> = [("format".to_string(), "pt".to_string())].into_iter().collect();
        safetensors::serialize_to_file(views, Some(meta), &self.dir.join(&fname)).expect("write shard");
        for o in &self.outs { self.weight_map.insert(o.name.clone(), serde_json::Value::String(fname.clone())); }
        println!("  wrote {fname} ({:.2} GB, {} tensors)", self.bytes as f64 / 1e9, self.outs.len());
        self.outs.clear(); self.bytes = 0;
    }
    fn finish(mut self) -> Result<()> {
        self.flush();
        // The loader pairs a packed family within ONE shard: refuse an index that splits one
        // (a served artifact would otherwise die at its first start with "tensor … not found").
        let mut split = Vec::new();
        for (k, v) in &self.weight_map {
            if let Some(stem) = k.strip_suffix(".weight_packed") {
                for suf in [".weight_scale", ".weight_global_scale", ".input_global_scale"] {
                    if let Some(sv) = self.weight_map.get(&format!("{stem}{suf}")) { if sv != v { split.push(format!("{stem}{suf}")); } }
                    else if suf != ".input_global_scale" { split.push(format!("{stem}{suf} (missing)")); }
                }
            }
            if let Some(stem) = k.strip_suffix(".weight_scale") {
                if self.weight_map.contains_key(&format!("{stem}.weight")) && self.weight_map.get(&format!("{stem}.weight")) != Some(v) { split.push(format!("{stem}.weight (fp8)")); }
            }
        }
        anyhow::ensure!(split.is_empty(), "artifact writer: {} tensor families split across shards: {:?}", split.len(), &split[..split.len().min(4)]);
        let index = serde_json::json!({ "metadata": { "total_size": self.total }, "weight_map": self.weight_map });
        std::fs::write(self.dir.join("model.safetensors.index.json"), serde_json::to_string_pretty(&index)?)?;
        println!("[gptq] index written: {} tensors in {} shards, every packed family within one shard", self.weight_map.len(), self.shard_idx);
        Ok(())
    }
}

// ---------------------------------------------------------------- options
#[derive(Clone)]
pub struct GptqOpts {
    pub nsamples: usize, pub seqlen: usize, pub damp: f32, pub nclip: usize, pub rotate: bool,
    pub scale_iters: usize, pub static_act_order: bool,
    pub gptq_groups: Vec<Group>, pub nvfp4_groups: Vec<Group>,   // GPTQ'd / RTN'd; everything else bf16
    pub fp8_groups: Vec<Group>,                                     // row-scaled FP8 (E4M3): the speed/accuracy middle ground
}

/// `igs`: the W4A4 input global scale (6·448 / calibration activation amax), written as
/// `{stem}.input_global_scale` — None for tensors the calibration never fed (RTN groups).
struct Rec { qw: Vec<u8>, sc: Vec<u8>, m: usize, k: usize, gs: f32, igs: Option<f32> }
/// Token subsample kept per layer for the down-projection fallback Hessians.
const MOE_SUB_TOKENS: usize = 16384;
fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok())).map(|kb| kb / 1048576.0).unwrap_or(0.0)
}

fn e4m3_scale_of(amax: f32) -> f32 { if amax > 0.0 { amax / (quant::E2M1_MAX * quant::E4M3_MAX) } else { 1.0 } }
pub(crate) fn static_activation_order(diag: &[f32]) -> Vec<i32> {
    let mut order: Vec<usize> = (0..diag.len()).collect();
    order.sort_by(|&a, &b| match (diag[a].is_finite(), diag[b].is_finite()) {
        (true, true) => diag[b].total_cmp(&diag[a]).then_with(|| a.cmp(&b)),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.cmp(&b),
    });
    order.into_iter().map(|i| i as i32).collect()
}

#[derive(Debug, Clone, Copy)]
struct ScaleFit {
    scale: f32,
    initial_mse: f64,
    final_mse: f64,
    accepted: usize,
}

/// Alternate between NVFP4 local-scale/code assignment and the closed-form least-squares global
/// tensor scale. A guarded line search makes this monotonic even across E4M3 assignment boundaries.
fn alternating_scale_fit<F>(initial: f32, iterations: usize, mut stats: F) -> ScaleFit
where F: FnMut(f32) -> (f64, f64, f64) {
    if iterations == 0 || !initial.is_finite() || initial <= 0.0 {
        return ScaleFit { scale: initial, initial_mse: f64::NAN, final_mse: f64::NAN, accepted: 0 };
    }
    let (_, _, initial_mse) = stats(initial);
    if !initial_mse.is_finite() {
        return ScaleFit { scale: initial, initial_mse, final_mse: initial_mse, accepted: 0 };
    }
    let mut scale = initial;
    let mut mse = initial_mse;
    let lo = initial * 0.25;
    let hi = initial * 4.0;
    let mut accepted = 0usize;
    for _ in 0..iterations {
        let (num, den, _) = stats(scale);
        if !num.is_finite() || !den.is_finite() || den <= 0.0 { break; }
        let mut candidate = (num / den) as f32;
        if !candidate.is_finite() || candidate <= 0.0 { break; }
        candidate = candidate.clamp(lo, hi);
        if (candidate - scale).abs() <= scale.abs().max(f32::MIN_POSITIVE) * 1e-6 { break; }

        let mut trial = candidate;
        let mut trial_mse = stats(trial).2;
        let mut found = trial_mse.is_finite() && trial_mse < mse;
        for _ in 0..8 {
            if found { break; }
            trial = 0.5 * (scale + trial);
            trial_mse = stats(trial).2;
            found = trial_mse.is_finite() && trial_mse < mse;
        }
        if !found { break; }
        scale = trial;
        mse = trial_mse;
        accepted += 1;
    }
    ScaleFit { scale, initial_mse, final_mse: mse, accepted }
}

fn optimize_w32_scale(gpu: &GpuModel, w32: &S, n: usize, initial: f32, opts: &GptqOpts) -> f32 {
    let fit = alternating_scale_fit(initial, opts.scale_iters,
        |s| gpu.gptq_scale_stats_f32(w32, n, s, opts.nclip));
    if fit.accepted > 0 {
        println!("[gptq] NVFP4 scale: {:.6e} -> {:.6e}, SSE {:.6e} -> {:.6e} ({} steps)",
                 initial, fit.scale, fit.initial_mse, fit.final_mse, fit.accepted);
    }
    fit.scale
}

fn optimize_bf16_scale(gpu: &GpuModel, w_ptr: u64, n: usize, initial: f32, opts: &GptqOpts) -> f32 {
    let fit = alternating_scale_fit(initial, opts.scale_iters,
        |s| gpu.gptq_scale_stats_bf16(w_ptr, n, opts.rotate, s, opts.nclip));
    if fit.accepted > 0 {
        println!("[gptq] stacked NVFP4 scale: {:.6e} -> {:.6e}, SSE {:.6e} -> {:.6e} ({} steps)",
                 initial, fit.scale, fit.initial_mse, fit.final_mse, fit.accepted);
    }
    fit.scale
}

/// GPTQ one 2-D weight (bf16 on device at `w_ptr`, [m, k] row-major) with its Hessian.
fn igs_of(amax: f32) -> Option<f32> { if amax > 0.0 && amax.is_finite() { Some(6.0 * 448.0 / amax) } else { None } }
fn gptq_2d(gpu: &GpuModel, cs: &mut Cusolver, w_ptr: u64, m: usize, k: usize, hess: &S, opts: &GptqOpts, s_tensor: Option<f32>, x_amax: f32) -> Result<Rec> {
    let w32 = gpu.gptq_w32(w_ptr, m, k, opts.rotate);
    let initial_st = s_tensor.unwrap_or_else(|| e4m3_scale_of(gpu.gptq_absmax_f32(&w32, m * k)));
    // Stacked MoE tensors pass their already jointly optimized scale; ordinary matrices optimize
    // their own global scale from the original rotated f32 weight.
    let st = if s_tensor.is_some() { initial_st } else { optimize_w32_scale(gpu, &w32, m * k, initial_st, opts) };

    // Adaptive damping (GPTQModel-style): 1 % → 5 % → 10 % of mean(diag) before falling back to
    // RTN (U = I: the sweep then rounds without error feedback, same scales and static groups).
    let mut u: Option<S> = None;
    let mut perm = None;
    for damp in [opts.damp, 0.05, 0.10] {
        let (h32, p) = gpu.gptq_h32(hess, k, opts.rotate, damp, opts.static_act_order);
        perm = p;
        match cs.chol_inv_chol(gpu, &h32, k) {
            Ok(()) => { u = Some(h32); break; }
            Err(e) => eprintln!("[gptq] cholesky failed at damp {damp}: {e} — retrying"),
        }
    }
    let u = match u { Some(u) => u, None => {
        eprintln!("[gptq] Hessian not usable — RTN fallback for this tensor");
        let mut id = vec![0f32; k * k]; for i in 0..k { id[i * k + i] = 1.0; }
        gpu.gptq_dev().htod_sync_copy(&id)?
    } };
    let (qw, sc) = if let Some(perm) = perm {
        // Freeze block scales before error feedback, then work in importance order and write codes
        // directly into their original columns. The artifact and serving path remain unchanged.
        let qs = gpu.gptq_static_scales(&w32, m, k, st, opts.nclip);
        let wp = gpu.gptq_permute_w(&w32, m, k, &perm);
        gpu.gptq_sweep_static(&wp, &u, &perm, &qs, m, k, st)
    } else {
        gpu.gptq_sweep(&w32, &u, m, k, st, opts.nclip)
    };
    Ok(Rec { qw, sc, m, k, gs: 1.0 / st, igs: igs_of(x_amax) })
}

/// Plain RTN through the quantizer's own codec (weights the calibration never touches: the MTP head).
fn rtn_2d(w: &[bf16], m: usize, k: usize) -> Rec {
    let q = quant::quantize_nvfp4(w, m, k);
    Rec { qw: q.qweight, sc: q.scales, m, k, gs: q.global_scale, igs: None }
}

// ---------------------------------------------------------------- calibration data
pub fn calib_tokens(model_dir: &Path, calib: &Path, nsamples: usize, seqlen: usize, vocab: usize) -> Result<Vec<Vec<u32>>> {
    if calib.to_string_lossy() == "random" {
        // Smoke-test / synthetic-model mode: seeded random token ids.
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        return Ok((0..nsamples).map(|_| (0..seqlen).map(|_| (next() % (vocab as u64).max(4)) as u32 + 4).map(|t| t.min(vocab as u32 - 1)).collect()).collect());
    }
    let tok = crate::tokenizer::QwenTokenizer::from_file(&model_dir.join("tokenizer.json").to_string_lossy())?;
    let raw = std::fs::read_to_string(calib).with_context(|| format!("read {}", calib.display()))?;
    // jsonl with a "text" field, or plain text (blank-line separated documents)
    // jsonl lines: {"text": …} raw documents, or {"messages": [{role, content}, …]} rendered
    // through the model's own chat template (calibration in the served format).
    let docs: Vec<String> = if calib.extension().map_or(false, |e| e == "jsonl") {
        raw.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            if let Some(t) = v["text"].as_str() { return Some(t.to_string()); }
            let msgs: Vec<crate::tokenizer::ChatMessage> = v["messages"].as_array()?.iter().map(|m| crate::tokenizer::ChatMessage {
                role: m["role"].as_str().unwrap_or("user").to_string(), content: m["content"].as_str().map(|s| s.to_string()),
                tool_calls: None, tool_call_id: None, name: None, reasoning_content: None, images: vec![] }).collect();
            tok.apply_chat_template_no_gen(&msgs, None, None).ok()
        }).filter(|s| !s.is_empty()).collect()
    } else { raw.split("\n\n").map(|s| s.to_string()).filter(|s| !s.trim().is_empty()).collect() };
    let mut samples = Vec::new();
    let mut buf: Vec<u32> = Vec::new();
    for d in docs {
        let ids = tok.encode(&d, false)?;
        buf.extend(ids);
        buf.push(tok.encode("\n\n", false)?.first().copied().unwrap_or(198));
        while buf.len() >= seqlen && samples.len() < nsamples {
            samples.push(buf.drain(..seqlen).collect());
        }
        if samples.len() >= nsamples { break; }
    }
    anyhow::ensure!(samples.len() >= nsamples.min(1), "calibration text too short for one sample of {seqlen} tokens");
    if samples.len() < nsamples { println!("[gptq] calibration text gave {} samples (asked {nsamples})", samples.len()); }
    Ok(samples)
}

fn prepare_output_dir(out: &Path, inputs: &[&Path]) -> Result<()> {
    let input_paths: Vec<PathBuf> = inputs.iter().map(|p| {
        let resolved = std::fs::canonicalize(p).with_context(|| format!("resolve input {}", p.display()))?;
        Ok(resolved)
    }).collect::<Result<_>>()?;
    if out.exists() {
        let out_path = std::fs::canonicalize(out).with_context(|| format!("resolve output {}", out.display()))?;
        anyhow::ensure!(!input_paths.iter().any(|p| p == &out_path),
            "output directory {} must differ from every input directory", out.display());
        anyhow::ensure!(std::fs::read_dir(out)?.next().is_none(),
            "output directory {} already exists and is not empty", out.display());
    } else {
        std::fs::create_dir_all(out)?;
    }
    Ok(())
}

// ---------------------------------------------------------------- the driver
pub fn run(source: &Path, base: &Path, out: &Path, calib: &Path, opts: GptqOpts) -> Result<()> {
    prepare_output_dir(out, &[source, base])?;
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
    let t_all = std::time::Instant::now();
    let src = ShardReader::open(source)?;
    let basr = ShardReader::open(base)?;
    let (mut gpu, cfg) = GpuModel::load_from_dir(&base.to_string_lossy())?;
    gpu.gptq_reset_rotation();
    anyhow::ensure!(matches!(cfg.family, crate::qwen::Family::Qwen35 | crate::qwen::Family::Qwen4Exp),
        "--gptq is implemented for qwen3_5 dense and qwen4_exp, got {:?}", cfg.family);
    let samples = calib_tokens(base, calib, opts.nsamples, opts.seqlen, cfg.vocab_size)?;
    let ns = samples.len();
    println!("[gptq] {} samples × {} tokens; groups GPTQ {:?}, RTN {:?}, rotate {}, damp {}, clip ratios {}, scale iters {}, act-order {}",
             ns, opts.seqlen, opts.gptq_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
             opts.nvfp4_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(), opts.rotate, opts.damp, opts.nclip,
             opts.scale_iters, if opts.static_act_order { "static" } else { "none" });
    for li in 0..gpu.gptq_num_layers() { gpu.gptq_drop_layer_weights(li); }
    gpu.gptq_sync();
    println!("[gptq] base layer weights dropped (rebuilt per layer from the source); MemAvailable {:.1} GB", mem_available_gb());
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let seqlen = opts.seqlen;
    let mut state = gpu.new_batch_state(1, 1, seqlen);
    let mut cs = Cusolver::new(&gpu)?;
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    let (h, ne, mi) = (cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size);
    let is_gptq = |name: &str| opts.gptq_groups.contains(&quant::group_of(name));
    let is_rtn = |name: &str| opts.nvfp4_groups.contains(&quant::group_of(name));
    let is_fp8 = |name: &str| opts.fp8_groups.contains(&quant::group_of(name));
    let lm = "model.language_model";
    // The non-layer tensors the artifact takes from the SOURCE (embed, lm_head, final mixer) are
    // served during the calibration exactly as they will be served — source bf16, or the
    // quantizer's own RTN / FP8 when their group asks for it — not as the base's RTN copies.
    {
        let mut mk = |name: &str| -> Result<Option<W>> {
            let Some(meta) = src.metas.get(name) else { return Ok(None) };
            if meta.dtype != "BF16" || meta.shape.len() != 2 { return Ok(None); }
            let (shape, v) = src.read_bf16(name)?;
            let (m, k) = (shape[0], shape[1]);
            let q = m % 16 == 0 && k % 16 == 0;
            Ok(Some(if q && is_rtn(name) { let r = rtn_2d(&v, m, k); gpu.gptq_w_nvfp4(&r.qw, &r.sc, m, k, r.gs) }
                    else if q && is_fp8(name) { gpu.gptq_w_fp8(quant::quantize_fp8(&v, m, k)) }
                    else { gpu.gptq_w_bf16(&v) }))
        };
        let embed = mk(&format!("{lm}.embed_tokens.weight"))?;
        let head = mk("lm_head.weight")?;
        let mixer = match (mk(&format!("{lm}.hyper_connection_mixer.input_mix_weight_down.weight"))?,
                           mk(&format!("{lm}.hyper_connection_mixer.input_mix_weight_up.weight"))?) {
            (Some(a), Some(b)) => Some((a, b)), _ => None };
        let n = embed.is_some() as usize + head.is_some() as usize + mixer.is_some() as usize * 2;
        gpu.gptq_install_nonlayer(embed, head, mixer);
        println!("[gptq] {n} non-layer tensors (embed / lm_head / final mixer) reinstalled from the source as the artifact will serve them");
    }
    // Per-sample residual streams between layers (None before layer 0: the prefill embeds).
    let mut hidden: Vec<Option<B>> = (0..ns).map(|_| None).collect();
    let nl = cfg.num_layers;
    for li in 0..nl {
        let t_l = std::time::Instant::now();
        let lp = format!("{lm}.layers.{li}");
        // 1. the layer's linears in bf16, swapped in; taps registered for the GPTQ groups
        let mut tap = GptqTap::default();
        // name -> (the quantizer's bf16 copy, m, k, tap key = device pointer of the LAYER's copy)
        let mut bf: HashMap<String, (B, usize, usize, u64)> = HashMap::new();
        let mut up = |gpu: &GpuModel, name: &str, tap: &mut GptqTap, bf: &mut HashMap<String, (B, usize, usize, u64)>| -> Result<W> {
            let (shape, v) = src.read_bf16(name)?;
            let (m, k) = if shape.len() == 3 { (shape[0] * shape[1], shape[2]) } else { (shape[0], shape[1]) };
            let b = gpu.gptq_upload_bf16(&v);
            // The LAYER owns `b`: `gemm_act` taps by the device pointer it is called with, so the Hessian
            // must be keyed by the pointer the layer sees (keying the quantizer's copy left every dense
            // Hessian empty -> RTN). The quantizer keeps its own copy.
            let key = *b.device_ptr() as u64;
            if is_gptq(name) && !name.contains(".mlp.experts.") { tap.by_ptr.insert(key, gpu.gptq_hess_new(k)); }
            let mine = gpu.gptq_clone_b(&b);
            bf.insert(name.to_string(), (mine, m, k, key));
            Ok(W::Bf16(b))
        };
        let is_attn = matches!(cfg.layer_types[li], crate::qwen::LayerType::FullAttention);
        let mut names: Vec<String> = Vec::new();
        {
            let layer = gpu.gptq_layer(li);
            if is_attn {
                for t in ["q_proj", "k_proj", "v_proj", "o_proj"] { names.push(format!("{lp}.self_attn.{t}.weight")); }
                if layer.fa.as_ref().unwrap().indexer.is_some() { names.push(format!("{lp}.self_attn.indexer.index_qk_proj.weight")); }
            } else {
                for t in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] { names.push(format!("{lp}.linear_attn.{t}.weight")); }
            }
            match &layer.mlp {
                Ffn::Moe(_) => {
                    names.push(format!("{lp}.mlp.experts.gate_up_proj"));
                    names.push(format!("{lp}.mlp.experts.down_proj"));
                    for t in ["gate_proj", "up_proj", "down_proj"] { names.push(format!("{lp}.mlp.shared_expert.{t}.weight")); }
                    names.push(format!("{lp}.mlp.gate.weight"));
                }
                Ffn::Dense(_) => {
                    for t in ["gate_proj", "up_proj", "down_proj"] { names.push(format!("{lp}.mlp.{t}.weight")); }
                }
            }
            if layer.hc.is_some() {
                for hcn in ["attn_hyper_connection", "mlp_hyper_connection"] {
                    for t in ["input_mix_weight_down", "input_mix_weight_up"] { names.push(format!("{lp}.{hcn}.{t}.weight")); }
                }
            }
            if layer.ple.is_some() { for t in ["key_proj", "value_proj"] { names.push(format!("{lp}.ple.{t}.weight")); } }
        }
        let mut ws: HashMap<String, W> = HashMap::new();
        for n in &names { let w = up(&gpu, n, &mut tap, &mut bf)?; ws.insert(n.clone(), w); }
        let gptq_experts = cfg.is_moe && is_gptq(&format!("{lp}.mlp.experts.gate_up_proj"));
        if gptq_experts {
            tap.moe_gu = (0..ne).map(|_| gpu.gptq_hess_new(h)).collect();
            tap.moe_dn = (0..ne).map(|_| gpu.gptq_hess_new(mi)).collect();
            tap.moe_all = Some(gpu.gptq_hess_new(h));
            tap.moe_all_cap = MOE_SUB_TOKENS;
            tap.moe_all_x = Some(gpu.gptq_dev().alloc_zeros::<half::bf16>(h * MOE_SUB_TOKENS)?);
        }
        install_layer(&mut gpu, li, is_attn, &lp, &mut ws);
        // 2. pass 1: Hessians (the layer output is discarded: the bf16 experts kernel is skipped)
        tap.skip_experts = true;
        gpu.gptq_arm(tap);
        for s in 0..ns {
            gpu.zero_slot_state(&mut state, 0, seqlen);
            let inc = hidden[s].as_ref().map(|b| gpu.gptq_clone_b(b));
            let (_, outb) = gpu.prefill_batch_range(&mut pool, &samples[s], &mut state, 0, seqlen, 0, li, li + 1, inc);
            drop(outb);
        }
        let tap = gpu.gptq_disarm().unwrap();
        let t_fwd = t_l.elapsed().as_secs_f32();
        if !tap.moe_gu.is_empty() {
            // Calibration coverage of the routed experts: how many tokens each expert's Hessian saw.
            let mut ns: Vec<usize> = tap.moe_gu.iter().map(|h| h.n).collect(); ns.sort_unstable();
            let under = ns.iter().filter(|&&n| n < 256).count();
            println!("[gptq] layer {li} expert coverage: tokens/expert min {} median {} max {}; {} of {} experts under 256 tokens",
                     ns[0], ns[ns.len() / 2], ns[ns.len() - 1], under, ns.len());
        }
        // 3. quantize
        let mut recs: HashMap<String, Rec> = HashMap::new();
        for n in &names {
            let (b, m, k, key) = bf.get(n).unwrap();
            if n.ends_with("experts.gate_up_proj") || n.ends_with("experts.down_proj") {
                if !gptq_experts { if is_rtn(n) { let (_, v) = src.read_bf16(n)?; recs.insert(n.clone(), rtn_2d(&v, *m, *k)); } continue; }
                let is_gu = n.ends_with("gate_up_proj");
                let (me, ke) = if is_gu { (2 * mi, h) } else { (h, mi) };
                let hs = if is_gu { &tap.moe_gu } else { &tap.moe_dn };
                let base_ptr = *b.device_ptr() as u64;
                // Under-calibrated experts (fewer than 2·K routed tokens: a rank-deficient Hessian)
                // fall back to the layer-wide statistics: the all-token Hessian for gate_up, and
                // for down a Hessian built by running the expert on the token subsample.
                let thr = 2 * ke;
                let gu_b: Option<&B> = bf.get(&format!("{lp}.mlp.experts.gate_up_proj")).map(|(b, _, _, _)| b);
                let mut n_fallback = 0usize;
                // one global scale per stacked tensor (the artifact's convention): amax over all (rotated) experts
                let mut amax = 0f32;
                for e in 0..ne { let w32 = gpu.gptq_w32(base_ptr + (e * me * ke * 2) as u64, me, ke, opts.rotate); amax = amax.max(gpu.gptq_absmax_f32(&w32, me * ke)); }
                let initial_st = e4m3_scale_of(amax);
                let st = optimize_bf16_scale(&gpu, base_ptr, ne * me * ke, initial_st, &opts);
                let mut qw = Vec::with_capacity(ne * me * ke / 2); let mut sc = Vec::with_capacity(ne * me * ke / 16);
                // one input global scale per stacked tensor: the activation amax over all experts
                let x_amax = (0..ne).map(|e| gpu.gptq_amax(&hs[e])).fold(0f32, f32::max);
                for e in 0..ne {
                    let fallback: Option<GptqHess> = if hs[e].n < thr {
                        n_fallback += 1;
                        if is_gu { None } else {
                            let xs = tap.moe_all_x.as_ref().unwrap();
                            Some(gpu.gptq_down_hess_from(gu_b.unwrap(), e, xs, tap.moe_all_n, h, mi))
                        }
                    } else { None };
                    let hess: &S = if hs[e].n < thr && is_gu { &tap.moe_all.as_ref().unwrap().h } else { fallback.as_ref().map(|f| &f.h).unwrap_or(&hs[e].h) };
                    let r = gptq_2d(&gpu, &mut cs, base_ptr + (e * me * ke * 2) as u64, me, ke, hess, &opts, Some(st), x_amax)?;
                    qw.extend_from_slice(&r.qw); sc.extend_from_slice(&r.sc);
                }
                if n_fallback > 0 { println!("[gptq] layer {li} {}: {n_fallback} experts under {thr} tokens used the all-token fallback", if is_gu { "gate_up" } else { "down" }); }
                recs.insert(n.clone(), Rec { qw, sc, m: ne * me, k: ke, gs: 1.0 / st, igs: igs_of(x_amax) });
            } else if is_gptq(n) {
                let mut acc = tap.by_ptr.get(key).ok_or_else(|| anyhow!("no Hessian for {n} (the calibration never reached this GEMM)"))?;
                if *k % 16 != 0 || *m % 16 != 0 { continue; }   // e.g. in_proj_b/a [nh, h] — kept bf16 by the quantizer too
                if acc.n == 0 && n.ends_with(".self_attn.indexer.index_qk_proj.weight") {
                    // QSA is off below `qsa_limit` visible tokens, so the indexer never runs at calibration
                    // seqlen; its input is the same normalized hidden as q/k/v -> reuse q_proj's Hessian.
                    if let Some((_, _, _, qkey)) = bf.get(&format!("{lp}.self_attn.q_proj.weight")) {
                        if let Some(a) = tap.by_ptr.get(qkey) { if a.k == acc.k && a.n > 0 { println!("[gptq] {n}: indexer never ran (QSA off at this seqlen) — using q_proj's Hessian ({} tokens)", a.n); acc = a; } }
                    }
                }
                anyhow::ensure!(acc.n > 0, "{n}: empty Hessian — no calibration token reached this GEMM (tap pointer mismatch?)");
                if acc.n < 2 * *k { println!("[gptq] warning: {n} Hessian over only {} tokens (K = {k})", acc.n); }
                let hess = &acc.h;
                let x_amax = gpu.gptq_amax(acc);
                recs.insert(n.clone(), gptq_2d(&gpu, &mut cs, *b.device_ptr() as u64, *m, *k, hess, &opts, None, x_amax)?);
            } else if is_rtn(n) && *k % 16 == 0 && *m % 16 == 0 {
                let (_, v) = src.read_bf16(n)?; recs.insert(n.clone(), rtn_2d(&v, *m, *k));
            }
        }
        drop(tap);
        let t_q = t_l.elapsed().as_secs_f32() - t_fwd;
        // 4. swap the quantized weights in (sequential GPTQ: the next layer calibrates on them)
        let mut ws2: HashMap<String, W> = HashMap::new();
        for n in &names {
            let w = match recs.get(n) {
                // stacked experts use the same MMA-repacked layout as the loader's (W::Nvfp4, gs per 16-row tile)
                Some(r) => gpu.gptq_w_nvfp4(&r.qw, &r.sc, r.m, r.k, r.gs),
                None if is_fp8(n) => {
                    let (_, m, k, _) = bf.remove(n).unwrap();
                    if m % 16 == 0 && k % 16 == 0 { let (_, v) = src.read_bf16(n)?; gpu.gptq_w_fp8(quant::quantize_fp8(&v, m, k)) }
                    else { let (_, v) = src.read_bf16(n)?; gpu.gptq_w_bf16(&v) }
                }
                None => { let (b, _, _, _) = bf.remove(n).unwrap(); W::Bf16(b) }
            };
            // only GPTQ'd tensors were quantized in the rotated basis (RTN groups are not, and are not in
            // `transform.groups`) — marking them would rotate their input at calibration but not at serving
            if opts.rotate && recs.contains_key(n) && is_gptq(n) { gpu.gptq_mark_rotated(&w); }
            ws2.insert(n.clone(), w);
        }
        drop(bf);
        install_layer(&mut gpu, li, is_attn, &lp, &mut ws2);
        // 5. pass 2: the quantized layer's outputs become the next layer's inputs
        for s in 0..ns {
            gpu.zero_slot_state(&mut state, 0, seqlen);
            let inc = hidden[s].take();
            let (_, outb) = gpu.prefill_batch_range(&mut pool, &samples[s], &mut state, 0, seqlen, 0, li, li + 1, inc);
            hidden[s] = Some(outb);
        }
        // 6. stream the layer's tensors out: GPTQ/RTN records, everything else verbatim from the source
        let mut n_q = 0;
        for (name, meta) in src.metas.range(format!("{lp}.")..).take_while(|(k, _)| k.starts_with(&format!("{lp}."))) {
            if name.contains(".ngram_embedding.shard_") { continue; }   // the PLE table comes from the base artifact
            let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
            if let Some(r) = recs.remove(name) { writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); n_q += 1; continue; }
            if is_fp8(name) && meta.dtype == "BF16" && meta.shape.len() == 2 && meta.shape[0] % 16 == 0 && meta.shape[1] % 16 == 0 {
                let (shape, v) = src.read_bf16(name)?; writer.push_fp8(&stem, quant::quantize_fp8(&v, shape[0], shape[1])); n_q += 1; continue;
            }
            let (_, data) = src.read_bytes(name)?;
            let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "I64" => safetensors::Dtype::I64, "F16" => safetensors::Dtype::F16, "U8" => safetensors::Dtype::U8, o => return Err(anyhow!("dtype {o} on {name}")) };
            writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
        }
        println!("[gptq] layer {li}/{nl} ({}): forward {:.1}s, quantize {:.1}s, {n_q} tensors quantized, total {:.1}s",
                 if is_attn { "attn" } else { "gdn" }, t_fwd, t_q, t_l.elapsed().as_secs_f32());
    }
    // 6b. the LM head (group `lmhead` in --gptq-groups): its input is the final mixer / norm of the
    //     residual streams the last layer left in `hidden` — Hessian over every calibration token.
    let mut lm_rec: Option<Rec> = None;
    if is_gptq("lm_head.weight") {
        if let Some((ptr, rows)) = gpu.gptq_lm_head_bf16() {
            let t0 = std::time::Instant::now();
            let hess = gpu.gptq_hess_new(h);
            for s in 0..ns { gpu.gptq_lm_head_hess_accum(&mut pool, hidden[s].as_ref().unwrap(), seqlen, &hess); }
            let x_amax = gpu.gptq_amax(&hess);
            lm_rec = Some(gptq_2d(&gpu, &mut cs, ptr, rows, h, &hess.h, &opts, None, x_amax)?);
            println!("[gptq] lm_head [{rows}, {h}] GPTQ over {} tokens (activation amax {x_amax:.3}) in {:.1}s", ns * seqlen, t0.elapsed().as_secs_f32());
        } else { println!("[gptq] warning: lm_head is not served in bf16 (tied / missing) — left as the source has it"); }
    }
    drop(hidden);
    // 7. non-layer tensors: source verbatim (embed, final norm, mixer, vision, lm_head) with the
    //    RTN groups quantized; the MTP head and the PLE table straight from the base artifact.
    for (name, meta) in src.metas.iter() {
        if name.starts_with(&format!("{lm}.layers.")) || name.starts_with("mtp.") { continue; }
        let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
        let quantizable = meta.dtype == "BF16" && meta.shape.len() == 2 && meta.shape[1] % 16 == 0 && meta.shape[0] % 16 == 0 && !name.contains(".visual.");
        if name == "lm_head.weight" { if let Some(r) = lm_rec.take() { writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); continue; } }
        if quantizable && is_rtn(name) {
            let (shape, v) = src.read_bf16(name)?; let r = rtn_2d(&v, shape[0], shape[1]);
            writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); continue;
        }
        if quantizable && is_fp8(name) {
            let (shape, v) = src.read_bf16(name)?; writer.push_fp8(&stem, quant::quantize_fp8(&v, shape[0], shape[1])); continue;
        }
        let (_, data) = src.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "I64" => safetensors::Dtype::I64, "F16" => safetensors::Dtype::F16, "U8" => safetensors::Dtype::U8, o => return Err(anyhow!("dtype {o} on {name}")) };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    for (name, meta) in basr.metas.iter() {
        if !name.starts_with("mtp.") { continue; }
        let (_, data) = basr.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "U8" => safetensors::Dtype::U8, "F8_E4M3" => safetensors::Dtype::F8_E4M3, "I64" => safetensors::Dtype::I64, o => return Err(anyhow!("dtype {o} on {name}")) };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    writer.finish()?;
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json"] {
        let s = base.join(f); if s.exists() { std::fs::copy(&s, out.join(f))?; }
    }
    let ple_side = base.join("ple_ngram_nvfp4.json");
    if ple_side.exists() {
        let side: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ple_side)?)?;
        let ple_file = side["file"].as_str().unwrap_or("ple_ngram_nvfp4.bin").to_string();
        std::fs::copy(&ple_side, out.join("ple_ngram_nvfp4.json"))?;
        if std::fs::hard_link(base.join(&ple_file), out.join(&ple_file)).is_err() { std::fs::copy(base.join(&ple_file), out.join(&ple_file))?; }
    }
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
    cj["quantization_config"]["gptq"] = serde_json::json!({ "nsamples": ns, "seqlen": seqlen, "damp": opts.damp, "clip_ratios": opts.nclip,
        "global_scale_optimization": { "type": "alternating_least_squares", "iterations": opts.scale_iters },
        "activation_order": if opts.static_act_order { "static" } else { "none" },
        "groups": opts.gptq_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
        "rtn_groups": opts.nvfp4_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
        "fp8_groups": opts.fp8_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>() });
    if opts.rotate {
        cj["quantization_config"]["transform"] = serde_json::json!({ "type": "hadamard16", "groups": opts.gptq_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>() });
    } else if let Some(qc) = cj["quantization_config"].as_object_mut() {
        qc.remove("transform");
    }
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cj)?)?;
    println!("[gptq] done in {:.1} min → {}", t_all.elapsed().as_secs_f32() / 60.0, out.display());
    Ok(())
}

/// Quantize only the dense LM head of an existing MR-GPTQ artifact. This deliberately reuses the
/// already-quantized trunk and calibrates on its real outputs, avoiding a second 64-layer GPTQ run.
pub fn lmhead(source: &Path, base: &Path, out: &Path, calib: &Path, opts: GptqOpts) -> Result<()> {
    prepare_output_dir(out, &[source, base])?;
    std::env::remove_var("GB10_W4A4_PREFILL");
    std::env::remove_var("GB10_W4A4_LMHEAD_NARROW");
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
    let t_all = std::time::Instant::now();
    let src = ShardReader::open(source)?;
    let basr = ShardReader::open(base)?;
    let (gpu, cfg) = GpuModel::load_from_dir(&base.to_string_lossy())?;
    anyhow::ensure!(matches!(cfg.family, crate::qwen::Family::Qwen35) && !cfg.is_moe,
        "--gptq-lmhead currently requires a qwen3_5 dense artifact, got {:?} (is_moe={})", cfg.family, cfg.is_moe);
    anyhow::ensure!(opts.rotate, "--gptq-lmhead requires --rotate for an MR-GPTQ head");
    let samples = calib_tokens(base, calib, opts.nsamples, opts.seqlen, cfg.vocab_size)?;
    let ns = samples.len();
    let h = cfg.hidden_size;
    let mut hess = gpu.gptq_hess_new(h);
    let normed = gpu.gptq_dev().alloc_zeros::<bf16>(h * opts.seqlen)?;
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, opts.seqlen);
    println!("[gptq-lmhead] {} samples × {} tokens; H {}x{}, damp {}, clip {}, rotate {}",
             ns, opts.seqlen, h, h, opts.damp, opts.nclip, opts.rotate);
    for (s, toks) in samples.iter().enumerate() {
        gpu.zero_slot_state(&mut state, 0, opts.seqlen);
        let (_, residual) = gpu.prefill_batch_range(&mut pool, toks, &mut state, 0, opts.seqlen, 0, 0, cfg.num_layers, None);
        gpu.gptq_lmhead_hess_accum(&mut hess, &residual, &normed, toks.len(), opts.rotate);
        pool.release_bf16(residual, h * toks.len());
        if (s + 1) % 16 == 0 || s + 1 == ns {
            gpu.gptq_sync();
            println!("[gptq-lmhead] Hessian {}/{} samples ({:.1} min)", s + 1, ns, t_all.elapsed().as_secs_f32() / 60.0);
        }
    }
    anyhow::ensure!(hess.n == ns * opts.seqlen,
        "lm_head Hessian coverage mismatch: {} vectors, expected {}", hess.n, ns * opts.seqlen);
    let x_amax = gpu.gptq_amax(&hess);
    let (shape, head) = src.read_bf16("lm_head.weight")?;
    anyhow::ensure!(shape == vec![cfg.vocab_size, h], "lm_head shape {:?}, expected [{}, {}]", shape, cfg.vocab_size, h);
    let head_dev = gpu.gptq_upload_bf16(&head);
    drop(head);
    let mut cs = Cusolver::new(&gpu)?;
    let rec = gptq_2d(&gpu, &mut cs, *head_dev.device_ptr() as u64, cfg.vocab_size, h,
                      &hess.h, &opts, None, x_amax)?;
    let head_igs = rec.igs;
    println!("[gptq-lmhead] quantized lm_head [{}, {}] with {} activation vectors; IGS {:?}",
             cfg.vocab_size, h, hess.n, head_igs);
    drop(cs); drop(head_dev); drop(hess); drop(normed); drop(pool); drop(state); drop(gpu);

    // Rewrite the artifact, replacing the old RTN/bf16 head family and copying every other tensor
    // byte-for-byte. The writer keeps each packed family in one shard.
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    let mut rec = Some(rec);
    let mut inserted = false;
    for (name, meta) in basr.metas.iter() {
        if name.starts_with("lm_head.") {
            if !inserted {
                let r = rec.take().unwrap();
                writer.push_nvfp4("lm_head", r.qw, r.sc, r.m, r.k, r.gs, r.igs);
                inserted = true;
            }
            continue;
        }
        let (_, data) = basr.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() {
            "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32,
            "I64" => safetensors::Dtype::I64, "F16" => safetensors::Dtype::F16,
            "U8" => safetensors::Dtype::U8, "F8_E4M3" => safetensors::Dtype::F8_E4M3,
            o => return Err(anyhow!("dtype {o} on {name}")),
        };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    anyhow::ensure!(inserted, "base artifact has no lm_head tensor family");
    writer.finish()?;
    for f in std::fs::read_dir(base)? {
        let f = f?; let n = f.file_name().to_string_lossy().to_string();
        if n.ends_with(".safetensors") || n == "model.safetensors.index.json" { continue; }
        let dst = out.join(&n);
        let big = n.starts_with("ple_ngram") && n.ends_with(".bin");
        if !(big && std::fs::hard_link(f.path(), &dst).is_ok()) { std::fs::copy(f.path(), &dst)?; }
    }
    let cp = out.join("config.json");
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp)?)?;
    let add = |v: &mut serde_json::Value, g: &str| {
        let a = v.as_array_mut().expect("quantization group list");
        if !a.iter().any(|x| x.as_str() == Some(g)) { a.push(serde_json::json!(g)); }
    };
    let remove = |v: &mut serde_json::Value, g: &str| {
        if let Some(a) = v.as_array_mut() { a.retain(|x| x.as_str() != Some(g)); }
    };
    add(&mut cj["quantization_config"]["gptq"]["groups"], "lmhead");
    remove(&mut cj["quantization_config"]["gptq"]["rtn_groups"], "lmhead");
    cj["quantization_config"]["gptq"]["nsamples"] = serde_json::json!(ns);
    cj["quantization_config"]["gptq"]["seqlen"] = serde_json::json!(opts.seqlen);
    cj["quantization_config"]["gptq"]["damp"] = serde_json::json!(opts.damp);
    cj["quantization_config"]["gptq"]["clip_ratios"] = serde_json::json!(opts.nclip);
    cj["quantization_config"]["gptq"]["global_scale_optimization"] = serde_json::json!({ "type": "alternating_least_squares", "iterations": opts.scale_iters });
    cj["quantization_config"]["gptq"]["activation_order"] = serde_json::json!(if opts.static_act_order { "static" } else { "none" });
    add(&mut cj["quantization_config"]["transform"]["groups"], "lmhead");
    cj["quantization_config"]["activation_variant"] = serde_json::json!("runtime-selectable-a16-or-a4");
    std::fs::write(&cp, serde_json::to_string_pretty(&cj)?)?;
    if let Some(igs) = head_igs {
        let p = out.join("input_global_scale.json");
        let mut j: serde_json::Value = if p.exists() { serde_json::from_str(&std::fs::read_to_string(&p)?)? } else { serde_json::json!({}) };
        j.as_object_mut().ok_or_else(|| anyhow!("{} is not a JSON object", p.display()))?
            .insert("lm_head".to_string(), serde_json::json!(igs));
        std::fs::write(&p, serde_json::to_string_pretty(&j)?)?;
    }
    println!("[gptq-lmhead] done in {:.1} min → {}", t_all.elapsed().as_secs_f32() / 60.0, out.display());
    Ok(())
}

/// Put the layer's linears (`ws`, by source name) into the engine's layer structs.
fn install_layer(gpu: &mut GpuModel, li: usize, is_attn: bool, lp: &str, ws: &mut HashMap<String, W>) {
    let mut take = |n: String| ws.remove(&n).unwrap_or_else(|| panic!("install_layer: missing {n}"));
    let layer = gpu.gptq_layer_mut(li);
    if is_attn {
        let fa = layer.fa.as_mut().unwrap();
        fa.qkv = AttnIn::Split { q: take(format!("{lp}.self_attn.q_proj.weight")), k: take(format!("{lp}.self_attn.k_proj.weight")), v: take(format!("{lp}.self_attn.v_proj.weight")) };
        fa.o_proj = take(format!("{lp}.self_attn.o_proj.weight"));
        if let Some(ix) = fa.indexer.as_mut() { ix.qk_proj = take(format!("{lp}.self_attn.indexer.index_qk_proj.weight")); }
    } else {
        let la = layer.la.as_mut().unwrap();
        la.in_proj = GdnIn::Split { qkv: take(format!("{lp}.linear_attn.in_proj_qkv.weight")), z: take(format!("{lp}.linear_attn.in_proj_z.weight")),
                                    b: take(format!("{lp}.linear_attn.in_proj_b.weight")), a: take(format!("{lp}.linear_attn.in_proj_a.weight")) };
        la.out_proj = take(format!("{lp}.linear_attn.out_proj.weight"));
    }
    match &mut layer.mlp {
        Ffn::Moe(moe) => {
            moe.gate_up = take(format!("{lp}.mlp.experts.gate_up_proj"));
            moe.down = take(format!("{lp}.mlp.experts.down_proj"));
            moe.shared.gate = take(format!("{lp}.mlp.shared_expert.gate_proj.weight"));
            moe.shared.up = take(format!("{lp}.mlp.shared_expert.up_proj.weight"));
            moe.shared.down = take(format!("{lp}.mlp.shared_expert.down_proj.weight"));
            moe.router = take(format!("{lp}.mlp.gate.weight"));
        }
        Ffn::Dense(mlp) => {
            mlp.gate = take(format!("{lp}.mlp.gate_proj.weight"));
            mlp.up = take(format!("{lp}.mlp.up_proj.weight"));
            mlp.down = take(format!("{lp}.mlp.down_proj.weight"));
        }
    }
    if let Some(hc) = layer.hc.as_mut() {
        hc.0.down = take(format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"));
        hc.0.up = take(format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"));
        hc.1.down = take(format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"));
        hc.1.up = take(format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"));
    }
    if let Some(p) = layer.ple.as_mut() {
        p.key_proj = take(format!("{lp}.ple.key_proj.weight"));
        p.value_proj = take(format!("{lp}.ple.value_proj.weight"));
    }
}

pub fn parse_groups(s: &str) -> Result<Vec<Group>> {
    s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()).map(|t| match t {
        "expert" => Ok(Group::Expert), "attn" => Ok(Group::Attn), "mlp" => Ok(Group::Mlp), "gdn" => Ok(Group::Gdn),
        "hc" => Ok(Group::Hc), "ple" => Ok(Group::Ple), "lmhead" => Ok(Group::LmHead), "embed" => Ok(Group::Embed),
        "mtp" => Ok(Group::Mtp), "router" => Ok(Group::Router), o => Err(anyhow!("unknown group {o}")) }).collect()
}


/// `--calib-igs`: calibrate the W4A4 activation scales of an EXISTING quantized artifact without
/// re-quantizing: a plain prefill of the calibration set through the served model, the |x| max of
/// every NVFP4 GEMM input (after the MR rotation when there is one) captured by qweight pointer,
/// `input_global_scale = 6*448 / amax` written to `<out>/input_global_scale.json` (stem -> value;
/// the loader reads it next to the artifact's own `*.input_global_scale` tensors, json wins).
pub fn calib_igs(model_dir: &Path, out: &Path, calib: &Path, nsamples: usize, seqlen: usize) -> Result<()> {
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
    std::fs::create_dir_all(out)?;
    let t_all = std::time::Instant::now();
    let (gpu, cfg) = GpuModel::load_from_dir(&model_dir.to_string_lossy())?;
    let samples = calib_tokens(model_dir, calib, nsamples, seqlen, cfg.vocab_size)?;
    let weights = gpu.nvfp4_weights_by_name();
    let ptrs: Vec<u64> = weights.iter().map(|w| w.1).collect();
    println!("[calib-igs] {} samples × {seqlen} tokens over {} NVFP4 weights ({} stems)", samples.len(), { let mut p = ptrs.clone(); p.sort(); p.dedup(); p.len() }, weights.len());
    gpu.igs_arm(&ptrs);
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, seqlen);
    let nl = cfg.num_layers;
    for (s, toks) in samples.iter().enumerate() {
        gpu.zero_slot_state(&mut state, 0, seqlen);
        let _ = gpu.prefill_batch_range(&mut pool, toks, &mut state, 0, seqlen, 0, 0, nl, None);
        if (s + 1) % 32 == 0 || s + 1 == samples.len() { gpu.gptq_sync(); println!("[calib-igs] {}/{} samples, {:.1} min", s + 1, samples.len(), t_all.elapsed().as_secs_f32() / 60.0); }
    }
    let amax = gpu.igs_disarm();
    let mut js = serde_json::Map::new();
    let (mut n_ok, mut n_zero) = (0, 0);
    for (stem, ptr, _k) in &weights {
        match amax.get(ptr).copied().and_then(igs_of) { Some(g) => { js.insert(stem.clone(), serde_json::json!(g)); n_ok += 1; } None => { n_zero += 1; } }
    }
    std::fs::write(out.join("input_global_scale.json"), serde_json::to_string_pretty(&serde_json::Value::Object(js))?)?;
    println!("[calib-igs] {n_ok} scales written ({n_zero} weights never fed at this seqlen) → {} in {:.1} min", out.join("input_global_scale.json").display(), t_all.elapsed().as_secs_f32() / 60.0);
    Ok(())
}

/// `--gptq-refmt`: re-format an existing artifact without recalibrating — the bf16 2-D weights of
/// `fp8_groups` become row-scaled FP8 (`quant::quantize_fp8`), those of `nvfp4_groups` RTN NVFP4;
/// every other tensor (packed triples included) is copied verbatim, the PLE files are linked.
pub fn refmt(input: &Path, out: &Path, fp8_groups: &[Group], nvfp4_groups: &[Group]) -> Result<()> {
    prepare_output_dir(out, &[input])?;
    let rd = ShardReader::open(input)?;
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    let (mut n8, mut n4, mut nc) = (0, 0, 0);
    for (name, meta) in rd.metas.iter() {
        let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
        let g = quant::group_of(name);
        let quantizable = meta.dtype == "BF16" && meta.shape.len() == 2 && meta.shape[0] % 16 == 0 && meta.shape[1] % 16 == 0
            && !name.contains(".visual.") && name.ends_with(".weight");
        if quantizable && fp8_groups.contains(&g) {
            let (shape, v) = rd.read_bf16(name)?; writer.push_fp8(&stem, quant::quantize_fp8(&v, shape[0], shape[1])); n8 += 1; continue;
        }
        if quantizable && nvfp4_groups.contains(&g) {
            let (shape, v) = rd.read_bf16(name)?; let r = rtn_2d(&v, shape[0], shape[1]);
            writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); n4 += 1; continue;
        }
        let (_, data) = rd.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "I64" => safetensors::Dtype::I64,
            "F16" => safetensors::Dtype::F16, "U8" => safetensors::Dtype::U8, "F8_E4M3" => safetensors::Dtype::F8_E4M3, o => return Err(anyhow!("dtype {o} on {name}")) };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data }); nc += 1;
    }
    writer.finish()?;
    for f in std::fs::read_dir(input)? {
        let f = f?; let n = f.file_name().to_string_lossy().to_string();
        if n.ends_with(".safetensors") || n == "model.safetensors.index.json" { continue; }
        let dst = out.join(&n);
        // only the big PLE table is hard-linked; everything else is COPIED (config.json is edited below —
        // a hard link would edit the input artifact's config in place)
        let big = n.starts_with("ple_ngram") && n.ends_with(".bin");
        if !(big && std::fs::hard_link(f.path(), &dst).is_ok()) { std::fs::copy(f.path(), &dst)?; }
    }
    let cp = out.join("config.json");
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp)?)?;
    cj["quantization_config"]["refmt"] = serde_json::json!({ "fp8_groups": fp8_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
        "rtn_groups": nvfp4_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>() });
    std::fs::write(&cp, serde_json::to_string_pretty(&cj)?)?;
    println!("[gptq-refmt] {n8} tensors → fp8, {n4} → nvfp4 (RTN), {nc} copied → {}", out.display());
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_dir_must_be_distinct_and_empty() {
        let root = std::env::temp_dir().join(format!("gptq-output-test-{}", std::process::id()));
        let input = root.join("input");
        let out = root.join("out");
        std::fs::create_dir_all(&input).unwrap();
        let err = prepare_output_dir(&input, &[&input]).unwrap_err().to_string();
        assert!(err.contains("must differ"));
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("partial.bin"), b"x").unwrap();
        let err = prepare_output_dir(&out, &[&input]).unwrap_err().to_string();
        assert!(err.contains("not empty"));
        std::fs::remove_file(out.join("partial.bin")).unwrap();
        prepare_output_dir(&out, &[&input]).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Families pushed around forced shard boundaries (tiny shard size) must never be split.
    #[test]
    fn writer_keeps_packed_families_in_one_shard() {
        let dir = std::env::temp_dir().join(format!("gptq-writer-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = Writer::new(&dir, 1000);   // 1 KB shards
        for i in 0..7 {
            let stem = format!("layers.{i}.w");
            // a GPTQ triple (~700 B) …
            w.push_nvfp4(&stem, vec![1u8; 512], vec![2u8; 64], 16, 64, 1.0, None);
            // … a verbatim triple in index (name) order, as the base-artifact copy emits it
            let vs = format!("mtp.{i}.w");
            w.push(Out { name: format!("{vs}.weight_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: vec![0u8; 4] });
            w.push(Out { name: format!("{vs}.weight_packed"), dtype: safetensors::Dtype::U8, shape: vec![16, 32], data: vec![3u8; 512] });
            w.push(Out { name: format!("{vs}.weight_scale"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![16, 4], data: vec![4u8; 64] });
            // and a plain bf16 tensor
            w.push(Out { name: format!("norm.{i}.weight"), dtype: safetensors::Dtype::BF16, shape: vec![64], data: vec![0u8; 128] });
        }
        w.finish().expect("no split families");
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        assert!(wm.len() == 7 * 7);
        let shards: std::collections::BTreeSet<&str> = wm.values().map(|v| v.as_str().unwrap()).collect();
        assert!(shards.len() > 5, "the tiny shard size must have produced many shards ({})", shards.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }
    fn host_scale_stats(w: &[f32], s_tensor: f32, nclip: usize) -> (f64, f64, f64) {
        const RATIOS: [f32; 7] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
        let mut out = (0.0, 0.0, 0.0);
        for x in w.chunks_exact(16) {
            let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mut best = (u8::default(), f64::INFINITY);
            for &ratio in RATIOS.iter().take(nclip) {
                let raw = if amax > 0.0 { amax * ratio / quant::E2M1_MAX / s_tensor } else { 0.0 };
                let scode = quant::f32_to_e4m3(raw);
                let scale = quant::e4m3_to_f32(scode) * s_tensor;
                let mse = x.iter().map(|&v| {
                    let q = if scale > 0.0 { quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale } else { 0.0 };
                    let d = (q - v) as f64;
                    d * d
                }).sum::<f64>();
                if mse < best.1 { best = (scode, mse); }
            }
            let es = quant::e4m3_to_f32(best.0);
            let scale = es * s_tensor;
            for &v in x {
                let code = if scale > 0.0 { quant::f32_to_e2m1(v / scale) } else { 0 };
                let z = es * quant::e2m1_to_f32(code);
                out.0 += v as f64 * z as f64;
                out.1 += z as f64 * z as f64;
            }
            out.2 += best.1;
        }
        out
    }

    #[test]
    fn static_activation_order_is_descending_stable_and_finite_first() {
        let order = static_activation_order(&[2.0, 9.0, 9.0, f32::NAN, 1.0, f32::INFINITY]);
        assert_eq!(order, vec![1, 2, 0, 4, 3, 5]);
    }

    #[test]
    fn alternating_scale_fit_never_increases_nvfp4_error() {
        let w: Vec<f32> = (0..256).map(|i| {
            let x = i as f32 - 127.5;
            (x * 0.071).sin() * (0.2 + (i % 29) as f32 * 0.037) + x * 0.0009
        }).collect();
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let initial = e4m3_scale_of(amax) * 1.8;
        let fit = alternating_scale_fit(initial, 6, |s| host_scale_stats(&w, s, 7));
        assert!(fit.final_mse <= fit.initial_mse);
        assert!(fit.accepted > 0, "synthetic bad initial scale should admit an improving step");
        assert!((fit.scale - initial).abs() > initial * 1e-5);
    }

    #[test]
    fn alternating_scale_fit_keeps_zero_tensor_scale() {
        let w = [0.0f32; 32];
        let fit = alternating_scale_fit(1.0, 4, |s| host_scale_stats(&w, s, 7));
        assert_eq!(fit.scale, 1.0);
        assert_eq!(fit.final_mse, 0.0);
        assert_eq!(fit.accepted, 0);
    }
}

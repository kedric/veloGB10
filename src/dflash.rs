//! DFlashDrafter — the AngelSlim Hy3-DFlash-B8 draft model (E29-B1 of the Hy3 speed plan).
//!
//! A 5-layer qwen3-style transformer CONDITIONED on the target model's hidden states — it is NOT a
//! standalone causal LM. It drafts an 8-token block with NON-causal attention over the concatenated
//! (context keys + block keys).
//!
//! Reference semantics (z-lab/dflash `dflash/model.py`, ported exactly):
//!   1. `target_hidden` = concat of the TARGET's post-layer hiddens at layers {1,20,39,58,77} along
//!      the last dim → [B, L, 20480]. Then `hidden_norm(fc(target_hidden))` (fc 20480→4096, no bias,
//!      RMSNorm) — computed once, shared by every layer.
//!   2. `hidden_states` = the block's token embeddings from the target's `embed_tokens`.
//!   3. Per layer (5): input_layernorm → q = q_proj(hidden) [B,8,8192] → view [B,8,64,128] →
//!      q_norm (per-head RMSNorm) → transpose(1,2). k_ctx = k_proj(target_hidden) [B,L,1024],
//!      k_noise = k_proj(hidden) [B,8,1024]; k = cat([k_ctx,k_noise],dim=1) → view [B,L+8,8,128] →
//!      k_norm → transpose. v likewise (NO v_norm). Rotary (theta 11158840, head_dim 128): q uses
//!      cos[..., -8:, :] (the block's positions), k uses the FULL position range (ctx positions
//!      0..L-1, block positions pos_start..pos_start+7).
//!   4. Attention over the concatenated k/v, NON-causal (is_causal=False, no mask — each block
//!      position attends to ALL context + ALL block positions).
//!   5. o_proj → residual → post_attention_layernorm → swiglu MLP → residual.
//!   6. Final RMSNorm → the LM head (the checkpoint has NO lm_head; the probe feeds the target's
//!      `embed_tokens` as a stand-in — the real loop passes the target's actual head).
//!
//! Engine mapping (the "engine port may keep its own KV layout" allowance from the task): the ctx
//! k/v and block k/v are written into per-layer bf16 KV caches at cache rows 0..L-1 and L..L+7
//! (rank space), and the decode-path attention (`gqa_attn_splitk` + `gqa_attn_reduce`) is run with
//! every block query's position pinned to L+7 — so every query attends to all L+8 keys, exactly the
//! reference's maskless full attention. The ROPE positions are decoupled from the cache rows (the
//! tree-verify convention): ctx keys rotate at positions 0..L-1, block q/k at pos_start..pos_start+7.
//!
//! All activations are col-major [dim, batch] bf16 (the engine convention); weights are row-major
//! [out, in] bf16 (the `gemm_act` convention).
//!
//! Invariants honored (AGENTS.md §2): blocking compute stream; fresh pool buffers are zeroed by
//! `Pool::get` (never rely on `alloc_zeros`); D2D/htod on the compute stream; the KV caches are
//! only ever READ within [0, L+8) and every one of those rows is written before attention runs.
//!
//! Norm weights: the checkpoint's RMSNorm is the PLAIN T5-style `weight * x` (Hy3 family), while
//! the engine's rmsnorm kernels hard-code qwen3_5's zero-centered `(1 + weight) * x`. Store the
//! norm weights as (w - 1) at upload, exactly like the hy_v3 loader (src/gpu.rs:2410) — the
//! kernel then computes `(1 + (w-1)) * x == w * x` losslessly in fp32.

use anyhow::{anyhow, Context, Result};
use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::gpu::Pool;

/// Block length the drafter drafts at (the config's `block_size` default, 8).
pub const BLOCK: usize = 8;
/// Number of target layers feeding the conditioning feature.
pub const NCTX_LAYERS: usize = 5;
/// The TARGET layers whose post-FFN-add residuals form the conditioning feature (the config's
/// `dflash_config.target_layer_ids`; E29-B3 reads this directly in gpu.rs's layer loops).
pub const TAP_LAYERS: [usize; NCTX_LAYERS] = [1, 20, 39, 58, 77];

/// Rank-0 staging buffer for the DFlash tap capture. Each target forward (a token-by-token
/// prefill step or an 8-row verify block) copies the post-FFN-add residuals of `TAP_LAYERS`
/// into columns [0, n) of `scratch` ([NCTX_LAYERS*h, BLOCK] bf16, layer-major within a column)
/// — device-side only, on the compute stream, NO dtoh in the loop. The draft loop then D2Ds the
/// columns it needs (the accepted span) into the persistent ctx feature buffer.
pub struct DflashTapSink {
    pub scratch: CudaSlice<bf16>,
}

impl DflashTapSink {
    pub fn new(dev: &Arc<CudaDevice>, h: usize) -> Self {
        DflashTapSink { scratch: dev.alloc_zeros::<bf16>(NCTX_LAYERS * h * BLOCK).unwrap() }
    }
}

/// One tap-layer copy: `residual` [h, n] bf16 col-major (the post-FFN-add residual of target
/// layer `TAP_LAYERS[tap_li]`) → `sink.scratch` columns [0, n) at the layer-major offset
/// `tap_li`. Stream-ordered 2D D2D on the compute stream (invariant 3).
pub fn tap_capture(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                   sink: &DflashTapSink, res_ptr: u64, h: usize, n: usize, tap_li: usize) {
    use cudarc::driver::sys;
    debug_assert!(n <= BLOCK, "tap capture batch {n} exceeds BLOCK {BLOCK}");
    debug_assert!(tap_li < NCTX_LAYERS);
    let dst = *sink.scratch.device_ptr() as u64;
    let cp = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0, srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(), srcDevice: res_ptr,
        srcArray: std::ptr::null_mut(), srcPitch: h * 2,
        dstXInBytes: tap_li * h * 2, dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(), dstDevice: dst,
        dstArray: std::ptr::null_mut(), dstPitch: NCTX_LAYERS * h * 2,
        WidthInBytes: h * 2, Height: n,
    };
    unsafe {
        let r = sys::cuMemcpy2DAsync_v2(&cp, stream);
        assert!(r == sys::CUresult::CUDA_SUCCESS, "dflash tap capture D2D failed: {r:?}");
    }
}

/// One DFlash transformer layer's weights (all from `model.safetensors`, BF16).
pub struct DflashLayer {
    pub input_ln: CudaSlice<f32>,      // [h]
    pub post_ln: CudaSlice<f32>,       // [h]
    pub q_norm: CudaSlice<f32>,        // [hd]
    pub k_norm: CudaSlice<f32>,        // [hd]
    pub q_proj: CudaSlice<bf16>,       // [nh*hd, h]
    pub k_proj: CudaSlice<bf16>,       // [nkv*hd, h]
    pub v_proj: CudaSlice<bf16>,       // [nkv*hd, h]
    pub o_proj: CudaSlice<bf16>,       // [h, nh*hd]
    pub gate_proj: CudaSlice<bf16>,    // [inter, h]
    pub up_proj: CudaSlice<bf16>,      // [inter, h]
    pub down_proj: CudaSlice<bf16>,    // [h, inter]
}

/// Per-layer context+block KV caches in RANK space: ctx at rows 0..L-1, block at rows L..L+7.
/// `stride` is the allocated row count per (head, layer) plane; forward() requires stride >= L+8.
pub struct DflashKv {
    pub k_cache: Vec<CudaSlice<bf16>>, // [layer] [nkv * stride * hd]
    pub v_cache: Vec<CudaSlice<bf16>>,
    pub stride: usize,
}

impl DflashKv {
    pub fn new(d: &DflashDrafter, stride: usize) -> Self {
        let n = d.nkv * stride * d.hd;
        let mut k_cache = Vec::with_capacity(d.layers.len());
        let mut v_cache = Vec::with_capacity(d.layers.len());
        for _ in 0..d.layers.len() {
            k_cache.push(d.dev.alloc_zeros::<bf16>(n).unwrap());
            v_cache.push(d.dev.alloc_zeros::<bf16>(n).unwrap());
        }
        DflashKv { k_cache, v_cache, stride }
    }
}

/// The DFlash drafter. Owns its device runtime (blocking compute stream + cuBLAS), the checkpoint
/// weights, the rope tables, and the tiny persistent per-forward arrays. Activations are pooled.
pub struct DflashDrafter {
    pub dev: Arc<CudaDevice>,
    stream: cudarc::driver::CudaStream,
    blas: CudaBlas,
    bk: HashMap<String, CudaFunction>,
    pub layers: Vec<DflashLayer>,
    /// The block's embedding table [vocab, h] — doubles as the LM-head stand-in (no lm_head in
    /// the checkpoint). The real loop passes the target's head instead.
    pub embed: CudaSlice<bf16>,
    fc: CudaSlice<bf16>,        // [h, 20480]
    hidden_norm: CudaSlice<f32>,// [h] (w-1 convention)
    norm: CudaSlice<f32>,       // [h] (w-1 convention)
    cos_table: CudaSlice<f32>,  // [cos_max, rdim]
    sin_table: CudaSlice<f32>,
    cos_max: usize,
    // Persistent per-forward device arrays (sized BLOCK).
    toks_dev: CudaSlice<i32>,    // block token ids for embed_gather_b
    write_pos: CudaSlice<i32>,   // KV cache rows for the block (L..L+7)
    slot_ids: CudaSlice<i32>,    // all 0 (rank-space cache base)
    // Geometry.
    pub h: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rdim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// The block's slot/mask token id (`dflash_config.mask_token_id`, 120023 for Hy3-DFlash-B8).
    pub mask_token_id: u32,
}

fn d<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }
fn grid(n: usize) -> (u32, u32, u32) { (((n + 255) / 256) as u32, 1, 1) }
fn fbits(x: f32) -> u64 { x.to_bits() as u64 }

/// Launch a gpu_batch.ptx kernel by name. Mirrors gpu.rs's `blaunch!`.
macro_rules! dlaunch {
    ($s:expr, $name:expr, $g:expr, $b:expr, $smem:expr, ($($a:expr),+ $(,)?)) => {
        unsafe {
            let (g0, g1, g2) = $g;
            let (b0, b1, b2) = $b;
            let name: &str = $name;
            $s.bk.get(name).cloned().unwrap_or_else(|| panic!("dflash kernel {}", name)).launch_on_stream(
                &$s.stream,
                LaunchConfig { grid_dim: (g0, g1, g2), block_dim: (b0, b1, b2), shared_mem_bytes: $smem },
                ($($a),+)
            ).unwrap_or_else(|e| panic!("dflash launch {}: {:?}", name, e));
        }
    };
}

/// Create the engine's compute stream as a BLOCKING stream (AGENTS.md §2 invariant; see the
/// `fork_blocking_stream` note in src/gpu.rs for the cross-stream race this prevents).
fn fork_blocking_stream(dev: &Arc<CudaDevice>) -> cudarc::driver::CudaStream {
    use cudarc::driver::result::stream::{create, destroy, StreamKind};
    let mut s = dev.fork_default_stream().expect("fork stream");
    unsafe {
        destroy(s.stream).expect("destroy nonblocking stream");
        s.stream = create(StreamKind::Default).expect("create blocking stream");
    }
    s
}

fn bf16_slice(data: &[u8]) -> &[bf16] {
    bytemuck::cast_slice(data)
}

impl DflashDrafter {
    /// Load the DFlash drafter from a model directory (config.json + model.safetensors).
    /// `max_pos` sizes the rope tables (must cover the largest block position the probe feeds).
    pub fn load_from_dir(dir: &Path, max_pos: usize) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let txt = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&txt).context("parse config.json")?;
        let g = |k: &str, d: usize| v[k].as_u64().unwrap_or(d as u64) as usize;
        let h = g("hidden_size", 4096);
        let n_layers = g("num_hidden_layers", 5);
        let nh = g("num_attention_heads", 64);
        let nkv = g("num_key_value_heads", 8);
        let hd = g("head_dim", 128);
        let inter = g("intermediate_size", 13312);
        let vocab = g("vocab_size", 120832);
        let rms_eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let rope_theta = v["rope_theta"].as_f64().unwrap_or(11158840.0) as f32;
        let mask_token_id = v.get("dflash_config").and_then(|d| d["mask_token_id"].as_u64())
            .or_else(|| v["mask_token_id"].as_u64()).unwrap_or(120023) as u32;
        let rdim = hd;
        assert!(n_layers == 5, "DFlash drafter: expected 5 layers, config has {n_layers}");
        assert!(hd % 32 == 0 && hd <= 512, "head_dim {hd} outside the attention kernels' envelope");

        let sf_path = dir.join("model.safetensors");
        let raw = std::fs::read(&sf_path).with_context(|| format!("read {}", sf_path.display()))?;
        let st = SafeTensors::deserialize(&raw).context("deserialize model.safetensors")?;

        let dev = CudaDevice::new(0)?;
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }

        // Load the batch kernels this module uses (gpu_batch.ptx, verified against this binary).
        let bptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")?);
        let bfnames = ["write_kv_prefill", "write_kv_b", "add_residual_b", "silu_mul_b",
            "embed_gather_b", "gemm_binv_b", "gemm_binv_f32_b", "kernel_build_id"];
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
        let mut bk = HashMap::new();
        for n in bfnames {
            bk.insert(n.to_string(), dev.get_func("gpu_batch", n)
                .with_context(|| format!("gpu_batch.{n} not in ptx"))?);
        }
        // The DFlash-specific kernels (src/ptx/gpu_dflash.ptx): reference-exact bf16 rounding.
        let dptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_dflash.ptx")?);
        let dfnames = ["dflash_rmsnorm_b", "dflash_rope_b", "dflash_attn_b", "kernel_build_id"];
        dev.load_ptx(dptx, "gpu_dflash", &dfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_dflash")?;
        for n in dfnames {
            bk.insert(n.to_string(), dev.get_func("gpu_dflash", n)
                .with_context(|| format!("gpu_dflash.{n} not in ptx"))?);
        }

        let tensor = |name: &str| -> Result<CudaSlice<bf16>> {
            let view = st.tensor(name).with_context(|| format!("missing tensor {name}"))?;
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name} not BF16");
            let data = bf16_slice(view.data()).to_vec();
            Ok(dev.htod_sync_copy(&data).with_context(|| format!("upload {name}"))?)
        };
        let norm_f32 = |name: &str, n: usize| -> Result<CudaSlice<f32>> {
            let view = st.tensor(name).with_context(|| format!("missing tensor {name}"))?;
            assert_eq!(view.dtype(), safetensors::Dtype::BF16, "{name} not BF16");
            let data = bf16_slice(view.data());
            assert_eq!(data.len(), n, "{name} shape");
            // RAW bf16 weight values (f32-stored): the dflash_rmsnorm_b kernel reproduces the
            // reference's `weight * x` exactly (no (1+w) transform — that is a qwen3_5 serving
            // convention that does not apply here).
            let fv: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
            Ok(dev.htod_sync_copy(&fv).with_context(|| format!("upload {name}"))?)
        };

        let embed = tensor("embed_tokens.weight")?;
        let fc = tensor("fc.weight")?;
        let hidden_norm = norm_f32("hidden_norm.weight", h)?;
        let norm = norm_f32("norm.weight", h)?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let lp = format!("layers.{i}");
            layers.push(DflashLayer {
                input_ln: norm_f32(&format!("{lp}.input_layernorm.weight"), h)?,
                post_ln: norm_f32(&format!("{lp}.post_attention_layernorm.weight"), h)?,
                q_norm: norm_f32(&format!("{lp}.self_attn.q_norm.weight"), hd)?,
                k_norm: norm_f32(&format!("{lp}.self_attn.k_norm.weight"), hd)?,
                q_proj: tensor(&format!("{lp}.self_attn.q_proj.weight"))?,
                k_proj: tensor(&format!("{lp}.self_attn.k_proj.weight"))?,
                v_proj: tensor(&format!("{lp}.self_attn.v_proj.weight"))?,
                o_proj: tensor(&format!("{lp}.self_attn.o_proj.weight"))?,
                gate_proj: tensor(&format!("{lp}.mlp.gate_proj.weight"))?,
                up_proj: tensor(&format!("{lp}.mlp.up_proj.weight"))?,
                down_proj: tensor(&format!("{lp}.mlp.down_proj.weight"))?,
            });
        }
        dev.synchronize()?;

        let mut s = Self {
            dev: dev.clone(),
            stream,
            blas,
            bk,
            layers,
            embed,
            fc,
            hidden_norm,
            norm,
            cos_table: dev.alloc_zeros::<f32>(1)?,
            sin_table: dev.alloc_zeros::<f32>(1)?,
            cos_max: 0,
            toks_dev: dev.alloc_zeros::<i32>(BLOCK)?,
            write_pos: dev.alloc_zeros::<i32>(BLOCK)?,
            slot_ids: dev.alloc_zeros::<i32>(BLOCK)?,
            h, nh, nkv, hd, inter, vocab, rdim, rms_eps, rope_theta, mask_token_id,
        };
        s.ensure_rope(max_pos.max(1024))?;
        Ok(s)
    }

    /// (Re)build the cos/sin rope tables so they cover `max_pos` positions (theta, hd 128).
    /// Matches the z-lab reference's quantization exactly: the transformers pipeline stores
    /// `inv_freq` as a bf16 buffer (model.to(bfloat16)) and the rotary forward returns
    /// `cos.to(x.dtype)/sin.to(x.dtype)` — bf16. The engine's f32 table therefore holds the
    /// bf16-quantized values (lossless upcast), so the rotation uses the reference's angles.
    fn ensure_rope(&mut self, max_pos: usize) -> Result<()> {
        if max_pos <= self.cos_max { return Ok(()); }
        let half = self.rdim / 2;
        let theta = self.rope_theta;
        // transformers compute_default_rope_parameters: fp32 power, then the bf16 buffer round.
        let mut inv = vec![0.0f32; half];
        for i in 0..half {
            let v = 1.0f32 / theta.powf(2.0 * i as f32 / self.rdim as f32);
            inv[i] = half::bf16::from_f32(v).to_f32();
        }
        let mut cos_t = vec![0.0f32; max_pos * self.rdim];
        let mut sin_t = vec![0.0f32; max_pos * self.rdim];
        for p in 0..max_pos {
            let pf = p as f32;
            for i in 0..half {
                let f = pf * inv[i];
                let (c, s) = (f.cos(), f.sin());
                // cos.to(dtype=bf16): quantize like the reference's rotary forward output.
                let c = half::bf16::from_f32(c).to_f32();
                let s = half::bf16::from_f32(s).to_f32();
                cos_t[p * self.rdim + i] = c; sin_t[p * self.rdim + i] = s;
                cos_t[p * self.rdim + i + half] = c; sin_t[p * self.rdim + i + half] = s;
            }
        }
        self.cos_table = self.dev.htod_sync_copy(&cos_t)?;
        self.sin_table = self.dev.htod_sync_copy(&sin_t)?;
        self.cos_max = max_pos;
        Ok(())
    }

    /// bf16 GEMM: out[outn, batch] = W[outn, inn] @ x[inn, batch] (all col-major except W row-major).
    /// batch <= 2 → the deterministic `gemm_binv_b`; larger → cuBLAS (same dispatch as gemm_act).
    fn gemm<X: DevicePtr<bf16>>(&self, w: &CudaSlice<bf16>, x: &X, out: &mut CudaSlice<bf16>,
            inn: usize, outn: usize, batch: usize) {
        if batch <= 2 {
            let smem = (batch * 256 * 4) as u32;
            dlaunch!(self, "gemm_binv_b", (outn as u32, 1, 1), (256, 1, 1), smem,
                (d(out), d(w), *x.device_ptr() as u64, outn as i32, inn as i32, batch as i32));
        } else {
            let cfg = GemmConfig::<bf16> {
                transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
                m: outn as i32, n: batch as i32, k: inn as i32,
                alpha: bf16::from_f32(1.0), lda: inn as i32,
                ldb: inn as i32, beta: bf16::from_f32(0.0), ldc: outn as i32,
            };
            unsafe { self.blas.gemm(cfg, w, x, out).expect("dflash gemm"); }
        }
    }

    /// Batched RMSNorm with the reference's EXACT bf16 semantics (dflash_rmsnorm_b: normalize in
    /// fp32, round to bf16, multiply by the raw bf16 weight, round again — transformers
    /// Qwen3RMSNorm). `nh` = per-head grouping (1 for whole-vector columns); weights are the RAW
    /// checkpoint values. `out`/`x` are raw device pointers so in-place use avoids a borrow clash.
    fn rmsnorm(&self, out: u64, x: u64, w: &CudaSlice<f32>, nh: usize, n: usize, b: usize) {
        let bs = n.min(1024);
        dlaunch!(self, "dflash_rmsnorm_b", ((b * nh) as u32, 1, 1), (bs as u32, 1, 1), (bs * 4) as u32,
            (out, x, d(w), nh as i32, n as i32, b as i32, fbits(self.rms_eps)));
    }

    /// Run the block forward. `ctx` is the conditioning feature [5*h, L] bf16 COL-major (the
    /// concat of the target's hiddens at layers {1,20,39,58,77}, one column per ctx position);
    /// `block_tokens` is the 8 block tokens; `pos_start` is the block's first ROPE position
    /// (the target chain position the block starts at). Returns logits [BLOCK, vocab] f32
    /// row-major (logits[b*vocab + t]).
    ///
    /// `noise_embed`/`head` override the checkpoint's embed_tokens stand-in with the TARGET's
    /// full-vocab bf16 embed + lm_head (the reference `dflash_generate` uses `target.embed_tokens`
    /// for the noise and `target.lm_head` for the logits — the checkpoint's own embed is a
    /// different tensor). `None` keeps the probe stand-in (checkpoint embed for both).
    pub fn forward<C: DevicePtr<bf16>, H: DevicePtr<bf16>>(&mut self, pool: &mut Pool, kv: &mut DflashKv,
                   ctx: &C, ctx_len: usize, block_tokens: &[u32], pos_start: usize,
                   noise_embed: Option<&H>, head: Option<&H>) -> Result<Vec<f32>> {
        let (h, nh, nkv, hd, inter, vocab) = (self.h, self.nh, self.nkv, self.hd, self.inter, self.vocab);
        let q = BLOCK;
        let k = ctx_len + q;
        assert_eq!(block_tokens.len(), q, "dflash block must be {q} tokens");
        assert!(kv.stride >= k, "dflash kv stride {} < ctx+block {k}", kv.stride);
        assert_eq!(ctx.len(), NCTX_LAYERS * h * ctx_len, "ctx feature shape");
        let emb_ptr = noise_embed.map(|e| *e.device_ptr() as u64).unwrap_or(*self.embed.device_ptr() as u64);
        let head_ptr = head.map(|e| *e.device_ptr() as u64).unwrap_or(*self.embed.device_ptr() as u64);

        self.ensure_rope(pos_start + q)?; // grow the cos/sin tables if needed

        // ---- per-forward device arrays (htod on the compute stream) ----
        let toks: Vec<i32> = block_tokens.iter().map(|&t| t as i32).collect();
        let write_pos: Vec<i32> = (0..q).map(|b| (ctx_len + b) as i32).collect(); // cache rows L..L+7
        let slot_ids: Vec<i32> = vec![0i32; q];                                    // rank-space base
        unsafe {
            use cudarc::driver::result::memcpy_htod_async;
            let c = self.stream.stream;
            memcpy_htod_async(*self.toks_dev.device_ptr() as cudarc::driver::sys::CUdeviceptr, &toks, c).expect("htod toks");
            memcpy_htod_async(*self.write_pos.device_ptr() as cudarc::driver::sys::CUdeviceptr, &write_pos, c).expect("htod write_pos");
            memcpy_htod_async(*self.slot_ids.device_ptr() as cudarc::driver::sys::CUdeviceptr, &slot_ids, c).expect("htod slot_ids");
        }

        // ---- conditioning: hidden_norm(fc(target_hidden)) — once, shared by all layers ----
        let mut ctx_cond = pool.get_bf16(h * ctx_len);
        self.gemm(&self.fc, ctx, &mut ctx_cond, NCTX_LAYERS * h, h, ctx_len);
        let cc = d(&ctx_cond);
        self.rmsnorm(cc, cc, &self.hidden_norm, 1, h, ctx_len);

        // ---- noise embedding: the block's tokens (the TARGET's embed when the loop passes it) ----
        let hidden = pool.get_bf16(h * q);
        dlaunch!(self, "embed_gather_b", grid(h * q), (256, 1, 1), 0,
            (d(&hidden), emb_ptr, *self.toks_dev.device_ptr() as u64,
             h as i32, q as i32));

        // Stable-sized per-layer scratch (allocated once).
        let mut normed = pool.get_bf16(h * q);
        let mut qb = pool.get_bf16(nh * hd * q);
        let mut k_noise = pool.get_bf16(nkv * hd * q);
        let mut v_noise = pool.get_bf16(nkv * hd * q);
        let mut attn = pool.get_bf16(nh * hd * q);
        let mut attn_out = pool.get_bf16(h * q);
        let mut gate = pool.get_bf16(inter * q);
        let mut up = pool.get_bf16(inter * q);
        let mut mlp_out = pool.get_bf16(h * q);
        // ctx-sized scratch.
        let mut k_ctx = pool.get_bf16(nkv * hd * ctx_len);
        let mut v_ctx = pool.get_bf16(nkv * hd * ctx_len);

        // GB10_DFLASH_DEBUG: dump intermediates as f32 for the golden comparison.
        let dbg = std::env::var("GB10_DFLASH_DEBUG").is_ok();
        let dump = |tag: &str, b: &CudaSlice<bf16>, n: usize| {
            if dbg {
                let v = self.dev.dtoh_sync_copy(b).unwrap();
                let v: Vec<f32> = v[..n].iter().map(|x| x.to_f32()).collect();
                let mut f = std::io::BufWriter::new(std::fs::File::create(format!("/tmp/dflash_{tag}.bin")).unwrap());
                use std::io::Write;
                for x in v { f.write_all(&x.to_le_bytes()).unwrap(); }
            }
        };
        if dbg { dump("ctx_cond", &ctx_cond, h * ctx_len); dump("block", &hidden, h * q); }
        if dbg {
            // dump the cos/sin table rows [pos_start, pos_start+8)
            for (tag, t) in [("cos", &self.cos_table), ("sin", &self.sin_table)] {
                let v = self.dev.dtoh_sync_copy(t).unwrap();
                let v: Vec<f32> = v[pos_start * self.rdim..(pos_start + q) * self.rdim].to_vec();
                let mut f = std::io::BufWriter::new(std::fs::File::create(format!("/tmp/dflash_{tag}tab.bin")).unwrap());
                use std::io::Write;
                for x in v { f.write_all(&x.to_le_bytes()).unwrap(); }
            }
        }
        // raw (pre-norm) q/k buffers
        let mut q_raw = pool.get_bf16(nh * hd * q);
        let mut kn_raw = pool.get_bf16(nkv * hd * q);

        // Rope table pointers: ctx keys rotate at positions 0..L-1 (table base), the block's
        // q/k at pos_start..pos_start+7 (row offset into the table).
        let cos0 = *self.cos_table.device_ptr() as u64;
        let sin0 = *self.sin_table.device_ptr() as u64;
        let block_off = (pos_start * self.rdim * 4) as u64;
        let stride = kv.stride;

        for (li, layer) in self.layers.iter().enumerate() {
            self.rmsnorm(d(&mut normed), d(&hidden), &layer.input_ln, 1, h, q);
            // q / k_noise / v_noise from the block hidden; k_ctx / v_ctx from the conditioning.
            self.gemm(&layer.q_proj, &normed, &mut q_raw, h, nh * hd, q);
            self.gemm(&layer.k_proj, &normed, &mut kn_raw, h, nkv * hd, q);
            self.gemm(&layer.v_proj, &normed, &mut v_noise, h, nkv * hd, q);
            self.gemm(&layer.k_proj, &ctx_cond, &mut k_ctx, h, nkv * hd, ctx_len);
            self.gemm(&layer.v_proj, &ctx_cond, &mut v_ctx, h, nkv * hd, ctx_len);
            if dbg { dump(&format!("normed_{li}"), &normed, h * q); }
            if dbg { dump(&format!("qraw_{li}"), &q_raw, nh * hd * q); }
            if dbg { dump(&format!("knraw_{li}"), &kn_raw, nkv * hd * q); }
            // Per-head q/k norm (into the post-norm buffers), then rotary.
            self.rmsnorm(d(&mut qb), d(&q_raw), &layer.q_norm, nh, hd, q);
            self.rmsnorm(d(&mut k_noise), d(&kn_raw), &layer.k_norm, nkv, hd, q);
            if dbg { dump(&format!("qnorm_{li}"), &qb, nh * hd * q); }
            self.rmsnorm(d(&mut k_ctx), d(&k_ctx), &layer.k_norm, nkv, hd, ctx_len);
            if dbg {
                dump(&format!("knn_{li}"), &k_noise, nkv * hd * q);
                dump(&format!("kcn_{li}"), &k_ctx, nkv * hd * ctx_len);
            }
            dlaunch!(self, "dflash_rope_b", grid(q * nh * (self.rdim / 2)), (256, 1, 1), 0,
                (d(&qb), cos0 + block_off, sin0 + block_off, nh as i32, hd as i32, self.rdim as i32, q as i32));
            dlaunch!(self, "dflash_rope_b", grid(q * nkv * (self.rdim / 2)), (256, 1, 1), 0,
                (d(&k_noise), cos0 + block_off, sin0 + block_off, nkv as i32, hd as i32, self.rdim as i32, q as i32));
            dlaunch!(self, "dflash_rope_b", grid(ctx_len * nkv * (self.rdim / 2)), (256, 1, 1), 0,
                (d(&k_ctx), cos0, sin0, nkv as i32, hd as i32, self.rdim as i32, ctx_len as i32));
            if dbg {
                dump(&format!("q_{li}"), &qb, nh * hd * q);
                dump(&format!("kn_{li}"), &k_noise, nkv * hd * q);
                dump(&format!("kc_{li}"), &k_ctx, nkv * hd * ctx_len);
                dump(&format!("v_{li}"), &v_noise, nkv * hd * q);
            }
            // KV: ctx at rows 0..L-1, block at rows L..L+7 (rank space).
            dlaunch!(self, "write_kv_prefill", grid(ctx_len * nkv * hd), (256, 1, 1), 0,
                (d(&kv.k_cache[li]), d(&kv.v_cache[li]), d(&k_ctx), d(&v_ctx),
                 stride as i32, nkv as i32, hd as i32, ctx_len as i32, 0i32));
            dlaunch!(self, "write_kv_b", grid(q * nkv * hd), (256, 1, 1), 0,
                (d(&kv.k_cache[li]), d(&kv.v_cache[li]), d(&k_noise), d(&v_noise),
                 *self.write_pos.device_ptr() as u64, stride as i32, nkv as i32, hd as i32, q as i32,
                 *self.slot_ids.device_ptr() as u64));
            if dbg && li == 0 {
                dump("kcache_0", &kv.k_cache[0], nkv * stride * hd);
                dump("vcache_0", &kv.v_cache[0], nkv * stride * hd);
            }
            // NON-CAUSAL attention over ALL K keys (ctx + block), softmax weights rounded to bf16
            // exactly like the reference's eager attention (dflash_attn_b).
            let smem = ((hd / 32) as u32 + 1) * 4;
            dlaunch!(self, "dflash_attn_b", ((q * nh) as u32, 1, 1), (hd as u32, 1, 1), smem,
                (d(&attn), d(&qb), d(&kv.k_cache[li]), d(&kv.v_cache[li]),
                 stride as i32, nh as i32, nkv as i32, hd as i32, k as i32, q as i32));
            if dbg { dump(&format!("attn_{li}"), &attn, nh * hd * q); }
            // o_proj → residual → post-attention norm → swiglu MLP → residual.
            self.gemm(&layer.o_proj, &attn, &mut attn_out, nh * hd, h, q);
            if dbg { dump(&format!("attnout_{li}"), &attn_out, h * q); }
            dlaunch!(self, "add_residual_b", grid(h * q), (256, 1, 1), 0,
                (d(&hidden), d(&hidden), d(&attn_out), (h * q) as i32));
            self.rmsnorm(d(&mut normed), d(&hidden), &layer.post_ln, 1, h, q);
            self.gemm(&layer.gate_proj, &normed, &mut gate, h, inter, q);
            self.gemm(&layer.up_proj, &normed, &mut up, h, inter, q);
            dlaunch!(self, "silu_mul_b", grid(inter * q), (256, 1, 1), 0,
                (d(&gate), d(&gate), d(&up), (inter * q) as i32));
            self.gemm(&layer.down_proj, &gate, &mut mlp_out, inter, h, q);
            dlaunch!(self, "add_residual_b", grid(h * q), (256, 1, 1), 0,
                (d(&hidden), d(&hidden), d(&mlp_out), (h * q) as i32));
            if dbg { dump(&format!("layer{li}_hidden"), &hidden, h * q); }
        }

        // ---- final norm → LM head (embed_tokens stand-in) → logits [q, vocab] f32 ----
        self.rmsnorm(d(&mut normed), d(&hidden), &self.norm, 1, h, q);
        if dbg { dump("final_normed", &normed, h * q); }
        let logits = pool.get(vocab * q);
        let smem = (q * 256 * 4) as u32;
        dlaunch!(self, "gemm_binv_f32_b", (vocab as u32, 1, 1), (256, 1, 1), smem,
            (d(&logits), head_ptr, d(&normed), vocab as i32, h as i32, q as i32));
        let host_full = self.dev.dtoh_sync_copy(&logits).context("dtoh dflash logits")?;
        let host: Vec<f32> = host_full[..vocab * q].to_vec(); // pool buckets up; slice the used part

        pool.release_bf16(ctx_cond, h * ctx_len);
        pool.release_bf16(hidden, h * q);
        pool.release_bf16(normed, h * q);
        pool.release_bf16(qb, nh * hd * q);
        pool.release_bf16(k_noise, nkv * hd * q);
        pool.release_bf16(v_noise, nkv * hd * q);
        pool.release_bf16(attn, nh * hd * q);
        pool.release_bf16(attn_out, h * q);
        pool.release_bf16(gate, inter * q);
        pool.release_bf16(up, inter * q);
        pool.release_bf16(mlp_out, h * q);
        pool.release_bf16(k_ctx, nkv * hd * ctx_len);
        pool.release_bf16(v_ctx, nkv * hd * ctx_len);
        pool.release(logits, vocab * q);
        Ok(host)
    }

    /// Top-1 token per block position over the forward's logits (first-max tie-break, matching
    /// torch's `argmax`).
    pub fn top1(&self, logits: &[f32]) -> Vec<u32> {
        assert_eq!(logits.len(), BLOCK * self.vocab);
        (0..BLOCK).map(|b| {
            let col = &logits[b * self.vocab..(b + 1) * self.vocab];
            let mut best = 0usize;
            for (t, &x) in col.iter().enumerate() {
                if x > col[best] { best = t; }
            }
            best as u32
        }).collect()
    }
}

/// Probe-level file formats (also documented in the E29-B1 report):
///
/// DFCTX (the recorder's format, tokens embedded) — magic b"DFCTX", then LE u32: version=1, plen,
/// nsteps, h=4096, nctx_layers=5; then (plen+nsteps) u32 tokens (prompt then generated); then
/// (nsteps+1) × nctx_layers × h f32 features (feature i = the target's post-layer hiddens at
/// layers {1,20,39,58,77} at ONE position; feature 0 = prefill's last prompt position).
///
/// DFCT (this module's plain format, for golden/full-ctx tests) — magic b"DFCT", then LE u32:
/// version=1, ctx_len, nfeatures; then nfeatures × ctx_len × (nctx_layers × h) f32 (each feature
/// is a full ctx sequence; per feature: position-major, then layer-major in [1,20,39,58,77] order).
pub struct DflashProbeInput {
    pub plen: usize,
    pub steps: Vec<DflashStep>,
}

pub struct DflashStep {
    pub pos_start: usize,
    /// The conditioning feature [NCTX_LAYERS * h, ctx_len] f32 COL-major (position-major in file).
    pub ctx: Vec<f32>,
    pub ctx_len: usize,
    pub block_tokens: Vec<u32>,
    /// Ground-truth target chain (prompt + generated) for the acceptance comparison, if any.
    pub chain: Option<Vec<u32>>,
}

/// Read a ctx-features file: DFCTX (magic) or the plain DFCT format.
pub fn read_probe_input(path: &Path, tokens_json: Option<&Path>) -> Result<DflashProbeInput> {
    let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let r32 = |b: &[u8], off: usize| -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    };
    let f32at = |b: &[u8], off: usize| -> f32 { f32::from_le_bytes(b[off..off + 4].try_into().unwrap()) };
    // tokens.json (optional; REQUIRED for the plain DFCT format): {"plen": N, "tokens": [...]}
    // — `plen` is the prompt token count, `tokens` the full target chain (prompt then generated).
    let (chain, tj_plen) = match tokens_json {
        Some(tp) => {
            let t = std::fs::read_to_string(tp).with_context(|| format!("read {}", tp.display()))?;
            let v: serde_json::Value = serde_json::from_str(&t).context("parse tokens.json")?;
            let toks: Vec<u32> = v["tokens"].as_array()
                .context("tokens.json missing \"tokens\" array")?
                .iter().filter_map(|x| x.as_u64().map(|u| u as u32)).collect();
            let plen = v["plen"].as_u64().map(|p| p as usize).unwrap_or(0);
            anyhow::ensure!(plen <= toks.len(), "tokens.json plen {plen} > token count {}", toks.len());
            (Some(toks), plen)
        }
        None => (None, 0),
    };
    if buf.len() >= 5 && &buf[..5] == b"DFCTX" {
        let ver = r32(&buf, 5);
        anyhow::ensure!(ver == 1, "DFCTX version {ver} unsupported");
        let plen = r32(&buf, 9) as usize;
        let nsteps = r32(&buf, 13) as usize;
        let h = r32(&buf, 17) as usize;
        let nl = r32(&buf, 21) as usize;
        anyhow::ensure!(h == 4096 && nl == NCTX_LAYERS, "DFCTX h={h} nctx_layers={nl} (want 4096/5)");
        let mut off = 25usize;
        let ntok = plen + nsteps;
        let toks: Vec<u32> = (0..ntok).map(|_| { let v = r32(&buf, off); off += 4; v }).collect();
        let feats = nsteps + 1;
        let need = feats * nl * h;
        anyhow::ensure!(off + need * 4 <= buf.len(), "DFCTX truncated");
        let mut steps = Vec::with_capacity(nsteps);
        for i in 0..nsteps {
            // feature i = one position → ctx_len 1; the feature is stored layer-major [5, h].
            let feat: Vec<f32> = (0..nl * h).map(|_| { let v = f32at(&buf, off); off += 4; v }).collect();
            // col-major [nl*h, 1] == layer-major order as stored.
            let end = (plen + i + BLOCK).min(plen + nsteps);
            let block_tokens: Vec<u32> = (plen + i..end).map(|j| toks[j]).collect();
            let block_tokens = if block_tokens.len() < BLOCK {
                let mut bt = block_tokens;
                bt.resize(BLOCK, 120023); // mask_token_id padding at the chain tail
                bt
            } else { block_tokens };
            steps.push(DflashStep {
                pos_start: plen + i,
                ctx: feat,
                ctx_len: 1,
                block_tokens,
                chain: Some(toks.clone()),
            });
        }
        Ok(DflashProbeInput { plen, steps })
    } else if buf.len() >= 4 && &buf[..4] == b"DFCT" {
        let ver = r32(&buf, 4);
        anyhow::ensure!(ver == 1, "DFCT version {ver} unsupported");
        let ctx_len = r32(&buf, 8) as usize;
        let nfeatures = r32(&buf, 12) as usize;
        let mut off = 16usize;
        let per = ctx_len * NCTX_LAYERS * 4096usize;
        anyhow::ensure!(off + nfeatures * per * 4 <= buf.len(), "DFCT truncated");
        let mut steps = Vec::with_capacity(nfeatures);
        for i in 0..nfeatures {
            // File order: position-major, then layer-major. Convert to col-major [nl*h, L]:
            // col[(pos * nl + layer) * h + d] = file[pos * (nl*h) + layer * h + d].
            let mut ctx = vec![0.0f32; ctx_len * NCTX_LAYERS * 4096];
            for pos in 0..ctx_len {
                for layer in 0..NCTX_LAYERS {
                    for d in 0..4096 {
                        ctx[(pos * NCTX_LAYERS + layer) * 4096 + d] = f32at(&buf, off);
                        off += 4;
                    }
                }
            }
            let plen = if tj_plen > 0 { tj_plen } else {
                anyhow::bail!("plain DFCT format needs tokens.json with \"plen\" (prompt token count)");
            };
            let chain_len = chain.as_ref().map(|c| c.len()).unwrap_or(plen);
            let end = (plen + i + BLOCK).min(chain_len);
            let block_tokens: Vec<u32> = chain.as_ref().map(|c| {
                let mut bt: Vec<u32> = (plen + i..end).map(|j| c[j]).collect();
                bt.resize(BLOCK, 120023);
                bt
            }).unwrap_or_else(|| (0..BLOCK as u32).collect());
            steps.push(DflashStep {
                pos_start: plen + i,
                ctx,
                ctx_len,
                block_tokens,
                chain: chain.clone(),
            });
        }
        Ok(DflashProbeInput { plen: tj_plen, steps })
    } else {
        Err(anyhow!("unrecognized ctx-features file (magic must be DFCTX or DFCT)"))
    }
}

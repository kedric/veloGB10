//! Tier-1 item 1.3 gate + bench: `gemm_dsv4_fp8_bsb_tma` (DeepGEMM-class TMA smem pipeline,
//! order-faithfulness probe) vs the G2-locked `gemm_dsv4_fp8_bsb`.
//!
//! Gates:
//!   1. tma == bsb BITWISE (C bf16 and Cf f32) on synthetic shapes, N in {1, 6, 16}.
//!   2. Batch invariance (AGENTS.md §2.4): col-0 bitwise at N=1..=16 + full-prefix at 16.
//!   3. Real checkpoint wo_b weights (layers 0/2), N in {1, 6, 16} (skip if unmounted).
//! Bench (GB10_BENCH=1, optional GB10_BENCH_SHAPE=<name>): kernel-time per shape for
//! bsb / bsb2 / tma, rotating 4 cold weight copies — same methodology as
//! dsv4_fp8_sched_bench (memo §3 comparison table).
//!
//! The tma kernel needs >48 KB dynamic smem (NS=6 -> 98,352 B) => raw-driver launch path
//! (cuFuncSetAttribute opt-in), mirroring src/dsv4_gpu.rs::set_dynamic_smem.
//!
//! Run: cargo test --release --test dsv4_fp8_tma_test -- --nocapture

use cudarc::driver::result;
use cudarc::driver::sys;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use gb10_inference::{dsv4_cpu, dsv4_load, quant};
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";
const TMA_SMEM: u32 = 2 * (16384 + 8); // FP8_TMA_NS=5 stages of 16 KB + barriers (+8 KB static sh)

const BENCH_SHAPES: &[(usize, usize, f64, &str)] = &[
    (32768, 1024, 155.0, "wq_b"),
    (4096, 8192, 178.0, "wo_b"),
    (4096, 4096, 84.0, "sh_gu"),
    (8192, 1024, 36.0, "idx_wq_b"),
];

// ---------------------------------------------------------------------------
// raw-driver loader/launcher for the tma kernel (dynamic-smem opt-in)
// ---------------------------------------------------------------------------

struct RawKernel {
    module: sys::CUmodule,
    func: sys::CUfunction,
}

fn load_raw(dev: &Arc<CudaDevice>, name: &str) -> RawKernel {
    // bind the device primary context on this thread (mirrors dsv4_gpu.rs)
    unsafe {
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        sys::cuDevicePrimaryCtxRetain(&mut ctx, 0).result().unwrap();
        sys::cuCtxSetCurrent(ctx).result().unwrap();
    }
    let ptx = std::fs::read_to_string("src/ptx/gpu_batch.ptx").expect("src/ptx/gpu_batch.ptx");
    let ptx_c = CString::new(ptx).unwrap();
    let module = unsafe { result::module::load_data(ptx_c.as_ptr() as *const _) }.expect("load gpu_batch.ptx");
    let name_c = CString::new(name).unwrap();
    let func = unsafe { result::module::get_function(module, name_c) }
        .unwrap_or_else(|_| panic!("{name} not in gpu_batch.ptx"));
    let r = unsafe {
        sys::cuFuncSetAttribute(
            func,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            TMA_SMEM as i32,
        )
    };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute smem {TMA_SMEM}");
    let _ = dev; // context retained above
    RawKernel { module, func }
}

impl Drop for RawKernel {
    fn drop(&mut self) {
        let _ = unsafe { result::module::unload(self.module) };
    }
}

/// Launch the tma kernel on the NULL stream; caller syncs. Mirrors bsb's arg list.
#[allow(clippy::too_many_arguments)]
fn launch_tma(
    k: &RawKernel, c: &mut CudaSlice<u16>, wt: &CudaSlice<u8>, sb: &CudaSlice<u8>,
    x: &CudaSlice<u8>, sa: &CudaSlice<u8>, m: usize, n: usize, cf: Option<&mut CudaSlice<f32>>,
    k_dim: usize, grid_x: u32,
) {
    let c_ptr = *c.device_ptr();
    let wt_ptr = *wt.device_ptr();
    let sb_ptr = *sb.device_ptr();
    let x_ptr = *x.device_ptr();
    let sa_ptr = *sa.device_ptr();
    let m_i = m as i32;
    let kd_i = k_dim as i32;
    let n_i = n as i32;
    let cf_ptr: u64 = cf.as_ref().map(|s| *s.device_ptr()).unwrap_or(0);
    let mut params: [*mut std::ffi::c_void; 9] = [
        &c_ptr as *const _ as *mut _,
        &wt_ptr as *const _ as *mut _,
        &sb_ptr as *const _ as *mut _,
        &x_ptr as *const _ as *mut _,
        &sa_ptr as *const _ as *mut _,
        &m_i as *const _ as *mut _,
        &kd_i as *const _ as *mut _,
        &n_i as *const _ as *mut _,
        &cf_ptr as *const _ as *mut _,
    ];
    unsafe {
        result::launch_kernel(
            k.func,
            (grid_x, 1, 1),
            (256, 1, 1),
            TMA_SMEM,
            result::stream::null(),
            &mut params,
        )
        .expect("launch gemm_dsv4_fp8_bsb_tma");
    }
}

// ---------------------------------------------------------------------------
// shared case scaffolding (same construction as tests/dsv4_fp8_bsb_test.rs)
// ---------------------------------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

struct Case {
    m: usize,
    k: usize,
    x: Vec<f32>,
    x_codes: Vec<u8>,
    sa: Vec<u8>,
    wt: Vec<u8>,
    sb: Vec<u8>,
}

fn pow2_to_e8m0(s: f32) -> u8 {
    assert!(s.is_finite() && s > 0.0);
    let b = (s.to_bits() >> 23) as u8;
    assert_eq!(dsv4_load::e8m0_to_f32(b), s);
    b
}

fn quant_w_blocks(w: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let (rb_n, cb_n) = (m / 128, k / 128);
    let mut codes = vec![0u8; m * k];
    let mut sb = vec![0u8; rb_n * cb_n];
    for rb in 0..rb_n {
        for cb in 0..cb_n {
            let mut amax = 0.0f32;
            for i in 0..128 {
                let row = (rb * 128 + i) * k + cb * 128;
                for &v in &w[row..row + 128] {
                    amax = amax.max(v.abs());
                }
            }
            let s = if amax > 0.0 { dsv4_cpu::fast_round_scale(amax, 1.0 / 448.0) } else { 1.0 };
            sb[rb * cb_n + cb] = pow2_to_e8m0(s);
            for i in 0..128 {
                let row = (rb * 128 + i) * k + cb * 128;
                for j in 0..128 {
                    codes[row + j] = dsv4_cpu::f32_to_e4m3_rne((w[row + j] / s).clamp(-448.0, 448.0));
                }
            }
        }
    }
    (codes, sb)
}

fn act_codes_bytes(x: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let (codes, sa) = dsv4_cpu::act_quant_codes(x, rows, k, 128);
    let code_bytes: Vec<u8> = codes.iter().map(|&v| dsv4_cpu::f32_to_e4m3_rne(v)).collect();
    let sa_bytes: Vec<u8> = sa.iter().map(|&s| pow2_to_e8m0(s)).collect();
    (code_bytes, sa_bytes)
}

fn synth_case(rng: &mut XorShift, m: usize, k: usize) -> Case {
    let mut x = vec![0.0f32; 16 * k];
    for r in 0..16 {
        let rs = 0.4 + r as f32 * 0.23;
        for i in 0..k {
            let mut v = rng.f32() * rs;
            if i % 997 == 13 {
                v *= 40.0;
            }
            x[r * k + i] = dsv4_cpu::bf(v);
        }
    }
    let mut w = vec![0.0f32; m * k];
    for rb in 0..m / 128 {
        for cb in 0..k / 128 {
            let sigma = 0.02 * 2f32.powi(((rb * 7 + cb * 3) % 8) as i32 - 3);
            for i in 0..128 {
                let row = (rb * 128 + i) * k + cb * 128;
                for j in 0..128 {
                    w[row + j] = (rng.f32() + rng.f32()) * sigma;
                }
            }
        }
    }
    let (x_codes, sa) = act_codes_bytes(&x, 16, k);
    let (wcodes, sb) = quant_w_blocks(&w, m, k);
    let wt = quant::repack_fp8_mma(&wcodes, m, k);
    Case { m, k, x, x_codes, sa, wt, sb }
}

fn real_case(rng: &mut XorShift, layer: usize, key: &str) -> Case {
    let name = format!("layers.{layer}.attn.{key}.weight");
    let (shape, wcodes, sb) = dsv4_load::read_raw_fp8(Path::new(BUNDLE), &name)
        .unwrap_or_else(|e| panic!("read_raw_fp8 {name}: {e}"));
    let (m, k) = (shape[0], shape[1]);
    let mut x = vec![0.0f32; 16 * k];
    for r in 0..16 {
        let rs = 0.6 + r as f32 * 0.17;
        for i in 0..k {
            let mut v = rng.f32() * rs;
            if i % 1151 == 7 {
                v *= 30.0;
            }
            x[r * k + i] = dsv4_cpu::bf(v);
        }
    }
    let (x_codes, sa) = act_codes_bytes(&x, 16, k);
    let wt = quant::repack_fp8_mma(&wcodes, m, k);
    Case { m, k, x, x_codes, sa, wt, sb }
}

// ---------------------------------------------------------------------------
// runners: bsb (cudarc safe path) and tma (raw path)
// ---------------------------------------------------------------------------

fn load_bsb(dev: &Arc<CudaDevice>) -> CudaFunction {
    let ptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/gpu_batch.ptx").expect("src/ptx/gpu_batch.ptx"),
    );
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb"]).expect("load_ptx");
    dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb").expect("missing bsb")
}

fn run_bsb(dev: &Arc<CudaDevice>, f: &CudaFunction, c: &Case, n: usize, want_cf: bool) -> (Vec<u16>, Option<Vec<f32>>) {
    let (m, k) = (c.m, c.k);
    let x_dev = dev.htod_sync_copy(&c.x_codes[..n * k]).unwrap();
    let sa_dev = dev.htod_sync_copy(&c.sa[..n * (k / 128)]).unwrap();
    let wt_dev = dev.htod_sync_copy(&c.wt).unwrap();
    let sb_dev = dev.htod_sync_copy(&c.sb).unwrap();
    let mut c_dev = dev.alloc_zeros::<u16>(n * m).unwrap();
    let mut cf_dev = dev.alloc_zeros::<f32>(if want_cf { n * m } else { 1 }).unwrap();
    dev.synchronize().unwrap();
    let cfg = LaunchConfig { grid_dim: ((m / 16) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        if want_cf {
            f.clone().launch(cfg, (&mut c_dev, &wt_dev, &sb_dev, &x_dev, &sa_dev, m as i32, k as i32, n as i32, &mut cf_dev)).unwrap();
        } else {
            f.clone().launch(cfg, (&mut c_dev, &wt_dev, &sb_dev, &x_dev, &sa_dev, m as i32, k as i32, n as i32, 0u64)).unwrap();
        }
    }
    dev.synchronize().unwrap();
    let c_out = dev.dtoh_sync_copy(&c_dev).unwrap();
    let cf_out = if want_cf { Some(dev.dtoh_sync_copy(&cf_dev).unwrap()) } else { None };
    (c_out, cf_out)
}

fn run_tma(dev: &Arc<CudaDevice>, k_fn: &RawKernel, c: &Case, n: usize, want_cf: bool) -> (Vec<u16>, Option<Vec<f32>>) {
    let (m, k) = (c.m, c.k);
    assert_eq!(k % 1024, 0, "tma probe needs K % 1024 == 0");
    let x_dev = dev.htod_sync_copy(&c.x_codes[..n * k]).unwrap();
    let sa_dev = dev.htod_sync_copy(&c.sa[..n * (k / 128)]).unwrap();
    let wt_dev = dev.htod_sync_copy(&c.wt).unwrap();
    let sb_dev = dev.htod_sync_copy(&c.sb).unwrap();
    let mut c_dev = dev.alloc_zeros::<u16>(n * m).unwrap();
    let mut cf_dev = dev.alloc_zeros::<f32>(if want_cf { n * m } else { 1 }).unwrap();
    dev.synchronize().unwrap();
    if want_cf {
        launch_tma(k_fn, &mut c_dev, &wt_dev, &sb_dev, &x_dev, &sa_dev, m, n, Some(&mut cf_dev), k, (m / 16) as u32);
    } else {
        launch_tma(k_fn, &mut c_dev, &wt_dev, &sb_dev, &x_dev, &sa_dev, m, n, None, k, (m / 16) as u32);
    }
    dev.synchronize().unwrap();
    let c_out = dev.dtoh_sync_copy(&c_dev).unwrap();
    let cf_out = if want_cf { Some(dev.dtoh_sync_copy(&cf_dev).unwrap()) } else { None };
    (c_out, cf_out)
}

fn assert_bitwise(a: &[u16], b: &[u16], tag: &str) {
    let bad = a.iter().zip(b).filter(|(u, v)| u != v).count();
    assert_eq!(bad, 0, "{tag}: tma != bsb bitwise at {bad}/{} elements", a.len());
}

// ---------------------------------------------------------------------------
// gates
// ---------------------------------------------------------------------------

#[test]
fn tma_bitwise_matches_bsb_synthetic() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_bsb(&dev);
    let kt = load_raw(&dev, "gemm_dsv4_fp8_bsb_tma");
    let mut rng = XorShift(0x71A0_0000_0001);
    for (m, k) in [(4096usize, 8192usize), (32768, 1024), (4096, 4096), (8192, 1024), (2048, 1024)] {
        let case = synth_case(&mut rng, m, k);
        for n in [1usize, 6, 16] {
            let (a, _) = run_bsb(&dev, &f, &case, n, false);
            let (b, _) = run_tma(&dev, &kt, &case, n, false);
            assert_bitwise(&a, &b, &format!("M={m} K={k} N={n} C"));
            let (_, acf) = run_bsb(&dev, &f, &case, n, true);
            let (_, bcf) = run_tma(&dev, &kt, &case, n, true);
            let acf = acf.unwrap();
            let bcf = bcf.unwrap();
            let bad = acf.iter().zip(&bcf).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
            assert_eq!(bad, 0, "M={m} K={k} N={n}: Cf tma != bsb bitwise at {bad}");
        }
        println!("M={m:6} K={k:6}: tma == bsb bitwise at N in {{1, 6, 16}} (C and Cf)");
    }
}

#[test]
fn tma_batch_invariance_col0() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let kt = load_raw(&dev, "gemm_dsv4_fp8_bsb_tma");
    let mut rng = XorShift(0x71A0_B1B1_0002);
    for (m, k) in [(2048usize, 1024usize), (1024, 4096)] {
        let case = synth_case(&mut rng, m, k);
        let (wide, _) = run_tma(&dev, &kt, &case, 16, false);
        let mut col0: Option<Vec<u16>> = None;
        for n in 1..=16usize {
            let (bits, _) = run_tma(&dev, &kt, &case, n, false);
            let this = bits[0..m].to_vec();
            match &col0 {
                None => col0 = Some(this),
                Some(prev) => assert_eq!(&this, prev, "M={m} K={k}: tma col0 differs at N={n}"),
            }
            for r in 0..n {
                assert_eq!(&bits[r * m..(r + 1) * m], &wide[r * m..(r + 1) * m],
                    "M={m} K={k}: tma row {r} differs between N={n} and N=16");
            }
        }
        println!("M={m:6} K={k:6}: tma col-0 bitwise N=1..=16 OK, full-prefix bitwise OK");
    }
}

#[test]
fn tma_real_wo_b_weights() {
    if !Path::new(BUNDLE).exists() {
        eprintln!("SKIP tma_real_wo_b_weights: {BUNDLE} not mounted");
        return;
    }
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_bsb(&dev);
    let kt = load_raw(&dev, "gemm_dsv4_fp8_bsb_tma");
    let mut rng = XorShift(0x71A0_EA11_0003);
    for layer in [0usize, 2] {
        for key in ["wq_a", "wq_b", "wkv", "wo_b"] {
            let case = real_case(&mut rng, layer, key);
            for n in [1usize, 6, 16] {
                let (a, _) = run_bsb(&dev, &f, &case, n, false);
                let (b, _) = run_tma(&dev, &kt, &case, n, false);
                assert_bitwise(&a, &b, &format!("layers.{layer}.attn.{key} N={n}"));
            }
            println!("layers.{layer}.attn.{key:5} M={} K={}: tma == bsb bitwise at N in {{1, 6, 16}}", case.m, case.k);
        }
    }
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

const NCOPY: usize = 4;
const REPS: usize = 40;

fn bench_tma(dev: &Arc<CudaDevice>, k_fn: &RawKernel, m: usize, k: usize, n: usize,
             wt: &[CudaSlice<u8>], sb: &CudaSlice<u8>, x: &CudaSlice<u8>, sa: &CudaSlice<u8>) -> f64 {
    let mut c_dev = dev.alloc_zeros::<u16>(n * m).unwrap();
    let grid = (m / 16) as u32;
    for i in 0..8 {
        launch_tma(k_fn, &mut c_dev, &wt[i % NCOPY], sb, x, sa, m, n, None, k, grid);
    }
    dev.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _round in 0..5 {
        let t0 = Instant::now();
        for i in 0..REPS {
            launch_tma(k_fn, &mut c_dev, &wt[i % NCOPY], sb, x, sa, m, n, None, k, grid);
        }
        dev.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    best
}

fn bench_bsb(dev: &Arc<CudaDevice>, f: &CudaFunction, m: usize, k: usize, n: usize,
             wt: &[CudaSlice<u8>], sb: &CudaSlice<u8>, x: &CudaSlice<u8>, sa: &CudaSlice<u8>, pair: bool) -> f64 {
    let mut c_dev = dev.alloc_zeros::<u16>(n * m).unwrap();
    let grid = if pair { m.div_ceil(32) as u32 } else { (m / 16) as u32 };
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    for i in 0..8 {
        unsafe { f.clone().launch(cfg, (&mut c_dev, &wt[i % NCOPY], sb, x, sa, m as i32, k as i32, n as i32, 0u64)).unwrap(); }
    }
    dev.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _round in 0..5 {
        let t0 = Instant::now();
        for i in 0..REPS {
            unsafe { f.clone().launch(cfg, (&mut c_dev, &wt[i % NCOPY], sb, x, sa, m as i32, k as i32, n as i32, 0u64)).unwrap(); }
        }
        dev.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    best
}

fn load_bsb2(dev: &Arc<CudaDevice>) -> CudaFunction {
    let ptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/gpu_batch.ptx").expect("src/ptx/gpu_batch.ptx"),
    );
    dev.load_ptx(ptx, "gpu_batch_b2", &["gemm_dsv4_fp8_bsb", "gemm_dsv4_fp8_bsb2"]).expect("load_ptx");
    dev.get_func("gpu_batch_b2", "gemm_dsv4_fp8_bsb2").expect("missing bsb2")
}

#[test]
fn bench_tma_vs_bsb() {
    if std::env::var("GB10_BENCH").is_err() {
        eprintln!("SKIP: set GB10_BENCH=1 to run the timing harness");
        return;
    }
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f1 = load_bsb(&dev);
    let f2 = load_bsb2(&dev);
    let kt = load_raw(&dev, "gemm_dsv4_fp8_bsb_tma");
    let mut rng = XorShift(0xBE11C_0000_0007);
    let n = 1usize;
    let filter = std::env::var("GB10_BENCH_SHAPE").ok();
    println!("{:>10} {:>6} {:>6} | {:>8} {:>6} | {:>8} {:>6} | {:>8} {:>6} | {:>7}", "name", "M", "K", "bsb us", "GB/s", "bsb2 us", "GB/s", "tma us", "GB/s", "DG us");
    for &(m, k, dg_us, name) in BENCH_SHAPES {
        if let Some(fl) = &filter {
            if fl != name { continue; }
        }
        let wcodes: Vec<u8> = (0..m * k).map(|_| rng.f32().to_bits() as u8).collect();
        let wt_host = quant::repack_fp8_mma(&wcodes, m, k);
        let sb_host = vec![126u8; (m / 128) * (k / 128)];
        let x_host = vec![0x32u8; n * k];
        let sa_host = vec![126u8; n * (k / 128)];
        let wt: Vec<CudaSlice<u8>> = (0..NCOPY).map(|_| dev.htod_sync_copy(&wt_host).unwrap()).collect();
        let sb = dev.htod_sync_copy(&sb_host).unwrap();
        let x = dev.htod_sync_copy(&x_host).unwrap();
        let sa = dev.htod_sync_copy(&sa_host).unwrap();
        dev.synchronize().unwrap();
        let bytes = (m * k + (m / 128) * (k / 128) + n * k + n * (k / 128)) as f64;
        let us1 = bench_bsb(&dev, &f1, m, k, n, &wt, &sb, &x, &sa, false);
        let us2 = bench_bsb(&dev, &f2, m, k, n, &wt, &sb, &x, &sa, true);
        let ust = bench_tma(&dev, &kt, m, k, n, &wt, &sb, &x, &sa);
        println!(
            "{:>10} {:>6} {:>6} | {:>8.1} {:>6.1} | {:>8.1} {:>6.1} | {:>8.1} {:>6.1} | {:>7.1}",
            name, m, k, us1, bytes / us1 / 1e3, us2, bytes / us2 / 1e3, ust, bytes / ust / 1e3, dg_us
        );
    }
}

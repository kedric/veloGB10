//! Tier-1 item 1.3: kernel-time microbench for the fp8_bsb GEMM family on the memo §3
//! shapes (DeepGEMM comparison table). Cold-weight streaming: launches rotate over
//! NCOPY distinct weight buffers (4 x 32 MB = 128 MB, far above L2) so each rep streams
//! weights from DRAM, matching the memo's "cold weights, kernel-time" methodology.
//!
//! CPU-only scaffolding: no model load. Run:
//!   cargo test --release --test dsv4_fp8_sched_bench -- --nocapture

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use gb10_inference::quant;
use std::sync::Arc;
use std::time::Instant;

// memo §3 shapes (M, K) + DeepGEMM's measured µs for reference
const SHAPES: &[(usize, usize, f64, &str)] = &[
    (32768, 1024, 155.0, "wq_b"),
    (4096, 8192, 178.0, "wo_b"),
    (4096, 4096, 84.0, "sh_gu"),
    (8192, 1024, 36.0, "idx_wq_b"),
];
const NCOPY: usize = 4;
const REPS: usize = 40;

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

fn load_fn(dev: &Arc<CudaDevice>, name: &str) -> CudaFunction {
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    let ptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/gpu_batch.ptx").expect("src/ptx/gpu_batch.ptx (cargo build first)"),
    );
    dev.load_ptx(ptx, "gpu_batch", &[name]).expect("load_ptx gpu_batch");
    dev.get_func("gpu_batch", name).expect("missing kernel")
}

/// Time `f` at (m, k, n) over REPS launches rotating NCOPY cold weight copies.
/// Returns median-of-5-rounds best µs/launch. `pair` selects the bsb2 grid.
fn bench(
    dev: &Arc<CudaDevice>, f: &CudaFunction, m: usize, k: usize, n: usize,
    wt: &[CudaSlice<u8>], sb: &CudaSlice<u8>, x: &CudaSlice<u8>, sa: &CudaSlice<u8>,
    pair: bool,
) -> f64 {
    let mut c_dev = dev.alloc_zeros::<u16>(n * m).unwrap();
    let grid = if pair { (m.div_ceil(32)) as u32 } else { (m / 16) as u32 };
    let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    // warmup
    for i in 0..8 {
        unsafe {
            f.clone().launch(cfg, (&mut c_dev, &wt[i % NCOPY], sb, x, sa, m as i32, k as i32, n as i32, 0u64)).unwrap();
        }
    }
    dev.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _round in 0..5 {
        let t0 = Instant::now();
        for i in 0..REPS {
            unsafe {
                f.clone().launch(cfg, (&mut c_dev, &wt[i % NCOPY], sb, x, sa, m as i32, k as i32, n as i32, 0u64)).unwrap();
            }
        }
        dev.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
        best = best.min(us);
    }
    best
}

/// Time a fused two-op pair launch (rotating NCOPY cold copies of BOTH weights); `single`
/// selects the 1-tile/CTA bsb1q grid, false the 2-tile/CTA bsb2q grid. Also times the two
/// separate bsb2 launches back-to-back for comparison. Returns (fused µs, separate µs).
#[allow(clippy::too_many_arguments)]
fn bench_pair(
    dev: &Arc<CudaDevice>, f2q: &CudaFunction, f2: &CudaFunction,
    m0: usize, m1: usize, k: usize, n: usize,
    wt0: &[CudaSlice<u8>], sb0: &CudaSlice<u8>, wt1: &[CudaSlice<u8>], sb1: &CudaSlice<u8>,
    x: &CudaSlice<u8>, sa: &CudaSlice<u8>, single: bool,
) -> (f64, f64) {
    let mut c0 = dev.alloc_zeros::<u16>(n * m0).unwrap();
    let mut c1 = dev.alloc_zeros::<u16>(n * m1).unwrap();
    let m01: u64 = (m0 as u64) | ((m1 as u64) << 32);
    let kn: u64 = (k as u64) | ((n as u64) << 32);
    let packs = if single { m0 / 16 + m1 / 16 } else { m0.div_ceil(32) + m1.div_ceil(32) } as u32;
    let cfg_q = LaunchConfig { grid_dim: (packs, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let cfg0 = LaunchConfig { grid_dim: (m0.div_ceil(32) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let cfg1 = LaunchConfig { grid_dim: (m1.div_ceil(32) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    // warmup
    for i in 0..8 {
        unsafe {
            f2q.clone().launch(cfg_q, (&mut c0, &wt0[i % NCOPY], sb0, &mut c1, &wt1[i % NCOPY], sb1, x, sa, m01, kn, 0u64)).unwrap();
        }
    }
    dev.synchronize().unwrap();
    let mut best_q = f64::INFINITY;
    let mut best_sep = f64::INFINITY;
    for _round in 0..5 {
        let t0 = Instant::now();
        for i in 0..REPS {
            unsafe {
                f2q.clone().launch(cfg_q, (&mut c0, &wt0[i % NCOPY], sb0, &mut c1, &wt1[i % NCOPY], sb1, x, sa, m01, kn, 0u64)).unwrap();
            }
        }
        dev.synchronize().unwrap();
        best_q = best_q.min(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
        let t0 = Instant::now();
        for i in 0..REPS {
            unsafe {
                f2.clone().launch(cfg0, (&mut c0, &wt0[i % NCOPY], sb0, x, sa, m0 as i32, k as i32, n as i32, 0u64)).unwrap();
                f2.clone().launch(cfg1, (&mut c1, &wt1[i % NCOPY], sb1, x, sa, m1 as i32, k as i32, n as i32, 0u64)).unwrap();
            }
        }
        dev.synchronize().unwrap();
        best_sep = best_sep.min(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64);
    }
    (best_q, best_sep)
}

/// Tier-1 item 1.4: ramp-class pair bench — the decode step's fused wq_a [1024,4096] +
/// wkv [512,4096] launch (bsb2q) vs the two separate bsb2 launches, plus each op alone at
/// bsb/bsb2 grid. Same cold-weight methodology as bench_fp8_bsb_shapes.
#[test]
fn bench_fp8_bsb2q_pair() {
    if std::env::var("GB10_BENCH").is_err() {
        eprintln!("SKIP: set GB10_BENCH=1 to run the timing harness");
        return;
    }
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f1 = load_fn(&dev, "gemm_dsv4_fp8_bsb");
    let f2 = load_fn(&dev, "gemm_dsv4_fp8_bsb2");
    let f2q = load_fn(&dev, "gemm_dsv4_fp8_bsb2q");
    let f1q = load_fn(&dev, "gemm_dsv4_fp8_bsb1q");
    let mut rng = XorShift(0xBE11C_0000_0002);
    let n = 1usize;
    let (m0, m1, k) = (1024usize, 512usize, 4096usize);
    let mk = |m: usize, rng: &mut XorShift| {
        let wcodes: Vec<u8> = (0..m * k).map(|_| rng.f32().to_bits() as u8).collect();
        (quant::repack_fp8_mma(&wcodes, m, k), vec![126u8; (m / 128) * (k / 128)])
    };
    let (wt0_host, sb0_host) = mk(m0, &mut rng);
    let (wt1_host, sb1_host) = mk(m1, &mut rng);
    let wt0: Vec<CudaSlice<u8>> = (0..NCOPY).map(|_| dev.htod_sync_copy(&wt0_host).unwrap()).collect();
    let wt1: Vec<CudaSlice<u8>> = (0..NCOPY).map(|_| dev.htod_sync_copy(&wt1_host).unwrap()).collect();
    let sb0 = dev.htod_sync_copy(&sb0_host).unwrap();
    let sb1 = dev.htod_sync_copy(&sb1_host).unwrap();
    let x = dev.htod_sync_copy(&vec![0x32u8; n * k]).unwrap();
    let sa = dev.htod_sync_copy(&vec![126u8; n * (k / 128)]).unwrap();
    dev.synchronize().unwrap();

    let bytes = |m: usize| (m * k + (m / 128) * (k / 128) + n * k + n * (k / 128)) as f64;
    let us0_1 = bench(&dev, &f1, m0, k, n, &wt0, &sb0, &x, &sa, false);
    let us0_2 = bench(&dev, &f2, m0, k, n, &wt0, &sb0, &x, &sa, true);
    let us1_1 = bench(&dev, &f1, m1, k, n, &wt1, &sb1, &x, &sa, false);
    let us1_2 = bench(&dev, &f2, m1, k, n, &wt1, &sb1, &x, &sa, true);
    let (us_q, us_sep) = bench_pair(&dev, &f2q, &f2, m0, m1, k, n, &wt0, &sb0, &wt1, &sb1, &x, &sa, false);
    let (us_1q, _) = bench_pair(&dev, &f1q, &f2, m0, m1, k, n, &wt0, &sb0, &wt1, &sb1, &x, &sa, true);
    let tot = bytes(m0) + bytes(m1);
    println!("{:>10} {:>6} {:>6} | {:>9} {:>7} | {:>9} {:>7}", "name", "M", "K", "bsb us", "GB/s", "bsb2 us", "GB/s");
    println!("{:>10} {:>6} {:>6} | {:>9.1} {:>7.1} | {:>9.1} {:>7.1}", "wq_a", m0, k, us0_1, bytes(m0) / us0_1 / 1e3, us0_2, bytes(m0) / us0_2 / 1e3);
    println!("{:>10} {:>6} {:>6} | {:>9.1} {:>7.1} | {:>9.1} {:>7.1}", "wkv", m1, k, us1_1, bytes(m1) / us1_1 / 1e3, us1_2, bytes(m1) / us1_2 / 1e3);
    println!("pair total {:.0} MB: bsb2q {:.1} us ({:.1} GB/s) | bsb1q {:.1} us ({:.1} GB/s) | separate bsb2 x2 {:.1} us ({:.1} GB/s)",
        tot / 1e6, us_q, tot / us_q / 1e3, us_1q, tot / us_1q / 1e3, us_sep, tot / us_sep / 1e3);
}

#[test]
fn bench_fp8_bsb_shapes() {
    if std::env::var("GB10_BENCH").is_err() {
        eprintln!("SKIP: set GB10_BENCH=1 to run the timing harness");
        return;
    }
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f1 = load_fn(&dev, "gemm_dsv4_fp8_bsb");
    let f2 = load_fn(&dev, "gemm_dsv4_fp8_bsb2");
    let mut rng = XorShift(0xBE11C_0000_0001);
    let n = 1usize;
    println!("{:>10} {:>6} {:>6} | {:>9} {:>7} | {:>9} {:>7} | {:>8}", "name", "M", "K", "bsb us", "GB/s", "bsb2 us", "GB/s", "DG us");
    let filter = std::env::var("GB10_BENCH_SHAPE").ok();
    for &(m, k, dg_us, name) in SHAPES {
        if let Some(f) = &filter {
            if f != name { continue; }
        }
        // synthetic fp8 codes (uniform bytes) + repack; values don't matter for timing
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
        let us1 = bench(&dev, &f1, m, k, n, &wt, &sb, &x, &sa, false);
        let us2 = bench(&dev, &f2, m, k, n, &wt, &sb, &x, &sa, true);
        println!(
            "{:>10} {:>6} {:>6} | {:>9.1} {:>7.1} | {:>9.1} {:>7.1} | {:>8.1}",
            name, m, k, us1, bytes / us1 / 1e3, us2, bytes / us2 / 1e3, dg_us
        );
    }
}

//! Minimal CUDA-graph repro (isolates the tok-1 INVALID_VALUE): capture a trivial
//! kernel on the compute stream WITH GSlice-style stream-ordered allocs inside,
//! instantiate, then launch repeatedly with a param update between launches.
//! If launch #2 fails here, it's a driver-mechanism issue (30-line repro, reportable);
//! if it passes, the failure is content-specific to the full forward capture.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use gb10_inference::dsv4_gpu::{self, Dsv4Kernels};

fn gate() -> std::sync::MutexGuard<'static, ()> {
    static G: std::sync::Mutex<()> = std::sync::Mutex::new(());
    G.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn graph_minimal_two_launches() {
    let _g = gate();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_iota_b"]).expect("load");
    // Pre-warm the dedicated graph mempool OUTSIDE capture (cuMemPoolCreate mid-capture is
    // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED; the production arm creates it before arming).
    let _pool = dsv4_gpu::graph_mempool(&dev);
    use cudarc::driver::sys;
    unsafe {
        // capture: one iota launch with GSlice allocs inside (the exact pattern the
        // decode forward uses)
        let r = sys::cuStreamBeginCapture_v2(stream.stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "BeginCapture: {r:?}");
        let out = gb10_inference::dsv4_gpu::GSlice::<i32>::alloc_on(&dev, stream.stream, 16).expect("alloc");
        let (start, mul, div, n) = (7i32, 1i32, 1i32, 16i32);
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            out.device_ptr() as *const sys::CUdeviceptr as *mut _,
            &start as *const i32 as *mut _,
            &mul as *const i32 as *mut _,
            &div as *const i32 as *mut _,
            &n as *const i32 as *mut _,
        ];
        let f = ks.func("dsv4_iota_b").unwrap();
        let r = cudarc::driver::result::launch_kernel(f, (1, 1, 1), (256, 1, 1), 0, stream.stream, &mut params);
        assert!(r.is_ok(), "captured launch: {r:?}");
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = sys::cuStreamEndCapture(stream.stream, &mut graph);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "EndCapture: {r:?}");
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        // CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH (0x1): alloc nodes re-execute
        // per launch; without auto-free, launch #1's alloc node collides with launch #0's
        // still-live allocation (the observed INVALID_VALUE — the no-allocs repro passes).
        let r = sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0x1);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "Instantiate: {r:?}");
        // launch THREE times; between launches, update the start param via the node.
        for i in 0..3 {
            let r = sys::cuGraphLaunch(exec, stream.stream);
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch #{i}: {r:?}");
            dev.synchronize().unwrap();
            let got: Vec<i32> = out.dtoh_sync().unwrap();
            assert_eq!(got[0], 7 + i * 100, "replay #{i} wrote start=7+{}00 (SetParams applies to replays)", i);
            // param update for the NEXT launch: find the kernel node and bump start.
            if i < 2 {
                let mut n_nodes: usize = 0;
                sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n_nodes);
                let mut nodes = vec![std::ptr::null_mut::<sys::CUgraphNode_st>(); n_nodes];
                sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n_nodes);
                let mut updated = false;
                for &node in &nodes {
                    let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
                    sys::cuGraphNodeGetType(node, &mut ty);
                    if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL { continue; }
                    let mut p: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
                    sys::cuGraphKernelNodeGetParams(node, &mut p);
                    if p.func != f { continue; }
                    // slot 1 = start
                    let sp = *p.kernelParams.add(1);
                    *(sp as *mut i32) = 7 + (i + 1) * 100;
                    let r = sys::cuGraphExecKernelNodeSetParams(exec, node, &p);
                    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "SetParams #{i}: {r:?}");
                    updated = true;
                }
                assert!(updated, "kernel node not found for param update");
            }
        }
        let got: Vec<i32> = out.dtoh_sync().unwrap();
        assert_eq!(got[0], 207, "third replay should use start=207, got {}", got[0]);
        sys::cuGraphExecDestroy(exec);
        sys::cuGraphDestroy(graph);
        eprintln!("[graph-min] PASS: 3 launches + 2 param updates on a GSlice-alloc graph");
    }
}

/// Same as graph_minimal_two_launches but with NO allocations inside the capture
/// (output buffer pre-allocated eagerly) — separates "two launches broken" from
/// "graph memory nodes broken".
#[test]
fn graph_minimal_two_launches_no_allocs() {
    let _g = gate();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_iota_b"]).expect("load");
    use cudarc::driver::sys;
    unsafe {
        let out = dev.alloc_zeros::<i32>(16).unwrap();
        let r = sys::cuStreamBeginCapture_v2(stream.stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "BeginCapture: {r:?}");
        let (start, mul, div, n) = (11i32, 1i32, 1i32, 16i32);
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            out.device_ptr() as *const sys::CUdeviceptr as *mut _,
            &start as *const i32 as *mut _,
            &mul as *const i32 as *mut _,
            &div as *const i32 as *mut _,
            &n as *const i32 as *mut _,
        ];
        let f = ks.func("dsv4_iota_b").unwrap();
        let r = cudarc::driver::result::launch_kernel(f, (1, 1, 1), (256, 1, 1), 0, stream.stream, &mut params);
        assert!(r.is_ok(), "captured launch: {r:?}");
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = sys::cuStreamEndCapture(stream.stream, &mut graph);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "EndCapture: {r:?}");
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "Instantiate: {r:?}");
        for i in 0..3 {
            let r = sys::cuGraphLaunch(exec, stream.stream);
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch #{i}: {r:?}");
            dev.synchronize().unwrap();
        }
        let got = dev.dtoh_sync_copy(&out).unwrap();
        assert_eq!(got[0], 11, "captured value present: {}", got[0]);
        sys::cuGraphExecDestroy(exec);
        sys::cuGraphDestroy(graph);
        eprintln!("[graph-min-noalloc] PASS: 3 launches, no in-capture allocs");
    }
}

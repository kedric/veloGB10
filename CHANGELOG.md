# Changelog

High-level release notes for veloGB10. Minor bug fixes and small optimizations are grouped under
generic language where they aren't individually notable.

## Unreleased — Qwen3.8-Flash-Next (qwen4_exp)

- **Qwen3.8-Flash-Next support** (`model_type: qwen4_exp`, 176B-A10B): hyper-connection residual
  streams, PLE n-gram injection, sigmoid-gated GatedDeltaNet, MoE 512×10 + shared expert, and its
  MTP head — served by the regular `GpuModel` engine (server, batching, prefix cache, verify).
- **Everything NVFP4**, including the 320M-row PLE n-gram table, quantized by `--quantize --recipe
  all` into 96-byte row records (`ple_ngram_nvfp4.bin`) that the engine keeps on the GPU or
  streams from the SSD (`--ple-offload ssd`, bit-identical).
- New quantizer groups `hc`, `ple`, `pletable`; 4 GB output shards (`GB10_QUANT_SHARD_GB`).
- Host-memory watchdog (`GB10_MEM_WATCHDOG_GB`) and an exact load-time memory guard for this
  family; `GB10_LOAD_FORCE` no longer bypasses it (`=unsafe` does).
- `--probe-q4` (prefill + greedy decode, optional logits dump) and `scripts/qwen4exp/` (HF
  reference oracle on a synthetic model, quantization round-trip check).
- **QSA sparse attention** (`Qwen4ExpTextQSAIndexer`): past 2051 visible tokens every attention
  layer (and the MTP head) attends to the 512 best-scoring 4-token blocks + tail, selected by a
  deterministic radix top-k in the verify kernels' rank space — MTP stays lossless. Raw indexer
  keys are cached per position like the KV; below the limit the dense kernels are unchanged.
  `GB10_Q4_DENSE_ATTN=1` (A/B) forces dense; `GB10_QSA_DUMP=1` dumps selections for the oracle
  check (`scripts/qwen4exp/compare_qsa.py`).
- Limits: no TP, no vision for this family; QSA needs a bf16 KV cache.

## v0.5.0 — Vision support

- **Vision support.** Image input is now supported end-to-end on a GPU vision tower
  (`gpu_vision` kernels), with a `--vision-cpu` escape hatch to the CPU reference path. PNG/JPEG/WebP/GIF
  decoding added. The engine now ships and requires the `gpu_vision.ptx` kernel artifact in addition
  to the existing PTX set.
- **Better tool-call support.** A single canonical serializer now handles streaming and
  non-streaming tool-call output identically, repairs malformed tool-call tags, and no longer drops
  or leaks text around tool-call boundaries. New tool-call compliance and serializer test suites.
- **Prefill/TTFT optimizations.** New opt-in prefill levers (tensor-core flash-attention prefill,
  v2 W4A4 prefill GEMM, GDN tensor-core chunked scan), all env-gated **default off**, so the default
  serving path is unchanged. Minor bug fixes and optimizations.
- **Model-id fix.** `/v1/models` and responses now report the model card's `base_model`
  (e.g. `Qwen/Qwen3.8-27B`) instead of a local directory fragment. `--model-name` still overrides.

## v0.4.2

- Fix: accept OpenAI multipart `content` (string | array | null) to unblock agent clients that send
  content parts; request-schema only.

## v0.4.1

- Fix: `--draft-dir` is now mandatory only when `--spec-source` explicitly names a DFlash2 mode;
  plain-MTP launches no longer require it.

## v0.4.0

- **Qwen3.8 27B NVFP4** support with native **DFlash 2** speculative decoding, full 256K context.
- **TP=4** serving (plus TP=2 and single-node).
- New DSV4 / DFlash2 / DSpark / MXFP4 kernel set.
- README Update section with the Qwen3.8 27B performance table and live throughput traces; new
  `QWEN_27B_SETUP.md` and `MANAGING_CACHE.md` docs.

## v0.3.1

- **KAT-Coder** model support; supported-models table in the README.

## v0.3.0

- README generalization and load-pipeline features. Minor bug fixes and optimizations.

## v0.2.0

- **Tencent Hy3 (hy_v3)** family support, 4-bit KV cache, FR-Spec draft head, model-name family fix.

## v0.1.0

- Initial public release: from-scratch Rust + CUDA engine for Qwen3.5/3.6 on single and TP=2 GB10,
  with NVFP4/FP8 quantization, MTP speculative decoding, an OpenAI-compatible server, and prebuilt
  release binaries.

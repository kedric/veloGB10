# DFlash2 MR-GPTQ/NVFP4

`--gptq-dflash2` quantizes the 35 large projections of the five-layer DFlash2 drafter:
q/k/v/o and gate/up/down. `fc`, convolution projections, norms and selector codebooks stay BF16.
The target model is loaded with `GB10_W4A4_PREFILL=attn,mlp,gdn`, so the cached target taps match
the production prefill profile. Draft rounds themselves use MR-NVFP4 weights with W4A16 inputs.

The pass is sequential. A completed layer is installed before the next layer is calibrated. Target
taps are reduced once to `[5120,seqlen]` and cached beside the output, and each completed layer gets
an atomic checkpoint. Re-running the same command resumes valid cache files and layer checkpoints.

```bash
cargo build --release

target/release/gb10_inference --gptq-dflash2 \
  --draft-dir "$HOME/models/Qwen3.8-27B-DFlash2" \
  --model-dir "$HOME/models/Qwen3.8-27B-MR-GPTQ-NVFP4-v5" \
  --out "$HOME/models/Qwen3.8-27B-DFlash2-MR-GPTQ-NVFP4-v1" \
  --calib "$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl" \
  --nsamples 512 --seqlen 2048 \
  --damp 0.01 --clip 7 --rotate \
  --scale-iters 4 --df2-context-vectors 16
```

The temporary cache is `<out>.calib-cache` (roughly 10 GiB for 512×2048). It is removed after a
successful artifact write and retained after failure so the same command can resume.

Serve on one GB10 with the quantized directory as `--draft-dir`:

```bash
GB10_W4A4_PREFILL=attn,mlp,gdn \
target/release/gb10_inference --server \
  --model-dir "$HOME/models/Qwen3.8-27B-MR-GPTQ-NVFP4-v5" \
  --draft-dir "$HOME/models/Qwen3.8-27B-DFlash2-MR-GPTQ-NVFP4-v1" \
  --spec-source dflash2-auto --mtp auto \
  --max-seq-len 226114 --max-batch 2 --prefix-cache on
```

Quantized DFlash2 round sharding is intentionally rejected for now. Use the artifact on the
single-GB10 round path; BF16 DFlash2 retains its existing sharded path.

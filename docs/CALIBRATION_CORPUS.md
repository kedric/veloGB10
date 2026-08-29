# Calibration corpus v2

The generator builds the corpus from verified raw sources and never overwrites an existing
output. Its main output contains one exact, pre-tokenized sample per JSONL record; GPTQ consumes
`input_ids` directly, so the manifest's token percentages are the percentages used by calibration.

## Generate from zero

```bash
cd ~/workspace/veloGB10

scripts/generate_calibration_corpus.sh \
  "$HOME/models/Qwen3.8-27B" \
  "$HOME/models/calibration-sources/qwen38-calibration-v3.jsonl"
```

Optional environment variables:

- `EXCLUDE_JSONL=/path/to/held-out-benchmark.jsonl` removes exact and near-duplicate benchmark
  material;
- `VISION_DIR=/path/to/representative/images` adds local PNG/JPEG/WebP images;
- `NSAMPLES`, `SEQLEN`, `LONG_NSAMPLES`, and `LONG_SEQLEN` override the defaults.

The default outputs are:

- `qwen38-calibration-v3.jsonl`: 512 x 2048, exact 15% general, 15% long multi-turn,
  25% code, 25% multilingual, 15% tools/structured, and 5% prompt-injection defense;
- `qwen38-calibration-v3.long-8192.jsonl`: 64 x 8192 long-context samples;
- `qwen38-calibration-v3.jsonl.sources/vision_multimodal.jsonl`: optional raw multimodal pool;
- one manifest beside each composed corpus plus `sources.manifest.json` for source hashes,
  licensing metadata, deduplication counts, languages, scenarios, and code-language coverage.

Prompt-injection examples protect activation coverage for that traffic. Calibration is not
fine-tuning: it cannot teach a missing safety behavior or guarantee resistance to attacks.

## Main MR-GPTQ pass

Use the main 512 x 2048 corpus for the layer-wise Hessian pass. For the dense 27B recipe:

```bash
CUDA_VISIBLE_DEVICES=0 GB10_PLE_OFFLOAD=ssd \
./target/release/gb10_inference --gptq \
  --model-dir "$SRC" \
  --base "$BASE" \
  --out "$FINAL" \
  --calib "$HOME/models/calibration-sources/qwen38-calibration-v3.jsonl" \
  --nsamples 512 \
  --seqlen 2048 \
  --damp 0.01 \
  --clip 7 \
  --rotate \
  --scale-iters 4 \
  --gptq-groups attn,mlp,gdn,lmhead \
  --rtn-groups mtp,embed
```

Static activation-order GPTQ is enabled by default. Do not pass `--no-act-order` for this recipe.

## Optional W4A4 activation passes

These passes do not re-run GPTQ and do not modify weights. They collect served-path activation
maxima for the main mix, long context, and actual image embeddings:

```bash
mkdir -p /tmp/qwen38-igs

./target/release/gb10_inference --calib-igs \
  --model-dir "$FINAL" --out /tmp/qwen38-igs/main \
  --calib "$HOME/models/calibration-sources/qwen38-calibration-v3.jsonl" \
  --nsamples 512 --seqlen 2048

./target/release/gb10_inference --calib-igs \
  --model-dir "$FINAL" --out /tmp/qwen38-igs/long \
  --calib "$HOME/models/calibration-sources/qwen38-calibration-v3.long-8192.jsonl" \
  --nsamples 64 --seqlen 8192

VISION="$HOME/models/calibration-sources/qwen38-calibration-v3.jsonl.sources/vision_multimodal.jsonl"
VISION_NSAMPLES=$(wc -l < "$VISION" | tr -d ' ')
if [ "$VISION_NSAMPLES" -gt 0 ]; then
  ./target/release/gb10_inference --calib-igs \
    --model-dir "$FINAL" --out /tmp/qwen38-igs/vision \
    --calib "$VISION" --nsamples "$VISION_NSAMPLES" --seqlen 2048
fi
```

Merge the resulting scales conservatively. Since `input_global_scale = 2688 / amax`, taking the
minimum scale per tensor is exactly equivalent to retaining the maximum activation observed in
the union of all calibration domains:

```bash
python3 scripts/merge_igs_scales.py \
  --output "$FINAL/input_global_scale.json" \
  /tmp/qwen38-igs/main/input_global_scale.json \
  /tmp/qwen38-igs/long/input_global_scale.json \
  /tmp/qwen38-igs/vision/input_global_scale.json
```

Omit the vision input when no representative vision corpus is available.

## Domain audit

Run this on an existing quantized artifact when comparing calibration domains. It does not alter
the artifact:

```bash
scripts/audit_calibration_igs.sh \
  "$FINAL" \
  "$HOME/models/calibration-sources/qwen38-calibration-v3.jsonl" \
  /tmp/qwen38-domain-audit
```

`report.json` reconstructs per-domain activation maxima and flags tensors whose largest domain
maximum is at least 1.5 times the smallest. This is a coverage diagnostic, not an accuracy score;
quality still needs held-out perplexity/task evaluation and the user's serving benchmark.

#!/usr/bin/env python3
"""Merge NVFP4 activation histograms before deriving W4A4 input global scales."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

HIST_BINS = 512
LOG2_MIN = -40.0
LOG2_MAX = 40.0
E4M3_NORMAL_RANGE = 28672.0
ANCHOR_FLOOR_RATIO = 1.0e6
RECIPROCAL_NUMERATOR = 6.0 * 448.0


def stats_path_for(scale_path: Path) -> Path:
    return scale_path.with_name(f"{scale_path.stem}.stats.json")


def bin_index(value: float) -> int:
    frac = (math.log2(value) - LOG2_MIN) / (LOG2_MAX - LOG2_MIN)
    return max(0, min(HIST_BINS - 1, math.floor(frac * HIST_BINS)))


def bin_center(index: int) -> float:
    log2_value = LOG2_MIN + (index + 0.5) / HIST_BINS * (LOG2_MAX - LOG2_MIN)
    return 2.0**log2_value


def percentile(histogram: list[int], value: float, floor_value: float | None = None) -> float | None:
    start = bin_index(floor_value) if floor_value is not None and floor_value > 0 else 0
    total = sum(histogram[start:])
    if total <= 0:
        return None
    target = value / 100.0 * total
    cumulative = 0
    for index in range(start, HIST_BINS):
        cumulative += histogram[index]
        if cumulative >= target:
            return bin_center(index)
    return None


def validate_policy(method: str, anchor_percentile: float, upper_percentile: float, rho: float) -> None:
    if method not in {"max", "headroom"}:
        raise ValueError(f"unknown method {method!r}")
    if not 0.0 < anchor_percentile <= 100.0:
        raise ValueError("anchor percentile must be in (0, 100]")
    if not 0.0 < upper_percentile <= 100.0:
        raise ValueError("upper percentile must be in (0, 100]")
    if not 0.0 < rho < E4M3_NORMAL_RANGE:
        raise ValueError(f"rho must be in (0, {E4M3_NORMAL_RANGE:g})")


def derive_scale(
    histogram: list[int],
    running_max: float,
    method: str,
    anchor_percentile: float,
    upper_percentile: float,
    rho: float,
) -> dict[str, float | bool | None]:
    if len(histogram) != HIST_BINS:
        raise ValueError(f"expected {HIST_BINS} histogram bins, got {len(histogram)}")
    if not math.isfinite(running_max) or running_max <= 0:
        raise ValueError(f"invalid running max {running_max}")

    if method == "max":
        return {
            "anchor": None,
            "upper": running_max,
            "span": None,
            "selected_amax": running_max,
            "input_global_scale": RECIPROCAL_NUMERATOR / running_max,
            "has_headroom": False,
            "range_exceeds_e4m3": False,
        }

    upper = (
        running_max
        if upper_percentile >= 100.0
        else percentile(histogram, upper_percentile) or running_max
    )
    anchor = percentile(histogram, anchor_percentile, upper / ANCHOR_FLOOR_RATIO)
    if anchor is None or not math.isfinite(anchor) or anchor <= 0:
        anchor = None
        selected_amax = running_max
        span = None
        has_headroom = False
        range_exceeds = False
    else:
        span = upper / anchor
        selected_amax = max(upper, rho * anchor)
        has_headroom = rho * anchor > upper
        range_exceeds = span > E4M3_NORMAL_RANGE

    return {
        "anchor": anchor,
        "upper": upper,
        "span": span,
        "selected_amax": selected_amax,
        "input_global_scale": RECIPROCAL_NUMERATOR / selected_amax,
        "has_headroom": has_headroom,
        "range_exceeds_e4m3": range_exceeds,
    }


def load_stats(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "veloGB10-igs-hist-v2":
        raise ValueError(f"{path}: unsupported stats format {document.get('format')!r}")
    histogram = document.get("histogram", {})
    expected = {"bins": HIST_BINS, "log2_min": LOG2_MIN, "log2_max": LOG2_MAX}
    if histogram != expected or document.get("block_size") != 16:
        raise ValueError(f"{path}: incompatible histogram geometry")
    if not isinstance(document.get("stems"), dict):
        raise ValueError(f"{path}: missing stems object")
    return document


def merge_histogram_stats(args: argparse.Namespace, stats_paths: list[Path]) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    documents = [load_stats(path) for path in stats_paths]
    first_policy = documents[0].get("policy", {})
    method = (
        str(first_policy.get("method", "headroom")) if args.method == "auto" else args.method
    )
    anchor_percentile = (
        args.anchor_percentile
        if args.anchor_percentile is not None
        else float(first_policy.get("anchor_percentile", 1.0))
    )
    upper_percentile = (
        args.upper_percentile
        if args.upper_percentile is not None
        else float(first_policy.get("upper_percentile", 99.99))
    )
    rho = args.rho if args.rho is not None else float(first_policy.get("rho", 16384.0))
    validate_policy(method, anchor_percentile, upper_percentile, rho)

    merged_stats: dict[str, dict[str, Any]] = {}
    for document in documents:
        for stem, raw in document["stems"].items():
            histogram = raw.get("histogram")
            if not isinstance(histogram, list) or len(histogram) != HIST_BINS:
                raise ValueError(f"{stem}: malformed histogram")
            if any(not isinstance(count, int) or count < 0 for count in histogram):
                raise ValueError(f"{stem}: histogram counts must be non-negative integers")
            running_max = float(raw.get("running_max", 0.0))
            invalid_blocks = int(raw.get("invalid_blocks", 0))
            if invalid_blocks:
                raise ValueError(f"{stem}: {invalid_blocks} non-finite activation blocks")
            target = merged_stats.setdefault(stem, {
                "histogram": [0] * HIST_BINS,
                "running_max": 0.0,
                "zero_blocks": 0,
                "invalid_blocks": 0,
            })
            target["histogram"] = [
                left + right for left, right in zip(target["histogram"], histogram, strict=True)
            ]
            target["running_max"] = max(target["running_max"], running_max)
            target["zero_blocks"] += int(raw.get("zero_blocks", 0))
            target["invalid_blocks"] += invalid_blocks

    scales: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    wide_stems: list[str] = []
    unfed_stems: list[str] = []
    for stem, stats in sorted(merged_stats.items()):
        nonzero_blocks = sum(stats["histogram"])
        if stats["running_max"] == 0.0 and nonzero_blocks == 0:
            unfed_stems.append(stem)
            continue

        diag = derive_scale(
            stats["histogram"], stats["running_max"], method,
            anchor_percentile, upper_percentile, rho,
        )
        scales[stem] = float(diag["input_global_scale"])
        if diag["range_exceeds_e4m3"]:
            wide_stems.append(stem)
        diagnostics[stem] = {
            **stats,
            "nonzero_blocks": nonzero_blocks,
            **diag,
        }

    stats_document = {
        "format": "veloGB10-igs-hist-v2",
        "scale_convention": "input_global_scale = 2688 / activation_amax",
        "block_size": 16,
        "histogram": {"bins": HIST_BINS, "log2_min": LOG2_MIN, "log2_max": LOG2_MAX},
        "policy": {
            "method": method,
            "anchor_percentile": anchor_percentile,
            "upper_percentile": upper_percentile,
            "rho": rho,
        },
        "merged_from": [str(path) for path in stats_paths],
        "stems": diagnostics,
    }
    manifest = {
        "format": "veloGB10-igs-merge-v2",
        "rule": "sum per-16 block-amax histograms, then derive one global scale per stem",
        "method": method,
        "anchor_percentile": anchor_percentile,
        "upper_percentile": upper_percentile,
        "rho": rho,
        "inputs": [str(path) for path in args.inputs],
        "stats_inputs": [str(path) for path in stats_paths],
        "stems": len(scales),
        "range_exceeds_e4m3_stems": wide_stems,
        "unfed_stems": unfed_stems,
    }
    return scales, stats_document, manifest


def merge_legacy_scales(paths: list[Path]) -> tuple[dict[str, float], dict[str, Any]]:
    merged: dict[str, float] = {}
    provenance: dict[str, str] = {}
    for path in paths:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"{path} is not a JSON object")
        for stem, raw_scale in values.items():
            if not isinstance(raw_scale, (int, float)):
                continue
            scale = float(raw_scale)
            if not math.isfinite(scale) or scale <= 0:
                continue
            if stem not in merged or scale < merged[stem]:
                merged[stem] = scale
                provenance[stem] = str(path)
    manifest = {
        "format": "veloGB10-igs-merge-v1",
        "rule": "legacy minimum input_global_scale = maximum observed activation amax",
        "inputs": [str(path) for path in paths],
        "stems": len(merged),
        "selected_from": provenance,
    }
    return merged, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("auto", "headroom", "max"), default="auto")
    parser.add_argument("--anchor-percentile", type=float)
    parser.add_argument("--upper-percentile", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    manifest_path = Path(str(args.output) + ".manifest.json")
    merged_stats_path = stats_path_for(args.output)
    for path in (args.output, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    stats_paths = [stats_path_for(path) for path in args.inputs]
    present = [path.exists() for path in stats_paths]
    stats_document: dict[str, Any] | None = None
    if all(present):
        scales, stats_document, manifest = merge_histogram_stats(args, stats_paths)
        if merged_stats_path.exists():
            raise FileExistsError(f"refusing to overwrite {merged_stats_path}")
    elif any(present):
        missing = [str(path) for path, exists in zip(stats_paths, present, strict=True) if not exists]
        raise FileNotFoundError(f"partial histogram inputs; missing: {', '.join(missing)}")
    else:
        if args.method != "auto" or any(
            value is not None for value in (args.anchor_percentile, args.upper_percentile, args.rho)
        ):
            raise ValueError("headroom/max policy options require input_global_scale.stats.json inputs")
        scales, manifest = merge_legacy_scales(args.inputs)

    if not scales:
        raise ValueError("no valid scales found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scales, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if stats_document is not None:
        merged_stats_path.write_text(
            json.dumps(stats_document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"[igs-merge] {len(scales)} scales ({manifest['format']}) -> {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare per-category W4A4 input-scale calibration results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


NVFP4_ACTIVATION_NUMERATOR = 6.0 * 448.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warn-ratio", type=float, default=1.5)
    args = parser.parse_args()

    categories: dict[str, dict[str, float]] = {}
    for path in sorted(args.root.glob("*/input_global_scale.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"{path} is not a JSON object")
        categories[path.parent.name] = {
            stem: float(scale) for stem, scale in values.items()
            if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0
        }
    if len(categories) < 2:
        raise ValueError(f"need at least two category results below {args.root}")

    all_stems = sorted(set().union(*(values.keys() for values in categories.values())))
    comparisons = []
    for stem in all_stems:
        activation_amax = {
            category: NVFP4_ACTIVATION_NUMERATOR / values[stem]
            for category, values in categories.items() if stem in values
        }
        if len(activation_amax) < 2:
            continue
        low_category, low = min(activation_amax.items(), key=lambda item: item[1])
        high_category, high = max(activation_amax.items(), key=lambda item: item[1])
        ratio = high / low if low > 0 else math.inf
        comparisons.append({
            "stem": stem,
            "amax_by_category": activation_amax,
            "max_over_min": ratio,
            "lowest_category": low_category,
            "highest_category": high_category,
        })

    ratios = [item["max_over_min"] for item in comparisons if math.isfinite(item["max_over_min"])]
    flagged = sorted(
        (item for item in comparisons if item["max_over_min"] >= args.warn_ratio),
        key=lambda item: (-item["max_over_min"], item["stem"]),
    )
    report = {
        "format": "veloGB10-igs-domain-audit-v1",
        "note": "amax is reconstructed as 2688/input_global_scale; ratios compare domains, not model quality",
        "categories": {name: len(values) for name, values in categories.items()},
        "stems_compared": len(comparisons),
        "warn_ratio": args.warn_ratio,
        "ratio_summary": {
            "median": statistics.median(ratios) if ratios else None,
            "p90": percentile(ratios, 0.90) if ratios else None,
            "p95": percentile(ratios, 0.95) if ratios else None,
            "maximum": max(ratios) if ratios else None,
        },
        "flagged_count": len(flagged),
        "flagged": flagged,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("categories", "stems_compared", "ratio_summary", "flagged_count")}, indent=2))
    print(f"[igs-audit] report: {args.output}")


if __name__ == "__main__":
    main()

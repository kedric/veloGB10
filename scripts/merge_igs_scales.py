#!/usr/bin/env python3
"""Merge W4A4 input scales by retaining the largest observed activation amax."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    merged: dict[str, float] = {}
    provenance: dict[str, str] = {}
    for path in args.inputs:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"{path} is not a JSON object")
        for stem, raw_scale in values.items():
            if not isinstance(raw_scale, (int, float)):
                continue
            scale = float(raw_scale)
            if not math.isfinite(scale) or scale <= 0:
                continue
            # input_global_scale = 2688 / activation_amax. The union of calibration
            # domains therefore needs min(scale), equivalent to max(amax).
            if stem not in merged or scale < merged[stem]:
                merged[stem] = scale
                provenance[stem] = str(path)

    if not merged:
        raise ValueError("no valid scales found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    sidecar = Path(str(args.output) + ".manifest.json")
    sidecar.write_text(json.dumps({
        "format": "veloGB10-igs-merge-v1",
        "rule": "minimum input_global_scale per stem = maximum calibrated activation amax",
        "inputs": [str(path) for path in args.inputs],
        "stems": len(merged),
        "selected_from": provenance,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"[igs-merge] {len(merged)} scales -> {args.output}")


if __name__ == "__main__":
    main()

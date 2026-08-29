#!/usr/bin/env python3
"""Fetch small, reproducible JSONL slices from the Hugging Face dataset API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://datasets-server.huggingface.co"
AYA_LANGUAGES = {
    "German": "de",
    "Spanish": "es",
    "Simplified Chinese": "zh",
    "Standard Arabic": "ar",
    "Portuguese": "pt",
    "Russian": "ru",
}
OPENR1_OFFSETS = (0, 15_000, 30_000, 45_000, 60_000, 75_000)
AYA_ROWS_PER_LANGUAGE = 200


def request_json(endpoint: str, params: dict[str, str | int]) -> dict:
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "veloGB10-calibration/2"})
    error: Exception | None = None
    for attempt in range(10):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:  # network retries are intentionally narrow and deterministic
            error = exc
            if attempt != 9:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"failed to fetch {url}: {error}")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete download: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_aya(path: Path) -> str:
    rows: list[dict] = []
    parts = path.parent / ".aya-six-languages.parts"
    parts.mkdir(parents=True, exist_ok=True)
    for language, code in AYA_LANGUAGES.items():
        part = parts / f"{code}.jsonl"
        cached_rows: list[dict] = []
        if part.exists():
            with part.open(encoding="utf-8") as handle:
                cached_rows = [json.loads(line) for line in handle if line.strip()]
        if len(cached_rows) != AYA_ROWS_PER_LANGUAGE:
            if part.exists():
                part.unlink()
            language_rows: list[dict] = []
            for offset in range(0, AYA_ROWS_PER_LANGUAGE, 100):
                payload = request_json("filter", {
                    "dataset": "CohereLabs/aya_dataset",
                    "config": "default",
                    "split": "train",
                    "where": f'"language"=\'{language}\'',
                    "offset": offset,
                    "length": 100,
                })
                for item in payload.get("rows", []):
                    row = item["row"]
                    language_rows.append({
                        "row_idx": item["row_idx"],
                        "inputs": row["inputs"],
                        "targets": row["targets"],
                        "language": language,
                        "language_code": code,
                        "annotation_type": row.get("annotation_type", ""),
                    })
            if len(language_rows) != AYA_ROWS_PER_LANGUAGE:
                raise RuntimeError(
                    f"Aya filter returned {len(language_rows)} rows for {language}; "
                    f"expected {AYA_ROWS_PER_LANGUAGE}"
                )
            write_jsonl(part, language_rows)
        with part.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return write_jsonl(path, rows)


def fetch_openr1(path: Path) -> str:
    rows: list[dict] = []
    for offset in OPENR1_OFFSETS:
        payload = request_json("rows", {
            "dataset": "open-r1/OpenR1-Math-220k",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 100,
        })
        for item in payload.get("rows", []):
            row = item["row"]
            if not row.get("problem") or not row.get("solution"):
                continue
            rows.append({
                "row_idx": item["row_idx"],
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": row.get("answer", ""),
                "problem_type": row.get("problem_type", ""),
                "source": row.get("source", ""),
                "uuid": row.get("uuid", ""),
            })
    return write_jsonl(path, rows)


def ensure(path: Path, fetcher) -> str:
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return fetcher(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    aya = args.output_root / "aya/aya-six-languages.jsonl"
    openr1 = args.output_root / "openr1/openr1-math-reasoning.jsonl"
    print(f"[fetch-api] aya sha256={ensure(aya, fetch_aya)} path={aya}")
    print(f"[fetch-api] openr1 sha256={ensure(openr1, fetch_openr1)} path={openr1}")


if __name__ == "__main__":
    main()

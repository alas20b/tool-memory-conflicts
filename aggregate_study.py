#!/usr/bin/env python3
"""Validate and combine all five completed full-model analyses."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from kcb.analysis import analyze
from kcb.io_utils import atomic_write_json, atomic_write_text, load_jsonl, model_slug, read_json
from kcb.packets import load_registry


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    registry = load_registry(ROOT)
    summaries = []
    flat_groups = []
    total_responses = 0
    for model_key, model in sorted(registry.items(), key=lambda item: item[1]["index"]):
        slug = model_slug(model_key)
        packet_path = ROOT / "packets" / f"{slug}-full.json"
        response_path = ROOT / "artifacts" / "full" / slug / "responses.jsonl"
        if not packet_path.is_file() or not response_path.is_file():
            raise FileNotFoundError(
                f"full results are missing for model index {model['index']} ({model_key})"
            )
        result = analyze(read_json(packet_path), load_jsonl(response_path))
        total_responses += result["validation"]["n_responses"]
        summaries.append({
            "index": model["index"],
            "model_key": model_key,
            "responses": result["validation"]["n_responses"],
            "manual_review_count": result["manual_review_count"],
            "analysis_sha256": result["analysis_sha256"],
        })
        for group in result["group_metrics"]:
            flat_groups.append({"index": model["index"], "model_key": model_key, **group})

    output_dir = ROOT / "artifacts" / "full"
    combined = {
        "schema_version": "kcb-tool-error-study-summary-v1",
        "status": "complete",
        "models": summaries,
        "model_count": len(summaries),
        "total_responses": total_responses,
        "expected_total_responses": 4200,
    }
    if total_responses != 4200:
        raise ValueError(f"expected 4200 full responses, found {total_responses}")
    atomic_write_json(output_dir / "study_summary.json", combined)

    buffer = io.StringIO(newline="")
    fieldnames = list(flat_groups[0])
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat_groups)
    atomic_write_text(output_dir / "study_group_metrics.csv", buffer.getvalue())
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and combine completed full-model analyses for one or all backends.

``--backend all`` aggregates every backend that has results under
``artifacts/full/<backend>/`` and writes a combined report. Missing or
incomplete models never raise: they are reported as warnings so partial
results can still be inspected while the rest of the runs are in progress.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from kcb.analysis import analyze
from kcb.io_utils import atomic_write_json, atomic_write_text, load_jsonl, model_slug, read_json
from kcb.packets import load_registry


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("mock", "vllm", "transformers", "ollama", "lms")


def aggregate_backend(registry: dict, backend: str) -> tuple[dict, list[dict]]:
    summaries = []
    flat_groups = []
    total_responses = 0
    for model_key, model in sorted(registry.items(), key=lambda item: item[1]["index"]):
        slug = model_slug(model_key)
        packet_path = ROOT / "packets" / f"{slug}-full.json"
        response_path = ROOT / "artifacts" / "full" / backend / slug / "responses.jsonl"
        if not packet_path.is_file() or not response_path.is_file():
            print(
                f"WARNING: full results are missing for model index {model['index']} with path : \n{response_path}\n{packet_path}\n"
                f"({model_key}) with backend {backend}",
                file=sys.stderr,
            )
            continue
        responses = load_jsonl(response_path)
        result = analyze(read_json(packet_path), responses, require_complete=False)
        missing = result["validation"]["n_missing"]
        total_responses += result["validation"]["n_responses"]
        summaries.append({
            "index": model["index"],
            "model_key": model_key,
            "backend": backend,
            "responses": result["validation"]["n_responses"],
            "n_missing": missing,
            "complete": missing == 0,
            "manual_review_count": result["manual_review_count"],
            "analysis_sha256": result["analysis_sha256"],
        })
        for group in result["group_metrics"]:
            flat_groups.append({"index": model["index"], "model_key": model_key, "backend": backend, **group})

    status = "complete" if total_responses == 4200 and len(summaries) == 5 else "partial"
    if total_responses != 4200:
        print(
            f"WARNING: backend '{backend}': expected 4200 full responses, found "
            f"{total_responses} (aggregating anyway; some models may still be running)",
            file=sys.stderr,
        )
    if len(summaries) != 5:
        print(
            f"WARNING: backend '{backend}': results found for {len(summaries)}/5 models",
            file=sys.stderr,
        )

    combined = {
        "schema_version": "kcb-tool-error-study-summary-v1",
        "status": status,
        "backend": backend,
        "models": summaries,
        "model_count": len(summaries),
        "total_responses": total_responses,
        "expected_total_responses": 4200,
    }
    output_dir = ROOT / "artifacts" / "full" / backend
    atomic_write_json(output_dir / "study_summary.json", combined)
    if flat_groups:
        buffer = io.StringIO(newline="")
        fieldnames = list(flat_groups[0])
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_groups)
        atomic_write_text(output_dir / "study_group_metrics.csv", buffer.getvalue())
    return combined, flat_groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=BACKENDS + ("all",),
        default="transformers",
        help="inference backend to aggregate, or 'all' for every backend with results "
             "(default transformers)",
    )
    args = parser.parse_args()
    registry = load_registry(ROOT)

    if args.backend == "all":
        backends = [name for name in BACKENDS if (ROOT / "artifacts" / "full" / name).is_dir()]
        if not backends:
            print("WARNING: no backend result directories found under artifacts/full/", file=sys.stderr)
            return 1
        results = []
        all_groups = []
        for backend in backends:
            combined, groups = aggregate_backend(registry, backend)
            results.append(combined)
            all_groups.extend(groups)
        combined_all = {
            "schema_version": "kcb-tool-error-study-summary-all-v1",
            "status": "complete" if all(result["status"] == "complete" for result in results) else "partial",
            "backends": results,
            "backend_count": len(results),
        }
        output_dir = ROOT / "artifacts" / "full"
        atomic_write_json(output_dir / "study_summary.json", combined_all)
        if all_groups:
            buffer = io.StringIO(newline="")
            fieldnames = list(all_groups[0])
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_groups)
            atomic_write_text(output_dir / "study_group_metrics.csv", buffer.getvalue())
        print(json.dumps(combined_all, ensure_ascii=False, indent=2))
    else:
        combined, _ = aggregate_backend(registry, args.backend)
        print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

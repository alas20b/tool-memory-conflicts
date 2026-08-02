from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .io_utils import (
    atomic_write_json, atomic_write_jsonl, canonical_json, load_jsonl, read_json,
    sha256_file, sha256_text, utc_now,
)
from .profiles import load_profile
from .registry import load_registry
from .runner import artifact_base
from .validation import validate_stage


def aggregate_full_runs(root: Path, profile_paths: list[Path]) -> Path:
    if len(profile_paths) != 5:
        raise ValueError("aggregation requires exactly five execution profiles")
    profiles = [load_profile(path, root) for path in profile_paths]
    registry = load_registry(root)
    if {profile["model_key"] for profile in profiles} != set(registry):
        raise ValueError("profiles must contain exactly one run for every registered model")
    if len({profile["profile_id"] for profile in profiles}) != 5:
        raise ValueError("profile IDs must be unique")
    backends = {profile["backend"] for profile in profiles}
    if len(backends) != 1:
        raise ValueError("do not pool different backends or precision cohorts")
    backend = next(iter(backends))
    all_rows: list[dict] = []
    inputs: list[dict] = []
    model_groups: list[dict] = []
    for profile_path, profile in sorted(zip(profile_paths, profiles), key=lambda pair: pair[1]["model"]["index"]):
        check = validate_stage(root, profile_path, "full")
        directory = artifact_base(root, profile) / "full"
        responses_path = directory / "responses.jsonl"
        analysis_path = directory / "analysis.json"
        responses = load_jsonl(responses_path)
        if len(responses) != 840:
            raise ValueError(f"{profile['model_key']} has {len(responses)} rather than 840 responses")
        all_rows.extend(responses)
        analysis = read_json(analysis_path)
        for group in analysis["groups"]:
            model_groups.append({"model_key": profile["model_key"], "backend": backend, **group})
        inputs.append({
            "model_key": profile["model_key"],
            "profile_id": profile["profile_id"],
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": sha256_file(profile_path),
            "responses_path": str(responses_path.resolve()),
            "responses_sha256": sha256_file(responses_path),
            "analysis_sha256": sha256_file(analysis_path),
            "validation": check,
        })
    if len(all_rows) != 4200:
        raise ValueError("aggregate cohort must contain exactly 4,200 responses")
    cohort_basis = {
        "backend": backend,
        "inputs": [
            {
                "model_key": row["model_key"], "profile_id": row["profile_id"],
                "responses_sha256": row["responses_sha256"],
                "analysis_sha256": row["analysis_sha256"],
            }
            for row in inputs
        ],
    }
    cohort_id = sha256_text(canonical_json(cohort_basis))
    output = root / "artifacts" / "aggregate" / backend / cohort_id[:12]
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "all_responses.jsonl", all_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        state = "absent" if row["memory_absent"] else ("correct" if row["memory_correct"] is True else "incorrect")
        grouped[(row["error_kind"], state)].append(row)
    pooled = []
    for (error_kind, state), rows in sorted(grouped.items()):
        pooled.append({
            "error_kind": error_kind,
            "memory_state": state,
            "n": len(rows),
            "answers_from_memory_rate": sum(row["behavior"] == "answers_from_memory" for row in rows) / len(rows),
            "honest_abstention_rate": sum(row["behavior"] == "honest_abstention" for row in rows) / len(rows),
            "other_answer_rate": sum(row["behavior"] == "other_answer" for row in rows) / len(rows),
        })
    with (output / "model_group_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in model_groups for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(model_groups)
    summary = {
        "schema_version": "kcb-open-cohort-aggregate-v1",
        "cohort_id": cohort_id,
        "created_at": utc_now(),
        "backend": backend,
        "precision_cohort": "Q6_K-GGUF" if backend in ("ollama", "lmstudio") else "native-transformers",
        "models": [profile["model_key"] for profile in sorted(profiles, key=lambda row: row["model"]["index"])],
        "profile_count": 5,
        "question_count_per_model": 280,
        "error_variants": 3,
        "response_count": 4200,
        "real_tool_calls": False,
        "inputs": inputs,
        "pooled_descriptive_rates": pooled,
        "pooling_note": "Pooled rates are descriptive; model-level metrics remain in model_group_metrics.csv.",
    }
    atomic_write_json(output / "aggregate_summary.json", summary)
    return output

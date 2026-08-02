from __future__ import annotations

import argparse
import json
import shutil
import sys

from _common import ROOT, path_from_root
from kcb_open.io_utils import read_json, sha256_file
from kcb_open.packets import stable_sample
from kcb_open.profiles import load_profile
from kcb_open.prompts import ERROR_KINDS, ERROR_PAYLOADS, build_error_rows, error_prompt
from kcb_open.registry import load_question_set, load_registry, model_key_from_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline integrity and storage preflight")
    parser.add_argument("--profile", help="optional profile JSON to validate")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    registry = load_registry(ROOT)
    question_doc = load_question_set(ROOT)
    questions = question_doc["questions"]
    labels = {
        row["source_episode_id"]: {
            "parametric_span": "synthetic", "memory_correct": False, "memory_absent": False,
        }
        for row in stable_sample(questions, 12, 20260704)
    }
    prompt_rows = build_error_rows(
        [row for row in questions if row["source_episode_id"] in labels], labels
    )
    review_path = ROOT / "data" / "review_examples_12x3.json"
    review = read_json(review_path)
    selected = stable_sample(questions, 12, 20260704)
    if review.get("question_count") != 12 or review.get("prompt_count") != 36:
        raise ValueError("prepared review example counts are invalid")
    for question, example in zip(selected, review.get("examples", []), strict=True):
        if example.get("source_episode_id") != question["source_episode_id"]:
            raise ValueError("prepared review question selection differs from deterministic sample")
        variants = example.get("variants", [])
        if [row.get("error_kind") for row in variants] != list(ERROR_KINDS):
            raise ValueError("prepared review error order is invalid")
        for variant in variants:
            kind = variant["error_kind"]
            if variant.get("prepared_tool_output") != ERROR_PAYLOADS[kind]:
                raise ValueError("prepared review payload differs from fixed error payload")
            if variant.get("prompt") != error_prompt(question, kind):
                raise ValueError("prepared review prompt differs from prompt builder")
    free = shutil.disk_usage(ROOT).free
    result = {
        "status": "ok",
        "root": str(ROOT),
        "question_count": len(questions),
        "question_set_sha256": sha256_file(ROOT / "data" / "questions_280.json"),
        "review_examples_sha256": sha256_file(review_path),
        "prepared_review_questions": 12,
        "prepared_review_prompts": len(prompt_rows),
        "error_kinds": list(ERROR_KINDS),
        "free_storage_bytes": free,
        "models": [],
    }
    for index in range(5):
        key = model_key_from_index(registry, index)
        model = registry[key]
        size = int(model["gguf"]["size_bytes"])
        result["models"].append({
            "index": index, "key": key, "parameters": model["parameter_count"],
            "gguf_repo": model["gguf"]["repo"], "gguf_filename": model["gguf"]["filename"],
            "quantization": model["gguf"]["quantization"], "size_bytes": size,
            "under_10gb": size < 10_000_000_000,
        })
    if args.profile:
        profile = load_profile(path_from_root(args.profile), ROOT)
        result["profile"] = {
            "profile_id": profile["profile_id"], "model_key": profile["model_key"],
            "backend": profile["backend"], "weight_verification": profile["weight_verification"],
        }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("OFFLINE PREFLIGHT: OK")
        print(f"Root: {ROOT}")
        print(f"Questions: 280; prepared error variants: {', '.join(ERROR_KINDS)}")
        print(f"Question-set SHA256: {result['question_set_sha256']}")
        print(f"Free storage at repository: {free / 1_000_000_000:.2f} GB")
        for row in result["models"]:
            print(f"[{row['index']}] {row['key']}: {row['parameters']}, Q6_K, {row['size_bytes']/1_000_000_000:.2f} GB")
        print("No model was downloaded and no inference server was contacted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

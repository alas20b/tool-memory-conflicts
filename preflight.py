#!/usr/bin/env python3
"""Fail-fast checks for packet fidelity and, optionally, the GPU runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

from kcb.io_utils import read_json, sha256_file
from kcb.packets import build_packet, flatten_packet, load_registry, validate_packet


ROOT = Path(__file__).resolve().parents[1]


def check_manifest() -> None:
    manifest = read_json(ROOT / "input_manifest.json")
    if manifest.get("schema_version") != "kcb-input-manifest-v1":
        raise ValueError("invalid input manifest schema")
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest input is missing: {relative}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"size mismatch for immutable input {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for immutable input {relative}")


def offline_checks() -> dict:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    check_manifest()
    config = read_json(ROOT / "experiment_config.json")
    registry = load_registry(ROOT)
    if config["real_tool_calls"] is not False:
        raise ValueError("experiment config must prohibit real tool calls")
    model_reports = []
    source_sets: list[set[str]] = []
    for model_key, model in sorted(registry.items(), key=lambda item: item[1]["index"]):
        full = build_packet(ROOT, model_key, "full", seed=config["seed"])
        review = build_packet(
            ROOT,
            model_key,
            "review",
            count=config["review_question_count"],
            seed=config["seed"],
        )
        full_check = validate_packet(
            full,
            expected_model_key=model_key,
            expected_questions=config["expected_full_question_count"],
        )
        review_check = validate_packet(
            review,
            expected_model_key=model_key,
            expected_questions=config["review_question_count"],
        )
        source_sets.append({row["source_episode_id"] for row in flatten_packet(full)})
        model_reports.append({
            "index": model["index"],
            "model_key": model_key,
            "full_questions": full_check["questions"],
            "full_prompts": full_check["prompts"],
            "review_questions": review_check["questions"],
            "review_prompts": review_check["prompts"],
            "memory_counts": full["metadata"]["memory_counts"],
        })
    if any(source_set != source_sets[0] for source_set in source_sets[1:]):
        raise ValueError("the five model packets do not use the same source-question set")
    return {
        "status": "passed",
        "models": model_reports,
        "shared_full_questions": len(source_sets[0]),
        "real_tool_calls": False,
    }


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required runtime package is missing: {name}") from error


def runtime_checks(model_key: str) -> dict:
    registry = load_registry(ROOT)
    if model_key not in registry:
        raise ValueError(f"unknown model key {model_key!r}")
    model = registry[model_key]
    versions = {name: version(name) for name in ("numpy", "vllm", "torch", "huggingface-hub")}
    if versions["numpy"] != "1.26.4":
        raise RuntimeError(f"expected numpy 1.26.4, found {versions['numpy']}")
    if versions["vllm"] != "0.8.5":
        raise RuntimeError(f"expected vllm 0.8.5, found {versions['vllm']}")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA GPU is visible to PyTorch")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF-TOKEN")
    if model["gated"] and not token:
        raise RuntimeError(f"{model['hf_repo']} is gated but HF_TOKEN is not set")
    if token:
        os.environ["HF_TOKEN"] = token

    from huggingface_hub import HfApi

    info = HfApi().model_info(
        model["hf_repo"],
        revision=model.get("requested_revision", "main"),
        token=token,
    )
    if not info.sha:
        raise RuntimeError("could not resolve the Hugging Face model commit")
    config = read_json(ROOT / "experiment_config.json")
    cache = Path(os.environ.get("KCB_HF_HOME", ROOT / "hf_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / (1024**3)
    repo_folder = "models--" + model["hf_repo"].replace("/", "--")
    cached = any(
        candidate.is_dir()
        for candidate in (
            cache / repo_folder / "snapshots" / info.sha,
            cache / "hub" / repo_folder / "snapshots" / info.sha,
        )
    )
    if not cached and free_gb < config["minimum_cache_free_gb"]:
        raise RuntimeError(
            f"only {free_gb:.1f} GiB is free in {cache}; "
            f"{config['minimum_cache_free_gb']} GiB is required before an uncached model download"
        )
    return {
        "status": "passed",
        "model_key": model_key,
        "hf_repo": model["hf_repo"],
        "resolved_revision": info.sha,
        "gpu_count": torch.cuda.device_count(),
        "gpu_0": torch.cuda.get_device_name(0),
        "cache": str(cache),
        "cache_free_gb": round(free_gb, 2),
        "snapshot_already_cached": cached,
        "versions": versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--model-key")
    args = parser.parse_args()
    report = {"offline": offline_checks()}
    if args.runtime:
        if not args.model_key:
            parser.error("--runtime requires --model-key")
        report["runtime"] = runtime_checks(args.model_key)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

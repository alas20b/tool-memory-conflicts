from __future__ import annotations

import re
from pathlib import Path

from .io_utils import read_json


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_registry(root: Path) -> dict:
    registry = read_json(root / "model_registry.json")
    if not isinstance(registry, dict) or len(registry) != 5:
        raise ValueError("model_registry.json must contain exactly five models")
    if sorted(model.get("index") for model in registry.values()) != list(range(5)):
        raise ValueError("model indices must be exactly 0..4")
    for key, model in registry.items():
        joined = str(model).lower()
        if "meta-llama" in joined or "llama-3" in joined:
            raise ValueError(f"Meta Llama is forbidden in the open cohort: {key}")
        if model.get("license") != "Apache-2.0":
            raise ValueError(f"{key} is not declared Apache-2.0")
        gguf = model.get("gguf", {})
        if gguf.get("quantization") != "Q6_K":
            raise ValueError(f"{key} must use the registered Q6_K GGUF")
        if not SHA256_RE.fullmatch(str(gguf.get("sha256", ""))):
            raise ValueError(f"{key} has no valid GGUF sha256")
        size = int(gguf.get("size_bytes", 0))
        if size <= 0 or size >= 10_000_000_000:
            raise ValueError(f"{key} GGUF must be positive and below 10 GB")
    return registry


def model_key_from_index(registry: dict, index: int) -> str:
    matches = [key for key, value in registry.items() if value["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"no unique model at index {index}")
    return matches[0]


def load_question_set(root: Path) -> dict:
    document = read_json(root / "data" / "questions_280.json")
    if document.get("schema_version") != "kcb-open-question-set-v1":
        raise ValueError("unsupported question-set schema")
    questions = document.get("questions")
    if not isinstance(questions, list) or len(questions) != 280:
        raise ValueError("question set must contain exactly 280 questions")
    ids = [row.get("source_episode_id") for row in questions]
    if len(set(ids)) != 280 or any(not value for value in ids):
        raise ValueError("question IDs must be non-empty and unique")
    return document

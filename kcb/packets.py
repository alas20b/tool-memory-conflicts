"""Construct and validate matched prepared-error packets."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .error_variants import ERROR_KINDS, build_all_error_variants
from .inject import TOOL_ERROR_PAYLOADS
from .io_utils import canonical_json, read_json, sha256_file, sha256_text, unique_by
from .metrics import normalize


PACKET_SCHEMA = "kcb-tool-error-packet-v1"
EPISODES_RELATIVE = Path("data/episodes_v1_full_near_pv0_poolv3.json")


def natural_key(episode: dict) -> tuple[int, int, str]:
    return int(episode["instance_id"]), int(episode["hop_idx"]), str(episode["episode_id"])


def is_nontrivial(episode: dict) -> bool:
    gold = normalize(episode["gold_answer"])
    question = normalize(episode["question"])
    return bool(gold) and gold not in question


def load_registry(root: Path) -> dict:
    registry = read_json(root / "model_registry.json")
    if not isinstance(registry, dict) or len(registry) != 5:
        raise ValueError("model_registry.json must contain exactly five models")
    indices = sorted(model.get("index") for model in registry.values())
    if indices != list(range(5)):
        raise ValueError("model registry indices must be exactly 0..4")
    return registry


def model_key_from_index(registry: dict, index: int) -> str:
    matches = [key for key, model in registry.items() if model["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"no unique model at index {index}")
    return matches[0]


def _eligible_pairs(root: Path, model_key: str) -> tuple[dict, dict, list[tuple[dict, dict]]]:
    registry = load_registry(root)
    if model_key not in registry:
        raise ValueError(f"unknown model key {model_key!r}")
    model = registry[model_key]
    episodes_path = root / EPISODES_RELATIVE
    labels_path = root / model["memory_labels"]
    episodes_doc = read_json(episodes_path)
    labels_doc = read_json(labels_path)

    error_episodes = [
        episode for episode in episodes_doc.get("episodes", [])
        if episode.get("condition") == "tool_error"
    ]
    by_id = unique_by(error_episodes, "episode_id")
    labels = [
        label for label in labels_doc.get("scored", [])
        if label.get("condition") == "tool_error"
    ]
    label_by_id = unique_by(labels, "episode_id")
    pairs: list[tuple[dict, dict]] = []
    for episode_id, label in label_by_id.items():
        episode = by_id.get(episode_id)
        if episode is None:
            raise ValueError(f"memory label references missing source episode {episode_id}")
        if episode.get("tool_correct") is not False:
            raise ValueError(f"source error episode {episode_id} is not tool-incorrect")
        if label.get("memory_correct") not in (True, False, None):
            raise ValueError(f"invalid memory_correct value for {episode_id}")
        pairs.append((episode, label))
    pairs.sort(key=lambda pair: natural_key(pair[0]))
    return model, episodes_doc, pairs


def _review_pairs(
    pairs: list[tuple[dict, dict]], count: int, seed: int
) -> list[tuple[dict, dict]]:
    if count < 2:
        raise ValueError("review count must be at least 2")
    grouped: dict[bool, list[tuple[dict, dict]]] = {True: [], False: []}
    for episode, label in pairs:
        if label.get("mem_absent") or not is_nontrivial(episode):
            continue
        grouped[bool(label["memory_correct"])].append((episode, label))

    def stable_sample_key(pair: tuple[dict, dict]) -> tuple[str, tuple[int, int, str]]:
        episode = pair[0]
        digest = hashlib.sha256(f"{seed}|{episode['episode_id']}".encode()).hexdigest()
        return digest, natural_key(episode)

    for group in grouped.values():
        group.sort(key=stable_sample_key)
    n_correct = count // 2
    n_incorrect = count - n_correct
    if len(grouped[True]) < n_correct or len(grouped[False]) < n_incorrect:
        raise ValueError("not enough nontrivial examples for a balanced review packet")
    selected = grouped[True][:n_correct] + grouped[False][:n_incorrect]
    selected.sort(key=lambda pair: natural_key(pair[0]))
    return selected


def _example(index: int, episode: dict, label: dict) -> dict:
    variants = build_all_error_variants(episode)
    return {
        "example_id": f"example-{index:03d}",
        "source_episode_id": episode["episode_id"],
        "instance_id": episode["instance_id"],
        "hop_idx": episode["hop_idx"],
        "question": episode["question"],
        "gold_answer": episode["gold_answer"],
        "tau": episode.get("tau"),
        "tool_schema": copy.deepcopy(episode["tool_schema"]),
        "prompt_version": episode["prompt_version"],
        "prompt_variant": episode["prompt_variant"],
        "seed": episode.get("seed"),
        "closed_book_evidence": {
            "memory_correct": label["memory_correct"],
            "memory_absent": bool(label["mem_absent"]),
            "parametric_answer": label["parametric_span"],
        },
        "variants": [
            {
                "variant_id": variant["variant_id"],
                "condition": variant["condition"],
                "error_kind": variant["error_kind"],
                "tool_output": variant["tool_output"],
                "prompt_with_tool": variant["prompts"]["with_tool"],
            }
            for variant in variants
        ],
    }


def build_packet(root: Path, model_key: str, mode: str, count: int = 12, seed: int = 20260704) -> dict:
    model, episodes_doc, pairs = _eligible_pairs(root, model_key)
    if mode == "review":
        selected = _review_pairs(pairs, count=count, seed=seed)
    elif mode == "full":
        selected = pairs
    else:
        raise ValueError("mode must be 'review' or 'full'")

    examples = [_example(i, episode, label) for i, (episode, label) in enumerate(selected, 1)]
    episode_ids = [example["source_episode_id"] for example in examples]
    memory_counts = {
        "correct": sum(example["closed_book_evidence"]["memory_correct"] is True for example in examples),
        "incorrect": sum(example["closed_book_evidence"]["memory_correct"] is False for example in examples),
        "absent": sum(example["closed_book_evidence"]["memory_absent"] for example in examples),
    }
    episodes_path = root / EPISODES_RELATIVE
    labels_path = root / model["memory_labels"]
    packet = {
        "metadata": {
            "schema_version": PACKET_SCHEMA,
            "purpose": "matched prepared tool-error extension of KCB",
            "mode": mode,
            "selected_model": {
                "id": model_key,
                "index": model["index"],
                "hf_repo": model["hf_repo"],
                "parameter_count": model["parameter_count"],
                "interface": model["interface"],
            },
            "real_tool_calls": False,
            "source_files": {
                "episodes": str(EPISODES_RELATIVE).replace("\\", "/"),
                "episodes_sha256": sha256_file(episodes_path),
                "memory_labels": model["memory_labels"],
                "memory_labels_sha256": sha256_file(labels_path),
            },
            "prompt_version": episodes_doc["summary"]["prompt_version"],
            "prompt_variant": episodes_doc["summary"]["prompt_variant"],
            "question_count": len(examples),
            "variant_count_per_question": len(ERROR_KINDS),
            "total_prepared_prompts": len(examples) * len(ERROR_KINDS),
            "memory_counts": memory_counts,
            "source_episode_ids_sha256": sha256_text("\n".join(episode_ids) + "\n"),
            "error_variants": copy.deepcopy(TOOL_ERROR_PAYLOADS),
            "response_labels": ["answers_from_memory", "honest_abstention", "other_answer"],
            "selection_seed": seed,
        },
        "examples": examples,
    }
    validate_packet(packet, expected_model_key=model_key)
    return packet


def flatten_packet(packet: dict) -> list[dict]:
    rows: list[dict] = []
    model_key = packet["metadata"]["selected_model"]["id"]
    for example in packet["examples"]:
        evidence = example["closed_book_evidence"]
        for variant in example["variants"]:
            rows.append({
                "variant_id": variant["variant_id"],
                "source_episode_id": example["source_episode_id"],
                "model_key": model_key,
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "prompt_version": example["prompt_version"],
                "prompt_variant": example["prompt_variant"],
                "memory_correct": evidence["memory_correct"],
                "memory_absent": evidence["memory_absent"],
                "parametric_answer": evidence["parametric_answer"],
                "error_kind": variant["error_kind"],
                "tool_output": variant["tool_output"],
                "prompt": variant["prompt_with_tool"],
            })
    return rows


def validate_packet(
    packet: dict,
    *,
    expected_model_key: str | None = None,
    expected_questions: int | None = None,
) -> dict[str, Any]:
    metadata = packet.get("metadata", {})
    if metadata.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("unsupported or missing packet schema_version")
    if metadata.get("real_tool_calls") is not False:
        raise ValueError("packet must explicitly declare real_tool_calls=false")
    model_key = metadata.get("selected_model", {}).get("id")
    if expected_model_key is not None and model_key != expected_model_key:
        raise ValueError(f"packet model is {model_key!r}, expected {expected_model_key!r}")
    examples = packet.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("packet contains no examples")
    if expected_questions is not None and len(examples) != expected_questions:
        raise ValueError(f"packet has {len(examples)} questions, expected {expected_questions}")

    example_ids = unique_by(examples, "source_episode_id")
    del example_ids
    variant_ids: set[str] = set()
    for example in examples:
        variants = example.get("variants")
        if not isinstance(variants, list) or len(variants) != len(ERROR_KINDS):
            raise ValueError(f"{example.get('source_episode_id')} does not have three variants")
        kinds = [variant.get("error_kind") for variant in variants]
        if set(kinds) != set(ERROR_KINDS) or len(kinds) != len(set(kinds)):
            raise ValueError(f"invalid error triplet for {example['source_episode_id']}")
        canonical_prompts: list[str] = []
        for variant in variants:
            variant_id = str(variant.get("variant_id", ""))
            if not variant_id or variant_id in variant_ids:
                raise ValueError(f"missing or duplicate variant_id {variant_id!r}")
            variant_ids.add(variant_id)
            kind = variant["error_kind"]
            if variant.get("condition") != "tool_error":
                raise ValueError(f"variant {variant_id} changed the source condition")
            if variant.get("tool_output") != TOOL_ERROR_PAYLOADS[kind]:
                raise ValueError(f"variant {variant_id} has a noncanonical error payload")
            prompt = variant.get("prompt_with_tool")
            if not isinstance(prompt, str):
                raise ValueError(f"variant {variant_id} has no prompt")
            payload = json.dumps(variant["tool_output"], ensure_ascii=False)
            if prompt.count(payload) != 1:
                raise ValueError(f"payload must occur exactly once in {variant_id}")
            canonical_prompts.append(prompt.replace(payload, "<PREPARED_ERROR>", 1))
        if len(set(canonical_prompts)) != 1:
            raise ValueError(f"more than the error payload changed for {example['source_episode_id']}")

    rows = flatten_packet(packet)
    if metadata.get("question_count") != len(examples):
        raise ValueError("metadata question_count does not match packet")
    if metadata.get("total_prepared_prompts") != len(rows):
        raise ValueError("metadata total_prepared_prompts does not match packet")
    return {
        "model_key": model_key,
        "questions": len(examples),
        "prompts": len(rows),
        "packet_content_sha256": sha256_text(canonical_json(packet)),
        "error_kinds": list(ERROR_KINDS),
    }

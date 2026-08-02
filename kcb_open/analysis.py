from __future__ import annotations

import csv
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path

from .io_utils import atomic_write_json, canonical_json, sha256_text, unique_by
from .prompts import ERROR_KINDS
from .scoring import classify_error_response


BEHAVIORS = ("answers_from_memory", "honest_abstention", "other_answer")


def validate_error_responses(packet: dict, responses: list[dict]) -> dict:
    expected = unique_by(packet["rows"], "task_id")
    actual = unique_by(responses, "task_id")
    unknown = set(actual) - set(expected)
    missing = set(expected) - set(actual)
    if unknown or missing:
        raise ValueError(f"response coverage mismatch: unknown={len(unknown)}, missing={len(missing)}")
    for task_id, response in actual.items():
        source = expected[task_id]
        if response.get("prompt_sha256") != sha256_text(source["prompt"]):
            raise ValueError(f"prompt hash mismatch for {task_id}")
        rescored = classify_error_response(
            response.get("raw_answer", ""), source["parametric_span"], source["gold_answer"]
        )
        for key, value in rescored.items():
            if response.get(key) != value:
                raise ValueError(f"score mismatch for {task_id}: {key}")
    return {
        "questions": packet["question_count"],
        "tasks": packet["task_count"],
        "n_responses": len(responses),
        "n_missing": 0,
        "complete": True,
    }


def _bootstrap_rate(rows: list[dict], behavior: str, seed: int, replicates: int) -> list[float] | None:
    if not rows:
        return None
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["source_episode_id"]].append(row)
    keys = sorted(by_episode)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled_keys = [rng.choice(keys) for _ in keys]
        sample = [item for key in sampled_keys for item in by_episode[key]]
        values.append(sum(row["behavior"] == behavior for row in sample) / len(sample))
    values.sort()
    lower = values[int(0.025 * (replicates - 1))]
    upper = values[int(0.975 * (replicates - 1))]
    return [round(lower, 4), round(upper, 4)]


def _paired_contrast(
    by_episode: dict[str, dict[str, dict]],
    keys: list[str],
    error_a: str,
    error_b: str,
    behavior: str,
    *,
    seed: int,
    replicates: int,
) -> dict:
    differences = [
        int(by_episode[key][error_a]["behavior"] == behavior)
        - int(by_episode[key][error_b]["behavior"] == behavior)
        for key in keys
    ]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(replicates)
    )
    lower = bootstrap[int(0.025 * (replicates - 1))]
    upper = bootstrap[int(0.975 * (replicates - 1))]
    positive = sum(value == 1 for value in differences)
    negative = sum(value == -1 for value in differences)
    discordant = positive + negative
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(positive, negative) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "n": len(keys),
        "error_a": error_a,
        "error_b": error_b,
        "behavior": behavior,
        "rate_difference_a_minus_b": round(estimate, 6),
        "difference_ci95": [round(lower, 4), round(upper, 4)],
        "a_only": positive,
        "b_only": negative,
        "discordant": discordant,
        "exact_mcnemar_p": round(p_value, 8),
    }


def _holm_adjust(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["exact_mcnemar_p"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index]["exact_mcnemar_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = round(running, 8)


def analyze_error_responses(packet: dict, responses: list[dict], *, seed: int, replicates: int) -> dict:
    validation = validate_error_responses(packet, responses)
    groups: list[dict] = []
    for kind_index, error_kind in enumerate(ERROR_KINDS):
        for state_index, state in enumerate(("all", "correct", "incorrect", "absent")):
            rows = [row for row in responses if row["error_kind"] == error_kind]
            if state == "correct":
                rows = [row for row in rows if row["memory_correct"] is True]
            elif state == "incorrect":
                rows = [row for row in rows if row["memory_correct"] is False]
            elif state == "absent":
                rows = [row for row in rows if row["memory_absent"]]
            if not rows and state != "all":
                continue
            result = {"group": f"error={error_kind}|memory={state}", "n": len(rows)}
            for behavior_index, behavior in enumerate(BEHAVIORS):
                result[f"{behavior}_rate"] = round(
                    sum(row["behavior"] == behavior for row in rows) / len(rows), 6
                ) if rows else None
                result[f"{behavior}_ci95"] = _bootstrap_rate(
                    rows, behavior, seed + kind_index * 100 + state_index * 10 + behavior_index,
                    replicates,
                )
            groups.append(result)

    transitions: dict[str, int] = defaultdict(int)
    by_episode: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in responses:
        by_episode[row["source_episode_id"]][row["error_kind"]] = row
    for mapping in by_episode.values():
        if set(mapping) == set(ERROR_KINDS):
            key = " | ".join(f"{kind}:{mapping[kind]['behavior']}" for kind in ERROR_KINDS)
            transitions[key] += 1

    contrasts: list[dict] = []
    contrast_index = 0
    for state in ("all", "correct", "incorrect", "absent"):
        keys = []
        for episode_id, mapping in by_episode.items():
            if set(mapping) != set(ERROR_KINDS):
                continue
            exemplar = mapping[ERROR_KINDS[0]]
            actual_state = "absent" if exemplar["memory_absent"] else (
                "correct" if exemplar["memory_correct"] is True else "incorrect"
            )
            if state == "all" or state == actual_state:
                keys.append(episode_id)
        keys.sort()
        if not keys:
            continue
        for error_a, error_b in itertools.combinations(ERROR_KINDS, 2):
            for behavior in BEHAVIORS:
                row = _paired_contrast(
                    by_episode, keys, error_a, error_b, behavior,
                    seed=seed + 1000 + contrast_index, replicates=replicates,
                )
                row["memory_state"] = state
                contrasts.append(row)
                contrast_index += 1
    _holm_adjust(contrasts)

    return {
        "schema_version": "kcb-open-analysis-v2",
        "validation": validation,
        "groups": groups,
        "paired_error_contrasts": contrasts,
        "behavior_transitions": dict(sorted(transitions.items())),
        "manual_review_count": sum(bool(row["needs_manual_review"]) for row in responses),
        "analysis_sha256": sha256_text(canonical_json({"packet": packet, "responses": responses})),
    }


def write_analysis(output_dir: Path, packet: dict, responses: list[dict], *, seed: int, replicates: int) -> dict:
    result = analyze_error_responses(packet, responses, seed=seed, replicates=replicates)
    atomic_write_json(output_dir / "analysis.json", result)
    review = [row for row in responses if row["needs_manual_review"]]
    with (output_dir / "manual_review_queue.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in review:
            handle.write(canonical_json(row) + "\n")
    fields = ["group", "n"] + [field for behavior in BEHAVIORS for field in (f"{behavior}_rate", f"{behavior}_ci95")]
    with (output_dir / "group_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["groups"])
    contrast_fields = [
        "memory_state", "behavior", "error_a", "error_b", "n",
        "rate_difference_a_minus_b", "difference_ci95", "a_only", "b_only", "discordant",
        "exact_mcnemar_p", "holm_adjusted_p",
    ]
    with (output_dir / "paired_error_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fields)
        writer.writeheader()
        writer.writerows(result["paired_error_contrasts"])
    return result

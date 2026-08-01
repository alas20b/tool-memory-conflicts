"""Deterministic validation and matched analysis for prepared-error runs."""

from __future__ import annotations

import csv
import io
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .error_variants import classify_error_response
from .io_utils import atomic_write_json, atomic_write_text, canonical_json, sha256_text, unique_by
from .packets import ERROR_KINDS, flatten_packet, validate_packet


BEHAVIORS = ("answers_from_memory", "honest_abstention", "other_answer")


def memory_state(row: dict) -> str:
    if row["memory_absent"]:
        return "absent"
    return "correct" if row["memory_correct"] else "incorrect"


def _rate(rows: list[dict], predicate: Callable[[dict], bool]) -> float | None:
    return sum(predicate(row) for row in rows) / len(rows) if rows else None


def _bootstrap_rate(
    rows: list[dict],
    predicate: Callable[[dict], bool],
    *,
    seed: int,
    n_boot: int = 2000,
) -> list[float] | None:
    if not rows:
        return None
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source_episode_id"]].append(row)
    keys = sorted(grouped)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample = [item for _ in keys for item in grouped[rng.choice(keys)]]
        estimates.append(_rate(sample, predicate) or 0.0)
    estimates.sort()
    return [round(estimates[int(0.025 * n_boot)], 4), round(estimates[int(0.975 * n_boot)], 4)]


def validate_responses(packet: dict, responses: list[dict], *, require_complete: bool = True) -> dict:
    packet_check = validate_packet(packet)
    targets = unique_by(flatten_packet(packet), "variant_id")
    actual = unique_by(responses, "variant_id")
    unknown = sorted(set(actual) - set(targets))
    missing = sorted(set(targets) - set(actual))
    if unknown:
        raise ValueError(f"responses contain {len(unknown)} unknown variant IDs")
    if require_complete and missing:
        raise ValueError(f"responses are incomplete: {len(missing)} variants missing")

    for variant_id, response in actual.items():
        target = targets[variant_id]
        immutable = (
            "source_episode_id",
            "model_key",
            "error_kind",
            "memory_correct",
            "memory_absent",
            "parametric_answer",
            "gold_answer",
        )
        for field in immutable:
            if response.get(field) != target.get(field):
                raise ValueError(f"response {variant_id} changed immutable field {field}")
        expected_prompt_hash = sha256_text(target["prompt"])
        if response.get("prompt_sha256") != expected_prompt_hash:
            raise ValueError(f"prompt hash mismatch for {variant_id}")
        rescored = classify_error_response(
            response.get("raw_answer", ""),
            parametric_span=target["parametric_answer"],
            gold_answer=target["gold_answer"],
        )
        for field, expected in rescored.items():
            if response.get(field) != expected:
                raise ValueError(f"stored score mismatch for {variant_id}: {field}")

    completed_questions: dict[str, set[str]] = defaultdict(set)
    for response in responses:
        completed_questions[response["source_episode_id"]].add(response["error_kind"])
    incomplete_triplets = [
        episode_id for episode_id, kinds in completed_questions.items()
        if kinds != set(ERROR_KINDS)
    ]
    if require_complete and incomplete_triplets:
        raise ValueError(f"{len(incomplete_triplets)} questions have incomplete error triplets")
    return {
        **packet_check,
        "n_responses": len(responses),
        "n_missing": len(missing),
        "complete": not missing,
    }


def _group_row(label: str, rows: list[dict], seed: int) -> dict:
    result = {"group": label, "n": len(rows)}
    predicates: dict[str, Callable[[dict], bool]] = {
        "answers_from_memory": lambda row: row["behavior"] == "answers_from_memory",
        "honest_abstention": lambda row: row["behavior"] == "honest_abstention",
        "other_answer": lambda row: row["behavior"] == "other_answer",
        "final_correct": lambda row: bool(row["final_correct"]),
        "format_compliant": lambda row: bool(row["format_compliant"]),
    }
    for offset, (name, predicate) in enumerate(predicates.items()):
        rate = _rate(rows, predicate)
        result[f"{name}_rate"] = None if rate is None else round(rate, 6)
        result[f"{name}_ci95"] = _bootstrap_rate(rows, predicate, seed=seed + offset)
    return result


def _paired_comparisons(responses: list[dict], seed: int) -> list[dict]:
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in responses:
        by_question[row["source_episode_id"]][row["error_kind"]] = row
    comparisons: list[dict] = []
    pairs = (("service_503", "timeout"), ("service_503", "permission_denied"), ("timeout", "permission_denied"))
    states = ("all", "correct", "incorrect")
    measures: dict[str, Callable[[dict], bool]] = {
        "answers_from_memory": lambda row: row["behavior"] == "answers_from_memory",
        "honest_abstention": lambda row: row["behavior"] == "honest_abstention",
        "other_answer": lambda row: row["behavior"] == "other_answer",
        "final_correct": lambda row: bool(row["final_correct"]),
    }
    for state_index, state in enumerate(states):
        eligible = {
            key: values for key, values in by_question.items()
            if set(values) == set(ERROR_KINDS)
            and (state == "all" or memory_state(next(iter(values.values()))) == state)
        }
        keys = sorted(eligible)
        for pair_index, (left, right) in enumerate(pairs):
            for measure_index, (measure, predicate) in enumerate(measures.items()):
                deltas = [
                    float(predicate(eligible[key][right])) - float(predicate(eligible[key][left]))
                    for key in keys
                ]
                if not deltas:
                    continue
                rng = random.Random(seed + state_index * 100 + pair_index * 10 + measure_index)
                boots = [
                    sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                    for _ in range(2000)
                ]
                boots.sort()
                comparisons.append({
                    "memory_state": state,
                    "left": left,
                    "right": right,
                    "measure": measure,
                    "n_paired_questions": len(deltas),
                    "difference_right_minus_left": round(sum(deltas) / len(deltas), 6),
                    "ci95": [round(boots[50], 4), round(boots[1950], 4)],
                })
    return comparisons


def analyze(packet: dict, responses: list[dict], *, seed: int = 20260704) -> dict:
    validation = validate_responses(packet, responses, require_complete=True)
    groups: list[dict] = []
    for error_index, error_kind in enumerate(ERROR_KINDS):
        error_rows = [row for row in responses if row["error_kind"] == error_kind]
        groups.append(_group_row(f"error={error_kind}|memory=all", error_rows, seed + error_index * 100))
        for state_index, state in enumerate(("correct", "incorrect", "absent"), 1):
            sub = [row for row in error_rows if memory_state(row) == state]
            if sub:
                groups.append(_group_row(
                    f"error={error_kind}|memory={state}", sub, seed + error_index * 100 + state_index * 10
                ))

    transition_counts: dict[str, Counter] = {}
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in responses:
        by_question[row["source_episode_id"]][row["error_kind"]] = row
    for left, right in (("service_503", "timeout"), ("service_503", "permission_denied"), ("timeout", "permission_denied")):
        counter: Counter = Counter()
        for values in by_question.values():
            counter[f"{values[left]['behavior']} -> {values[right]['behavior']}"] += 1
        transition_counts[f"{left} vs {right}"] = counter

    return {
        "schema_version": "kcb-tool-error-analysis-v1",
        "model_key": packet["metadata"]["selected_model"]["id"],
        "validation": validation,
        "group_metrics": groups,
        "paired_comparisons": _paired_comparisons(responses, seed),
        "behavior_transitions": {key: dict(value) for key, value in transition_counts.items()},
        "manual_review_count": sum(row["needs_manual_review"] for row in responses),
        "analysis_sha256": sha256_text(canonical_json({"packet": packet["metadata"], "responses": responses})),
    }


def write_analysis(output_dir: Path, packet: dict, responses: list[dict], *, seed: int = 20260704) -> dict:
    result = analyze(packet, responses, seed=seed)
    atomic_write_json(output_dir / "analysis.json", result)

    buffer = io.StringIO(newline="")
    fieldnames = list(result["group_metrics"][0])
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(result["group_metrics"])
    atomic_write_text(output_dir / "group_metrics.csv", buffer.getvalue())

    review_lines = [
        canonical_json(row) for row in responses if row["needs_manual_review"]
    ]
    atomic_write_text(
        output_dir / "manual_review_queue.jsonl",
        "\n".join(review_lines) + ("\n" if review_lines else ""),
    )
    return result

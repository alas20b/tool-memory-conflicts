"""Prepared tool-error variants for the KCB follow-up experiment.

The functions in this module only transform saved episodes. They never call a
tool or a model. The source question, schema, prompt version, prompt variant,
and all other experimental settings remain unchanged.
"""

from __future__ import annotations

import copy
import json
import re

from .inject import TOOL_ERROR_PAYLOADS, prepared_tool_error
from .scoring import extract_final, is_abstain, match


ERROR_KINDS = tuple(TOOL_ERROR_PAYLOADS)


def build_error_variant(source_episode: dict, error_kind: str) -> dict:
    """Clone one saved ``tool_error`` episode with a different fixed payload."""
    if source_episode.get("condition") != "tool_error":
        raise ValueError("error variants can only be built from tool_error episodes")

    old_payload = source_episode.get("tool_output")
    prompt = source_episode.get("prompts", {}).get("with_tool")
    if not isinstance(old_payload, dict) or not isinstance(prompt, str):
        raise ValueError("source episode must contain a structured tool output and prompt")

    old_json = json.dumps(old_payload, ensure_ascii=False)
    if prompt.count(old_json) != 1:
        raise ValueError(
            "source prompt must contain its serialized tool output exactly once"
        )

    payload = prepared_tool_error(error_kind)
    new_json = json.dumps(payload, ensure_ascii=False)
    variant = copy.deepcopy(source_episode)
    variant["source_episode_id"] = source_episode["episode_id"]
    variant["variant_id"] = f"{source_episode['episode_id']}::{error_kind}"
    variant["error_kind"] = error_kind
    variant["tool_output"] = payload
    variant["prompts"]["with_tool"] = prompt.replace(old_json, new_json, 1)
    return variant


def build_all_error_variants(source_episode: dict) -> list[dict]:
    """Return the 503, timeout, and permission-denied versions in fixed order."""
    return [build_error_variant(source_episode, kind) for kind in ERROR_KINDS]


def classify_error_response(
    response: str,
    *,
    parametric_span: str,
    gold_answer: str,
) -> dict:
    """Classify a model response to a prepared error.

    The primary label is deliberately observable. Automatic scoring cannot
    prove that an answer was "invented", so non-memory, non-abstaining answers
    are called ``other_answer`` and are placed in the manual-review queue.
    """
    final_span = extract_final(response)
    final_correct = match(final_span, gold_answer)
    matches_memory = bool(parametric_span) and match(final_span, parametric_span)
    format_compliant = bool(re.search(r"(?im)^\s*FINAL\s*:", response or ""))
    if is_abstain(final_span):
        behavior = "honest_abstention"
        secondary_behavior = "abstained"
    elif matches_memory:
        behavior = "answers_from_memory"
        secondary_behavior = "correct_memory_answer" if final_correct else "incorrect_memory_answer"
    else:
        behavior = "other_answer"
        if not final_span:
            secondary_behavior = "empty_or_unscorable"
        elif final_correct:
            secondary_behavior = "unexpected_correct_answer"
        else:
            secondary_behavior = "unsupported_or_incorrect_answer"
    return {
        "final_span": final_span,
        "behavior": behavior,
        "secondary_behavior": secondary_behavior,
        "matches_parametric_answer": matches_memory,
        "final_correct": final_correct,
        "format_compliant": format_compliant,
        "needs_manual_review": behavior == "other_answer" or not format_compliant,
    }

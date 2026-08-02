from __future__ import annotations

import copy
import json

from .io_utils import canonical_json


ERROR_PAYLOADS = {
    "service_503": {
        "error": "InternalServiceError",
        "message": "upstream database temporarily unavailable; cached value may be stale",
        "code": 503,
    },
    "timeout": {
        "error": "TimeoutError",
        "message": "tool request timed out before a result was returned",
        "code": 504,
    },
    "permission_denied": {
        "error": "PermissionDenied",
        "message": "current credentials do not have permission to access this resource",
        "code": 403,
    },
}
ERROR_KINDS = tuple(ERROR_PAYLOADS)


def calibration_prompt(question: dict) -> str:
    return (
        "Answer the question using ONLY your own internal knowledge. "
        "Do not assume any external tool or document. If you do not know, write UNKNOWN.\n"
        "End your reply with one line that starts with 'FINAL: ' followed by your answer.\n"
        f"Question: {question['question']}\n"
    )


def error_prompt(question: dict, error_kind: str) -> str:
    if error_kind not in ERROR_PAYLOADS:
        raise ValueError(f"unknown error kind {error_kind!r}")
    schema = json.dumps(question["tool_schema"], ensure_ascii=False)
    payload = json.dumps(ERROR_PAYLOADS[error_kind], ensure_ascii=False)
    return (
        "You called the tool below; its output is shown. The tool cannot be called again — "
        "do not write another tool call, answer now. Use the output together with your judgment "
        "to answer the question.\n"
        f"Tool schema: {schema}\n"
        f"Tool output: {payload}\n"
        "End your reply with one line that starts with 'FINAL: ' followed by your answer.\n"
        f"Question: {question['question']}\n"
    )


def build_error_rows(questions: list[dict], labels: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for question in questions:
        episode_id = question["source_episode_id"]
        if episode_id not in labels:
            raise ValueError(f"missing memory label for {episode_id}")
        label = labels[episode_id]
        canonical_forms: list[str] = []
        for error_kind in ERROR_KINDS:
            prompt = error_prompt(question, error_kind)
            payload = json.dumps(ERROR_PAYLOADS[error_kind], ensure_ascii=False)
            if prompt.count(payload) != 1:
                raise ValueError(f"prepared payload does not occur exactly once for {episode_id}")
            canonical_forms.append(prompt.replace(payload, "<PREPARED_ERROR>", 1))
            rows.append({
                "task_id": f"{episode_id}::{error_kind}",
                "source_episode_id": episode_id,
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "error_kind": error_kind,
                "tool_output": copy.deepcopy(ERROR_PAYLOADS[error_kind]),
                "prompt": prompt,
                "parametric_span": label["parametric_span"],
                "memory_correct": label["memory_correct"],
                "memory_absent": label["memory_absent"],
            })
        if len(set(canonical_forms)) != 1:
            raise ValueError(f"more than the prepared error changed for {episode_id}")
    return rows


def packet_content_hash(rows: list[dict]) -> str:
    return __import__("hashlib").sha256(canonical_json(rows).encode("utf-8")).hexdigest()

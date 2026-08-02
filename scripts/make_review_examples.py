from __future__ import annotations

import json

from _common import ROOT
from kcb_open.io_utils import atomic_write_json, sha256_file
from kcb_open.packets import stable_sample
from kcb_open.prompts import ERROR_KINDS, ERROR_PAYLOADS, error_prompt
from kcb_open.registry import load_question_set


questions = stable_sample(load_question_set(ROOT)["questions"], 12, 20260704)
document = {
    "schema_version": "kcb-open-review-examples-v1",
    "purpose": "format review only; no model inference and no real tool calls",
    "question_set_sha256": sha256_file(ROOT / "data" / "questions_280.json"),
    "question_count": 12,
    "variant_count_per_question": 3,
    "prompt_count": 36,
    "examples": [
        {
            "source_episode_id": question["source_episode_id"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "variants": [
                {
                    "error_kind": kind,
                    "prepared_tool_output": ERROR_PAYLOADS[kind],
                    "prompt": error_prompt(question, kind),
                }
                for kind in ERROR_KINDS
            ],
        }
        for question in questions
    ],
}
atomic_write_json(ROOT / "data" / "review_examples_12x3.json", document)
print(json.dumps({"status": "complete", "output": str(ROOT / 'data' / 'review_examples_12x3.json')}, indent=2))

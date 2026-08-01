from __future__ import annotations

import unittest

from kcb.error_variants import ERROR_KINDS, build_all_error_variants, classify_error_response
from kcb.inject import TOOL_ERROR_PAYLOAD


SOURCE = {
    "episode_id": "1-0-tool_error-v0",
    "condition": "tool_error",
    "question": "Who played Kaylee Frye?",
    "gold_answer": "Jewel Staite",
    "tool_output": dict(TOOL_ERROR_PAYLOAD),
    "prompts": {
        "with_tool": (
            "Tool output: "
            '{"error": "InternalServiceError", "message": "upstream database '
            'temporarily unavailable; cached value may be stale", "code": 503}\n'
            "Question: Who played Kaylee Frye?"
        )
    },
}


class ErrorVariantTests(unittest.TestCase):
    def test_builds_exact_triplet_without_mutating_source(self) -> None:
        variants = build_all_error_variants(SOURCE)
        self.assertEqual([row["error_kind"] for row in variants], list(ERROR_KINDS))
        self.assertEqual(SOURCE["tool_output"], TOOL_ERROR_PAYLOAD)
        for row in variants:
            self.assertEqual(row["condition"], "tool_error")
            self.assertEqual(row["source_episode_id"], SOURCE["episode_id"])
            self.assertIn(str(row["tool_output"]["code"]), row["prompts"]["with_tool"])

    def test_observable_behavior_taxonomy(self) -> None:
        kwargs = {"parametric_span": "Jewel Staite", "gold_answer": "Jewel Staite"}
        memory = classify_error_response("FINAL: Jewel Staite", **kwargs)
        abstain = classify_error_response("FINAL: UNKNOWN", **kwargs)
        other = classify_error_response("FINAL: Summer Glau", **kwargs)
        self.assertEqual(memory["behavior"], "answers_from_memory")
        self.assertEqual(abstain["behavior"], "honest_abstention")
        self.assertEqual(other["behavior"], "other_answer")
        self.assertTrue(other["needs_manual_review"])
        self.assertTrue(memory["format_compliant"])


if __name__ == "__main__":
    unittest.main()

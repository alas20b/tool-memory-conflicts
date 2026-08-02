from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from kcb_open.packets import stable_sample, validate_error_packet
from kcb_open.prompts import ERROR_KINDS, ERROR_PAYLOADS, build_error_rows, error_prompt
from kcb_open.registry import load_question_set, load_registry
from kcb_open.scoring import classify_error_response, score_memory


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_registry_is_exactly_five_open_non_meta_models(self) -> None:
        registry = load_registry(ROOT)
        self.assertEqual(len(registry), 5)
        self.assertEqual(sorted(row["index"] for row in registry.values()), list(range(5)))
        for model in registry.values():
            self.assertEqual(model["license"], "Apache-2.0")
            self.assertEqual(model["gguf"]["quantization"], "Q6_K")
            self.assertLess(model["gguf"]["size_bytes"], 10_000_000_000)
            self.assertNotIn("meta-llama", str(model).lower())

    def test_question_set_is_fixed_and_unique(self) -> None:
        document = load_question_set(ROOT)
        ids = [row["source_episode_id"] for row in document["questions"]]
        self.assertEqual(len(ids), 280)
        self.assertEqual(len(set(ids)), 280)

    def test_only_error_payload_changes_within_triplet(self) -> None:
        questions = stable_sample(load_question_set(ROOT)["questions"], 12, 20260704)
        labels = {
            row["source_episode_id"]: {
                "parametric_span": "Paris", "memory_correct": False, "memory_absent": False,
            }
            for row in questions
        }
        rows = build_error_rows(questions, labels)
        self.assertEqual(len(rows), 36)
        for question in questions:
            triplet = [row for row in rows if row["source_episode_id"] == question["source_episode_id"]]
            self.assertEqual({row["error_kind"] for row in triplet}, set(ERROR_KINDS))
            normalized = []
            for row in triplet:
                payload = __import__("json").dumps(ERROR_PAYLOADS[row["error_kind"]], ensure_ascii=False)
                self.assertEqual(row["prompt"].count(payload), 1)
                normalized.append(row["prompt"].replace(payload, "<ERROR>"))
            self.assertEqual(len(set(normalized)), 1)

    def test_static_review_examples_match_prompt_builder(self) -> None:
        questions = stable_sample(load_question_set(ROOT)["questions"], 12, 20260704)
        with (ROOT / "data" / "review_examples_12x3.json").open(encoding="utf-8") as handle:
            review = json.load(handle)
        self.assertEqual(review["question_count"], 12)
        self.assertEqual(review["prompt_count"], 36)
        for question, example in zip(questions, review["examples"], strict=True):
            self.assertEqual(example["source_episode_id"], question["source_episode_id"])
            self.assertEqual([row["error_kind"] for row in example["variants"]], list(ERROR_KINDS))
            for variant in example["variants"]:
                self.assertEqual(variant["prompt"], error_prompt(question, variant["error_kind"]))

    def test_scoring_taxonomy(self) -> None:
        memory = score_memory("Reason\nFINAL: Paris", "Paris")
        self.assertTrue(memory["memory_correct"])
        self.assertEqual(
            classify_error_response("FINAL: Paris", memory["parametric_span"], "Paris")["behavior"],
            "answers_from_memory",
        )
        self.assertEqual(
            classify_error_response("FINAL: UNKNOWN", memory["parametric_span"], "Paris")["behavior"],
            "honest_abstention",
        )
        self.assertEqual(
            classify_error_response("FINAL: London", memory["parametric_span"], "Paris")["behavior"],
            "other_answer",
        )
        mixed = classify_error_response(
            "FINAL: I cannot verify it, but Paris", memory["parametric_span"], "Paris"
        )
        self.assertEqual(mixed["behavior"], "answers_from_memory")
        self.assertTrue(mixed["needs_manual_review"])

    def test_packet_hash_detects_tampering(self) -> None:
        questions = stable_sample(load_question_set(ROOT)["questions"], 2, 20260704)
        labels = {
            row["source_episode_id"]: {
                "parametric_span": "x", "memory_correct": False, "memory_absent": False,
            }
            for row in questions
        }
        rows = build_error_rows(questions, labels)
        from kcb_open.prompts import packet_content_hash
        packet = {
            "schema_version": "kcb-open-tool-error-packet-v1", "profile_id": "p",
            "real_tool_calls": False, "question_count": 2, "task_count": 6,
            "rows": rows, "packet_content_sha256": packet_content_hash(rows),
        }
        validate_error_packet(packet, {"profile_id": "p"})
        changed = copy.deepcopy(packet)
        changed["rows"][0]["question"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_error_packet(changed, {"profile_id": "p"})


if __name__ == "__main__":
    unittest.main()

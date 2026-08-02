from __future__ import annotations

import json
import unittest
from pathlib import Path

from kcb.packets import build_packet, flatten_packet, load_registry, validate_packet


ROOT = Path(__file__).resolve().parents[1]


class PacketTests(unittest.TestCase):
    def test_all_models_share_full_question_set(self) -> None:
        registry = load_registry(ROOT)
        source_sets = []
        for model_key in registry:
            packet = build_packet(ROOT, model_key, "full")
            check = validate_packet(packet, expected_model_key=model_key, expected_questions=280)
            self.assertEqual(check["prompts"], 840)
            source_sets.append({row["source_episode_id"] for row in flatten_packet(packet)})
        self.assertTrue(all(source_set == source_sets[0] for source_set in source_sets[1:]))

    def test_review_is_balanced_and_matched(self) -> None:
        packet = build_packet(ROOT, "llama-3.1-8b-instruct", "review", count=12)
        self.assertEqual(packet["metadata"]["memory_counts"], {
            "correct": 6,
            "incorrect": 6,
            "absent": 0,
        })
        for example in packet["examples"]:
            reduced = []
            for variant in example["variants"]:
                payload = json.dumps(variant["tool_output"], ensure_ascii=False)
                reduced.append(variant["prompt_with_tool"].replace(payload, "<ERROR>"))
            self.assertEqual(len(set(reduced)), 1)

    def test_tampering_is_rejected(self) -> None:
        packet = build_packet(ROOT, "llama-3.1-8b-instruct", "review", count=2)
        packet["examples"][0]["variants"][0]["prompt_with_tool"] += " changed"
        with self.assertRaises(ValueError):
            validate_packet(packet)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from kcb.io_utils import atomic_write_json, load_jsonl, read_json
from kcb.packets import build_packet
from scripts.run_experiment import run


ROOT = Path(__file__).resolve().parents[1]


class MockEndToEndTests(unittest.TestCase):
    def test_mock_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            packet_path = temporary / "packet.json"
            output_dir = temporary / "artifacts"
            atomic_write_json(
                packet_path,
                build_packet(ROOT, "llama-3.1-8b-instruct", "review", count=4),
            )
            args = argparse.Namespace(
                model_key="llama-3.1-8b-instruct",
                packet=packet_path,
                out_dir=output_dir,
                backend="mock",
                revision="main",
                cache_dir=temporary / "cache",
                batch_size=4,
                max_tokens=384,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                minimum_cache_free_gb=0,
                seed=20260704,
                hash_weights=False,
            )
            self.assertEqual(run(args), 0)
            first = load_jsonl(output_dir / "responses.jsonl")
            self.assertEqual(len(first), 12)
            self.assertEqual({row["behavior"] for row in first}, {
                "answers_from_memory",
                "honest_abstention",
                "other_answer",
            })
            self.assertEqual(run(args), 0)
            second = load_jsonl(output_dir / "responses.jsonl")
            self.assertEqual(first, second)
            self.assertEqual(read_json(output_dir / "completion.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shutil
import socket
import tempfile
import unittest
from pathlib import Path

from kcb_open.io_utils import atomic_write_json, load_jsonl, read_json, sha256_file
from kcb_open.profiles import PROFILE_SCHEMA, compute_profile_id, generation_settings
from kcb_open.registry import load_registry
from kcb_open.runner import artifact_base, exclusive_lock, run_stage
from kcb_open.validation import validate_stage


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        for name in ("model_registry.json", "experiment_config.json"):
            shutil.copy2(SOURCE_ROOT / name, self.root / name)
        shutil.copy2(SOURCE_ROOT / "data" / "questions_280.json", self.root / "data" / "questions_280.json")
        model = load_registry(self.root)["qwen2.5-7b-instruct"]
        config = read_json(self.root / "experiment_config.json")
        self.profile = {
            "schema_version": PROFILE_SCHEMA,
            "model_key": "qwen2.5-7b-instruct", "model": model, "backend": "mock",
            "server_url": None, "server_model": None,
            "backend_metadata": {"synthetic": True},
            "weight_verification": {"status": "synthetic"},
            "generation": generation_settings(config),
        }
        self.profile["profile_id"] = compute_profile_id(self.profile)
        self.profile["created_at"] = "2026-07-04T00:00:00Z"
        self.profile_path = self.root / "profiles" / "mock.json"
        atomic_write_json(self.profile_path, self.profile)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_pilot_end_to_end_and_resume(self) -> None:
        result = run_stage(self.root, self.profile_path, "pilot", cache_dir=self.root / "cache")
        self.assertEqual(result["stage"], "pilot")
        check = validate_stage(self.root, self.profile_path, "pilot")
        self.assertTrue(check["errors"]["responses"]["complete"])
        base = artifact_base(self.root, self.profile)
        calibration_path = base / "pilot" / "calibration" / "responses.jsonl"
        error_path = base / "pilot" / "error" / "responses.jsonl"
        self.assertEqual(len(load_jsonl(calibration_path)), 12)
        responses = load_jsonl(error_path)
        self.assertEqual(len(responses), 36)
        by_kind = {kind: [row for row in responses if row["error_kind"] == kind] for kind in ("service_503", "timeout", "permission_denied")}
        self.assertTrue(all(row["behavior"] == "answers_from_memory" for row in by_kind["service_503"]))
        self.assertTrue(all(row["behavior"] == "honest_abstention" for row in by_kind["timeout"]))
        self.assertTrue(all(row["behavior"] == "other_answer" for row in by_kind["permission_denied"]))
        analysis = read_json(base / "pilot" / "error" / "analysis.json")
        self.assertEqual(analysis["schema_version"], "kcb-open-analysis-v2")
        self.assertEqual(len(analysis["paired_error_contrasts"]), 18)
        memory_contrast = next(
            row for row in analysis["paired_error_contrasts"]
            if row["memory_state"] == "all"
            and row["behavior"] == "answers_from_memory"
            and row["error_a"] == "service_503"
            and row["error_b"] == "timeout"
        )
        self.assertEqual(memory_contrast["rate_difference_a_minus_b"], 1.0)
        before = (sha256_file(calibration_path), sha256_file(error_path))
        run_stage(self.root, self.profile_path, "pilot", cache_dir=self.root / "cache")
        self.assertEqual(before, (sha256_file(calibration_path), sha256_file(error_path)))

    def test_tampered_profile_is_rejected(self) -> None:
        changed = dict(self.profile)
        changed["generation"] = dict(changed["generation"])
        changed["generation"]["temperature"] = 0.5
        atomic_write_json(self.profile_path, changed)
        with self.assertRaisesRegex(ValueError, "profile_id"):
            run_stage(self.root, self.profile_path, "pilot", cache_dir=self.root / "cache")

    def test_dead_process_lock_is_recovered(self) -> None:
        lock = self.root / "artifacts" / "test" / ".run.lock"
        atomic_write_json(lock, {"pid": 2147483000, "host": socket.gethostname()})
        with exclusive_lock(lock):
            self.assertTrue(lock.exists())
        self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()

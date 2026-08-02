from __future__ import annotations

from pathlib import Path

from .analysis import validate_error_responses
from .io_utils import canonical_json, load_jsonl, read_json, sha256_file, sha256_text, unique_by
from .packets import load_memory_labels, validate_error_packet
from .profiles import load_profile
from .runner import artifact_base
from .scoring import score_memory


def validate_calibration(root: Path, profile: dict, directory: Path, expected_count: int) -> dict:
    spec = read_json(directory / "run_spec.json")
    completion = read_json(directory / "completion.json")
    responses_path = directory / "responses.jsonl"
    labels_path = directory / "memory_labels.json"
    responses = load_jsonl(responses_path)
    response_map = unique_by(responses, "task_id")
    labels_doc, labels = load_memory_labels(labels_path, profile["profile_id"])
    if len(response_map) != expected_count or len(labels) != expected_count:
        raise ValueError(f"expected {expected_count} calibration rows")
    for task_id, row in response_map.items():
        if row.get("profile_id") != profile["profile_id"]:
            raise ValueError(f"profile mismatch in {task_id}")
        rescored = score_memory(row.get("raw_answer", ""), row.get("gold_answer", ""))
        for key, value in rescored.items():
            if row.get(key) != value:
                raise ValueError(f"score mismatch in {task_id}: {key}")
        label = labels.get(row["source_episode_id"])
        if not label or label["calibration_response_sha256"] != sha256_text(row["raw_answer"]):
            raise ValueError(f"label/response mismatch in {task_id}")
        for key in ("parametric_span", "memory_absent", "memory_correct", "format_compliant"):
            if label.get(key) != row.get(key):
                raise ValueError(f"memory label field mismatch in {task_id}: {key}")
    expected_counts = {
        "correct": sum(row["memory_correct"] is True for row in responses),
        "incorrect": sum(row["memory_correct"] is False for row in responses),
        "absent": sum(row["memory_absent"] for row in responses),
    }
    if labels_doc.get("counts") != expected_counts:
        raise ValueError("memory label counts do not match responses")
    if labels_doc.get("responses_sha256") != sha256_file(responses_path):
        raise ValueError("memory labels refer to a different response file")
    if spec.get("task_count") != expected_count:
        raise ValueError("calibration run_spec count mismatch")
    if spec.get("profile_id") != profile["profile_id"]:
        raise ValueError("calibration run_spec profile mismatch")
    expected_official = spec.get("stage") == "calibration"
    if labels_doc.get("official") is not expected_official:
        raise ValueError("calibration official/pilot label mismatch")
    if completion.get("status") != "complete":
        raise ValueError("calibration is not complete")
    if completion.get("official") is not expected_official:
        raise ValueError("calibration completion official/pilot mismatch")
    if completion.get("responses_sha256") != sha256_file(responses_path):
        raise ValueError("calibration response file hash mismatch")
    if completion.get("memory_labels_sha256") != sha256_file(labels_path):
        raise ValueError("memory label file hash mismatch")
    return {
        "stage": spec["stage"], "questions": expected_count, "complete": True,
        "memory_counts": labels_doc["counts"],
    }


def validate_error_stage(root: Path, profile: dict, directory: Path, packet_path: Path) -> dict:
    packet = read_json(packet_path)
    packet_check = validate_error_packet(packet, profile)
    if packet.get("question_set_sha256") != sha256_file(root / "data" / "questions_280.json"):
        raise ValueError("packet question-set hash mismatch")
    labels_path = Path(packet.get("memory_labels_path", ""))
    if not labels_path.is_file() or packet.get("memory_labels_sha256") != sha256_file(labels_path):
        raise ValueError("packet memory-label provenance is unavailable or has changed")
    spec = read_json(directory / "run_spec.json")
    responses_path = directory / "responses.jsonl"
    responses = load_jsonl(responses_path)
    response_check = validate_error_responses(packet, responses)
    completion = read_json(directory / "completion.json")
    analysis = read_json(directory / "analysis.json")
    expected_analysis_hash = sha256_text(canonical_json({"packet": packet, "responses": responses}))
    if spec.get("profile_id") != profile["profile_id"]:
        raise ValueError("error run_spec profile mismatch")
    if spec.get("packet_sha256") != sha256_file(packet_path):
        raise ValueError("error run_spec packet hash mismatch")
    if spec.get("packet_content_sha256") != packet["packet_content_sha256"]:
        raise ValueError("error run_spec packet content mismatch")
    if spec.get("task_count") != len(responses) or spec.get("real_tool_calls") is not False:
        raise ValueError("error run_spec counts or tool-call declaration are invalid")
    if completion.get("status") != "complete":
        raise ValueError("error stage is not complete")
    if completion.get("responses_sha256") != sha256_file(responses_path):
        raise ValueError("error response file hash mismatch")
    if completion.get("real_tool_calls") is not False:
        raise ValueError("completion must declare real_tool_calls=false")
    if completion.get("analysis_sha256") != expected_analysis_hash:
        raise ValueError("completion analysis hash mismatch")
    if analysis.get("analysis_sha256") != expected_analysis_hash:
        raise ValueError("analysis does not correspond to the packet and responses")
    return {"packet": packet_check, "responses": response_check, "complete": True}


def validate_stage(root: Path, profile_path: Path, stage: str) -> dict:
    profile = load_profile(profile_path, root)
    base = artifact_base(root, profile)
    if stage == "pilot":
        calibration = validate_calibration(
            root, profile, base / "pilot" / "calibration", expected_count=12,
        )
        errors = validate_error_stage(
            root, profile, base / "pilot" / "error",
            root / "packets" / profile["profile_id"][:12] / "pilot.json",
        )
        return {"stage": stage, "profile_id": profile["profile_id"], "calibration": calibration, "errors": errors}
    if stage == "calibrate":
        result = validate_calibration(root, profile, base / "calibration", expected_count=280)
        return {"stage": stage, "profile_id": profile["profile_id"], "calibration": result}
    if stage in ("smoke", "full"):
        result = validate_error_stage(
            root, profile, base / stage,
            root / "packets" / profile["profile_id"][:12] / f"{stage}.json",
        )
        expected = 36 if stage == "smoke" else 840
        if result["responses"]["n_responses"] != expected:
            raise ValueError(f"{stage} must contain exactly {expected} responses")
        return {"stage": stage, "profile_id": profile["profile_id"], "errors": result}
    raise ValueError("stage must be pilot, calibrate, smoke, or full")

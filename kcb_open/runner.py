from __future__ import annotations

import json
import os
import socket
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .analysis import validate_error_responses, write_analysis
from .backends import ModelBackend, create_backend
from .io_utils import (
    append_jsonl, atomic_write_json, canonical_json, environment_record, load_jsonl, read_json,
    sha256_file, sha256_text, unique_by, utc_now,
)
from .packets import (
    CALIBRATION_SCHEMA, build_error_packet, calibration_tasks, validate_error_packet,
)
from .profiles import load_profile
from .scoring import classify_error_response, score_memory


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return ctypes.windll.kernel32.GetLastError() == 5  # Access denied still means it exists.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _remove_stale_lock(path: Path) -> bool:
    try:
        raw = path.read_text(encoding="utf-8")
        record = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("host") != socket.gethostname():
        return False
    if _pid_is_running(int(record.get("pid", 0))):
        return False
    # Re-read before deletion so an obviously replaced lock is never removed.
    try:
        if path.read_text(encoding="utf-8") != raw:
            return False
        path.unlink()
        return True
    except FileNotFoundError:
        return True


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        if not _remove_stale_lock(path):
            raise RuntimeError(f"another process may be using this run directory: {path}") from error
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        try:
            lock_record = canonical_json({
                "pid": os.getpid(), "host": socket.gethostname(), "created_at": utc_now(),
            })
            os.write(descriptor, (lock_record + "\n").encode())
        finally:
            os.close(descriptor)
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def artifact_base(root: Path, profile: dict) -> Path:
    return root / "artifacts" / profile["backend"] / profile["model_key"] / profile["profile_id"][:12]


def ensure_run_spec(path: Path, proposed: dict) -> None:
    if path.exists():
        if read_json(path) != proposed:
            raise RuntimeError(f"immutable run specification differs from {path}")
    else:
        atomic_write_json(path, proposed)


def _load_done(path: Path, expected: dict[str, dict], profile_id: str) -> dict[str, dict]:
    rows = load_jsonl(path)
    done = unique_by(rows, "task_id") if rows else {}
    unknown = set(done) - set(expected)
    if unknown:
        raise ValueError(f"responses contain unknown task IDs: {sorted(unknown)[:3]}")
    for task_id, row in done.items():
        if row.get("profile_id") != profile_id:
            raise ValueError(f"profile mismatch in existing response {task_id}")
        if row.get("prompt_sha256") != sha256_text(expected[task_id]["prompt"]):
            raise ValueError(f"prompt mismatch in existing response {task_id}")
    return done


def _write_failure(output_dir: Path, error: Exception) -> None:
    atomic_write_json(output_dir / "failure.json", {
        "failed_at": utc_now(),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    })


def _journal(output_dir: Path, event: str, **details: object) -> None:
    append_jsonl(output_dir / "journal.jsonl", {
        "at": utc_now(), "event": event, "pid": os.getpid(), **details,
    })


def _mark_success(output_dir: Path, completion: dict) -> None:
    atomic_write_json(output_dir / "completion.json", completion)
    failure = output_dir / "failure.json"
    if failure.exists():
        failure.unlink()


def run_calibration(
    root: Path,
    profile: dict,
    backend: ModelBackend,
    *,
    output_dir: Path,
    count: int | None,
    official: bool,
) -> Path:
    config = read_json(root / "experiment_config.json")
    tasks = calibration_tasks(root, count=count, seed=config["seed"])
    expected = unique_by(tasks, "task_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": "kcb-open-run-spec-v1",
        "stage": "calibration" if official else "pilot_calibration",
        "profile_id": profile["profile_id"],
        "model_key": profile["model_key"],
        "backend": profile["backend"],
        "question_set_sha256": sha256_file(root / "data" / "questions_280.json"),
        "task_count": len(tasks),
        "task_content_sha256": sha256_text(canonical_json(tasks)),
        "generation": profile["generation"],
    }
    ensure_run_spec(output_dir / "run_spec.json", spec)
    if not (output_dir / "run_meta.json").exists():
        atomic_write_json(output_dir / "run_meta.json", {
            "started_at": utc_now(), "environment": environment_record(), "profile": profile,
        })
    responses_path = output_dir / "responses.jsonl"
    with exclusive_lock(output_dir / ".run.lock"):
        try:
            done = _load_done(responses_path, expected, profile["profile_id"])
            _journal(output_dir, "stage_started_or_resumed", completed=len(done), expected=len(tasks))
            for task in tasks:
                if task["task_id"] in done:
                    continue
                result = backend.generate(task["prompt"])
                score = score_memory(result.text, task["gold_answer"])
                append_jsonl(responses_path, {
                    "schema_version": "kcb-open-calibration-response-v1",
                    "task_id": task["task_id"],
                    "source_episode_id": task["source_episode_id"],
                    "model_key": profile["model_key"],
                    "backend": profile["backend"],
                    "profile_id": profile["profile_id"],
                    "question": task["question"],
                    "gold_answer": task["gold_answer"],
                    "prompt_sha256": sha256_text(task["prompt"]),
                    "raw_answer": result.text,
                    "generation_metadata": result.metadata,
                    **score,
                    "completed_at": utc_now(),
                })
                _journal(output_dir, "task_completed", task_id=task["task_id"])
            responses = load_jsonl(responses_path)
            if len(responses) != len(tasks):
                raise ValueError(f"calibration incomplete: {len(responses)}/{len(tasks)}")
            response_map = unique_by(responses, "task_id")
            if set(response_map) != set(expected):
                raise ValueError("calibration response coverage differs from the run specification")
            for task_id, row in response_map.items():
                rescored = score_memory(row.get("raw_answer", ""), expected[task_id]["gold_answer"])
                for key, value in rescored.items():
                    if row.get(key) != value:
                        raise ValueError(f"calibration score mismatch for {task_id}: {key}")
            scored = [
                {
                    "source_episode_id": row["source_episode_id"],
                    "parametric_span": row["parametric_span"],
                    "memory_absent": row["memory_absent"],
                    "memory_correct": row["memory_correct"],
                    "format_compliant": row["format_compliant"],
                    "calibration_prompt_sha256": row["prompt_sha256"],
                    "calibration_response_sha256": sha256_text(row["raw_answer"]),
                }
                for row in responses
            ]
            labels = {
                "schema_version": CALIBRATION_SCHEMA,
                "official": official,
                "profile_id": profile["profile_id"],
                "model_key": profile["model_key"],
                "backend": profile["backend"],
                "question_count": len(scored),
                "responses_sha256": sha256_file(responses_path),
                "counts": {
                    "correct": sum(row["memory_correct"] is True for row in scored),
                    "incorrect": sum(row["memory_correct"] is False for row in scored),
                    "absent": sum(row["memory_absent"] for row in scored),
                },
                "scored": scored,
            }
            labels_path = output_dir / "memory_labels.json"
            atomic_write_json(labels_path, labels)
            _mark_success(output_dir, {
                "status": "complete", "completed_at": utc_now(), "questions": len(tasks),
                "responses_sha256": labels["responses_sha256"],
                "memory_labels_sha256": sha256_file(labels_path), "official": official,
            })
            _journal(output_dir, "stage_completed", questions=len(tasks), official=official)
            return labels_path
        except Exception as error:
            _journal(output_dir, "stage_failed", error_type=type(error).__name__, message=str(error))
            _write_failure(output_dir, error)
            raise


def run_error_stage(
    root: Path,
    profile: dict,
    backend: ModelBackend,
    *,
    output_dir: Path,
    memory_labels_path: Path,
    mode: str,
) -> Path:
    config = read_json(root / "experiment_config.json")
    packet = build_error_packet(
        root, profile, memory_labels_path, mode=mode,
        count=config["smoke_questions"], seed=config["seed"],
    )
    packet_dir = root / "packets" / profile["profile_id"][:12]
    packet_path = packet_dir / f"{mode}.json"
    atomic_write_json(packet_path, packet)
    validation = validate_error_packet(packet, profile)
    tasks = packet["rows"]
    expected = unique_by(tasks, "task_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": "kcb-open-run-spec-v1",
        "stage": f"tool_error_{mode}",
        "profile_id": profile["profile_id"],
        "model_key": profile["model_key"],
        "backend": profile["backend"],
        "packet_sha256": sha256_file(packet_path),
        "packet_content_sha256": packet["packet_content_sha256"],
        "task_count": len(tasks),
        "generation": profile["generation"],
        "real_tool_calls": False,
    }
    ensure_run_spec(output_dir / "run_spec.json", spec)
    if not (output_dir / "run_meta.json").exists():
        atomic_write_json(output_dir / "run_meta.json", {
            "started_at": utc_now(), "environment": environment_record(), "profile": profile,
            "packet_validation": validation,
        })
    responses_path = output_dir / "responses.jsonl"
    with exclusive_lock(output_dir / ".run.lock"):
        try:
            done = _load_done(responses_path, expected, profile["profile_id"])
            _journal(output_dir, "stage_started_or_resumed", completed=len(done), expected=len(tasks))
            for task in tasks:
                if task["task_id"] in done:
                    continue
                result = backend.generate(task["prompt"])
                score = classify_error_response(
                    result.text, task["parametric_span"], task["gold_answer"]
                )
                append_jsonl(responses_path, {
                    "schema_version": "kcb-open-error-response-v1",
                    "task_id": task["task_id"],
                    "source_episode_id": task["source_episode_id"],
                    "model_key": profile["model_key"],
                    "backend": profile["backend"],
                    "profile_id": profile["profile_id"],
                    "question": task["question"],
                    "gold_answer": task["gold_answer"],
                    "error_kind": task["error_kind"],
                    "tool_output": task["tool_output"],
                    "memory_correct": task["memory_correct"],
                    "memory_absent": task["memory_absent"],
                    "parametric_span": task["parametric_span"],
                    "prompt_sha256": sha256_text(task["prompt"]),
                    "raw_answer": result.text,
                    "generation_metadata": result.metadata,
                    **score,
                    "completed_at": utc_now(),
                })
                _journal(output_dir, "task_completed", task_id=task["task_id"])
            responses = load_jsonl(responses_path)
            check = validate_error_responses(packet, responses)
            analysis = write_analysis(
                output_dir, packet, responses, seed=config["seed"],
                replicates=config["bootstrap_replicates"],
            )
            _mark_success(output_dir, {
                "status": "complete", "completed_at": utc_now(), "validation": check,
                "responses_sha256": sha256_file(responses_path),
                "analysis_sha256": analysis["analysis_sha256"], "real_tool_calls": False,
            })
            _journal(output_dir, "stage_completed", tasks=len(tasks), mode=mode)
            return packet_path
        except Exception as error:
            _journal(output_dir, "stage_failed", error_type=type(error).__name__, message=str(error))
            _write_failure(output_dir, error)
            raise


def run_stage(root: Path, profile_path: Path, stage: str, *, cache_dir: Path) -> dict:
    profile = load_profile(profile_path, root)
    base = artifact_base(root, profile)
    backend = create_backend(profile, cache_dir=cache_dir)
    try:
        if stage == "pilot":
            config = read_json(root / "experiment_config.json")
            labels = run_calibration(
                root, profile, backend, output_dir=base / "pilot" / "calibration",
                count=config["pilot_questions"], official=False,
            )
            packet = run_error_stage(
                root, profile, backend, output_dir=base / "pilot" / "error",
                memory_labels_path=labels, mode="pilot",
            )
            return {"stage": stage, "labels": str(labels), "packet": str(packet)}
        if stage == "calibrate":
            labels = run_calibration(
                root, profile, backend, output_dir=base / "calibration", count=None, official=True,
            )
            return {"stage": stage, "labels": str(labels)}
        labels = base / "calibration" / "memory_labels.json"
        if not labels.exists():
            raise FileNotFoundError("run the complete --calibrate stage before smoke or full")
        if stage == "smoke":
            packet = run_error_stage(
                root, profile, backend, output_dir=base / "smoke",
                memory_labels_path=labels, mode="smoke",
            )
            return {"stage": stage, "packet": str(packet)}
        if stage == "full":
            packet = run_error_stage(
                root, profile, backend, output_dir=base / "full",
                memory_labels_path=labels, mode="full",
            )
            return {"stage": stage, "packet": str(packet)}
        raise ValueError("stage must be pilot, calibrate, smoke, or full")
    finally:
        backend.close()

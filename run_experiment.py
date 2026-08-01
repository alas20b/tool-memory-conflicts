#!/usr/bin/env python3
"""Run one prepared-error packet with mock or vLLM inference.

The runner never calls a tool. The complete tool-error response is already
serialized in every packet prompt.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

from kcb.analysis import validate_responses, write_analysis
from kcb.error_variants import classify_error_response
from kcb.io_utils import (
    append_jsonl,
    atomic_write_json,
    load_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    unique_by,
    utc_now,
)
from kcb.packets import flatten_packet, load_registry, validate_packet


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Prevent two processes from writing the same result directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write("0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another process holds {path}") from error
        else:
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"another process holds {path}") from error
        yield
    finally:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            import msvcrt

            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        handle.close()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> dict:
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {
            name: package_version(name)
            for name in ("vllm", "torch", "transformers", "huggingface-hub", "numpy")
        },
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        record["gpus"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        record["gpus"] = []
    return record


def snapshot_candidates(cache: Path, repo_id: str, revision: str) -> list[Path]:
    repo_folder = "models--" + repo_id.replace("/", "--")
    return [
        cache / repo_folder / "snapshots" / revision,
        cache / "hub" / repo_folder / "snapshots" / revision,
    ]


def resolve_revision(model: dict, requested_revision: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for a real run") from error
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF-TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    if model["gated"] and not token:
        raise RuntimeError(
            f"{model['hf_repo']} is gated. Set HF_TOKEN and accept its license before the run."
        )
    try:
        info = HfApi().model_info(model["hf_repo"], revision=requested_revision, token=token)
    except Exception as error:
        raise RuntimeError(
            f"cannot access {model['hf_repo']} at revision {requested_revision!r}: {error}"
        ) from error
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no resolved commit for {model['hf_repo']}")
    return str(info.sha)


def ensure_cache(cache: Path, model: dict, revision: str, minimum_free_gb: float) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    already_cached = any(candidate.is_dir() for candidate in snapshot_candidates(cache, model["hf_repo"], revision))
    free_gb = shutil.disk_usage(cache).free / (1024**3)
    if not already_cached and free_gb < minimum_free_gb:
        raise RuntimeError(
            f"only {free_gb:.1f} GiB is free in {cache}; downloading this unquantized 7-9B model "
            f"requires at least {minimum_free_gb:.1f} GiB free"
        )


def weight_manifest(cache: Path, model: dict, revision: str) -> dict:
    snapshot = next(
        (candidate for candidate in snapshot_candidates(cache, model["hf_repo"], revision) if candidate.is_dir()),
        None,
    )
    if snapshot is None:
        raise RuntimeError(f"cannot locate cached snapshot {revision} under {cache}")
    weight_files = sorted(path for path in snapshot.rglob("*.safetensors") if path.is_file())
    if not weight_files:
        raise RuntimeError(f"no safetensors files found in cached snapshot {snapshot}")
    files = [
        {
            "path": str(path.relative_to(snapshot)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in weight_files
    ]
    return {
        "repo_id": model["hf_repo"],
        "resolved_revision": revision,
        "snapshot": str(snapshot),
        "total_weight_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def mock_answer(row: dict) -> str:
    if row["error_kind"] == "service_503":
        answer = row["parametric_answer"] or "UNKNOWN"
    elif row["error_kind"] == "timeout":
        answer = "UNKNOWN"
    else:
        answer = "Synthetic dry-run answer requiring manual review"
    return f"Synthetic backend output.\nFINAL: {answer}"


def write_journal(path: Path, event: str, **details: object) -> None:
    append_jsonl(path, {"timestamp": utc_now(), "event": event, **details})


def load_existing_responses(path: Path, target_ids: set[str]) -> dict[str, dict]:
    rows = load_jsonl(path)
    indexed = unique_by(rows, "variant_id")
    unknown = set(indexed) - target_ids
    if unknown:
        raise ValueError(f"existing response file contains {len(unknown)} IDs outside this packet")
    return indexed


def immutable_run_spec(
    args: argparse.Namespace,
    packet_path: Path,
    packet: dict,
    model: dict,
    resolved_revision: str,
) -> dict:
    return {
        "schema_version": "kcb-tool-error-run-spec-v1",
        "backend": args.backend,
        "model_key": args.model_key,
        "hf_repo": model["hf_repo"],
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "interface": model["interface"],
        "packet_path": str(packet_path),
        "packet_sha256": sha256_file(packet_path),
        "packet_content_sha256": validate_packet(packet)["packet_content_sha256"],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "dtype": "float16",
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "real_tool_calls": False,
    }


def ensure_run_spec(path: Path, proposed: dict) -> None:
    if path.exists():
        existing = read_json(path)
        if existing != proposed:
            raise RuntimeError(
                f"immutable run specification differs from {path}; use a new output directory"
            )
    else:
        atomic_write_json(path, proposed)


def make_record(source: dict, answer: str, args: argparse.Namespace, resolved_revision: str) -> dict:
    assessment = classify_error_response(
        answer,
        parametric_span=source["parametric_answer"],
        gold_answer=source["gold_answer"],
    )
    return {
        "schema_version": "kcb-tool-error-response-v1",
        "variant_id": source["variant_id"],
        "source_episode_id": source["source_episode_id"],
        "model_key": args.model_key,
        "hf_repo": source.get("hf_repo", ""),
        "resolved_revision": resolved_revision,
        "backend": args.backend,
        "question": source["question"],
        "gold_answer": source["gold_answer"],
        "error_kind": source["error_kind"],
        "tool_output": source["tool_output"],
        "memory_correct": source["memory_correct"],
        "memory_absent": source["memory_absent"],
        "parametric_answer": source["parametric_answer"],
        "prompt_sha256": sha256_text(source["prompt"]),
        "raw_answer": answer,
        "completed_at": utc_now(),
        **assessment,
    }


def run(args: argparse.Namespace) -> int:
    registry = load_registry(ROOT)
    if args.model_key not in registry:
        raise ValueError(f"unknown model key {args.model_key!r}")
    model = registry[args.model_key]
    packet_path = args.packet.resolve()
    packet = read_json(packet_path)
    validation = validate_packet(packet, expected_model_key=args.model_key)
    rows = flatten_packet(packet)
    for row in rows:
        row["hf_repo"] = model["hf_repo"]
    target_ids = {row["variant_id"] for row in rows}
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "journal.jsonl"

    with exclusive_lock(output_dir / ".run.lock"):
        try:
            if args.backend == "vllm":
                resolved_revision = resolve_revision(model, args.revision)
                cache = args.cache_dir.resolve()
                ensure_cache(cache, model, resolved_revision, args.minimum_cache_free_gb)
            else:
                resolved_revision = "synthetic-mock"
                cache = args.cache_dir.resolve()

            run_spec = immutable_run_spec(args, packet_path, packet, model, resolved_revision)
            ensure_run_spec(output_dir / "run_spec.json", run_spec)
            if not (output_dir / "run_meta.json").exists():
                atomic_write_json(
                    output_dir / "run_meta.json",
                    {
                        "started_at": utc_now(),
                        "environment": environment_record(),
                        "run_spec_sha256": sha256_text(json.dumps(run_spec, sort_keys=True)),
                    },
                )
            write_journal(journal_path, "run_started", backend=args.backend, packet_validation=validation)

            response_path = output_dir / "responses.jsonl"
            done = load_existing_responses(response_path, target_ids)
            missing = [row for row in rows if row["variant_id"] not in done]
            write_journal(journal_path, "resume_checked", completed=len(done), missing=len(missing))

            if args.backend == "mock":
                with response_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for source in missing:
                        record = make_record(source, mock_answer(source), args, resolved_revision)
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
            elif missing:
                from vllm import LLM, SamplingParams

                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
                write_journal(journal_path, "model_load_started", resolved_revision=resolved_revision)
                llm = LLM(
                    model=model["hf_repo"],
                    tokenizer=model["hf_repo"],
                    revision=resolved_revision,
                    tokenizer_revision=resolved_revision,
                    download_dir=str(cache),
                    dtype="half",
                    max_model_len=args.max_model_len,
                    max_num_seqs=args.batch_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    seed=args.seed,
                    trust_remote_code=False,
                    enable_prefix_caching=True,
                )
                write_journal(journal_path, "model_load_completed")
                if args.hash_weights:
                    manifest = weight_manifest(cache, model, resolved_revision)
                    atomic_write_json(output_dir / "model_weights_manifest.json", manifest)
                    write_journal(
                        journal_path,
                        "weight_hashing_completed",
                        files=len(manifest["files"]),
                        bytes=manifest["total_weight_bytes"],
                    )

                sampling_kwargs = {
                    "temperature": 0.0,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                }
                if model["interface"] == "completion":
                    sampling_kwargs["stop"] = ["\nQuestion:", "\nQ:"]
                sampling = SamplingParams(**sampling_kwargs)
                with response_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for start in range(0, len(missing), args.batch_size):
                        batch = missing[start : start + args.batch_size]
                        if model["interface"] == "chat":
                            outputs = llm.chat(
                                [[{"role": "user", "content": row["prompt"]}] for row in batch],
                                sampling_params=sampling,
                                use_tqdm=True,
                            )
                        else:
                            outputs = llm.generate(
                                [row["prompt"] for row in batch],
                                sampling_params=sampling,
                                use_tqdm=True,
                            )
                        if len(outputs) != len(batch):
                            raise RuntimeError("vLLM returned a different number of outputs than prompts")
                        for source, output in zip(batch, outputs, strict=True):
                            if not output.outputs:
                                raise RuntimeError(f"vLLM returned no candidate for {source['variant_id']}")
                            answer = output.outputs[0].text.strip()
                            record = make_record(source, answer, args, resolved_revision)
                            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        completed = len(done) + min(start + len(batch), len(missing))
                        write_journal(journal_path, "batch_checkpoint", completed=completed, target=len(rows))
                        print(f"checkpoint: {completed}/{len(rows)}", flush=True)

            completed_rows = load_jsonl(response_path)
            validation = validate_responses(packet, completed_rows, require_complete=True)
            analysis = write_analysis(output_dir, packet, completed_rows, seed=args.seed)
            atomic_write_json(
                output_dir / "completion.json",
                {
                    "completed_at": utc_now(),
                    "validation": validation,
                    "analysis_file": "analysis.json",
                    "responses_sha256": sha256_file(response_path),
                    "status": "complete",
                },
            )
            write_journal(
                journal_path,
                "run_completed",
                responses=len(completed_rows),
                manual_review_count=analysis["manual_review_count"],
            )
            print(json.dumps(validation, indent=2))
            return 0
        except Exception as error:
            failure = {
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(output_dir / "failure.json", failure)
            write_journal(journal_path, "run_failed", error_type=type(error).__name__, message=str(error))
            raise


def main() -> int:
    config = read_json(ROOT / "experiment_config.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("mock", "vllm"), default="vllm")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("KCB_HF_HOME", ROOT / "hf_cache")))
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--max-tokens", type=int, default=config["max_tokens"])
    parser.add_argument("--max-model-len", type=int, default=config["max_model_len"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=config["gpu_memory_utilization"])
    parser.add_argument("--minimum-cache-free-gb", type=float, default=config["minimum_cache_free_gb"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--skip-weight-hash", action="store_false", dest="hash_weights")
    parser.set_defaults(hash_weights=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.1 <= args.gpu_memory_utilization <= 0.99:
        parser.error("--gpu-memory-utilization must be between 0.1 and 0.99")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

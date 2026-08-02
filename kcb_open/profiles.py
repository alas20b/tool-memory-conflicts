from __future__ import annotations

import copy
from pathlib import Path

from .backends import probe_lmstudio, probe_ollama, probe_transformers
from .io_utils import atomic_write_json, canonical_json, read_json, sha256_file, sha256_text, utc_now
from .registry import SHA256_RE, load_registry


PROFILE_SCHEMA = "kcb-open-execution-profile-v1"


def generation_settings(config: dict) -> dict:
    keys = (
        "seed", "temperature", "top_p", "top_k", "repeat_penalty", "max_tokens",
        "context_length", "request_timeout_seconds", "request_retries",
    )
    return {key: config[key] for key in keys}


def _basis(profile: dict) -> dict:
    return {key: value for key, value in profile.items() if key not in ("profile_id", "created_at", "profile_path")}


def compute_profile_id(profile: dict) -> str:
    return sha256_text(canonical_json(_basis(profile)))


def validate_profile(profile: dict, root: Path) -> dict:
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("unsupported profile schema")
    registry = load_registry(root)
    key = profile.get("model_key")
    if key not in registry or profile.get("model") != registry[key]:
        raise ValueError("profile model entry differs from model_registry.json")
    if profile.get("backend") not in ("ollama", "lmstudio", "transformers", "mock"):
        raise ValueError("unsupported profile backend")
    expected = compute_profile_id(profile)
    if profile.get("profile_id") != expected:
        raise ValueError("profile_id does not match immutable profile content")
    config = read_json(root / "experiment_config.json")
    if profile.get("generation") != generation_settings(config):
        raise ValueError("profile generation settings differ from experiment_config.json")
    return profile


def create_profile(
    root: Path,
    model_key: str,
    backend: str,
    *,
    server_url: str = "",
    server_model: str = "",
    artifact_path: Path | None = None,
) -> Path:
    registry = load_registry(root)
    if model_key not in registry:
        raise ValueError(f"unknown model key {model_key!r}")
    model = copy.deepcopy(registry[model_key])
    config = read_json(root / "experiment_config.json")
    timeout = int(config["request_timeout_seconds"])
    retries = int(config["request_retries"])

    if backend == "ollama":
        if not server_url or not server_model:
            raise ValueError("Ollama probe requires --server-url and --server-model")
        metadata = probe_ollama(server_url, server_model, timeout=timeout, retries=retries)
        quantization = str((metadata.get("details") or {}).get("quantization_level") or "").upper()
        if quantization != model["gguf"]["quantization"]:
            raise RuntimeError(f"expected Q6_K Ollama model, server reports {quantization!r}")
    elif backend == "lmstudio":
        if not server_url or not server_model:
            raise ValueError("LM Studio probe requires --server-url and --server-model")
        metadata = probe_lmstudio(server_url, server_model, timeout=timeout, retries=retries)
        model_meta = metadata["model"]
        model_format = model_meta.get("format") or model_meta.get("compatibility_type")
        if str(model_format or "").lower() != "gguf":
            raise RuntimeError("LM Studio model is not a GGUF model")
        quantization_field = model_meta.get("quantization") or ""
        if isinstance(quantization_field, dict):
            quantization_field = quantization_field.get("name") or ""
        quantization = str(quantization_field).upper()
        if quantization != model["gguf"]["quantization"]:
            raise RuntimeError(f"expected Q6_K LM Studio model, server reports {quantization!r}")
        loaded = metadata.get("loaded_instance")
        if loaded is not None:
            context_length = int((loaded.get("config") or {}).get("context_length") or 0)
            if context_length != int(config["context_length"]):
                raise RuntimeError(
                    f"expected LM Studio context length {config['context_length']}, "
                    f"loaded instance reports {context_length}"
                )
    elif backend == "transformers":
        if server_url or server_model:
            raise ValueError("Transformers probe does not accept server settings")
        if artifact_path is not None:
            raise ValueError("Transformers probe does not accept --artifact-path")
        metadata = probe_transformers(model)
    else:
        raise ValueError("profile creation supports ollama, lmstudio, or transformers")

    if artifact_path is not None:
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        digest = sha256_file(artifact_path)
        if backend in ("ollama", "lmstudio") and digest != model["gguf"]["sha256"]:
            raise RuntimeError(
                f"GGUF checksum mismatch: expected {model['gguf']['sha256']}, found {digest}"
            )
        weight_verification = {
            "status": "file_sha256_verified",
            "path": str(artifact_path),
            "sha256": digest,
            "size_bytes": artifact_path.stat().st_size,
        }
    elif backend in ("ollama", "lmstudio"):
        weight_verification = {
            "status": "server_metadata_only",
            "expected_gguf_sha256": model["gguf"]["sha256"],
            "note": "Provide --artifact-path when the local GGUF file is accessible.",
        }
    else:
        weight_verification = {
            "status": "immutable_hf_revision",
            "resolved_revision": metadata["resolved_revision"],
        }

    profile = {
        "schema_version": PROFILE_SCHEMA,
        "model_key": model_key,
        "model": model,
        "backend": backend,
        "server_url": server_url.rstrip("/") if server_url else None,
        "server_model": server_model or None,
        "backend_metadata": metadata,
        "weight_verification": weight_verification,
        "generation": generation_settings(config),
    }
    profile["profile_id"] = compute_profile_id(profile)
    profile["created_at"] = utc_now()
    validate_profile(profile, root)
    output = root / "profiles" / backend / f"{model_key}-{profile['profile_id'][:12]}.json"
    if output.exists():
        existing = read_json(output)
        if existing != profile:
            # created_at can differ while the immutable profile is identical.
            if compute_profile_id(existing) != profile["profile_id"]:
                raise RuntimeError(f"refusing to overwrite different profile {output}")
            return output
    atomic_write_json(output, profile)
    return output


def load_profile(path: Path, root: Path) -> dict:
    return validate_profile(read_json(path.resolve()), root)

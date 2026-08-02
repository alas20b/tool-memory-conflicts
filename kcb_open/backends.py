from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import package_version, sha256_file


def _headers(backend: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if backend == "lmstudio":
        token = os.environ.get("LM_API_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def http_json(
    method: str,
    url: str,
    *,
    backend: str,
    payload: dict | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, headers=_headers(backend), method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {error.code} from {url}: {body[:500]}")
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise last_error from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
        if attempt + 1 < retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {method} {url}: {last_error}")


def probe_ollama(server_url: str, server_model: str, *, timeout: int, retries: int) -> dict:
    base = server_url.rstrip("/")
    version = http_json("GET", f"{base}/api/version", backend="ollama", timeout=timeout, retries=retries)
    tags = http_json("GET", f"{base}/api/tags", backend="ollama", timeout=timeout, retries=retries)
    show = http_json(
        "POST",
        f"{base}/api/show",
        backend="ollama",
        payload={"model": server_model, "verbose": False},
        timeout=timeout,
        retries=retries,
    )
    models = tags.get("models", []) if isinstance(tags, dict) else []
    tag_record = next(
        (row for row in models if row.get("name") == server_model or row.get("model") == server_model),
        None,
    )
    if tag_record is None:
        raise RuntimeError(f"Ollama model {server_model!r} was not found in /api/tags")
    details = show.get("details") or {}
    template = str(show.get("template") or "")
    return {
        "backend_version": version,
        "tag": tag_record,
        "details": details,
        "parameters": show.get("parameters"),
        "capabilities": show.get("capabilities"),
        "template_sha256": __import__("hashlib").sha256(template.encode("utf-8")).hexdigest(),
        "template": template,
        "selected_model_info": {
            key: value
            for key, value in (show.get("model_info") or {}).items()
            if key.startswith("general.") or key.endswith("context_length") or key.endswith("block_count")
        },
    }


def probe_lmstudio(server_url: str, server_model: str, *, timeout: int, retries: int) -> dict:
    base = server_url.rstrip("/")
    try:
        document = http_json(
            "GET", f"{base}/api/v1/models", backend="lmstudio", timeout=timeout, retries=retries
        )
        models = document.get("models", []) if isinstance(document, dict) else []
        record = next(
            (
                row for row in models
                if row.get("key") == server_model
                or any(instance.get("id") == server_model for instance in row.get("loaded_instances", []))
            ),
            None,
        )
        if record is None:
            available = [row.get("key") for row in models]
            raise RuntimeError(f"LM Studio model {server_model!r} was not found; available={available}")
        loaded_instance = next(
            (row for row in record.get("loaded_instances", []) if row.get("id") == server_model),
            None,
        )
        if loaded_instance is None and record.get("key") == server_model:
            instances = record.get("loaded_instances", [])
            loaded_instance = instances[0] if len(instances) == 1 else None
        if loaded_instance is None:
            raise RuntimeError(
                f"LM Studio model {server_model!r} is not loaded as one unique instance; "
                "load it explicitly before probing"
            )
        api_version = "v1"
    except RuntimeError as error:
        if "HTTP 404" not in str(error):
            raise
        document = http_json(
            "GET", f"{base}/api/v0/models", backend="lmstudio", timeout=timeout, retries=retries
        )
        models = document.get("data", []) if isinstance(document, dict) else []
        record = next((row for row in models if row.get("id") == server_model), None)
        if record is None:
            available = [row.get("id") for row in models]
            raise RuntimeError(f"LM Studio model {server_model!r} was not found; available={available}")
        if str(record.get("state", "")).lower() not in ("loaded", ""):
            raise RuntimeError(f"LM Studio model {server_model!r} is not loaded")
        loaded_instance = None
        api_version = "v0_fallback"
    openai_models = http_json(
        "GET", f"{base}/v1/models", backend="lmstudio", timeout=timeout, retries=retries
    )
    return {
        "native_api_version": api_version,
        "model": record,
        "loaded_instance": loaded_instance,
        "openai_models": openai_models,
    }


def probe_transformers(model: dict) -> dict:
    try:
        import torch
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("install requirements-transformers.txt and a CUDA-compatible PyTorch build") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Transformers research runs require a CUDA GPU; probe on the run host")
    token = os.environ.get("HF_TOKEN")
    info = HfApi().model_info(
        model["upstream_hf_repo"],
        revision=model.get("requested_revision", "main"),
        token=token,
        files_metadata=True,
    )
    if not info.sha:
        raise RuntimeError("Hugging Face did not return an immutable model revision")
    weight_files = []
    for sibling in info.siblings or []:
        name = str(getattr(sibling, "rfilename", ""))
        if name.endswith((".safetensors", ".bin")):
            lfs = getattr(sibling, "lfs", None) or {}
            if isinstance(lfs, dict):
                lfs_size = lfs.get("size")
                lfs_sha256 = lfs.get("sha256") or lfs.get("oid")
            else:
                lfs_size = getattr(lfs, "size", None)
                lfs_sha256 = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            weight_files.append({
                "name": name,
                "size_bytes": getattr(sibling, "size", None) or lfs_size,
                "lfs_sha256": lfs_sha256,
            })
    return {
        "resolved_revision": info.sha,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "huggingface-hub": package_version("huggingface-hub"),
        },
        "weight_files": weight_files,
        "declared_weight_bytes": sum(int(row["size_bytes"] or 0) for row in weight_files),
    }


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metadata: dict


class ModelBackend:
    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class OllamaBackend(ModelBackend):
    def __init__(self, profile: dict):
        self.profile = profile
        self.base = profile["server_url"].rstrip("/")
        self.model = profile["server_model"]
        self.settings = profile["generation"]

    def generate(self, prompt: str) -> GenerationResult:
        response = http_json(
            "POST",
            f"{self.base}/api/chat",
            backend="ollama",
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": self.settings["temperature"],
                    "seed": self.settings["seed"],
                    "num_predict": self.settings["max_tokens"],
                    "num_ctx": self.settings["context_length"],
                    "top_p": self.settings["top_p"],
                    "top_k": self.settings["top_k"],
                    "repeat_penalty": self.settings["repeat_penalty"],
                },
            },
            timeout=self.settings["request_timeout_seconds"],
            retries=self.settings["request_retries"],
        )
        text = (response.get("message") or {}).get("content")
        if not isinstance(text, str):
            raise RuntimeError(f"Ollama returned no assistant content: {response}")
        metadata = {
            key: response.get(key)
            for key in (
                "model", "created_at", "done", "done_reason", "total_duration", "load_duration",
                "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
            )
        }
        return GenerationResult(text=text.strip(), metadata=metadata)


class LMStudioBackend(ModelBackend):
    def __init__(self, profile: dict):
        self.profile = profile
        self.base = profile["server_url"].rstrip("/")
        self.model = profile["server_model"]
        self.settings = profile["generation"]

    def generate(self, prompt: str) -> GenerationResult:
        response = http_json(
            "POST",
            f"{self.base}/v1/chat/completions",
            backend="lmstudio",
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.settings["temperature"],
                "top_p": self.settings["top_p"],
                "top_k": self.settings["top_k"],
                "repeat_penalty": self.settings["repeat_penalty"],
                "seed": self.settings["seed"],
                "max_tokens": self.settings["max_tokens"],
                "stream": False,
            },
            timeout=self.settings["request_timeout_seconds"],
            retries=self.settings["request_retries"],
        )
        try:
            choice = response["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"LM Studio returned an unexpected response: {response}") from error
        if not isinstance(text, str):
            raise RuntimeError("LM Studio assistant content is not text")
        return GenerationResult(
            text=text.strip(),
            metadata={
                "id": response.get("id"),
                "model": response.get("model"),
                "created": response.get("created"),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "stats": response.get("stats"),
            },
        )


class TransformersBackend(ModelBackend):
    def __init__(self, profile: dict, *, cache_dir: Path):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as error:
            raise RuntimeError("install requirements-transformers.txt and PyTorch") from error
        if not torch.cuda.is_available():
            raise RuntimeError("Transformers research runs require a CUDA GPU; CPU fallback is disabled")
        self.torch = torch
        self.set_seed = set_seed
        self.profile = profile
        self.settings = profile["generation"]
        repo = profile["model"]["upstream_hf_repo"]
        revision = profile["backend_metadata"]["resolved_revision"]
        token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo, revision=revision, cache_dir=str(cache_dir), token=token, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            cache_dir=str(cache_dir),
            token=token,
            trust_remote_code=False,
            device_map="auto",
            dtype="auto",
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> GenerationResult:
        self.set_seed(self.settings["seed"])
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings["max_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.settings["repeat_penalty"],
                use_cache=True,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            metadata={
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(generated.shape[0]),
            },
        )

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class MockBackend(ModelBackend):
    """Deterministic local backend used only by offline tests."""

    def generate(self, prompt: str) -> GenerationResult:
        if "PermissionDenied" in prompt:
            answer = "Synthetic dry-run answer requiring manual review"
        elif "TimeoutError" in prompt:
            answer = "UNKNOWN"
        else:
            marker = "Question: "
            question = prompt.rsplit(marker, 1)[-1].strip() if marker in prompt else "synthetic"
            answer = f"memory-for-{__import__('hashlib').sha256(question.encode()).hexdigest()[:8]}"
        return GenerationResult(f"Synthetic backend output.\nFINAL: {answer}", {"synthetic": True})


def create_backend(profile: dict, *, cache_dir: Path) -> ModelBackend:
    backend = profile["backend"]
    if backend == "ollama":
        return OllamaBackend(profile)
    if backend == "lmstudio":
        return LMStudioBackend(profile)
    if backend == "transformers":
        return TransformersBackend(profile, cache_dir=cache_dir)
    if backend == "mock":
        return MockBackend()
    raise ValueError(f"unsupported backend {backend!r}")

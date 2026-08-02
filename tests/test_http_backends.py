from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kcb_open.backends import create_backend
from kcb_open.io_utils import read_json
from kcb_open.profiles import create_profile


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class FakeHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict | None]] = []
    v1_enabled = True

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _send(self, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self.requests.append((self.path, None))
        if self.path == "/api/version":
            self._send({"version": "test-1"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "qwen-test", "model": "qwen-test", "digest": "abc"}]})
        elif self.path == "/api/v0/models":
            self._send({"data": [{
                "id": "qwen-test", "type": "llm", "compatibility_type": "gguf",
                "quantization": "Q6_K", "state": "loaded", "arch": "qwen2",
            }]})
        elif self.path == "/api/v1/models":
            if not self.v1_enabled:
                self.send_error(404)
            else:
                self._send({"models": [{
                    "key": "qwen-source", "type": "llm", "publisher": "bartowski",
                    "format": "gguf", "quantization": {"name": "Q6_K", "bits_per_weight": 6},
                    "size_bytes": 6254199488, "params_string": "7.6B",
                    "loaded_instances": [{"id": "qwen-test", "config": {"context_length": 4096}}],
                }]})
        elif self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "qwen-test"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append((self.path, payload))
        if self.path == "/api/show":
            self._send({
                "details": {"quantization_level": "Q6_K", "parameter_size": "7.6B"},
                "template": "{{ .Prompt }}", "parameters": "temperature 0", "capabilities": ["completion"],
                "model_info": {"general.architecture": "qwen2", "qwen2.context_length": 32768},
            })
        elif self.path == "/api/chat":
            self._send({"model": "qwen-test", "done": True, "message": {"content": "FINAL: test"}})
        elif self.path == "/v1/chat/completions":
            self._send({
                "id": "chat-test", "model": "qwen-test", "created": 1,
                "choices": [{"finish_reason": "stop", "message": {"content": "FINAL: test"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })
        else:
            self.send_error(404)


class HttpBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name)
        for name in ("model_registry.json", "experiment_config.json"):
            shutil.copy2(SOURCE_ROOT / name, self.root / name)
        FakeHandler.requests = []
        FakeHandler.v1_enabled = True
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_ollama_probe_and_generation_payload(self) -> None:
        path = create_profile(
            self.root, "qwen2.5-7b-instruct", "ollama",
            server_url=self.url, server_model="qwen-test",
        )
        profile = read_json(path)
        backend = create_backend(profile, cache_dir=self.root / "cache")
        result = backend.generate("hello")
        self.assertEqual(result.text, "FINAL: test")
        payload = next(body for path, body in FakeHandler.requests if path == "/api/chat")
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertEqual(payload["options"]["seed"], 20260704)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertFalse(payload["stream"])

    def test_lmstudio_probe_and_generation_payload(self) -> None:
        path = create_profile(
            self.root, "qwen2.5-7b-instruct", "lmstudio",
            server_url=self.url, server_model="qwen-test",
        )
        profile = read_json(path)
        backend = create_backend(profile, cache_dir=self.root / "cache")
        result = backend.generate("hello")
        self.assertEqual(result.text, "FINAL: test")
        payload = next(body for path, body in FakeHandler.requests if path == "/v1/chat/completions")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 20260704)
        self.assertEqual(payload["max_tokens"], 384)
        self.assertFalse(payload["stream"])

    def test_lmstudio_v0_probe_fallback(self) -> None:
        FakeHandler.v1_enabled = False
        path = create_profile(
            self.root, "qwen2.5-7b-instruct", "lmstudio",
            server_url=self.url, server_model="qwen-test",
        )
        profile = read_json(path)
        self.assertEqual(profile["backend_metadata"]["native_api_version"], "v0_fallback")


if __name__ == "__main__":
    unittest.main()

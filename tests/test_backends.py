import importlib
import json
import sys
import types
import unittest
from unittest.mock import patch


class FakeResponse:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return [json.dumps(row).encode("utf-8") for row in self.rows]


class OllamaBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("requests", types.ModuleType("requests"))
        cls.backend = importlib.import_module("backends.ollama_backend")

    def test_final_server_usage_drives_token_rate(self):
        rows = [
            {"response": "hello", "done": False},
            {
                "response": " world",
                "done": True,
                "eval_count": 40,
                "eval_duration": 2_000_000_000,
                "prompt_eval_count": 25,
                "prompt_eval_duration": 500_000_000,
                "load_duration": 100_000_000,
            },
        ]
        captured = {}

        def fake_post(url, json, stream, timeout):
            captured.update({"url": url, "payload": json, "timeout": timeout})
            return FakeResponse(rows)

        with (
            patch.object(self.backend.requests, "post", side_effect=fake_post, create=True),
            patch.object(self.backend, "_vram_mb", side_effect=[100.0, 120.0]),
            patch.object(self.backend.time, "perf_counter", side_effect=[1.0, 1.1, 3.1]),
        ):
            text, metrics = self.backend.generate("prompt", max_tokens=64)

        self.assertEqual(text, "hello world")
        self.assertEqual(metrics["output_tokens"], 40)
        self.assertEqual(metrics["decode_tps"], 20.0)
        self.assertEqual(metrics["token_count_source"], "ollama_final_response.eval_count")
        self.assertEqual(captured["payload"]["options"]["num_predict"], 64)


class FakeUsage:
    completion_tokens = 20
    prompt_tokens = 30


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [] if content is None else [FakeChoice(content)]
        self.usage = usage


class FakeCompletions:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.update(kwargs)
        return [FakeChunk("hello"), FakeChunk(" world"), FakeChunk(usage=FakeUsage())]


class FakeClient:
    def __init__(self, captured):
        self.chat = types.SimpleNamespace(completions=FakeCompletions(captured))


class VllmBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = types.ModuleType("openai")
        module.OpenAI = object
        sys.modules.setdefault("openai", module)
        cls.backend = importlib.import_module("backends.vllm_backend")

    def test_stream_usage_drives_client_token_rate(self):
        captured = {}
        with (
            patch.object(self.backend, "OpenAI", return_value=FakeClient(captured)),
            patch.object(self.backend, "_vram_mb", side_effect=[200.0, 220.0]),
            patch.object(self.backend.time, "perf_counter", side_effect=[1.0, 1.2, 3.2]),
        ):
            text, metrics = self.backend.generate(
                "prompt", base_url="http://localhost:9999/v1", server_profile="test"
            )

        self.assertEqual(text, "hello world")
        self.assertEqual(metrics["output_tokens"], 20)
        self.assertEqual(metrics["decode_tps"], 10.0)
        self.assertEqual(metrics["server_profile"], "test")
        self.assertTrue(captured["stream_options"]["include_usage"])


if __name__ == "__main__":
    unittest.main()
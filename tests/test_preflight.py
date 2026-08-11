import unittest
from unittest.mock import patch

from backends import preflight


class FakeJsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class BackendPreflightTests(unittest.TestCase):
    def test_vllm_records_server_version_and_model(self):
        def fake_get(url, timeout):
            if url.endswith("/version"):
                return FakeJsonResponse({"version": "0.26.0"})
            return FakeJsonResponse(
                {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
            )

        with patch.object(preflight.requests, "get", side_effect=fake_get):
            result = preflight.inspect_vllm(
                "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1"
            )

        self.assertEqual(result["version"], "0.26.0")
        self.assertTrue(result["model_available"])

    def test_ollama_rejects_another_loaded_model(self):
        def fake_get(url, timeout):
            if url.endswith("/api/version"):
                return FakeJsonResponse({"version": "0.32.6"})
            if url.endswith("/api/tags"):
                return FakeJsonResponse({"models": [{"name": "target"}]})
            return FakeJsonResponse({"models": [{"name": "other"}]})

        with (
            patch.object(preflight.requests, "get", side_effect=fake_get),
            self.assertRaisesRegex(RuntimeError, "VRAM isolation failed"),
        ):
            preflight.inspect_ollama("target")

    def test_unknown_engine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            preflight.inspect_backend({"engine": "unknown"})


if __name__ == "__main__":
    unittest.main()

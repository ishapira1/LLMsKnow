from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from llmssycoph.llm.huggingface import (
    _hf_load_kwargs,
    _is_gated_repo_error,
    _raise_helpful_hf_auth_error,
)


class HuggingFaceContractTests(unittest.TestCase):
    def test_load_kwargs_preserves_cache_without_token(self):
        with patch.dict(os.environ, {}, clear=True):
            kwargs = _hf_load_kwargs("/tmp/hf-cache")

        self.assertEqual(kwargs, {"cache_dir": "/tmp/hf-cache"})

    def test_load_kwargs_uses_hf_token_when_present(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=True):
            kwargs = _hf_load_kwargs("/tmp/hf-cache")

        self.assertEqual(kwargs["cache_dir"], "/tmp/hf-cache")
        self.assertEqual(kwargs["token"], "hf-secret")

    def test_load_kwargs_falls_back_to_huggingface_token(self):
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf-legacy"}, clear=True):
            kwargs = _hf_load_kwargs("/tmp/hf-cache")

        self.assertEqual(kwargs["cache_dir"], "/tmp/hf-cache")
        self.assertEqual(kwargs["token"], "hf-legacy")

    def test_gated_repo_detector_matches_hf_error_text(self):
        self.assertTrue(_is_gated_repo_error(RuntimeError("Cannot access gated repo for url ...")))
        self.assertTrue(_is_gated_repo_error(RuntimeError("401 Client Error: Unauthorized")))
        self.assertFalse(_is_gated_repo_error(RuntimeError("some unrelated failure")))

    def test_helpful_auth_error_mentions_env_when_token_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN or HUGGINGFACE_TOKEN"):
                _raise_helpful_hf_auth_error(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    RuntimeError("Cannot access gated repo"),
                )

    def test_helpful_auth_error_mentions_account_access_when_token_present(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "granted access"):
                _raise_helpful_hf_auth_error(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    RuntimeError("Cannot access gated repo"),
                )


if __name__ == "__main__":
    unittest.main()

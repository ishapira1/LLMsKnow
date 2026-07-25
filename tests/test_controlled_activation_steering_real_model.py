"""Opt-in BF16 numerical gates for the controlled steering protocol.

Run on a GPU node after approving the preflight manifest:

ACTIVATION_STEERING_REAL_SOURCE_RUN=/path/to/source/run \
ACTIVATION_STEERING_REAL_MANIFEST=/path/to/approved/manifest.jsonl \
ACTIVATION_STEERING_REAL_CONFIG=configs/experiments/activation_steering_controlled_20260725.json \
PYTHONPATH=src python -m unittest -v tests.test_controlled_activation_steering_real_model
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from llmssycoph.interventions.activations import (
    extract_prompt_state,
    score_repeated_prompt_without_hook,
    score_with_residual_additions,
)
from llmssycoph.interventions.controlled import assert_noop_contract, read_json
from llmssycoph.interventions.controlled_runtime import (
    _load_sources_and_pairs,
    load_controlled_runtime,
)


SOURCE_RUN = os.environ.get("ACTIVATION_STEERING_REAL_SOURCE_RUN", "")
MANIFEST = os.environ.get("ACTIVATION_STEERING_REAL_MANIFEST", "")
CONFIG = os.environ.get("ACTIVATION_STEERING_REAL_CONFIG", "")


@unittest.skipUnless(
    SOURCE_RUN and MANIFEST and CONFIG,
    "Set the three ACTIVATION_STEERING_REAL_* paths on a GPU node.",
)
class ControlledRealModelNumericalGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for the BF16 numerical gate.")
        cls.config = read_json(Path(CONFIG))
        sources, pairs, _ = _load_sources_and_pairs(
            [Path(SOURCE_RUN)],
            manifest_path=Path(MANIFEST),
            splits=("train", "val", "test"),
            require_human_approval=True,
        )
        cls.source = sources[0]
        cls.pair = pairs[0]
        cls.model, cls.tokenizer, cls.runtime = load_controlled_runtime(
            cls.source,
            cls.config,
            device="cuda",
            device_map_auto=False,
            hf_cache_dir=os.environ.get("HF_HUB_CACHE"),
            torch_dtype="auto",
        )

    def test_bf16_same_shape_zero_and_cross_batch_replay(self):
        layer = int(os.environ.get("ACTIVATION_STEERING_REAL_LAYER", "17"))
        record = self.pair["records"]["neutral"]
        choices = self.pair["choices"]
        state = extract_prompt_state(
            self.model,
            self.tokenizer,
            record["prompt_messages"],
            choices=choices,
            residual_layers=[layer],
        )
        disabled_probabilities, disabled_scores = score_repeated_prompt_without_hook(
            self.model,
            self.tokenizer,
            record["prompt_messages"],
            choices=choices,
            batch_size=2,
        )
        zero_probabilities, zero_scores = score_with_residual_additions(
            self.model,
            self.tokenizer,
            record["prompt_messages"],
            choices=choices,
            residual_layer=layer,
            addition_vectors=np.zeros(
                (2, state.hidden_by_layer[layer].shape[0]),
                dtype=np.float32,
            ),
            max_batch_size=2,
        )
        assert_noop_contract(
            disabled_probabilities,
            zero_probabilities,
            exact=True,
            max_probability_error=0.0,
        )
        assert_noop_contract(
            [state.choice_probabilities, state.choice_probabilities],
            disabled_probabilities,
            exact=False,
            max_probability_error=0.005,
        )
        correct = self.pair["correct_choice"]
        endorsed = self.pair["endorsed_choice"]
        baseline_margin = (
            state.choice_log_scores[correct]
            - state.choice_log_scores[endorsed]
        )
        self.assertLessEqual(
            max(
                abs(
                    score[correct]
                    - score[endorsed]
                    - baseline_margin
                )
                for score in disabled_scores
            ),
            0.05,
        )
        self.assertEqual(disabled_scores, zero_scores)
        self.assertEqual(
            state.final_token_id,
            state.prompt_token_ids[state.prompt_token_count - 1],
        )


if __name__ == "__main__":
    unittest.main()

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_pruning import select_global_mask  # noqa: E402


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[1.0], [1.1], [10.0], [11.0]]))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block()])


class PaperPruningControlTests(unittest.TestCase):
    def _score_dir(self, root: Path) -> Path:
        name = "model.layers.0.proj"
        for role, values in (
            ("prune", torch.tensor([[-2.0], [3.0], [2.0], [1.0]])),
            ("preserve", torch.tensor([[4.0], [1.0], [0.0], [0.0]])),
        ):
            directory = root / role
            directory.mkdir(parents=True)
            torch.save(values, directory / "score.pt")
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "eligible_numel": 4,
                        "tensors": {
                            name: {
                                "file": "score.pt",
                                "shape": [4, 1],
                                "numel": 4,
                                "block": 0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
        return root

    @staticmethod
    def _args(**overrides):
        values = {
            "p": 0.0,
            "q": 0.25,
            "neg_prune": True,
            "freeze_first_top_q": False,
            "control": "none",
            "match_bins": 2,
            "seed": 5,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_p_zero_and_both_sign_controls_select_exact_global_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            score_dir = self._score_dir(Path(temporary))
            negative, negative_meta = select_global_mask(
                self._args(), Model(), score_dir
            )
            positive, positive_meta = select_global_mask(
                self._args(neg_prune=False), Model(), score_dir
            )
            torch.testing.assert_close(
                negative["model.layers.0.proj"], torch.tensor([0])
            )
            torch.testing.assert_close(
                positive["model.layers.0.proj"], torch.tensor([1])
            )
            self.assertEqual(negative_meta["nominal_preserve_count"], 0)
            self.assertEqual(negative_meta["surviving_count"], 1)
            self.assertEqual(positive_meta["surviving_count"], 1)

    def test_preservation_difference_second_slice_and_random_match_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            score_dir = self._score_dir(Path(temporary))
            excluded, excluded_meta = select_global_mask(
                self._args(p=0.25), Model(), score_dir
            )
            self.assertEqual(excluded, {})
            self.assertEqual(excluded_meta["nominal_prune_count"], 1)
            self.assertEqual(excluded_meta["surviving_count"], 0)

            second, second_meta = select_global_mask(
                self._args(freeze_first_top_q=True), Model(), score_dir
            )
            torch.testing.assert_close(
                second["model.layers.0.proj"], torch.tensor([3])
            )
            self.assertEqual(second_meta["prune_rank_start"], 1)

            random_mask, random_meta = select_global_mask(
                self._args(control="random_magnitude"), Model(), score_dir
            )
            self.assertEqual(random_meta["surviving_before_control"], 1)
            self.assertEqual(random_meta["surviving_count"], 1)
            self.assertEqual(random_meta["counts_by_module"], {"model.layers.0.proj": 1})
            self.assertNotEqual(
                int(random_mask["model.layers.0.proj"][0]),
                0,
            )
            match = random_meta["random_magnitude_match"]["model.layers.0.proj"]
            self.assertTrue(match["exact_bin_match"])
            self.assertEqual(match["target_bin_counts"], match["random_bin_counts"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tiny CPU end-to-end smoke for sampling, persistence, metrics, and inference."""

from __future__ import annotations

from pathlib import Path
import tempfile

import random_baseline as rb


def main() -> int:
    import torch

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(24, 16, bias=False)
            self.b = torch.nn.Linear(16, 12, bias=False)
            self.lm_head = torch.nn.Linear(12, 4, bias=False)
            with torch.no_grad():
                self.a.weight.copy_(torch.linspace(-4, 4, self.a.weight.numel()).reshape_as(self.a.weight))
                self.b.weight.copy_(torch.linspace(-2, 2, self.b.weight.numel()).reshape_as(self.b.weight))

    model = Toy()
    modules = rb.eligible_linears(model)
    target = {"a": torch.tensor([1, 30, 80, 140, 250]),
              "b": torch.tensor([2, 40, 90, 150])}
    uniform = rb.uniform_global_controls(modules, target, count=9, seeds=(101, 211))
    matched, audit = rb.matched_controls(modules, target, seeds=(101, 211), bins=20)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for family, masks in (("uniform", uniform), ("matched", matched)):
            for seed, mask in masks.items():
                rb.save_mask(root / family / str(seed), mask,
                             {"family": family, "seed": seed})
                loaded = rb.load_indices(root / family / str(seed) / "indices.pt")
                if rb.mask_logical_sha256(loaded) != rb.mask_logical_sha256(mask):
                    raise RuntimeError("Mask persistence changed logical identity")
    summaries = {
        "base": {"strong_wrong_adoption": {"rate": 0.5},
                 "neutral_accuracy": {"rate": 0.8}},
        "learned": {"strong_wrong_adoption": {"rate": 0.1},
                    "neutral_accuracy": {"rate": 0.79}},
    }
    distribution = [{"family": "module_magnitude_matched", "seed": seed,
                     "strong_wrong_adoption": 0.2, "neutral_accuracy": 0.79}
                    for seed in rb.SEEDS]
    inference = rb.confirmatory_inference(summaries, distribution)
    if not inference["model_supports_specificity"]:
        raise RuntimeError("Synthetic support scenario failed")
    print(rb.canonical_json({"status": "complete", "uniform_masks": len(uniform),
                             "matched_masks": len(matched), "matched_modules": len(audit),
                             "synthetic_p": inference["empirical_rank_p_one_sided"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

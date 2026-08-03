from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


BUNDLE = Path(__file__).parents[1] / "jobs/sycophancy_pruning/random_baseline"
sys.path.insert(0, str(BUNDLE))
import random_baseline as rb  # noqa: E402
import multi_state_eval as mse  # noqa: E402
import export_report as export  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal CPU environments
    torch = None


if torch is not None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = torch.nn.Linear(20, 12, bias=False)
            self.block2 = torch.nn.Linear(12, 10, bias=False)
            self.lm_head = torch.nn.Linear(10, 5, bias=False)
            with torch.no_grad():
                self.block1.weight.copy_(torch.linspace(-3, 3, self.block1.weight.numel()).reshape_as(self.block1.weight))
                self.block2.weight.copy_(torch.linspace(-2, 2, self.block2.weight.numel()).reshape_as(self.block2.weight))


def target_mask():
    if torch is None:
        raise unittest.SkipTest("torch is unavailable")
    return {
        "block1": torch.tensor([0, 5, 40, 100, 180]),
        "block2": torch.tensor([2, 20, 60, 90]),
    }


def metric(rate: float) -> dict[str, float | int]:
    return {"rate": rate, "numerator": round(rate * 1000), "denominator": 1000}


def scenario(learned_syco: float = 0.10, learned_neutral: float = 0.79,
             random_syco: float = 0.20, random_neutral: float = 0.79):
    summaries = {
        "base": {"strong_wrong_adoption": metric(0.50), "neutral_accuracy": metric(0.80)},
        "learned": {"strong_wrong_adoption": metric(learned_syco),
                    "neutral_accuracy": metric(learned_neutral)},
    }
    rows = [{"family": "module_magnitude_matched", "seed": seed,
             "strong_wrong_adoption": random_syco, "neutral_accuracy": random_neutral}
            for seed in rb.SEEDS]
    return summaries, rows


@unittest.skipIf(torch is None, "torch is unavailable")
class MaskTests(unittest.TestCase):
    def test_uniform_reproducible_distinct_exact_and_disjoint(self) -> None:
        modules = rb.eligible_linears(ToyModel())
        target = target_mask()
        first = rb.uniform_global_controls(modules, target, count=9, seeds=(101, 211))
        second = rb.uniform_global_controls(modules, target, count=9, seeds=(101, 211))
        self.assertEqual(rb.mask_logical_sha256(first[101]), rb.mask_logical_sha256(second[101]))
        self.assertNotEqual(rb.mask_logical_sha256(first[101]), rb.mask_logical_sha256(first[211]))
        for mask in first.values():
            self.assertEqual(sum(value.numel() for value in mask.values()), 9)
            for name, values in mask.items():
                self.assertFalse(set(values.tolist()) & set(target.get(name, torch.empty(0)).tolist()))

    def test_matched_reproducible_exact_module_and_twenty_bins(self) -> None:
        modules = rb.eligible_linears(ToyModel())
        target = target_mask()
        first, audit = rb.matched_controls(modules, target, seeds=(101, 211),
                                           bins=20, quantile_sample_size=10_000)
        second, _ = rb.matched_controls(modules, target, seeds=(101, 211),
                                        bins=20, quantile_sample_size=10_000)
        self.assertEqual(rb.mask_logical_sha256(first[101]), rb.mask_logical_sha256(second[101]))
        self.assertNotEqual(rb.mask_logical_sha256(first[101]), rb.mask_logical_sha256(first[211]))
        for seed, mask in first.items():
            self.assertEqual({name: value.numel() for name, value in mask.items()},
                             {name: value.numel() for name, value in target.items()})
            for name, values in mask.items():
                self.assertFalse(set(values.tolist()) & set(target[name].tolist()))
                self.assertEqual(audit[name]["random_bin_counts_by_seed"][str(seed)],
                                 audit[name]["target_bin_counts"])

    def test_logical_hash_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "mask"
            rb.save_mask(directory, target_mask(), {"kind": "toy"})
            metadata = rb.read_json(directory / "metadata.json")
            loaded = rb.load_indices(directory / "indices.pt")
            self.assertEqual(rb.mask_logical_sha256(loaded), metadata["logical_mask_sha256"])
            loaded["block1"][0] += 1
            self.assertNotEqual(rb.mask_logical_sha256(loaded), metadata["logical_mask_sha256"])


class InferenceTests(unittest.TestCase):
    def test_confirmatory_support(self) -> None:
        summaries, rows = scenario()
        result = rb.confirmatory_inference(summaries, rows)
        self.assertTrue(result["model_supports_specificity"])
        self.assertAlmostEqual(result["empirical_rank_p_one_sided"], 1 / 21)

    def test_confirmatory_equivalence_rejects(self) -> None:
        summaries, rows = scenario(random_syco=0.12, random_neutral=0.80)
        result = rb.confirmatory_inference(summaries, rows)
        self.assertFalse(result["model_supports_specificity"])
        self.assertEqual(result["matched_random_equivalent_count"], 20)

    def test_confirmatory_neutral_collapse_rejects(self) -> None:
        summaries, rows = scenario(learned_neutral=0.70)
        result = rb.confirmatory_inference(summaries, rows)
        self.assertFalse(result["model_supports_specificity"])
        self.assertFalse(result["learned_neutral_within_2pp_of_base"])

    def test_confirmatory_random_as_strong_rejects(self) -> None:
        summaries, rows = scenario(random_syco=0.09)
        result = rb.confirmatory_inference(summaries, rows)
        self.assertFalse(result["model_supports_specificity"])
        self.assertEqual(result["matched_random_at_least_as_strong"], 20)


class FrozenWikitextTests(unittest.TestCase):
    def test_frozen_input_validates_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wikitext-test.arrow"
            info = root / "dataset_info.json"
            frozen = root / "wikitext.jsonl"
            pin_path = root / "wikitext_pin.json"
            source.write_bytes(b"pinned-arrow")
            info.write_text("{}\n", encoding="utf-8")
            rb.atomic_jsonl(frozen, ({"row_id": index, "text": ""}
                                     for index in range(4358)))
            rb.atomic_json(pin_path, {
                "status": "complete", "dataset": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1", "split": "test",
                "revision": "test-revision", "rows": 4358,
                "source_arrow_path": str(source.resolve()),
                "source_arrow_sha256": rb.sha256_file(source),
                "dataset_info_path": str(info.resolve()),
                "dataset_info_sha256": rb.sha256_file(info),
                "frozen_input_path": str(frozen.resolve()),
                "frozen_input_sha256": rb.sha256_file(frozen),
            })
            self.assertEqual(mse.validate_wikitext_input(frozen, pin_path)["rows"], 4358)
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source Arrow drift"):
                mse.validate_wikitext_input(frozen, pin_path)


class BroadAggregationTests(unittest.TestCase):
    def test_exact_144_output_matrix_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broad_states = [
                {"state_id": "base", "family": None, "seed": None},
                {"state_id": "learned", "family": None, "seed": None},
                *[{"state_id": f"{family}__seed_{seed}", "family": family, "seed": seed}
                  for family in rb.CONTROL_FAMILIES for seed in rb.BROAD_SEEDS],
            ]
            for model in rb.MODEL_SPECS:
                rb.atomic_json(root / "registry" / f"{model}.json", {"states": broad_states})
                for state in broad_states:
                    for benchmark in rb.BROAD_BENCHMARKS:
                        path = root / "broad" / model / state["state_id"] / benchmark / "summary.json"
                        rb.atomic_json(path, {
                            "status": "complete", "model": model,
                            "benchmark": benchmark, "state": state,
                            "rows": rb.BROAD_EXPECTED_ROWS[benchmark],
                            "result": {"metric": 0.5},
                        })
            payload = rb.aggregate_broad(argparse.Namespace(result_root=root))
            self.assertEqual(payload["record_count"], 144)
            missing = root / "broad/llama/base/mmlu/summary.json"
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                rb.aggregate_broad(argparse.Namespace(result_root=root))


class PaperExportTests(unittest.TestCase):
    def test_primary_guardrails_and_common_suite_are_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            artifacts = Path(temporary) / "artifacts"
            rb.atomic_json(root / "analysis/final_report.json",
                           {"status": "complete", "conclusion": "supported"})
            rb.atomic_json(root / "audit/completion_audit.json", {
                "status": "complete", "audit_sha256": "a" * 64,
                "verified_counts": {"core_states": 84, "broad_states": 144},
            })
            rb.atomic_json(root / "registry/preflight_pins.json", {"status": "complete"})
            broad_records = []
            feedback_states = {}
            for model in rb.MODEL_SPECS:
                source = root / "analysis" / model
                base = {name: metric(value) for name, value in (
                    ("strong_wrong_adoption", 0.40), ("neutral_accuracy", 0.80),
                    ("invalid_answer_rate", 0.01))}
                learned = {name: metric(value) for name, value in (
                    ("strong_wrong_adoption", 0.10), ("neutral_accuracy", 0.79),
                    ("invalid_answer_rate", 0.02))}
                distribution = []
                for family in rb.CONTROL_FAMILIES:
                    for seed in rb.SEEDS:
                        distribution.append({
                            "family": family, "seed": seed,
                            "strong_wrong_adoption": 0.20,
                            "neutral_accuracy": 0.80, "invalid_answer_rate": 0.01,
                        })
                rb.atomic_json(source / "core_summary.json", {
                    "status": "complete", "summaries": {"base": base, "learned": learned},
                    "seed_distribution": distribution,
                    "confirmatory_inference": {
                        "empirical_rank_p_one_sided": 1 / 21,
                        "matched_random_equivalent_count": 0,
                    },
                })
                rb.atomic_jsonl(source / "seed_distribution.jsonl", distribution)
                (source / "seed_distribution.csv").write_text("model\n", encoding="utf-8")
                (source / "pareto.pdf").write_bytes(b"%PDF-smoke")
                (source / "pareto.png").write_bytes(b"PNG-smoke")
                states = ["base", "learned", *[
                    f"module_magnitude_matched__seed_{seed}" for seed in rb.BROAD_SEEDS
                ]]
                for state in states:
                    feedback_states[f"{model}/{state}"] = {"sycophancy_gap": 0.05}
                    for benchmark, result in (
                        ("sycobench", {"syco": 0.20}),
                        ("mmlu", {"accuracy": 0.70}),
                        ("icl", {"macro_accuracy": 0.60}),
                        ("alpaca_wikitext", {"alpaca_mean_response_loss": 1.2,
                                              "wikitext_perplexity": 8.5}),
                        ("elephant", {"accuracy": 0.75}),
                    ):
                        broad_records.append({"model": model, "state_id": state,
                                              "benchmark": benchmark, "result": result})
            rb.atomic_json(root / "analysis/broad_summary.json",
                           {"status": "complete", "records": broad_records})
            rb.atomic_json(root / "analysis/feedback_summary.json",
                           {"status": "complete", "states": feedback_states})
            with mock.patch.object(sys, "argv", ["export_report.py", "--result-root",
                                                  str(root), "--artifact-root", str(artifacts)]):
                self.assertEqual(export.main(), 0)
            tex = (artifacts / "random_mask_baselines.tex").read_text(encoding="utf-8")
            self.assertIn("Invalid rate", tex)
            self.assertIn("Matched random", tex)
            self.assertIn("Common-suite supporting outcomes", tex)
            self.assertIn("For Llama", tex)
            self.assertIn("For Qwen", tex)


if __name__ == "__main__":
    unittest.main()

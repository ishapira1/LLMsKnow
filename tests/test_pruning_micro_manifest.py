from __future__ import annotations

import argparse
import unittest

from scripts.build_strict_sycophancy_manifests import parse_sizes
from scripts.subset_pruning_evaluation_manifest import (
    DEFAULT_CONDITIONS,
    select_rows,
)


def _row(
    dataset: str,
    question_id: str,
    condition: str,
    *,
    strict_flip: bool,
    split: str = "val",
) -> dict[str, object]:
    return {
        "example_id": f"{dataset}:{split}:{question_id}:{condition}",
        "dataset": dataset,
        "split": split,
        "question_id": question_id,
        "draw_idx": 0,
        "condition": condition,
        "baseline_strict_flip": strict_flip,
    }


class CustomManifestSizeTests(unittest.TestCase):
    def test_micro_size_parses_without_changing_defaults(self) -> None:
        self.assertEqual(parse_sizes("micro=8"), (("micro", 8),))
        self.assertEqual(
            parse_sizes("smoke=16,pilot=128,main=412"),
            (("smoke", 16), ("pilot", 128), ("main", 412)),
        )

    def test_size_parser_rejects_ambiguous_or_non_nested_values(self) -> None:
        for value in ("", "micro", "micro=x", "micro=0", "large=16,small=8"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_sizes(value)


class EvaluationSubsetTests(unittest.TestCase):
    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for dataset in ("arc_challenge", "commonsense_qa"):
            for index in range(5):
                for condition in DEFAULT_CONDITIONS:
                    rows.append(
                        _row(
                            dataset,
                            f"q{index}",
                            condition,
                            strict_flip=index < 4,
                        )
                    )
        return rows

    def test_strict_subset_is_balanced_complete_and_deterministic(self) -> None:
        selected_a, audit_a = select_rows(
            self._rows(),
            questions=4,
            seed=5,
            splits=("val",),
            conditions=DEFAULT_CONDITIONS,
            require_baseline_strict_flip=True,
        )
        selected_b, audit_b = select_rows(
            reversed(self._rows()),
            questions=4,
            seed=5,
            splits=("val",),
            conditions=DEFAULT_CONDITIONS,
            require_baseline_strict_flip=True,
        )
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(len(selected_a), 4 * len(DEFAULT_CONDITIONS))
        self.assertEqual(
            audit_a["selected_questions_by_dataset"],
            {"arc_challenge": 2, "commonsense_qa": 2},
        )
        self.assertTrue(all(bool(row["baseline_strict_flip"]) for row in selected_a))

    def test_missing_condition_fails_instead_of_shrinking_requested_total(self) -> None:
        rows = self._rows()
        rows = [
            row
            for row in rows
            if not (
                row["dataset"] == "arc_challenge"
                and row["question_id"] in {"q0", "q1", "q2", "q3"}
                and row["condition"] == "suggest_correct_strong"
            )
        ]
        with self.assertRaisesRegex(ValueError, "insufficient eligible held-out"):
            select_rows(
                rows,
                questions=4,
                seed=5,
                splits=("val",),
                conditions=DEFAULT_CONDITIONS,
                require_baseline_strict_flip=True,
            )


if __name__ == "__main__":
    unittest.main()

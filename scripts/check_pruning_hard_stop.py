#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _relative_increase(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate <= 0 else float("inf")
    return (candidate - baseline) / abs(baseline)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record an operational early-stop sentinel after a hard utility violation."
    )
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-neutral-accuracy-drop", type=float, required=True)
    parser.add_argument("--max-neutral-probability-drop", type=float, required=True)
    parser.add_argument("--max-correction-accuracy-drop", type=float, required=True)
    parser.add_argument("--max-agreement-accuracy-drop", type=float, required=True)
    parser.add_argument("--max-preservation-loss-increase", type=float, required=True)
    parser.add_argument("--max-wikitext-perplexity-increase", type=float, required=True)
    parser.add_argument("--max-other-wrong-invalid-increase", type=float, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.selection_summary)
    baseline = frame[(frame["p"].astype(float) == 0) & (frame["q"].astype(float) == 0)]
    candidate = frame[~((frame["p"].astype(float) == 0) & (frame["q"].astype(float) == 0))]
    if len(baseline) != 1 or len(candidate) != 1:
        raise ValueError(
            f"Expected one baseline and one candidate row, found {len(baseline)} and {len(candidate)}"
        )
    base = baseline.iloc[0]
    cand = candidate.iloc[0]
    reasons: list[str] = []
    neutral_drop = float(base["neutral_accuracy"] - cand["neutral_accuracy"])
    neutral_probability_drop = float(
        base["neutral_correct_probability"] - cand["neutral_correct_probability"]
    )
    correction_drop = float(base["correction_accuracy"] - cand["correction_accuracy"])
    agreement_drop = float(base["agreement_accuracy"] - cand["agreement_accuracy"])
    preservation_increase = _relative_increase(
        float(cand["preservation_loss"]), float(base["preservation_loss"])
    )
    wikitext_increase = _relative_increase(
        float(cand["wikitext_perplexity"]), float(base["wikitext_perplexity"])
    )
    other_wrong_invalid_increase = float(
        cand["other_wrong_invalid_rate"] - base["other_wrong_invalid_rate"]
    )
    checks = (
        ("neutral_accuracy_drop", neutral_drop, args.max_neutral_accuracy_drop),
        (
            "neutral_correct_probability_drop",
            neutral_probability_drop,
            args.max_neutral_probability_drop,
        ),
        (
            "correction_accuracy_drop",
            correction_drop,
            args.max_correction_accuracy_drop,
        ),
        (
            "agreement_accuracy_drop",
            agreement_drop,
            args.max_agreement_accuracy_drop,
        ),
        (
            "preservation_loss_relative_increase",
            preservation_increase,
            args.max_preservation_loss_increase,
        ),
        (
            "wikitext_perplexity_relative_increase",
            wikitext_increase,
            args.max_wikitext_perplexity_increase,
        ),
        (
            "other_wrong_invalid_increase",
            other_wrong_invalid_increase,
            args.max_other_wrong_invalid_increase,
        ),
    )
    for name, value, limit in checks:
        if value > limit and not math.isclose(value, limit, rel_tol=1e-9, abs_tol=1e-12):
            reasons.append(f"{name}={value:.8g} > {limit:.8g}")
    payload = {
        "stop": bool(reasons),
        "p": float(cand["p"]),
        "q": float(cand["q"]),
        "reasons": reasons,
        "observed": {name: value for name, value, _ in checks},
        "limits": {name: limit for name, _, limit in checks},
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

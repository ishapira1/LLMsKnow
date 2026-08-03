#!/usr/bin/env python3
"""Export lightweight verified random_baseline assets and a LaTeX subsection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics

import random_baseline as rb


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    final = rb.read_json(args.result_root / "analysis/final_report.json")
    if final.get("status") != "complete":
        raise RuntimeError("Final report is incomplete")
    completion_audit = rb.read_json(args.result_root / "audit/completion_audit.json")
    if completion_audit.get("status") != "complete":
        raise RuntimeError("Completion audit is incomplete")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in rb.MODEL_SPECS:
        source = args.result_root / "analysis" / model
        summary = rb.read_json(source / "core_summary.json")
        for name in ("core_summary.json", "seed_distribution.jsonl", "seed_distribution.csv",
                     "pareto.pdf", "pareto.png"):
            shutil.copy2(source / name, args.artifact_root / f"{model}_{name}")
        learned = summary["summaries"]["learned"]
        base = summary["summaries"]["base"]
        matched = [row for row in summary["seed_distribution"]
                   if row["family"] == "module_magnitude_matched"]
        rows.append({"model": model.title(),
                     "base_syco": rb.metric_rate(base, "strong_wrong_adoption"),
                     "learned_syco": rb.metric_rate(learned, "strong_wrong_adoption"),
                     "learned_neutral": rb.metric_rate(learned, "neutral_accuracy"),
                     "matched_median": statistics.median(row["strong_wrong_adoption"] for row in matched),
                     "matched_min": min(row["strong_wrong_adoption"] for row in matched),
                     "matched_max": max(row["strong_wrong_adoption"] for row in matched),
                     "p": summary["confirmatory_inference"]["empirical_rank_p_one_sided"],
                     "equivalents": summary["confirmatory_inference"]["matched_random_equivalent_count"]})
    rb.atomic_json(args.artifact_root / "final_report.json", final)
    for source_name, output_name in (
        ("analysis/broad_summary.json", "broad_summary.json"),
        ("analysis/feedback_summary.json", "feedback_summary.json"),
        ("audit/completion_audit.json", "completion_audit.json"),
    ):
        shutil.copy2(args.result_root / source_name, args.artifact_root / output_name)
    rb.atomic_json(args.artifact_root / "provenance.json", {
        "status": "complete", "experiment": rb.EXPERIMENT,
        "source_completion_audit": str(args.result_root / "audit/completion_audit.json"),
        "source_completion_audit_sha256": rb.sha256_file(
            args.result_root / "audit/completion_audit.json"
        ),
        "source_completion_audit_logical_sha256": completion_audit["audit_sha256"],
        "verified_counts": completion_audit["verified_counts"],
        "preflight_pins_sha256": rb.sha256_file(args.result_root / "registry/preflight_pins.json"),
        "feedback_summary_sha256": rb.sha256_file(args.result_root / "analysis/feedback_summary.json"),
        "exported_at": rb.utc_now(),
    })
    table_rows = "\n".join(
        f"{row['model']} & {pct(row['base_syco'])} & {pct(row['learned_syco'])} & "
        f"{pct(row['learned_neutral'])} & {pct(row['matched_median'])} "
        f"[{pct(row['matched_min'])}, {pct(row['matched_max'])}] & {row['p']:.4f} & "
        f"{row['equivalents']} \\\\" for row in rows)
    conclusion = {
        "supported": "Both models support weight-selection specificity under the preregistered rule.",
        "model-specific": "The preregistered result is model-specific; only one model passes all criteria.",
        "unsupported": "The preregistered cross-model specificity hypothesis is not supported.",
    }[final["conclusion"]]
    tex = rf"""\subsection{{Random-Mask Baselines}}
\label{{sec:random-mask-baselines}}
We preregistered the hypothesis that the behavioral change depends on selecting
specific weights, rather than zeroing an arbitrary set of the same size.  For
Mixed-996 and the 3,139-weight Qwen replication mask, we compared the learned
mask with 20 fixed-seed uniform-global controls and 20 controls matched exactly
on per-matrix counts and a 20-bin absolute-weight-magnitude distribution.  All
controls were disjoint from their learned mask, and eligibility was fixed by
the base model's neutral-correct cohort.  The primary outcome was adoption of a
strong wrong suggestion; neutral accuracy and invalid-answer rate were
guardrails.  A random mask was considered equivalent within 3 percentage
points on adoption and 2 points on neutral accuracy.

\begin{{table}}[t]
\centering
\caption{{Preregistered random-mask baseline results (percent).  Matched values
show the seed median and range; $p$ is the one-sided empirical rank test.}}
\label{{tab:random-mask-baselines}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Model & Base adoption & Learned adoption & Learned neutral & Matched adoption & $p$ & Equiv. \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

Figure~\ref{{fig:random-mask-pareto}} reports the seed-level distributions and
the sycophancy--neutral-accuracy Pareto relationship for both models.
\begin{{figure}}[t]
\centering
\includegraphics[width=.49\linewidth]{{plots/random_baseline_llama_pareto.pdf}}
\includegraphics[width=.49\linewidth]{{plots/random_baseline_qwen_pareto.pdf}}
\caption{{Learned masks (teal), magnitude-matched random controls (orange), and
uniform controls across fixed seeds.}}
\label{{fig:random-mask-pareto}}
\end{{figure}}

\paragraph{{Preregistered conclusion.}} {conclusion}
"""
    rb.atomic_text(args.artifact_root / "random_mask_baselines.tex", tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

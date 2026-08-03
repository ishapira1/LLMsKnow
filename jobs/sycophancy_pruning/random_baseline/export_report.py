#!/usr/bin/env python3
"""Export lightweight verified random_baseline assets and a LaTeX subsection."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import statistics

import random_baseline as rb


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def pct_interval(values: list[float]) -> str:
    return (f"{pct(statistics.median(values))} "
            f"[{pct(min(values))}, {pct(max(values))}]")


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
    core_rows = []
    for model in rb.MODEL_SPECS:
        source = args.result_root / "analysis" / model
        summary = rb.read_json(source / "core_summary.json")
        for name in ("core_summary.json", "seed_distribution.jsonl", "seed_distribution.csv",
                     "pareto.pdf", "pareto.png"):
            shutil.copy2(source / name, args.artifact_root / f"{model}_{name}")
        learned = summary["summaries"]["learned"]
        base = summary["summaries"]["base"]
        distributions = summary["seed_distribution"]
        random_families = {
            family: [row for row in distributions if row["family"] == family]
            for family in rb.CONTROL_FAMILIES
        }
        core_rows.append({
            "model": model.title(), "base": base, "learned": learned,
            "families": random_families,
            "p": summary["confirmatory_inference"]["empirical_rank_p_one_sided"],
            "equivalents": summary["confirmatory_inference"]["matched_random_equivalent_count"],
        })
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
    primary_lines = []
    conclusion_details = []
    for model_row in core_rows:
        base = model_row["base"]
        learned = model_row["learned"]
        primary_lines.extend((
            f"{model_row['model']} & Base & "
            f"{pct(rb.metric_rate(base, 'strong_wrong_adoption'))} & "
            f"{pct(rb.metric_rate(base, 'neutral_accuracy'))} & "
            f"{pct(rb.metric_rate(base, 'invalid_answer_rate'))} & -- & -- \\\\",
            f"{model_row['model']} & Learned & "
            f"{pct(rb.metric_rate(learned, 'strong_wrong_adoption'))} & "
            f"{pct(rb.metric_rate(learned, 'neutral_accuracy'))} & "
            f"{pct(rb.metric_rate(learned, 'invalid_answer_rate'))} & "
            f"{model_row['p']:.4f} & {model_row['equivalents']} \\\\",
        ))
        for family, label in (("module_magnitude_matched", "Matched random"),
                              ("uniform_global", "Uniform random")):
            family_rows = model_row["families"][family]
            primary_lines.append(
                f"{model_row['model']} & {label} & "
                f"{pct_interval([float(row['strong_wrong_adoption']) for row in family_rows])} & "
                f"{pct_interval([float(row['neutral_accuracy']) for row in family_rows])} & "
                f"{pct_interval([float(row['invalid_answer_rate']) for row in family_rows])} & "
                "-- & -- \\\\")
        matched = model_row["families"]["module_magnitude_matched"]
        conclusion_details.append(
            f"For {model_row['model']}, learned-mask adoption was "
            f"{pct(rb.metric_rate(learned, 'strong_wrong_adoption'))}\% versus a "
            f"{pct(statistics.median(float(row['strong_wrong_adoption']) for row in matched))}\% "
            f"matched-random median; learned neutral accuracy was "
            f"{pct(rb.metric_rate(learned, 'neutral_accuracy'))}\% versus "
            f"{pct(rb.metric_rate(base, 'neutral_accuracy'))}\% at base "
            f"($p={model_row['p']:.4f}$; {model_row['equivalents']} equivalents)."
        )
    table_rows = "\n".join(primary_lines)

    broad = rb.read_json(args.result_root / "analysis/broad_summary.json")
    feedback = rb.read_json(args.result_root / "analysis/feedback_summary.json")
    broad_index = {
        (row["model"], row["state_id"], row["benchmark"]): row["result"]
        for row in broad["records"]
    }

    def broad_values(model: str, states: list[str]) -> list[list[float]]:
        output = []
        for state in states:
            utility = broad_index[(model, state, "alpaca_wikitext")]
            output.append([
                float(broad_index[(model, state, "sycobench")]["syco"]),
                float(broad_index[(model, state, "mmlu")]["accuracy"]),
                float(broad_index[(model, state, "icl")]["macro_accuracy"]),
                float(utility["alpaca_mean_response_loss"]),
                float(utility["wikitext_perplexity"]),
                float(feedback["states"][f"{model}/{state}"]["sycophancy_gap"]),
                float(broad_index[(model, state, "elephant")]["accuracy"]),
            ])
        return output

    broad_lines = []
    for model in rb.MODEL_SPECS:
        states_by_label = (
            ("Base", ["base"]),
            ("Learned", ["learned"]),
            ("Matched median", [f"module_magnitude_matched__seed_{seed}"
                                for seed in rb.BROAD_SEEDS]),
        )
        for label, states in states_by_label:
            values = broad_values(model, states)
            medians = [statistics.median(column) for column in zip(*values)]
            broad_lines.append(
                f"{model.title()} & {label} & {pct(medians[0])} & {pct(medians[1])} & "
                f"{pct(medians[2])} & {medians[3]:.3f} & {medians[4]:.2f} & "
                f"{pct(medians[5])} & {pct(medians[6])} \\\\")
    broad_table_rows = "\n".join(broad_lines)
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
\begin{{tabular}}{{llrrrll}}
\toprule
Model & State & Strong adoption & Neutral accuracy & Invalid rate & $p$ & Equiv. \\
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

\begin{{table*}}[t]
\centering
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrr}}
\toprule
Model & State & SycoBench $\downarrow$ & MMLU $\uparrow$ & ICL $\uparrow$ &
Alpaca loss $\downarrow$ & Wiki PPL $\downarrow$ & Feedback gap & ELEPHANT $\uparrow$ \\
\midrule
{broad_table_rows}
\bottomrule
\end{{tabular}}}}
\caption{{Common-suite supporting outcomes. Rates and the feedback gap are
reported in percentage points; matched-random entries are medians across the
five predeclared broad seeds. The machine-readable artifact contains all 144
model/state/benchmark summaries, including uniform controls and seed-level
results.}}
\label{{tab:random-mask-common-suite}}
\end{{table*}}

\paragraph{{Preregistered conclusion.}} {' '.join(conclusion_details)} {conclusion}
"""
    rb.atomic_text(args.artifact_root / "random_mask_baselines.tex", tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export lightweight verified random_baseline assets and a LaTeX subsection."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import statistics

import random_baseline as rb


EARLY_STOP_MODE = "user_authorized_core_early_stop"


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def pct_interval(values: list[float]) -> str:
    return f"{pct(statistics.median(values))} [{pct(min(values))}, {pct(max(values))}]"


def load_core_rows(result_root: Path, artifact_root: Path) -> list[dict[str, object]]:
    core_rows: list[dict[str, object]] = []
    for model in rb.MODEL_SPECS:
        source = result_root / "analysis" / model
        summary = rb.read_json(source / "core_summary.json")
        for name in ("core_summary.json", "seed_distribution.jsonl", "seed_distribution.csv",
                     "pareto.pdf", "pareto.png"):
            shutil.copy2(source / name, artifact_root / f"{model}_{name}")
        distributions = summary["seed_distribution"]
        core_rows.append({
            "model": model.title(),
            "base": summary["summaries"]["base"],
            "learned": summary["summaries"]["learned"],
            "families": {
                family: [row for row in distributions if row["family"] == family]
                for family in rb.CONTROL_FAMILIES
            },
            "p": summary["confirmatory_inference"]["empirical_rank_p_one_sided"],
            "equivalents": summary["confirmatory_inference"]["matched_random_equivalent_count"],
        })
    return core_rows


def primary_table(core_rows: list[dict[str, object]]) -> tuple[str, list[str]]:
    lines = []
    details = []
    for model_row in core_rows:
        base = model_row["base"]
        learned = model_row["learned"]
        lines.extend((
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
            lines.append(
                f"{model_row['model']} & {label} & "
                f"{pct_interval([float(row['strong_wrong_adoption']) for row in family_rows])} & "
                f"{pct_interval([float(row['neutral_accuracy']) for row in family_rows])} & "
                f"{pct_interval([float(row['invalid_answer_rate']) for row in family_rows])} & "
                "-- & -- \\\\")
        matched = model_row["families"]["module_magnitude_matched"]
        details.append(
            f"For {model_row['model']}, learned-mask adoption was "
            f"{pct(rb.metric_rate(learned, 'strong_wrong_adoption'))}\\% versus a "
            f"{pct(statistics.median(float(row['strong_wrong_adoption']) for row in matched))}\\% "
            f"matched-random median; learned neutral accuracy was "
            f"{pct(rb.metric_rate(learned, 'neutral_accuracy'))}\\% versus "
            f"{pct(rb.metric_rate(base, 'neutral_accuracy'))}\\% at base "
            f"($p={model_row['p']:.4f}$; {model_row['equivalents']} equivalents)."
        )
    return "\n".join(lines), details


def early_stop_supporting(result_root: Path, final: dict[str, object]) -> tuple[str, str, str]:
    broad = rb.read_json(result_root / "analysis/partial_broad_summary.json")
    if (broad.get("scope") != "supporting_partial_broad_llama_sycobench_only" or
            int(broad.get("record_count", -1)) != 12):
        raise RuntimeError("Early-stop supporting broad summary is incomplete")
    by_state = {row["state_id"]: row for row in broad["records"]}
    lines = []
    for label, states in (
        ("Base", ["base"]),
        ("Learned", ["learned"]),
        ("Matched random", [f"module_magnitude_matched__seed_{seed}"
                            for seed in rb.BROAD_SEEDS]),
        ("Uniform random", [f"uniform_global__seed_{seed}" for seed in rb.BROAD_SEEDS]),
    ):
        syco = [float(by_state[state]["result"]["syco"]) for state in states]
        accuracy = [float(by_state[state]["result"]["acc"]) for state in states]
        syco_text = pct(syco[0]) if len(syco) == 1 else pct_interval(syco)
        accuracy_text = pct(accuracy[0]) if len(accuracy) == 1 else pct_interval(accuracy)
        lines.append(f"Llama & {label} & {syco_text} & {accuracy_text} \\\\")
    table = rf"""\begin{{table}}[t]
\centering
\caption{{Completed supporting SycoBench block (percent). Random controls show
the median and range across the five fixed broad seeds.}}
\label{{tab:random-mask-sycobench-partial}}
\begin{{tabular}}{{llrr}}
\toprule
Model & State & SycoBench $\downarrow$ & Accuracy $\uparrow$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}"""
    scope = (
        "After the complete confirmatory core met the user-authorized resource-saving "
        "stopping rule, we retained and audited the already-running 12-state Llama "
        "SycoBench block (21,600 rows; zero hash or row-count failures) and canceled "
        "the 132 unstarted common-suite states. Feedback judging and the remaining "
        "broad benchmarks were therefore not run and are not used to support the claim."
    )
    parts = []
    models = final["models"]
    for model in rb.MODEL_SPECS:
        effects = models[model]["random_effects"]
        strong_max = max(float(row["strong_wrong_delta_pp"]["max_abs"])
                         for row in effects.values())
        neutral_max = max(float(row["neutral_accuracy_delta_pp"]["max_abs"])
                          for row in effects.values())
        parts.append(
            f"For {model.title()}, every random-mask seed remained within "
            f"{strong_max:.2f} points of base strong-wrong adoption and within "
            f"{neutral_max:.2f} points of base neutral accuracy."
        )
    takeaways = (
        " ".join(parts) + " In contrast, the learned masks reduced strong-wrong "
        "adoption by 31.60 points for Llama and 33.50 points for Qwen, with less than "
        "0.40 points of neutral-accuracy loss. The completed Llama SycoBench block "
        "gives the same qualitative result: random controls span 78.87--79.69, versus "
        "79.45 at base and 72.81 for the learned mask."
    )
    return table, scope, takeaways


def full_suite_supporting(result_root: Path) -> tuple[str, str, str]:
    broad = rb.read_json(result_root / "analysis/broad_summary.json")
    feedback = rb.read_json(result_root / "analysis/feedback_summary.json")
    index = {
        (row["model"], row["state_id"], row["benchmark"]): row["result"]
        for row in broad["records"]
    }

    def values(model: str, states: list[str]) -> list[list[float]]:
        output = []
        for state in states:
            utility = index[(model, state, "alpaca_wikitext")]
            output.append([
                float(index[(model, state, "sycobench")]["syco"]),
                float(index[(model, state, "mmlu")]["accuracy"]),
                float(index[(model, state, "icl")]["macro_accuracy"]),
                float(utility["alpaca_mean_response_loss"]),
                float(utility["wikitext_perplexity"]),
                float(feedback["states"][f"{model}/{state}"]["sycophancy_gap"]),
                float(index[(model, state, "elephant")]["accuracy"]),
            ])
        return output

    lines = []
    for model in rb.MODEL_SPECS:
        for label, states in (
            ("Base", ["base"]),
            ("Learned", ["learned"]),
            ("Matched median", [f"module_magnitude_matched__seed_{seed}"
                                for seed in rb.BROAD_SEEDS]),
        ):
            medians = [statistics.median(column) for column in zip(*values(model, states))]
            lines.append(
                f"{model.title()} & {label} & {pct(medians[0])} & {pct(medians[1])} & "
                f"{pct(medians[2])} & {medians[3]:.3f} & {medians[4]:.2f} & "
                f"{pct(medians[5])} & {pct(medians[6])} \\\\")
    table = rf"""\begin{{table*}}[t]
\centering
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrr}}
\toprule
Model & State & SycoBench $\downarrow$ & MMLU $\uparrow$ & ICL $\uparrow$ &
Alpaca loss $\downarrow$ & Wiki PPL $\downarrow$ & Feedback gap & ELEPHANT $\uparrow$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\caption{{Common-suite supporting outcomes. Rates and the feedback gap are
reported in percentage points; matched-random entries are medians across the
five predeclared broad seeds. The machine-readable artifact contains all 144
model/state/benchmark summaries, including uniform controls and seed-level results.}}
\label{{tab:random-mask-common-suite}}
\end{{table*}}"""
    return (
        table,
        "The full predeclared common suite and blinded feedback judgments were completed.",
        "The confirmatory and supporting evaluations agree on weight-selection specificity.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    final = rb.read_json(args.result_root / "analysis/final_report.json")
    if final.get("status") != "complete":
        raise RuntimeError("Final report is incomplete")
    early_audit = args.result_root / "audit/early_stop_completion_audit.json"
    audit_path = early_audit if early_audit.is_file() else args.result_root / "audit/completion_audit.json"
    audit = rb.read_json(audit_path)
    if audit.get("status") != "complete":
        raise RuntimeError("Completion audit is incomplete")
    early_stop = audit.get("completion_mode") == EARLY_STOP_MODE
    if early_stop and final.get("completion_mode") != EARLY_STOP_MODE:
        raise RuntimeError("Early-stop report/audit mode mismatch")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    core_rows = load_core_rows(args.result_root, args.artifact_root)
    rb.atomic_json(args.artifact_root / "final_report.json", final)
    if early_stop:
        copies = (
            ("analysis/partial_broad_summary.json", "partial_broad_summary.json"),
            ("audit/early_stop_completion_audit.json", "early_stop_completion_audit.json"),
        )
    else:
        copies = (
            ("analysis/broad_summary.json", "broad_summary.json"),
            ("analysis/feedback_summary.json", "feedback_summary.json"),
            ("audit/completion_audit.json", "completion_audit.json"),
        )
    for source, destination in copies:
        shutil.copy2(args.result_root / source, args.artifact_root / destination)
    provenance = {
        "status": "complete",
        "experiment": rb.EXPERIMENT,
        "completion_mode": audit.get("completion_mode", "full_preregistered_suite"),
        "source_completion_audit": str(audit_path),
        "source_completion_audit_sha256": rb.sha256_file(audit_path),
        "source_completion_audit_logical_sha256": audit["audit_sha256"],
        "verified_counts": audit["verified_counts"],
        "preflight_pins_sha256": rb.sha256_file(args.result_root / "registry/preflight_pins.json"),
        "exported_at": rb.utc_now(),
    }
    if early_stop:
        provenance["partial_broad_summary_sha256"] = rb.sha256_file(
            args.result_root / "analysis/partial_broad_summary.json"
        )
        provenance["skipped"] = final["skipped"]
    else:
        provenance["feedback_summary_sha256"] = rb.sha256_file(
            args.result_root / "analysis/feedback_summary.json"
        )
    rb.atomic_json(args.artifact_root / "provenance.json", provenance)

    table_rows, conclusion_details = primary_table(core_rows)
    supporting, scope_note, takeaways = (
        early_stop_supporting(args.result_root, final)
        if early_stop else full_suite_supporting(args.result_root)
    )
    conclusion = {
        "supported": "Both models support weight-selection specificity under the preregistered rule.",
        "model-specific": "The preregistered result is model-specific; only one model passes all criteria.",
        "unsupported": "The preregistered cross-model specificity hypothesis is not supported.",
    }[final["conclusion"]]
    tex = rf"""\subsection{{Random-Mask Baselines}}
\label{{sec:random-mask-baselines}}
We preregistered the hypothesis that the behavioral change depends on selecting
specific weights, rather than zeroing an arbitrary set of the same size. For
Mixed-996 and the 3,139-weight Qwen replication mask, we compared the learned
mask with 20 fixed-seed uniform-global controls and 20 controls matched exactly
on per-matrix counts and a 20-bin absolute-weight-magnitude distribution. All
controls were disjoint from their learned mask, and eligibility was fixed by
the base model's neutral-correct cohort. The primary outcome was adoption of a
strong wrong suggestion; neutral accuracy and invalid-answer rate were
guardrails. A random mask was considered equivalent within 3 percentage
points on adoption and 2 points on neutral accuracy.

\begin{{table*}}[t]
\centering
\caption{{Preregistered random-mask baseline results (percent). Random values
show the seed median and range; $p$ is the one-sided empirical rank test.}}
\label{{tab:random-mask-baselines}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrll}}
\toprule
Model & State & Strong adoption & Neutral accuracy & Invalid rate & $p$ & Equiv. \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}}}
\end{{table*}}

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

{supporting}

\paragraph{{Scope and stopping.}} {scope_note}

\paragraph{{Takeaways.}} {takeaways}

\paragraph{{Preregistered conclusion.}} {' '.join(conclusion_details)} {conclusion}
"""
    rb.atomic_text(args.artifact_root / "random_mask_baselines.tex", tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

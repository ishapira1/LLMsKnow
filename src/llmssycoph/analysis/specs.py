from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class AnalysisCellSpec:
    kind: str
    text: str = ""
    function_name: str = ""
    output_stem: str = ""
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisSubsectionSpec:
    title: str
    cells: List[AnalysisCellSpec]
    intro: str = ""


@dataclass(frozen=True)
class AnalysisSectionSpec:
    title: str
    subsections: List[AnalysisSubsectionSpec]
    intro: str = ""


@dataclass(frozen=True)
class AnalysisNotebookSpec:
    name: str
    title: str
    sections: List[AnalysisSectionSpec]


DEFAULT_MC_NOTEBOOK_SPEC = AnalysisNotebookSpec(
    name="default_mc_skeleton",
    title="Multiple-Choice Run Analysis Skeleton",
    sections=[
        AnalysisSectionSpec(
            title="Run Overview",
            intro="This section anchors the run: what was evaluated, how many rows exist, and what template families are present.",
            subsections=[
                AnalysisSubsectionSpec(
                    title="Run Metadata",
                    intro="High-level counts and run metadata.",
                    cells=[
                        AnalysisCellSpec(kind="table", function_name="table_run_overview", output_stem="run_overview"),
                    ],
                ),
                AnalysisSubsectionSpec(
                    title="Template Summary",
                    intro="Per-template summary of question counts and mean probabilities.",
                    cells=[
                        AnalysisCellSpec(
                            kind="table",
                            function_name="table_template_overview",
                            output_stem="template_overview",
                        ),
                    ],
                ),
            ],
        ),
        AnalysisSectionSpec(
            title="Probe Availability",
            intro="Probe-specific cells should be explicit about whether probe artifacts are available for this run.",
            subsections=[
                AnalysisSubsectionSpec(
                    title="Probe Inventory",
                    intro="Saved probe families and template coverage.",
                    cells=[
                        AnalysisCellSpec(
                            kind="table",
                            function_name="table_probe_inventory",
                            output_stem="probe_inventory",
                        ),
                    ],
                ),
            ],
        ),
        AnalysisSectionSpec(
            title="Behavior",
            intro="External behavior summaries should appear before deeper hidden-knowledge or susceptibility analysis.",
            subsections=[
                AnalysisSubsectionSpec(
                    title="Neutral Option Selection",
                    intro="Simple sanity check of which MC options are chosen under the neutral prompt.",
                    cells=[
                        AnalysisCellSpec(
                            kind="plot",
                            function_name="plot_neutral_option_selection",
                            output_stem="neutral_option_selection",
                        ),
                    ],
                ),
            ],
        ),
    ],
)

FULL_MC_NOTEBOOK_SPEC = AnalysisNotebookSpec(
    name="full_mc_report",
    title="Multiple-Choice Run Analysis",
    sections=[
        AnalysisSectionSpec(
            title="External",
            intro="This section uses only external model outputs from `sampling/sampled_responses.csv`.",
            subsections=[
                AnalysisSubsectionSpec(
                    title="Neutral Signals",
                    intro="High-level counts and external confidence/accuracy summaries under each template type.",
                    cells=[
                        AnalysisCellSpec(kind="table", function_name="table_external_summary_statistics", output_stem="01_01_external_summary_statistics"),
                        AnalysisCellSpec(kind="table", function_name="table_external_accuracy_metrics", output_stem="01_02_external_accuracy_metrics"),
                        AnalysisCellSpec(kind="plot", function_name="plot_effective_responses_histogram", output_stem="01_03_effective_responses_histogram", text="This histogram shows the effective number of responses, defined as `exp(H(p(.|x)))`, where `H` is the entropy of the model's full multiple-choice distribution. Larger values mean the mass is spread over more answers."),
                        AnalysisCellSpec(kind="plot", function_name="plot_accuracy_by_effective_responses_bucket", output_stem="01_04_accuracy_by_effective_responses_bucket", text="Here we split examples into 4 quantiles of the effective number of responses and plot empirical accuracy in each quantile. This checks whether flatter output distributions are associated with more mistakes."),
                        AnalysisCellSpec(kind="table", function_name="table_error_detection_auroc_by_effective_responses", output_stem="01_05_error_detection_auroc"),
                        AnalysisCellSpec(kind="plot", function_name="plot_reliability_diagram_top_probability", output_stem="01_06_reliability_diagram_top_probability", text="Reliability is computed from the top predicted probability `p_(1)(x)`. The diagonal is perfect calibration; curves below the diagonal indicate over-confidence."),
                    ],
                ),
                AnalysisSubsectionSpec(
                    title="Sycophancy",
                    intro="Paired neutral-vs-biased external behavior. Here `x` is the neutral prompt and `x'` is the matched bias-injected prompt for the same question.",
                    cells=[
                        AnalysisCellSpec(kind="plot", function_name="plot_sycophancy_delta_histograms", output_stem="01_07_sycophancy_delta_histograms", text="The top row plots `ΔP(correct) = P(correct|x) - P(correct|x')`. The second row plots the relative version `(P_x - P_x') / (P_x + P_x')`, which reduces the effect of simple headroom."),
                        AnalysisCellSpec(kind="table", function_name="table_sycophancy_flip_table", output_stem="01_08_sycophancy_flip_table"),
                        AnalysisCellSpec(kind="table", function_name="table_distribution_shift_metrics", output_stem="01_09_distribution_shift_metrics"),
                    ],
                ),
                AnalysisSubsectionSpec(
                    title="Does Confidence Explain Susceptibility to Sycophancy",
                    intro="These cells focus on whether neutral external confidence already predicts later movement toward the user-targeted answer.",
                    cells=[
                        AnalysisCellSpec(kind="plot", function_name="plot_incorrect_suggestion_transition_heatmap", output_stem="01_10_incorrect_suggestion_transition_heatmap", text="For `incorrect_suggestion`, rows are the neutral state and columns are the biased state. The three states are: correct, the specific user-suggested target `b`, and another wrong answer."),
                        AnalysisCellSpec(kind="plot", function_name="plot_incorrect_suggestion_top1_confidence_before_after", output_stem="01_11_incorrect_suggestion_top1_confidence_before_after", text="These side-by-side histograms compare the top-1 probability `p_(1)` before and after the incorrect-suggestion bias, with a shared y-axis for direct comparison."),
                        AnalysisCellSpec(kind="plot", function_name="plot_incorrect_suggestion_bias_target_gain_and_margin", output_stem="01_12_incorrect_suggestion_bias_target_gain_and_margin", text="Left: the bias-target gain `Δ_bias = P_{x'}(b) - P_x(b)`. Right: the neutral external margin `m_bias(x) = P_x(c) - P_x(b)`, which measures how strongly the correct answer outranked the suggested target before bias."),
                        AnalysisCellSpec(kind="plot", function_name="plot_target_adoption_by_prebias_chosen_margin", output_stem="01_13_target_adoption_by_prebias_chosen_margin", text="We bucket examples into 5 quantiles of the negative neutral chosen-answer margin `-m_top(x)`, where `m_top(x)=p_(1)(x)-p_(2)(x)`, and then plot the rate of adopting the user target under bias."),
                    ],
                ),
            ],
        ),
        AnalysisSectionSpec(
            title="Probe Analysis",
            intro="This section uses probe artifacts from `probes/probe_scores_by_prompt.csv`, `probes/chosen_probe/`, and `probes/all_probes/`.",
            subsections=[
                AnalysisSubsectionSpec(
                    title="Notation and Plotting Conventions",
                    cells=[
                        AnalysisCellSpec(
                            kind="markdown",
                            text="\n".join(
                                [
                                    "- `x` = neutral prompt",
                                    "- `x'` = bias-injected prompt",
                                    "- `c` = correct answer",
                                    "- `b` = bias-target answer when the template specifies one",
                                    "- `ŷ_x` = chosen answer under `x`",
                                    "- `ŷ_x'` = chosen answer under `x'`",
                                    "- `s(x, a)` = saved chosen-probe score for candidate `a` under prompt `x`",
                                    "- `rank_probe(x, a)` = descending rank of `a` by probe score",
                                    "- `m_probe^bias(x) = s(x, c) - s(x, b)` when an explicit `b` exists",
                                    "- `m_probe^truth(x) = s(x, c) - max_{i != c} s(x, i)`",
                                    "",
                                    "Important artifact note: the saved prompt-level probe table contains matched-template chosen probes. So cross-condition score comparisons are between the neutral chosen probe on `x` and the matched bias-template chosen probe on `x'`, not a single neutral probe re-scored on both prompts.",
                                ]
                            ),
                        ),
                    ],
                ),
                AnalysisSubsectionSpec(
                    title="Probe Sanity and Consistency",
                    intro="These cells check whether the probe is above chance, stable across layers, and reasonably aligned with held-out data.",
                    cells=[
                        AnalysisCellSpec(kind="plot", function_name="plot_probe_layerwise_performance", output_stem="02_01_probe_layerwise_performance", text="Each panel shows AUC by layer for the train, validation, and test splits. The vertical dashed line marks the chosen layer used for the saved prompt-level probe scores."),
                        AnalysisCellSpec(kind="table", function_name="table_chosen_probe_summary", output_stem="02_02_chosen_probe_summary"),
                        AnalysisCellSpec(kind="table", function_name="table_probe_validity_by_template_type", output_stem="02_03_probe_validity_by_template_type"),
                        AnalysisCellSpec(kind="plot", function_name="plot_probe_score_stability_under_bias", output_stem="02_04_probe_score_stability_under_bias", text="These matched-template scatter plots compare the saved probe score for the same candidate before and after bias: `s(x,c)` vs `s(x',c)`, `s(x,b)` vs `s(x',b)` when an explicit target exists, and `m_probe^bias(x)` vs `m_probe^bias(x')`."),
                        AnalysisCellSpec(kind="plot", function_name="plot_probe_margin_and_rank_distributions", output_stem="02_05_probe_margin_and_rank_distributions", text="The first row shows the neutral truth margin `m_probe^truth(x)=s(x,c)-max_{i\\neq c}s(x,i)`. The second row shows the rank of the correct answer under the probe, where rank 1 means the probe places the correct answer first."),
                        AnalysisCellSpec(kind="table", function_name="table_probe_chance_baseline_check", output_stem="02_06_probe_chance_baseline_check"),
                    ],
                ),
                AnalysisSubsectionSpec(
                    title="Hidden Knowledge Under Pressure",
                    intro="The core question here is whether bias changes what the model says more than what the probe internally supports.",
                    cells=[
                        AnalysisCellSpec(kind="plot", function_name="plot_internal_margin_shift_under_bias", output_stem="02_07_internal_margin_shift_under_bias", text="`Δm_probe^bias = m_probe^bias(x') - m_probe^bias(x)` measures how much internal support moves toward or away from the bias target. Negative values mean the internal correct-vs-target gap shrank under bias."),
                        AnalysisCellSpec(kind="plot", function_name="plot_hidden_knowledge_rate_biased_wrong", output_stem="02_08_hidden_knowledge_rate_biased_wrong", text="`HK` means the biased response is wrong but the probe still scores the correct answer above the biased chosen answer. `HK_strict` means the correct answer is probe-rank 1 under the biased prompt."),
                        AnalysisCellSpec(kind="plot", function_name="plot_susceptibility_vs_neutral_probe_margin", output_stem="02_09_susceptibility_vs_neutral_probe_margin", text="These curves use 5 quantiles of the neutral probe margin. For `incorrect_suggestion`, the target is the suggested wrong answer `b`; for `suggest_correct`, the target is the correct answer `c`."),
                        AnalysisCellSpec(kind="plot", function_name="plot_external_vs_internal_predictor_grid", output_stem="02_10_external_vs_internal_predictor_grid", text="This 2x2 grid compares low/high neutral external signal against low/high neutral internal probe signal, then reports the probability of adopting the user target in each cell."),
                        AnalysisCellSpec(kind="table", function_name="table_candidate_ranking_comparison", output_stem="02_11_candidate_ranking_comparison"),
                        AnalysisCellSpec(kind="plot", function_name="plot_confident_yet_compliant_slice", output_stem="02_12_confident_yet_compliant_slice", text="This slice keeps only the top quartile of neutral probe margin examples, then asks how often the model still adopts the bias target and how often hidden knowledge remains within that high-support subset."),
                    ],
                ),
            ],
        ),
    ],
)

NOTEBOOK_SPECS: Dict[str, AnalysisNotebookSpec] = {
    DEFAULT_MC_NOTEBOOK_SPEC.name: DEFAULT_MC_NOTEBOOK_SPEC,
    FULL_MC_NOTEBOOK_SPEC.name: FULL_MC_NOTEBOOK_SPEC,
}


def get_notebook_spec(name: str) -> AnalysisNotebookSpec:
    try:
        return NOTEBOOK_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown notebook spec: {name}") from exc


def list_notebook_specs() -> list[str]:
    return sorted(NOTEBOOK_SPECS)

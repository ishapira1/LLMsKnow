#!/usr/bin/env python3
"""Generate the appendix factual-transfer table from the Bonham campaign.

The published Bonham report contains all behavioral cells and OpenBookQA
accuracy, but it does not export neutral CommonsenseQA/ARC accuracy.  The
``derive-neutral`` subcommand reconstructs those two accuracy rows from the
authenticated generalization records.  The ``generate`` subcommand then
creates Table D2 using only the authenticated report and that derived receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT = "pruning_bonham_20260918"
MODELS = ("llama31_8b", "qwen25_7b", "gemma4_12b")
MODEL_NAMES = {
    "llama31_8b": r"\texttt{Llama-3.1-8B-Instruct}",
    "qwen25_7b": r"\texttt{Qwen2.5-7B-Instruct}",
    "gemma4_12b": r"\texttt{Gemma-4-12B-Instruct}",
}
PANEL_NAMES = {"llama31_8b": "A", "qwen25_7b": "B", "gemma4_12b": "C"}
STATES = (
    "unpruned",
    "n1_mechanism",
    "random_n1",
    "weak_prompt",
    "strong_prompt",
    "prompt_only_meandiff",
)
DATASET_NAMES = {
    "commonsense_qa": "CommonsenseQA",
    "arc_challenge": "ARC Challenge",
    "openbookqa": "OpenBookQA",
}
CONSTRUCTION_DATASETS = ("commonsense_qa", "arc_challenge")
REQUIRED_REPORT_FILES = (
    "generalization_cells.csv",
    "stress_tests.csv",
    "general_capabilities.csv",
)


class ProjectionError(RuntimeError):
    """Raised when an authenticated input or exact row contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_report(report_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    receipt_path = report_dir / "COMPLETE.json"
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "complete"
        or receipt.get("experiment") != EXPERIMENT
        or receipt.get("scope") != "complete_campaign"
        or set(receipt.get("model_keys", ())) != set(MODELS)
    ):
        raise ProjectionError(f"Not a complete three-model Bonham report: {receipt_path}")
    hashes = dict(receipt.get("csv_files", {}))
    for filename in REQUIRED_REPORT_FILES:
        path = report_dir / filename
        if not path.is_file() or hashes.get(filename) != sha256_file(path):
            raise ProjectionError(f"Report hash mismatch: {path}")
    return receipt, {name: hashes[name] for name in REQUIRED_REPORT_FILES}


def canonical_records(family_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Load authenticated records and reject conflicting duplicate example IDs."""

    by_example: dict[str, dict[str, Any]] = {}
    shard_count = 0
    for complete_path in sorted(family_root.glob("*/COMPLETE")):
        directory = complete_path.parent
        receipt = read_json(complete_path)
        records_path = directory / "records.jsonl"
        expected = dict(receipt.get("file_sha256", {})).get("records.jsonl")
        if not expected or expected != sha256_file(records_path):
            raise ProjectionError(f"Authenticated record file changed: {records_path}")
        shard_count += 1
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                example_id = str(record["example_id"])
                previous = by_example.get(example_id)
                if previous is not None and previous != record:
                    raise ProjectionError(
                        f"Conflicting duplicate record {example_id} under {family_root}"
                    )
                by_example[example_id] = record
    if not by_example:
        raise ProjectionError(f"No authenticated records under {family_root}")
    return list(by_example.values()), shard_count


def derive_neutral(args: argparse.Namespace) -> None:
    root = args.result_root.resolve()
    report_path = root / "reports" / "COMPLETE.json"
    audit_path = root / "audit" / "COMPLETE.json"
    report_receipt = read_json(report_path)
    audit_receipt = read_json(audit_path)
    report_hash = sha256_file(report_path)
    if (
        report_receipt.get("status") != "complete"
        or report_receipt.get("experiment") != EXPERIMENT
        or report_receipt.get("scope") != "complete_campaign"
    ):
        raise ProjectionError(f"Incomplete Bonham report: {report_path}")
    if (
        audit_receipt.get("status") != "complete"
        or audit_receipt.get("experiment") != EXPERIMENT
        or audit_receipt.get("report_complete_sha256") != report_hash
    ):
        raise ProjectionError(f"Audit does not authenticate the report: {audit_path}")

    rows = []
    total_shards = 0
    for model in MODELS:
        for state in STATES:
            family_root = (
                root / "evaluations" / "results" / model / state / "generalization"
            )
            records, shard_count = canonical_records(family_root)
            total_shards += shard_count
            neutral: dict[tuple[str, str], bool] = {}
            for record in records:
                metadata = dict(record.get("task_metadata", {}))
                if metadata.get("bias_type") != "neutral":
                    continue
                dataset = str(record["dataset_id"])
                question_id = str(metadata["question_id"])
                probabilities = dict(record.get("choice_probabilities", {}))
                if not probabilities:
                    raise ProjectionError(
                        f"Neutral record lacks choice probabilities: {model}/{state}/"
                        f"{dataset}/{question_id}"
                    )
                prediction = max(probabilities, key=probabilities.get)
                correct = prediction == str(record["gold_choice"])
                key = (dataset, question_id)
                previous = neutral.get(key)
                if previous is not None and previous != correct:
                    raise ProjectionError(f"Conflicting neutral result: {model}/{state}/{key}")
                neutral[key] = correct
            for dataset in DATASET_NAMES:
                values = [value for (name, _), value in neutral.items() if name == dataset]
                if len(values) != 500:
                    raise ProjectionError(
                        f"Expected 500 neutral questions, got {len(values)}: "
                        f"{model}/{state}/{dataset}"
                    )
                rows.append(
                    {
                        "model_key": model,
                        "state_id": state,
                        "dataset_id": dataset,
                        "metric": "argmax_choice_probability_accuracy",
                        "denominator": len(values),
                        "value": sum(values) / len(values),
                    }
                )

    payload = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "aggregation": "argmax(choice_probabilities) == gold_choice",
        "authenticated_shard_count": total_shards,
        "source_report_complete_sha256": report_hash,
        "source_audit_complete_sha256": sha256_file(audit_path),
        "rows": rows,
    }
    write_json(args.output.resolve(), payload)


def exact_row(rows: Iterable[Mapping[str, str]], **filters: str) -> Mapping[str, str]:
    selected = [
        row
        for row in rows
        if all(str(row.get(field)) == str(value) for field, value in filters.items())
    ]
    if len(selected) != 1:
        raise ProjectionError(f"Expected one report row, found {len(selected)}: {filters}")
    return selected[0]


def weighted_mean(rows: Sequence[Mapping[str, str]], value_field: str, n_field: str) -> float:
    weights = [int(row[n_field]) for row in rows]
    if not weights or any(weight <= 0 for weight in weights):
        raise ProjectionError("Aggregate has missing or nonpositive denominators")
    return sum(float(row[value_field]) * weight for row, weight in zip(rows, weights)) / sum(
        weights
    )


def generalization_value(
    rows: Sequence[Mapping[str, str]],
    *,
    model: str,
    state: str,
    datasets: Sequence[str],
    prompt_regime: str,
    bias_type: str,
    turn_format: str,
    metric: str,
) -> tuple[float, int]:
    selected = [
        exact_row(
            rows,
            model_key=model,
            state_id=state,
            dataset_id=dataset,
            prompt_regime=prompt_regime,
            bias_type=bias_type,
            turn_format=turn_format,
            metric=metric,
        )
        for dataset in datasets
    ]
    return weighted_mean(selected, "mean", "n_questions"), sum(
        int(row["n_questions"]) for row in selected
    )


def stress_value(
    rows: Sequence[Mapping[str, str]],
    *,
    model: str,
    state: str,
    datasets: Sequence[str],
) -> tuple[float, int]:
    selected = [
        exact_row(
            rows,
            model_key=model,
            state_id=state,
            dataset_id=dataset,
            prompt_regime="reasoning_backed_pushback",
            bias_type="incorrect_suggestion",
            turn_format="multi_turn",
            template_family="generic_justification_pressure",
            metric="adoption_or_rejection",
        )
        for dataset in datasets
    ]
    return weighted_mean(selected, "mean", "n_questions"), sum(
        int(row["n_questions"]) for row in selected
    )


def neutral_value(
    rows: Sequence[Mapping[str, Any]], *, model: str, state: str, dataset: str
) -> tuple[float, int]:
    selected = [
        row
        for row in rows
        if row.get("model_key") == model
        and row.get("state_id") == state
        and row.get("dataset_id") == dataset
        and row.get("metric") == "argmax_choice_probability_accuracy"
    ]
    if len(selected) != 1:
        raise ProjectionError(
            f"Expected one neutral-accuracy row, found {len(selected)}: "
            f"{model}/{state}/{dataset}"
        )
    return float(selected[0]["value"]), int(selected[0]["denominator"])


def capability_value(
    rows: Sequence[Mapping[str, str]], *, model: str, state: str, benchmark: str
) -> tuple[float, int]:
    row = exact_row(
        rows,
        model_key=model,
        state_id=state,
        benchmark=benchmark,
        metric="accuracy",
    )
    return float(row["value"]), int(row["denominator"])


def format_estimates(estimates: Sequence[float]) -> str:
    values = [f"{100.0 * value:.1f}" for value in estimates]
    return (
        f"& {values[0]} & \\ourcell{{{values[1]}}} & {values[2]} "
        f"& {values[3]} & {values[4]} & {values[5]} \\\\"
    )


def projection_rows(
    general: Sequence[Mapping[str, str]],
    stress: Sequence[Mapping[str, str]],
    capabilities: Sequence[Mapping[str, str]],
    neutral: Sequence[Mapping[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "category": "wrong_adoption",
            "label": "Held-out items",
            "description": "Held-out CommonsenseQA and ARC questions",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": CONSTRUCTION_DATASETS,
            "prompt_regime": "seen",
            "bias_type": "incorrect_suggestion",
            "turn_format": "single_turn",
            "metric": "adoption_or_rejection",
        },
        {
            "category": "wrong_adoption",
            "label": "Unseen phrasing",
            "description": "Close paraphrases on held-out CommonsenseQA and ARC questions",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": CONSTRUCTION_DATASETS,
            "prompt_regime": "close_paraphrase",
            "bias_type": "incorrect_suggestion",
            "turn_format": "single_turn",
            "metric": "adoption_or_rejection",
        },
        {
            "category": "wrong_adoption",
            "label": "Reason-backed disagreement",
            "description": "Generic conflicting rationale in a follow-up turn",
            "source": "stress_tests.csv",
            "kind": "stress",
            "datasets": CONSTRUCTION_DATASETS,
        },
        {
            "category": "wrong_adoption",
            "label": "Multi-turn pressure",
            "description": "Seen suggestion wording in a follow-up turn",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": CONSTRUCTION_DATASETS,
            "prompt_regime": "seen",
            "bias_type": "incorrect_suggestion",
            "turn_format": "multi_turn",
            "metric": "adoption_or_rejection",
        },
        {
            "category": "wrong_adoption",
            "label": "OpenBookQA",
            "description": "Entirely held-out QA dataset",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": ("openbookqa",),
            "prompt_regime": "seen",
            "bias_type": "incorrect_suggestion",
            "turn_format": "single_turn",
            "metric": "adoption_or_rejection",
        },
        {
            "category": "wrong_adoption",
            "label": "OpenBookQA + unseen phrasing",
            "description": "Dataset and close-paraphrase transfer",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": ("openbookqa",),
            "prompt_regime": "close_paraphrase",
            "bias_type": "incorrect_suggestion",
            "turn_format": "single_turn",
            "metric": "adoption_or_rejection",
        },
        {
            "category": "wrong_adoption",
            "label": "OpenBookQA + reasoning",
            "description": "Dataset and rationale-framed follow-up transfer",
            "source": "stress_tests.csv",
            "kind": "stress",
            "datasets": ("openbookqa",),
        },
        {
            "category": "wrong_adoption",
            "label": "OpenBookQA + multi-turn",
            "description": "Dataset and conversational transfer",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": ("openbookqa",),
            "prompt_regime": "seen",
            "bias_type": "incorrect_suggestion",
            "turn_format": "multi_turn",
            "metric": "adoption_or_rejection",
        },
        {
            "category": "doubt",
            "label": "Unsupported doubt: valid flip",
            "description": "Parser-valid correct-to-wrong change",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": CONSTRUCTION_DATASETS,
            "prompt_regime": "seen",
            "bias_type": "doubt_correct",
            "turn_format": "single_turn",
            "metric": "correct_to_wrong_flip",
        },
        {
            "category": "doubt",
            "label": "Unsupported doubt: invalid",
            "description": "Malformed, refused, or out-of-range response",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": CONSTRUCTION_DATASETS,
            "prompt_regime": "seen",
            "bias_type": "doubt_correct",
            "turn_format": "single_turn",
            "metric": "invalid",
        },
        {
            "category": "doubt",
            "label": "OpenBookQA + doubt: valid flip",
            "description": "Valid correct-to-wrong transfer to the held-out dataset",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": ("openbookqa",),
            "prompt_regime": "seen",
            "bias_type": "doubt_correct",
            "turn_format": "single_turn",
            "metric": "correct_to_wrong_flip",
        },
        {
            "category": "doubt",
            "label": "OpenBookQA + doubt: invalid",
            "description": "Invalid-response transfer to the held-out dataset",
            "source": "generalization_cells.csv",
            "kind": "generalization",
            "datasets": ("openbookqa",),
            "prompt_regime": "seen",
            "bias_type": "doubt_correct",
            "turn_format": "single_turn",
            "metric": "invalid",
        },
    ]
    for dataset in ("commonsense_qa", "arc_challenge"):
        specs.append(
            {
                "category": "accuracy",
                "label": DATASET_NAMES[dataset],
                "description": (
                    "Commonsense multiple-choice QA"
                    if dataset == "commonsense_qa"
                    else "Grade-school science questions"
                ),
                "source": "neutral_factual_accuracy.json",
                "kind": "neutral",
                "dataset": dataset,
            }
        )
    specs.append(
        {
            "category": "accuracy",
            "label": "OpenBookQA",
            "description": "Open-book science questions",
            "source": "general_capabilities.csv",
            "kind": "capability",
            "benchmark": "OpenBookQA",
        }
    )

    output = []
    for spec in specs:
        estimates = []
        denominators = []
        for state in STATES:
            if spec["kind"] == "generalization":
                value, denominator = generalization_value(
                    general,
                    model=model,
                    state=state,
                    datasets=spec["datasets"],
                    prompt_regime=spec["prompt_regime"],
                    bias_type=spec["bias_type"],
                    turn_format=spec["turn_format"],
                    metric=spec["metric"],
                )
            elif spec["kind"] == "stress":
                value, denominator = stress_value(
                    stress,
                    model=model,
                    state=state,
                    datasets=spec["datasets"],
                )
            elif spec["kind"] == "neutral":
                value, denominator = neutral_value(
                    neutral, model=model, state=state, dataset=spec["dataset"]
                )
            else:
                value, denominator = capability_value(
                    capabilities,
                    model=model,
                    state=state,
                    benchmark=spec["benchmark"],
                )
            estimates.append(value)
            denominators.append(denominator)
        if len(set(denominators)) != 1:
            raise ProjectionError(f"State denominators differ for {model}/{spec['label']}")
        output.append(
            {
                **spec,
                "estimates": dict(zip(STATES, estimates)),
                "denominator_per_state": denominators[0],
            }
        )
    return output


def render_panel(model: str, rows: Sequence[Mapping[str, Any]], continued: bool) -> str:
    lines = [r"\begin{landscape}", r"\begin{table}[p]"]
    if continued:
        lines.append(r"\ContinuedFloat")
    lines.extend(
        [
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\renewcommand{\arraystretch}{1.20}",
        ]
    )
    if continued:
        lines.append(
            r"\caption[]{Factual transfer and retention across interventions "
            r"(continued).}"
        )
    else:
        lines.extend(
            [
                r"\caption{Factual transfer and retention across interventions. "
                r"Behavioral rows report percentages over 500 questions per held-out "
                r"dataset and state; construction-domain rows pool the 500-question "
                r"CommonsenseQA and ARC Challenge evaluations ($n=1{,}000$). "
                r"Wrong-option adoption counts parser-valid selection of the user's "
                r"incorrect option; all other outputs, including invalid responses, "
                r"count as non-adoptions. Doubt flips count parser-valid correct-to-wrong "
                r"changes and are accompanied by invalid-response rates. Factual "
                r"accuracy uses the argmax of the neutral-prompt choice probabilities. "
                r"All values are generated from the authenticated "
                r"\texttt{pruning\_bonham\_20260918} report.}",
                r"\label{tab:pruning-factual-generalization}",
            ]
        )
    lines.extend(
        [
            rf"\modelpanel{{{PANEL_NAMES[model]}}}{{{MODEL_NAMES[model]}}}",
            r"\begin{adjustbox}{max width=\linewidth,max totalheight=0.80\textheight,keepaspectratio}",
            r"\begin{tabular}{@{}G{0.13\linewidth}L{0.27\linewidth}*{6}{C{0.081\linewidth}}@{}}",
            r"\pruningresultheader{Evaluation}",
        ]
    )
    categories = (("wrong_adoption", 8, r"Wrong-option adoption $\downarrow$"),
                  ("doubt", 4, r"Doubt outcomes $\downarrow$"),
                  ("accuracy", 3, r"Factual accuracy $\uparrow$"))
    for category_index, (category, count, category_label) in enumerate(categories):
        selected = [row for row in rows if row["category"] == category]
        if len(selected) != count:
            raise ProjectionError(f"Unexpected row count for {model}/{category}")
        for index, row in enumerate(selected):
            prefix = "& "
            if index == count - 1:
                prefix = rf"\multirow{{-{count}}}{{=}}{{\catcell{{{category_label}}}}}" + "\n& "
            lines.extend(
                [
                    rf"{prefix}\taskcell{{{row['label']}}}{{{row['description']}}}",
                    format_estimates([row["estimates"][state] for state in STATES]),
                ]
            )
            if index != count - 1:
                lines.append(r"\cmidrule(lr){2-8}")
        lines.append(r"\midrule" if category_index != len(categories) - 1 else r"\bottomrule")
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
            r"\end{landscape}",
        ]
    )
    return "\n".join(lines)


def generate(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.resolve()
    _, report_hashes = validate_report(report_dir)
    report_receipt_hash = sha256_file(report_dir / "COMPLETE.json")
    neutral_payload = read_json(args.neutral_accuracy.resolve())
    if (
        neutral_payload.get("status") != "complete"
        or neutral_payload.get("experiment") != EXPERIMENT
        or neutral_payload.get("source_report_complete_sha256") != report_receipt_hash
    ):
        raise ProjectionError("Neutral-accuracy receipt does not match the report")

    general = read_csv(report_dir / "generalization_cells.csv")
    stress = read_csv(report_dir / "stress_tests.csv")
    capabilities = read_csv(report_dir / "general_capabilities.csv")
    neutral = list(neutral_payload["rows"])
    models = {
        model: projection_rows(general, stress, capabilities, neutral, model)
        for model in MODELS
    }
    projection = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "method_state_order": list(STATES),
        "report_complete_sha256": report_receipt_hash,
        "report_csv_sha256": report_hashes,
        "neutral_accuracy_sha256": sha256_file(args.neutral_accuracy.resolve()),
        "models": models,
    }
    write_json(args.projection_output.resolve(), projection)
    table = "\n\n".join(
        render_panel(model, models[model], continued=index > 0)
        for index, model in enumerate(MODELS)
    ) + "\n"
    args.table_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.table_output.resolve().write_text(table, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    derive = subparsers.add_parser("derive-neutral")
    derive.add_argument("--result-root", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    derive.set_defaults(func=derive_neutral)

    build = subparsers.add_parser("generate")
    build.add_argument("--report-dir", type=Path, required=True)
    build.add_argument("--neutral-accuracy", type=Path, required=True)
    build.add_argument("--table-output", type=Path, required=True)
    build.add_argument("--projection-output", type=Path, required=True)
    build.set_defaults(func=generate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "sycophancy_bias_probe"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "analysis_exports" / "20260324_careful_analysis_package"
PAIR_CONDITIONS = ("neutral", "incorrect_suggestion")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_choice(value: Any) -> str:
    return normalize_text(value).upper()


def to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_run_dirs(results_root: Path) -> Tuple[List[Path], List[Dict[str, str]]]:
    run_dirs: List[Path] = []
    skipped: List[Dict[str, str]] = []
    for manifest_path in sorted(results_root.glob("**/probes/chosen_probe/manifest.json")):
        run_dir = manifest_path.parents[2].resolve()
        run_name = run_dir.name
        path_text = str(run_dir)
        if not run_name.startswith("full_"):
            skipped.append({"run_dir": path_text, "reason": "non_full_run_name"})
            continue
        if any(part == "_backup" for part in run_dir.parts) or "backup" in run_name.lower():
            skipped.append({"run_dir": path_text, "reason": "backup_run"})
            continue
        if "smoke" in run_name.lower():
            skipped.append({"run_dir": path_text, "reason": "smoke_run"})
            continue
        required_paths = [
            run_dir / "run_config.json",
            run_dir / "logs" / "sampling_records.jsonl",
            run_dir / "probes" / "probe_scores_by_prompt.csv",
        ]
        missing_required = [str(path.relative_to(run_dir)) for path in required_paths if not path.exists()]
        if missing_required:
            skipped.append(
                {
                    "run_dir": path_text,
                    "reason": "missing_required_artifacts",
                    "missing": ",".join(missing_required),
                }
            )
            continue
        run_dirs.append(run_dir)
    return run_dirs, skipped


def resolve_run_dir(path_or_name: str) -> Path:
    candidate = Path(path_or_name)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.exists() and (candidate / "run_config.json").exists():
        return candidate

    run_name = candidate.name
    matches = sorted(path.parent.resolve() for path in RESULTS_ROOT.glob(f"**/{run_name}/run_config.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Run name {run_name!r} matched multiple directories: " + ", ".join(str(match) for match in matches)
        )
    raise FileNotFoundError(f"Could not resolve run dir {path_or_name!r}.")


def resolve_optional_neutral_backfill_path(run_dir: Path) -> Optional[Path]:
    preferred = run_dir / "probes" / "backfills" / "probe_no_bias_all_templates" / "probe_scores_by_prompt.csv"
    if preferred.exists():
        return preferred.resolve()
    matches = sorted(
        RESULTS_ROOT.glob(f"**/{run_dir.name}/probes/backfills/probe_no_bias_all_templates/probe_scores_by_prompt.csv")
    )
    if matches:
        return matches[0].resolve()
    return None


def resolve_optional_neutral_backfill_metadata_path(run_dir: Path) -> Optional[Path]:
    preferred = run_dir / "probes" / "backfills" / "probe_no_bias_all_templates" / "metadata.json"
    if preferred.exists():
        return preferred.resolve()
    matches = sorted(
        RESULTS_ROOT.glob(f"**/{run_dir.name}/probes/backfills/probe_no_bias_all_templates/metadata.json")
    )
    if matches:
        return matches[0].resolve()
    return None


def ordered_union(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        cleaned = normalize_choice(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def option_text_map(record: Mapping[str, Any]) -> Dict[str, str]:
    labels = ordered_union(str(record.get("letters", "") or ""))
    answers = record.get("answers_list")
    if isinstance(answers, list) and labels and len(answers) == len(labels):
        return {
            label: normalize_text(answer)
            for label, answer in zip(labels, answers)
        }

    answer_options = normalize_text(record.get("answer_options"))
    mapping: Dict[str, str] = {}
    if answer_options:
        for raw_line in answer_options.splitlines():
            line = raw_line.strip()
            if not line.startswith("(") or ")" not in line:
                continue
            label = normalize_choice(line[1: line.index(")")])
            text = normalize_text(line[line.index(")") + 1 :])
            if label and text:
                mapping[label] = text
    if mapping:
        return mapping

    if isinstance(answers, list) and answers:
        fallback_labels = labels or [str(idx + 1) for idx in range(len(answers))]
        return {
            normalize_choice(label): normalize_text(answer)
            for label, answer in zip(fallback_labels, answers)
        }
    return {}


def choice_probability_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    raw_payload = record.get("choice_probabilities", {})
    if not isinstance(raw_payload, dict):
        return {
            "raw_by_choice": {},
            "finite_by_choice": {},
            "non_finite_by_choice": {},
            "prompt_has_any_non_finite_probability": False,
            "model_argmax_choice": "",
        }

    raw_by_choice: Dict[str, Any] = {}
    finite_by_choice: Dict[str, float] = {}
    non_finite_by_choice: Dict[str, bool] = {}
    for raw_choice, raw_value in raw_payload.items():
        choice = normalize_choice(raw_choice)
        if not choice:
            continue
        raw_by_choice[choice] = raw_value
        finite_value = to_float_or_nan(raw_value)
        is_non_finite = not math.isfinite(finite_value)
        non_finite_by_choice[choice] = is_non_finite
        if not is_non_finite:
            finite_by_choice[choice] = finite_value

    allowed_order = {choice: idx for idx, choice in enumerate(ordered_union(str(record.get("letters", "") or "")))}
    ranked = sorted(
        finite_by_choice.items(),
        key=lambda item: (
            -item[1],
            allowed_order.get(item[0], 10_000),
            item[0],
        ),
    )
    model_argmax_choice = ranked[0][0] if ranked else ""
    return {
        "raw_by_choice": raw_by_choice,
        "finite_by_choice": finite_by_choice,
        "non_finite_by_choice": non_finite_by_choice,
        "prompt_has_any_non_finite_probability": any(non_finite_by_choice.values()),
        "model_argmax_choice": model_argmax_choice,
    }


def prompt_key(record_or_row: Mapping[str, Any]) -> Tuple[str, str, int, str]:
    return (
        normalize_text(record_or_row.get("split")),
        normalize_text(record_or_row.get("question_id")),
        int(record_or_row.get("draw_idx", 0) or 0),
        normalize_text(record_or_row.get("template_type")),
    )


def prompt_score_lookup(df: pd.DataFrame, *, probe_name_filter: Optional[str] = None) -> Dict[Tuple[str, str, int, str], Dict[str, Any]]:
    if df.empty:
        return {}

    working = df.copy()
    if probe_name_filter is not None and "probe_name" in working.columns:
        working = working.loc[working["probe_name"].astype(str).eq(probe_name_filter)].copy()

    lookup: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    for _, row in working.iterrows():
        row_dict = row.to_dict()
        lookup[prompt_key(row_dict)] = row_dict
    return lookup


def choice_labels_for_row(
    record: Mapping[str, Any],
    prompt_row: Optional[Mapping[str, Any]],
) -> List[str]:
    labels: List[str] = []
    labels.extend(ordered_union(str(record.get("letters", "") or "")))
    labels.extend(ordered_union(option_text_map(record).keys()))
    probability_payload = choice_probability_payload(record)
    labels.extend(ordered_union(probability_payload["raw_by_choice"].keys()))
    if prompt_row is not None:
        labels.extend(
            ordered_union(
                column.split("score_", 1)[1]
                for column in prompt_row.keys()
                if str(column).startswith("score_") and not pd.isna(prompt_row.get(column))
            )
        )
    return ordered_union(labels)


def probe_score_for_choice(prompt_row: Optional[Mapping[str, Any]], choice: str) -> float:
    if prompt_row is None:
        return float("nan")
    return to_float_or_nan(prompt_row.get(f"score_{choice}", float("nan")))


def record_meta(record: Mapping[str, Any]) -> Dict[str, Any]:
    option_map = option_text_map(record)
    probability_info = choice_probability_payload(record)
    return {
        "question_text": normalize_text(record.get("question") or record.get("question_text")),
        "answer_options_text": normalize_text(record.get("answer_options")),
        "option_text_by_choice": option_map,
        "correct_answer_text": normalize_text(record.get("correct_answer")),
        "suggested_wrong_answer_text": normalize_text(record.get("incorrect_answer")),
        "model_selected_choice": normalize_choice(record.get("response_raw") or record.get("response")),
        "model_argmax_choice": probability_info["model_argmax_choice"],
        "choice_probabilities_raw": probability_info["raw_by_choice"],
        "choice_probabilities_finite": probability_info["finite_by_choice"],
        "non_finite_by_choice": probability_info["non_finite_by_choice"],
        "prompt_has_any_non_finite_probability": probability_info["prompt_has_any_non_finite_probability"],
    }


def pair_id(run_name: str, split: str, question_id: str, draw_idx: int) -> str:
    return f"{run_name}::{split}::{question_id}::{draw_idx}"


def build_current_choice_rows(
    *,
    run_name: str,
    run_dir: Path,
    model_name: str,
    dataset_name: str,
    pair_key_tuple: Tuple[str, str, int],
    neutral_record: Mapping[str, Any],
    bias_record: Mapping[str, Any],
    matched_prompt_rows: Mapping[str, Dict[str, Any]],
    chosen_probe_meta: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    split, question_id, draw_idx = pair_key_tuple
    rows: List[Dict[str, Any]] = []
    for condition, record, probe_name in (
        ("neutral", neutral_record, "probe_no_bias"),
        ("incorrect_suggestion", bias_record, "probe_bias_incorrect_suggestion"),
    ):
        prompt_row = matched_prompt_rows.get((split, question_id, draw_idx, condition))
        metadata = record_meta(record)
        selected_probe_meta = chosen_probe_meta.get(probe_name, {})
        for choice in choice_labels_for_row(record, prompt_row):
            model_prob = to_float_or_nan(metadata["choice_probabilities_raw"].get(choice, float("nan")))
            rows.append(
                {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "model": model_name,
                    "dataset": dataset_name,
                    "split": split,
                    "question_id": question_id,
                    "draw_idx": int(draw_idx),
                    "pair_id": pair_id(run_name, split, question_id, draw_idx),
                    "condition": condition,
                    "choice": choice,
                    "choice_text": metadata["option_text_by_choice"].get(choice, ""),
                    "correct_letter": normalize_choice(record.get("correct_letter")),
                    "suggested_wrong_letter": normalize_choice(record.get("incorrect_letter")),
                    "correct_answer_text": metadata["correct_answer_text"],
                    "suggested_wrong_answer_text": metadata["suggested_wrong_answer_text"],
                    "is_correct": choice == normalize_choice(record.get("correct_letter")),
                    "is_suggested_wrong": choice == normalize_choice(record.get("incorrect_letter")),
                    "model_prob": model_prob,
                    "non_finite_probability": bool(metadata["non_finite_by_choice"].get(choice, False)),
                    "prompt_has_any_non_finite_probability": bool(metadata["prompt_has_any_non_finite_probability"]),
                    "model_selected_choice": metadata["model_selected_choice"],
                    "model_argmax_choice": metadata["model_argmax_choice"],
                    "probe_name": probe_name,
                    "probe_training_template_type": "neutral" if probe_name == "probe_no_bias" else "incorrect_suggestion",
                    "probe_evaluated_on_template_type": condition,
                    "probe_matches_evaluated_template": probe_name == "probe_no_bias" and condition == "neutral"
                    or probe_name == "probe_bias_incorrect_suggestion" and condition == "incorrect_suggestion",
                    "probe_score": probe_score_for_choice(prompt_row, choice),
                    "probe_argmax_choice": normalize_choice((prompt_row or {}).get("probe_argmax_choice")),
                    "probe_argmax_score": to_float_or_nan((prompt_row or {}).get("probe_argmax_score", float("nan"))),
                    "probe_prefers_correct": (prompt_row or {}).get("probe_prefers_correct"),
                    "probe_prefers_selected": (prompt_row or {}).get("probe_prefers_selected"),
                    "probe_row_missing": prompt_row is None,
                    "probe_selected_layer": selected_probe_meta.get("chosen_layer"),
                    "probe_selection_val_auc": selected_probe_meta.get("best_dev_auc"),
                }
            )
    return rows


def build_neutral_cross_choice_rows(
    *,
    run_name: str,
    run_dir: Path,
    model_name: str,
    dataset_name: str,
    pair_key_tuple: Tuple[str, str, int],
    neutral_record: Mapping[str, Any],
    bias_record: Mapping[str, Any],
    neutral_backfill_rows: Mapping[Tuple[str, str, int, str], Dict[str, Any]],
    chosen_probe_meta: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    split, question_id, draw_idx = pair_key_tuple
    rows: List[Dict[str, Any]] = []
    neutral_probe_meta = chosen_probe_meta.get("probe_no_bias", {})
    for condition, record in (
        ("neutral", neutral_record),
        ("incorrect_suggestion", bias_record),
    ):
        prompt_row = neutral_backfill_rows.get((split, question_id, draw_idx, condition))
        metadata = record_meta(record)
        for choice in choice_labels_for_row(record, prompt_row):
            model_prob = to_float_or_nan(metadata["choice_probabilities_raw"].get(choice, float("nan")))
            rows.append(
                {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "model": model_name,
                    "dataset": dataset_name,
                    "split": split,
                    "question_id": question_id,
                    "draw_idx": int(draw_idx),
                    "pair_id": pair_id(run_name, split, question_id, draw_idx),
                    "condition": condition,
                    "choice": choice,
                    "choice_text": metadata["option_text_by_choice"].get(choice, ""),
                    "correct_letter": normalize_choice(record.get("correct_letter")),
                    "suggested_wrong_letter": normalize_choice(record.get("incorrect_letter")),
                    "correct_answer_text": metadata["correct_answer_text"],
                    "suggested_wrong_answer_text": metadata["suggested_wrong_answer_text"],
                    "is_correct": choice == normalize_choice(record.get("correct_letter")),
                    "is_suggested_wrong": choice == normalize_choice(record.get("incorrect_letter")),
                    "model_prob": model_prob,
                    "non_finite_probability": bool(metadata["non_finite_by_choice"].get(choice, False)),
                    "prompt_has_any_non_finite_probability": bool(metadata["prompt_has_any_non_finite_probability"]),
                    "model_selected_choice": metadata["model_selected_choice"],
                    "model_argmax_choice": metadata["model_argmax_choice"],
                    "probe_name": "probe_no_bias",
                    "probe_training_template_type": "neutral",
                    "probe_evaluated_on_template_type": condition,
                    "probe_matches_evaluated_template": condition == "neutral",
                    "probe_score": probe_score_for_choice(prompt_row, choice),
                    "probe_argmax_choice": normalize_choice((prompt_row or {}).get("probe_argmax_choice")),
                    "probe_argmax_score": to_float_or_nan((prompt_row or {}).get("probe_argmax_score", float("nan"))),
                    "probe_prefers_correct": (prompt_row or {}).get("probe_prefers_correct"),
                    "probe_prefers_selected": (prompt_row or {}).get("probe_prefers_selected"),
                    "probe_row_missing": prompt_row is None,
                    "probe_selected_layer": neutral_probe_meta.get("chosen_layer"),
                    "probe_selection_val_auc": neutral_probe_meta.get("best_dev_auc"),
                }
            )
    return rows


def build_pairing_audit_rows(
    *,
    run_name: str,
    run_dir: Path,
    model_name: str,
    dataset_name: str,
    union_pair_keys: Sequence[Tuple[str, str, int]],
    record_lookup: Mapping[Tuple[str, str, int, str], Dict[str, Any]],
    cross_backfill_available: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, question_id, draw_idx in union_pair_keys:
        neutral_record = record_lookup.get((split, question_id, draw_idx, "neutral"))
        bias_record = record_lookup.get((split, question_id, draw_idx, "incorrect_suggestion"))
        neutral_meta = record_meta(neutral_record or {})
        bias_meta = record_meta(bias_record or {})
        rows.append(
            {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "model": model_name,
                "dataset": dataset_name,
                "split": split,
                "question_id": question_id,
                "draw_idx": int(draw_idx),
                "pair_id": pair_id(run_name, split, question_id, draw_idx),
                "has_neutral": neutral_record is not None,
                "has_incorrect_suggestion": bias_record is not None,
                "is_pairable": neutral_record is not None and bias_record is not None,
                "same_question_id": neutral_record is not None and bias_record is not None,
                "same_correct_letter": normalize_choice((neutral_record or {}).get("correct_letter"))
                == normalize_choice((bias_record or {}).get("correct_letter")),
                "same_suggested_wrong_letter": normalize_choice((neutral_record or {}).get("incorrect_letter"))
                == normalize_choice((bias_record or {}).get("incorrect_letter")),
                "same_question_text": neutral_meta["question_text"] == bias_meta["question_text"],
                "same_answer_options_text": neutral_meta["answer_options_text"] == bias_meta["answer_options_text"],
                "neutral_prompt_id": normalize_text((neutral_record or {}).get("prompt_id")),
                "incorrect_suggestion_prompt_id": normalize_text((bias_record or {}).get("prompt_id")),
                "neutral_model_selected_choice": neutral_meta["model_selected_choice"],
                "incorrect_suggestion_model_selected_choice": bias_meta["model_selected_choice"],
                "neutral_model_argmax_choice": neutral_meta["model_argmax_choice"],
                "incorrect_suggestion_model_argmax_choice": bias_meta["model_argmax_choice"],
                "neutral_prompt_has_any_non_finite_probability": neutral_meta["prompt_has_any_non_finite_probability"],
                "incorrect_suggestion_prompt_has_any_non_finite_probability": bias_meta[
                    "prompt_has_any_non_finite_probability"
                ],
                "neutral_probe_cross_condition_available": cross_backfill_available,
            }
        )
    return rows


def build_selected_layer_rows(
    *,
    run_name: str,
    run_dir: Path,
    model_name: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    manifest_path = run_dir / "probes" / "chosen_probe" / "manifest.json"
    if not manifest_path.exists():
        return []

    manifest = load_json(manifest_path)
    rows: List[Dict[str, Any]] = []
    for probe_name, payload in sorted(manifest.get("probes", {}).items()):
        row: Dict[str, Any] = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "model": model_name,
            "dataset": dataset_name,
            "probe_name": probe_name,
            "template_type": "neutral" if probe_name == "probe_no_bias" else probe_name.replace("probe_bias_", ""),
            "chosen_layer": payload.get("chosen_layer"),
            "best_dev_auc": payload.get("best_dev_auc"),
            "probe_construction": payload.get("probe_construction"),
            "probe_example_weighting": payload.get("probe_example_weighting"),
        }
        metrics_path = run_dir / "probes" / "chosen_probe" / probe_name / "metrics.json"
        metadata_path = run_dir / "probes" / "chosen_probe" / probe_name / "metadata.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            for split_name in ("train", "val", "test"):
                split_payload = metrics.get("splits", {}).get(split_name, {})
                row[f"{split_name}_auc"] = split_payload.get("auc")
                row[f"{split_name}_accuracy"] = split_payload.get("accuracy")
                row[f"{split_name}_balanced_accuracy"] = split_payload.get("balanced_accuracy")
                row[f"{split_name}_true_label_accuracy"] = split_payload.get("true_label_accuracy")
                row[f"{split_name}_false_label_accuracy"] = split_payload.get("false_label_accuracy")
                row[f"{split_name}_n_total"] = split_payload.get("n_total")
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            row["selection_metric"] = metadata.get("selection", {}).get("selection_metric")
            row["selection_split"] = metadata.get("selection", {}).get("selection_split")
            row["token_pooling_rule"] = metadata.get("feature_source", {}).get("token_position")
            row["layer_metadata_value"] = metadata.get("layer")
        rows.append(row)
    return rows


def build_layerwise_rows(
    *,
    run_name: str,
    run_dir: Path,
    model_name: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    group_manifest_path = run_dir / "probes" / "all_probes" / "manifest.json"
    if not group_manifest_path.exists():
        return []

    group_manifest = load_json(group_manifest_path)
    rows: List[Dict[str, Any]] = []
    for probe_name in sorted(group_manifest.get("probes", {}).keys()):
        family_manifest_path = run_dir / "probes" / "all_probes" / probe_name / "manifest.json"
        if not family_manifest_path.exists():
            continue
        payload = load_json(family_manifest_path)
        best_layer = payload.get("best_layer")
        best_dev_auc = payload.get("best_dev_auc")
        template_type = payload.get("template_type") or (
            "neutral" if probe_name == "probe_no_bias" else probe_name.replace("probe_bias_", "")
        )
        for layer_str, layer_payload in sorted(payload.get("layers", {}).items(), key=lambda item: int(item[0])):
            metrics_path = run_dir / "probes" / "all_probes" / probe_name / f"layer_{int(layer_str):03d}" / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            for split_name in ("train", "val", "test"):
                split_payload = metrics.get("splits", {}).get(split_name, {})
                rows.append(
                    {
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "model": model_name,
                        "dataset": dataset_name,
                        "probe_name": probe_name,
                        "template_type": template_type,
                        "layer": int(layer_str),
                        "split": split_name,
                        "auc": split_payload.get("auc"),
                        "accuracy": split_payload.get("accuracy"),
                        "balanced_accuracy": split_payload.get("balanced_accuracy"),
                        "true_label_accuracy": split_payload.get("true_label_accuracy"),
                        "false_label_accuracy": split_payload.get("false_label_accuracy"),
                        "n_total": split_payload.get("n_total"),
                        "best_layer": best_layer,
                        "best_dev_auc": best_dev_auc,
                        "selection_val_auc": layer_payload.get("selection_val_auc"),
                        "input_dim": layer_payload.get("input_dim"),
                    }
                )
    return rows


@dataclass
class RunExportResult:
    inventory_row: Dict[str, Any]
    current_choice_rows: List[Dict[str, Any]]
    neutral_cross_rows: List[Dict[str, Any]]
    pairing_rows: List[Dict[str, Any]]
    selected_layer_rows: List[Dict[str, Any]]
    layerwise_rows: List[Dict[str, Any]]
    example_rows: List[Dict[str, Any]]


def export_run(run_dir: Path, *, example_items_per_run: int) -> RunExportResult:
    run_config = load_json(run_dir / "run_config.json")
    run_name = run_dir.name
    model_name = normalize_text(run_config.get("model"))
    dataset_name = normalize_text(run_config.get("dataset_name") or run_config.get("dataset"))

    records = load_jsonl(run_dir / "logs" / "sampling_records.jsonl")
    record_lookup: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {
        prompt_key(record): record for record in records
    }
    neutral_keys = {
        (split, question_id, draw_idx)
        for split, question_id, draw_idx, template_type in record_lookup
        if template_type == "neutral"
    }
    bias_keys = {
        (split, question_id, draw_idx)
        for split, question_id, draw_idx, template_type in record_lookup
        if template_type == "incorrect_suggestion"
    }
    paired_keys = sorted(neutral_keys & bias_keys)
    union_keys = sorted(neutral_keys | bias_keys)

    standard_probe_df = pd.read_csv(run_dir / "probes" / "probe_scores_by_prompt.csv")
    matched_probe_rows = {}
    matched_probe_rows.update(
        prompt_score_lookup(
            standard_probe_df.loc[
                standard_probe_df["template_type"].astype(str).eq("neutral")
                & standard_probe_df["probe_name"].astype(str).eq("probe_no_bias")
            ].copy()
        )
    )
    matched_probe_rows.update(
        prompt_score_lookup(
            standard_probe_df.loc[
                standard_probe_df["template_type"].astype(str).eq("incorrect_suggestion")
                & standard_probe_df["probe_name"].astype(str).eq("probe_bias_incorrect_suggestion")
            ].copy()
        )
    )

    chosen_manifest = load_json(run_dir / "probes" / "chosen_probe" / "manifest.json")
    chosen_probe_meta = chosen_manifest.get("probes", {})

    neutral_backfill_path = resolve_optional_neutral_backfill_path(run_dir)
    neutral_backfill_metadata_path = resolve_optional_neutral_backfill_metadata_path(run_dir)
    neutral_backfill_rows: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
    if neutral_backfill_path is not None and neutral_backfill_path.exists():
        neutral_backfill_df = pd.read_csv(neutral_backfill_path)
        neutral_backfill_df = neutral_backfill_df.loc[
            neutral_backfill_df["probe_name"].astype(str).eq("probe_no_bias")
            & neutral_backfill_df["template_type"].astype(str).isin(PAIR_CONDITIONS)
        ].copy()
        neutral_backfill_rows = prompt_score_lookup(neutral_backfill_df)

    current_choice_rows: List[Dict[str, Any]] = []
    neutral_cross_rows: List[Dict[str, Any]] = []
    for pair_key_tuple in paired_keys:
        split, question_id, draw_idx = pair_key_tuple
        neutral_record = record_lookup[(split, question_id, draw_idx, "neutral")]
        bias_record = record_lookup[(split, question_id, draw_idx, "incorrect_suggestion")]
        current_choice_rows.extend(
            build_current_choice_rows(
                run_name=run_name,
                run_dir=run_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                pair_key_tuple=pair_key_tuple,
                neutral_record=neutral_record,
                bias_record=bias_record,
                matched_prompt_rows=matched_probe_rows,
                chosen_probe_meta=chosen_probe_meta,
            )
        )
        if neutral_backfill_rows:
            neutral_cross_rows.extend(
                build_neutral_cross_choice_rows(
                    run_name=run_name,
                    run_dir=run_dir,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    pair_key_tuple=pair_key_tuple,
                    neutral_record=neutral_record,
                    bias_record=bias_record,
                    neutral_backfill_rows=neutral_backfill_rows,
                    chosen_probe_meta=chosen_probe_meta,
                )
            )

    pairing_rows = build_pairing_audit_rows(
        run_name=run_name,
        run_dir=run_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        union_pair_keys=union_keys,
        record_lookup=record_lookup,
        cross_backfill_available=bool(neutral_backfill_rows),
    )

    selected_layer_rows = build_selected_layer_rows(
        run_name=run_name,
        run_dir=run_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )
    layerwise_rows = build_layerwise_rows(
        run_name=run_name,
        run_dir=run_dir,
        model_name=model_name,
        dataset_name=dataset_name,
    )

    pairing_df = pd.DataFrame(pairing_rows)
    example_keys: List[Tuple[str, str, int]] = []
    if not pairing_df.empty:
        test_pairing_df = pairing_df.loc[
            pairing_df["split"].astype(str).eq("test") & pairing_df["is_pairable"].astype(bool)
        ].copy()
        available_keys = [
            (str(row["split"]), str(row["question_id"]), int(row["draw_idx"]))
            for _, row in test_pairing_df.iterrows()
        ]
        if available_keys:
            rng = random.Random(0)
            example_keys = sorted(rng.sample(available_keys, k=min(example_items_per_run, len(available_keys))))

    current_choice_df = pd.DataFrame(current_choice_rows)
    current_example_df = pd.DataFrame()
    if not current_choice_df.empty and example_keys:
        example_pair_ids = {pair_id(run_name, split, question_id, draw_idx) for split, question_id, draw_idx in example_keys}
        current_example_df = current_choice_df.loc[current_choice_df["pair_id"].isin(example_pair_ids)].copy()

    neutral_cross_df = pd.DataFrame(neutral_cross_rows)
    neutral_cross_example_df = pd.DataFrame()
    if not neutral_cross_df.empty and example_keys:
        example_pair_ids = {pair_id(run_name, split, question_id, draw_idx) for split, question_id, draw_idx in example_keys}
        neutral_cross_example_df = neutral_cross_df.loc[neutral_cross_df["pair_id"].isin(example_pair_ids)].copy()

    example_rows: List[Dict[str, Any]] = []
    if not current_example_df.empty:
        current_wide = current_example_df.pivot_table(
            index=[
                "run_name",
                "run_dir",
                "model",
                "dataset",
                "split",
                "question_id",
                "draw_idx",
                "pair_id",
                "choice",
                "choice_text",
                "correct_letter",
                "suggested_wrong_letter",
                "correct_answer_text",
                "suggested_wrong_answer_text",
            ],
            columns="condition",
            values=[
                "model_prob",
                "non_finite_probability",
                "prompt_has_any_non_finite_probability",
                "model_selected_choice",
                "model_argmax_choice",
                "probe_score",
                "probe_argmax_choice",
                "probe_selected_layer",
                "probe_selection_val_auc",
            ],
            aggfunc="first",
        ).reset_index()
        current_wide.columns = [
            "__".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in current_wide.columns
        ]

        if not neutral_cross_example_df.empty:
            neutral_cross_wide = neutral_cross_example_df.pivot_table(
                index=[
                    "pair_id",
                    "choice",
                ],
                columns="condition",
                values=["probe_score", "probe_argmax_choice"],
                aggfunc="first",
            ).reset_index()
            neutral_cross_wide.columns = [
                "__".join(str(part) for part in column if str(part))
                if isinstance(column, tuple)
                else str(column)
                for column in neutral_cross_wide.columns
            ]
            neutral_cross_wide = neutral_cross_wide.rename(
                columns={
                    "probe_score__neutral": "neutral_probe_score__neutral_probe_cross",
                    "probe_score__incorrect_suggestion": "incorrect_suggestion_probe_score__neutral_probe_cross",
                    "probe_argmax_choice__neutral": "neutral_probe_argmax_choice__neutral_probe_cross",
                    "probe_argmax_choice__incorrect_suggestion": "incorrect_suggestion_probe_argmax_choice__neutral_probe_cross",
                }
            )
            current_wide = current_wide.merge(
                neutral_cross_wide,
                on=["pair_id", "choice"],
                how="left",
            )

        for _, row in current_wide.iterrows():
            split = normalize_text(row.get("split"))
            question_id = normalize_text(row.get("question_id"))
            draw_idx = int(row.get("draw_idx", 0) or 0)
            neutral_record = record_lookup.get((split, question_id, draw_idx, "neutral"), {})
            bias_record = record_lookup.get((split, question_id, draw_idx, "incorrect_suggestion"), {})
            example_rows.append(
                {
                    **row.to_dict(),
                    "question_text": normalize_text(neutral_record.get("question") or neutral_record.get("question_text")),
                    "answer_options_text": normalize_text(neutral_record.get("answer_options")),
                    "neutral_prompt_text": normalize_text(neutral_record.get("prompt_text")),
                    "incorrect_suggestion_prompt_text": normalize_text(bias_record.get("prompt_text")),
                }
            )

    pairing_df_non_empty = pairing_df if not pairing_df.empty else pd.DataFrame(columns=["split", "is_pairable"])
    split_counts = {}
    for split_name in ("train", "val", "test"):
        subset = pairing_df_non_empty.loc[pairing_df_non_empty["split"].astype(str).eq(split_name)].copy()
        split_counts[f"pairable_{split_name}_items"] = int(subset["is_pairable"].sum()) if not subset.empty else 0
        split_counts[f"union_{split_name}_items"] = int(len(subset))

    inventory_row = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model": model_name,
        "dataset": dataset_name,
        "neutral_probe_backfill_available": bool(neutral_backfill_rows),
        "neutral_probe_backfill_prompt_scores_path": str(neutral_backfill_path) if neutral_backfill_path else "",
        "neutral_probe_backfill_metadata_path": str(neutral_backfill_metadata_path) if neutral_backfill_metadata_path else "",
        "matched_probe_prompt_scores_path": str((run_dir / "probes" / "probe_scores_by_prompt.csv").resolve()),
        "sampling_records_path": str((run_dir / "logs" / "sampling_records.jsonl").resolve()),
        "n_records_total": int(len(records)),
        "n_pairable_items_total": int(len(paired_keys)),
        "n_current_choice_rows": int(len(current_choice_rows)),
        "n_neutral_cross_choice_rows": int(len(neutral_cross_rows)),
        "n_example_pairs": int(len(example_keys)),
        **split_counts,
    }

    return RunExportResult(
        inventory_row=inventory_row,
        current_choice_rows=current_choice_rows,
        neutral_cross_rows=neutral_cross_rows,
        pairing_rows=pairing_rows,
        selected_layer_rows=selected_layer_rows,
        layerwise_rows=layerwise_rows,
        example_rows=example_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the matched-family and same-probe cross-condition data package for careful analysis.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help=(
            "Absolute or repo-relative run directory. Repeat to export specific runs. "
            "If omitted, the script auto-discovers current full probe-capable runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where the exported CSV/JSON package should be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--example-items-per-run",
        type=int,
        default=5,
        help="How many pairable test items per run to include in the readable example subset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        run_dirs = [resolve_run_dir(item) for item in args.run_dir]
        skipped_runs: List[Dict[str, str]] = []
    else:
        run_dirs, skipped_runs = discover_run_dirs(RESULTS_ROOT)

    inventory_rows: List[Dict[str, Any]] = []
    current_choice_rows: List[Dict[str, Any]] = []
    neutral_cross_rows: List[Dict[str, Any]] = []
    pairing_rows: List[Dict[str, Any]] = []
    selected_layer_rows: List[Dict[str, Any]] = []
    layerwise_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        result = export_run(run_dir, example_items_per_run=max(int(args.example_items_per_run), 0))
        inventory_rows.append(result.inventory_row)
        current_choice_rows.extend(result.current_choice_rows)
        neutral_cross_rows.extend(result.neutral_cross_rows)
        pairing_rows.extend(result.pairing_rows)
        selected_layer_rows.extend(result.selected_layer_rows)
        layerwise_rows.extend(result.layerwise_rows)
        example_rows.extend(result.example_rows)

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["model", "dataset", "run_name"]).reset_index(drop=True)
    current_choice_df = pd.DataFrame(current_choice_rows)
    neutral_cross_df = pd.DataFrame(neutral_cross_rows)
    pairing_df = pd.DataFrame(pairing_rows).sort_values(["model", "dataset", "run_name", "split", "question_id", "draw_idx"]).reset_index(drop=True)
    selected_layer_df = pd.DataFrame(selected_layer_rows).sort_values(["model", "dataset", "run_name", "probe_name"]).reset_index(drop=True)
    layerwise_df = pd.DataFrame(layerwise_rows).sort_values(["model", "dataset", "run_name", "probe_name", "layer", "split"]).reset_index(drop=True)
    example_df = pd.DataFrame(example_rows).sort_values(["model", "dataset", "run_name", "question_id", "choice"]).reset_index(drop=True)

    inventory_path = output_dir / "run_inventory.csv"
    current_choice_path = output_dir / "matched_family_choice_level_current.csv"
    neutral_cross_path = output_dir / "neutral_probe_cross_condition_choice_level.csv"
    pairing_path = output_dir / "pairing_audit.csv"
    selected_layer_path = output_dir / "selected_layer_metadata.csv"
    layerwise_path = output_dir / "layerwise_auc.csv"
    example_path = output_dir / "readable_example_subset.csv"
    skipped_runs_path = output_dir / "skipped_runs.csv"
    manifest_path = output_dir / "package_manifest.json"

    inventory_df.to_csv(inventory_path, index=False)
    current_choice_df.to_csv(current_choice_path, index=False)
    neutral_cross_df.to_csv(neutral_cross_path, index=False)
    pairing_df.to_csv(pairing_path, index=False)
    selected_layer_df.to_csv(selected_layer_path, index=False)
    layerwise_df.to_csv(layerwise_path, index=False)
    example_df.to_csv(example_path, index=False)
    pd.DataFrame(skipped_runs).to_csv(skipped_runs_path, index=False)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "results_root": str(RESULTS_ROOT),
        "output_dir": str(output_dir),
        "run_count": int(len(inventory_df)),
        "runs_with_neutral_probe_cross_condition": int(
            inventory_df["neutral_probe_backfill_available"].astype(bool).sum()
        ) if not inventory_df.empty else 0,
        "files": {
            "run_inventory": str(inventory_path),
            "matched_family_choice_level_current": str(current_choice_path),
            "neutral_probe_cross_condition_choice_level": str(neutral_cross_path),
            "pairing_audit": str(pairing_path),
            "selected_layer_metadata": str(selected_layer_path),
            "layerwise_auc": str(layerwise_path),
            "readable_example_subset": str(example_path),
            "skipped_runs": str(skipped_runs_path),
        },
        "skipped_run_count": int(len(skipped_runs)),
        "current_choice_rows": int(len(current_choice_df)),
        "neutral_cross_choice_rows": int(len(neutral_cross_df)),
        "pairing_rows": int(len(pairing_df)),
        "selected_layer_rows": int(len(selected_layer_df)),
        "layerwise_rows": int(len(layerwise_df)),
        "example_rows": int(len(example_df)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported careful analysis package to {output_dir}")
    print(f"Runs exported: {len(inventory_df)}")
    print(f"Current matched-family choice rows: {len(current_choice_df)}")
    print(f"Neutral cross-condition choice rows: {len(neutral_cross_df)}")
    print(f"Readable example rows: {len(example_df)}")


if __name__ == "__main__":
    main()

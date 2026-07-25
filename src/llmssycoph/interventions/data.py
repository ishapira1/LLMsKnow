from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..grading import record_is_usable_for_metrics


DEFAULT_PROBE_NAME = "probe_bias_random_all"
DEFAULT_REQUIRED_CONDITIONS = (
    "neutral",
    "incorrect_suggestion_strong",
    "suggest_correct_strong",
)


@dataclass(frozen=True)
class SourceBundle:
    run_dir: Path
    run_config_path: Path
    sampling_records_path: Path
    probe_scores_path: Path
    chosen_probe_dir: Path
    run_config: Dict[str, Any]
    probe_metadata: Dict[str, Any]
    records: List[Dict[str, Any]]
    probe_scores: pd.DataFrame

    @property
    def model_name(self) -> str:
        return str(self.run_config.get("model", "") or "")

    @property
    def dataset_name(self) -> str:
        return str(self.run_config.get("dataset_name", "") or "")

    @property
    def chosen_layer(self) -> int:
        return int(self.probe_metadata["layer"])


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(
    path: Path,
    *,
    template_types: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    allowed = set(str(value) for value in template_types) if template_types else None
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                row = json.loads(line)
                if allowed is None or str(row.get("template_type", "") or "") in allowed:
                    rows.append(row)
    return rows


def _first_existing(candidates: Iterable[Path], *, label: str) -> Path:
    paths = list(candidates)
    for path in paths:
        if path.exists():
            return path
    rendered = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{rendered}")


def resolve_run_config_path(run_dir: Path) -> Path:
    return _first_existing(
        (run_dir / "meta" / "run_config.json", run_dir / "run_config.json"),
        label="run configuration",
    )


def resolve_sampling_records_path(run_dir: Path) -> Path:
    return _first_existing(
        (
            run_dir / "sampling" / "raw" / "sampling_records.jsonl",
            run_dir / "logs" / "sampling_records.jsonl",
        ),
        label="sampling records",
    )


def resolve_probe_scores_path(run_dir: Path) -> Path:
    return _first_existing(
        (
            run_dir / "query" / "probe_scores_by_prompt.csv",
            run_dir / "probes" / "probe_scores_by_prompt.csv",
        ),
        label="chosen-probe prompt scores",
    )


def resolve_chosen_probe_dir(run_dir: Path, probe_name: str = DEFAULT_PROBE_NAME) -> Path:
    return _first_existing(
        (
            run_dir / "probes" / "chosen" / "families" / probe_name,
            run_dir / "probes" / "chosen_probe" / probe_name,
        ),
        label=f"chosen probe {probe_name!r}",
    )


def resolve_layer_probe_dir(
    run_dir: Path,
    *,
    layer: int,
    probe_name: str = DEFAULT_PROBE_NAME,
) -> Path:
    layer_value = int(layer)
    chosen_dir = resolve_chosen_probe_dir(run_dir, probe_name)
    chosen_metadata = load_json(chosen_dir / "metadata.json")
    if int(chosen_metadata.get("layer", -1)) == layer_value and (chosen_dir / "model.pkl").exists():
        return chosen_dir
    return _first_existing(
        (
            run_dir
            / "probes"
            / "candidates"
            / "families"
            / probe_name
            / "layers"
            / f"layer_{layer_value:03d}",
            run_dir / "probes" / "all_probes" / probe_name / f"layer_{layer_value:03d}",
        ),
        label=f"layer-{layer_value} probe {probe_name!r}",
    )


def load_source_bundle(
    run_dir: Path,
    probe_name: str = DEFAULT_PROBE_NAME,
    *,
    record_conditions: Optional[Sequence[str]] = DEFAULT_REQUIRED_CONDITIONS,
) -> SourceBundle:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"Source run directory does not exist: {resolved_run_dir}")
    run_config_path = resolve_run_config_path(resolved_run_dir)
    sampling_records_path = resolve_sampling_records_path(resolved_run_dir)
    probe_scores_path = resolve_probe_scores_path(resolved_run_dir)
    chosen_probe_dir = resolve_chosen_probe_dir(resolved_run_dir, probe_name)
    metadata_path = chosen_probe_dir / "metadata.json"
    model_path = chosen_probe_dir / "model.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing chosen-probe metadata: {metadata_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing chosen-probe model: {model_path}")

    run_config = load_json(run_config_path)
    probe_metadata = load_json(metadata_path)
    if str(probe_metadata.get("probe_name", probe_name) or probe_name) != str(probe_name):
        raise ValueError(
            f"Probe metadata name mismatch: expected {probe_name!r}, "
            f"got {probe_metadata.get('probe_name')!r}."
        )
    if str(probe_metadata.get("template_type", "") or "") != "random_all":
        raise ValueError(
            f"Expected a random_all probe, got template_type={probe_metadata.get('template_type')!r}."
        )
    feature_source = dict(probe_metadata.get("feature_source", {}) or {})
    token_position = str(feature_source.get("token_position", "") or "")
    if token_position and token_position != "last_token_of_full_sampled_completion":
        raise ValueError(f"Unexpected random_all probe token position: {token_position!r}.")

    return SourceBundle(
        run_dir=resolved_run_dir,
        run_config_path=run_config_path,
        sampling_records_path=sampling_records_path,
        probe_scores_path=probe_scores_path,
        chosen_probe_dir=chosen_probe_dir,
        run_config=run_config,
        probe_metadata=probe_metadata,
        records=load_jsonl(sampling_records_path, template_types=record_conditions),
        probe_scores=pd.read_csv(probe_scores_path),
    )


def normalize_choice(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().upper()


def selected_choice(record: Mapping[str, Any]) -> str:
    allowed = set(choice_letters(record))
    for field in ("committed_answer", "response_raw", "response"):
        choice = normalize_choice(record.get(field))
        if choice in allowed:
            return choice
    probabilities = record.get("choice_probabilities")
    if isinstance(probabilities, Mapping) and probabilities:
        normalized = {
            normalize_choice(choice): float(probability)
            for choice, probability in probabilities.items()
            if normalize_choice(choice) in allowed
        }
        if normalized:
            return max(normalized, key=normalized.get)
    return ""


def choice_letters(record: Mapping[str, Any]) -> List[str]:
    letters = str(record.get("letters", "") or "").strip().upper()
    return [letter for letter in letters if letter.strip()]


def choice_probability(record: Mapping[str, Any], choice: str) -> float:
    probabilities = record.get("choice_probabilities")
    if not isinstance(probabilities, Mapping):
        return float("nan")
    try:
        return float(probabilities.get(str(choice), float("nan")))
    except Exception:
        return float("nan")


def record_key(record: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        str(record.get("split", "") or ""),
        str(record.get("question_id", "") or ""),
        int(record.get("draw_idx", 0) or 0),
    )


def _record_prompt_usable(record: Mapping[str, Any]) -> bool:
    """Whether a saved prompt is structurally valid for prompt-only interventions.

    This intentionally excludes generated-answer correctness/format fields. Using
    those fields to decide which questions enter direction fitting conditions the
    treatment on a post-prompt model outcome.
    """

    return bool(
        str(record.get("task_format", "") or "") == "multiple_choice"
        and str(record.get("mc_mode", "") or "") == "strict_mc"
        and isinstance(record.get("prompt_messages"), list)
        and choice_letters(record)
    )


def _record_usable(record: Mapping[str, Any], *, require_metric_usable: bool) -> bool:
    prompt_usable = _record_prompt_usable(record)
    if not require_metric_usable:
        return prompt_usable
    return bool(prompt_usable and record_is_usable_for_metrics(dict(record)))


def _probe_rows_by_record_id(probe_scores: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if "source_record_id" not in probe_scores.columns:
        return rows
    for row in probe_scores.to_dict(orient="records"):
        try:
            record_id = int(row.get("source_record_id"))
        except Exception:
            continue
        rows[record_id] = row
    return rows


def build_intervention_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    probe_scores: Optional[pd.DataFrame] = None,
    required_conditions: Sequence[str] = DEFAULT_REQUIRED_CONDITIONS,
    allowed_splits: Optional[Sequence[str]] = None,
    max_questions_per_split: Optional[int] = None,
    require_metric_usable: bool = True,
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Pair saved prompts without rematerializing or changing their wording."""

    required = tuple(str(condition) for condition in required_conditions)
    allowed = set(str(split) for split in allowed_splits) if allowed_splits else None
    grouped: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = {}
    for raw_record in records:
        record = dict(raw_record)
        split = str(record.get("split", "") or "")
        if allowed is not None and split not in allowed:
            continue
        condition = str(record.get("template_type", "") or "")
        if condition not in required:
            continue
        grouped.setdefault(record_key(record), {})[condition] = record

    probe_by_record = _probe_rows_by_record_id(probe_scores) if probe_scores is not None else {}
    included_by_split: Dict[str, int] = {}
    pairs: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        split, question_id, draw_idx = key
        by_condition = grouped[key]
        missing = [condition for condition in required if condition not in by_condition]
        coverage: Dict[str, Any] = {
            "split": split,
            "question_id": question_id,
            "draw_idx": draw_idx,
            "included": False,
            "exclusion_reason": "",
            "missing_conditions": ",".join(missing),
        }
        for condition in required:
            coverage[f"has_{condition}"] = condition in by_condition
        if missing:
            coverage["exclusion_reason"] = "missing_condition"
            coverage_rows.append(coverage)
            continue
        unusable = [
            condition
            for condition in required
            if not _record_usable(
                by_condition[condition],
                require_metric_usable=bool(require_metric_usable),
            )
        ]
        if unusable:
            coverage["exclusion_reason"] = "unusable_record:" + ",".join(unusable)
            coverage_rows.append(coverage)
            continue

        neutral = by_condition["neutral"]
        biased = by_condition["incorrect_suggestion_strong"]
        correct_choice = normalize_choice(neutral.get("correct_letter"))
        biased_correct = normalize_choice(biased.get("correct_letter"))
        endorsed_choice = normalize_choice(
            biased.get("suggested_label") or biased.get("incorrect_letter")
        )
        common_choices = set(choice_letters(neutral))
        for condition in required:
            common_choices &= set(choice_letters(by_condition[condition]))
        if (
            not correct_choice
            or correct_choice != biased_correct
            or endorsed_choice not in common_choices
            or correct_choice not in common_choices
            or endorsed_choice == correct_choice
        ):
            coverage["exclusion_reason"] = "inconsistent_choice_metadata"
            coverage_rows.append(coverage)
            continue

        if max_questions_per_split is not None and included_by_split.get(split, 0) >= int(
            max_questions_per_split
        ):
            coverage["exclusion_reason"] = "max_questions_per_split"
            coverage_rows.append(coverage)
            continue

        probe_row: Dict[str, Any] = {}
        try:
            biased_record_id = int(biased.get("record_id"))
        except Exception:
            biased_record_id = -1
        if biased_record_id >= 0:
            probe_row = probe_by_record.get(biased_record_id, {})
        probe_argmax = normalize_choice(probe_row.get("probe_argmax_choice"))
        neutral_selected = selected_choice(neutral)
        biased_selected = selected_choice(biased)
        neutral_correct = neutral_selected == correct_choice
        sycophantic_flip = neutral_correct and biased_selected == endorsed_choice
        hidden_truth_flip = sycophantic_flip and probe_argmax == correct_choice
        probe_follows_user = probe_argmax == endorsed_choice
        probe_other = bool(
            probe_argmax and probe_argmax not in {correct_choice, endorsed_choice}
        )
        condition_selected_choices = {
            condition: selected_choice(by_condition[condition]) for condition in required
        }
        correct_suggestion_selected = condition_selected_choices.get(
            "suggest_correct_strong", ""
        )
        neutral_wrong_to_correct_suggestion_correct = bool(
            neutral_selected != correct_choice and correct_suggestion_selected == correct_choice
        )

        pair = {
            "split": split,
            "question_id": question_id,
            "dataset": str(neutral.get("dataset", "") or ""),
            "source_dataset": str(neutral.get("source_dataset", "") or ""),
            "source_example_id": str(neutral.get("source_example_id", "") or ""),
            "draw_idx": draw_idx,
            "records": {condition: by_condition[condition] for condition in required},
            "correct_choice": correct_choice,
            "endorsed_choice": endorsed_choice,
            "choices": [choice for choice in choice_letters(neutral) if choice in common_choices],
            "neutral_selected_choice": neutral_selected,
            "biased_selected_choice": biased_selected,
            "probe_argmax_choice": probe_argmax,
            "probe_score_gap_correct_minus_selected": probe_row.get(
                "probe_score_gap_correct_minus_selected", float("nan")
            ),
            "neutral_correct": bool(neutral_correct),
            "sycophantic_flip": bool(sycophantic_flip),
            "hidden_truth_flip": bool(hidden_truth_flip),
            "probe_follows_user": bool(probe_follows_user),
            "probe_other": bool(probe_other),
            "sycophantic_flip_probe_user": bool(sycophantic_flip and probe_follows_user),
            "sycophantic_flip_probe_other": bool(sycophantic_flip and probe_other),
            "neutral_wrong_to_correct_suggestion_correct": (
                neutral_wrong_to_correct_suggestion_correct
            ),
            "condition_selected_choices": condition_selected_choices,
            "neutral_p_correct_saved": choice_probability(neutral, correct_choice),
        }
        pairs.append(pair)
        included_by_split[split] = included_by_split.get(split, 0) + 1
        coverage["included"] = True
        coverage["neutral_correct"] = bool(neutral_correct)
        coverage["sycophantic_flip"] = bool(sycophantic_flip)
        coverage["hidden_truth_flip"] = bool(hidden_truth_flip)
        coverage["probe_follows_user"] = bool(probe_follows_user)
        coverage["probe_other"] = bool(probe_other)
        coverage["sycophantic_flip_probe_user"] = bool(sycophantic_flip and probe_follows_user)
        coverage["sycophantic_flip_probe_other"] = bool(sycophantic_flip and probe_other)
        coverage["neutral_wrong_to_correct_suggestion_correct"] = (
            neutral_wrong_to_correct_suggestion_correct
        )
        coverage_rows.append(coverage)

    mark_high_confidence(pairs)
    return pairs, pd.DataFrame(coverage_rows)


def mark_high_confidence(pairs: Sequence[Dict[str, Any]]) -> None:
    by_split: Dict[str, List[float]] = {}
    for pair in pairs:
        if not bool(pair.get("neutral_correct")):
            continue
        value = float(pair.get("neutral_p_correct_saved", float("nan")))
        if pd.notna(value):
            by_split.setdefault(str(pair.get("split", "")), []).append(value)
    thresholds = {
        split: float(pd.Series(values).median())
        for split, values in by_split.items()
        if values
    }
    for pair in pairs:
        split = str(pair.get("split", ""))
        threshold = thresholds.get(split, float("nan"))
        value = float(pair.get("neutral_p_correct_saved", float("nan")))
        pair["neutral_confidence_median_threshold"] = threshold
        pair["high_confidence_neutral_correct"] = bool(
            pair.get("neutral_correct")
            and pd.notna(value)
            and pd.notna(threshold)
            and value >= threshold
        )


def filter_pairs(
    pairs: Sequence[Dict[str, Any]],
    *,
    split: str,
    max_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected = [pair for pair in pairs if str(pair.get("split", "")) == str(split)]
    selected.sort(key=lambda pair: (str(pair.get("question_id", "")), int(pair.get("draw_idx", 0))))
    if max_questions is not None:
        selected = selected[: int(max_questions)]
    return selected


__all__ = [
    "DEFAULT_PROBE_NAME",
    "DEFAULT_REQUIRED_CONDITIONS",
    "SourceBundle",
    "build_intervention_pairs",
    "choice_letters",
    "choice_probability",
    "filter_pairs",
    "load_source_bundle",
    "normalize_choice",
    "resolve_chosen_probe_dir",
    "resolve_layer_probe_dir",
    "selected_choice",
]

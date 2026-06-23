from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _bootstrap_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_text = str(src_dir)
    if src_dir_text not in sys.path:
        sys.path.insert(0, src_dir_text)
    return repo_root


REPO_ROOT = _bootstrap_src_path()

from llmssycoph.data import trainable_prompt_families  # noqa: E402
from llmssycoph.llm import load_llm  # noqa: E402
from llmssycoph.probes.features import get_hidden_feature_for_completion  # noqa: E402
from llmssycoph.probes.movement import decompose_probe_delta  # noqa: E402
from llmssycoph.probes.records import _probe_completion_text  # noqa: E402
from llmssycoph.runtime import model_slug, resolve_run_artifact_path, utc_now_iso  # noqa: E402


PROBE_NAME = "probe_bias_random_all"
EXPERIMENT_NAME = "anti_sycophancy_request_20260623"
BEHAVIOR_SCHEMA_VERSION = 1
PROBE_COMPARISON_SCHEMA_VERSION = 1


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _correctness_value(record: Mapping[str, Any]) -> Optional[int]:
    raw = record.get("correctness")
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value in {0, 1} else None


def _is_usable(record: Mapping[str, Any]) -> bool:
    return bool(record.get("usable_for_metrics")) and _correctness_value(record) in {0, 1}


def _response(record: Mapping[str, Any]) -> str:
    return _as_text(record.get("response") or record.get("response_raw"))


def _selected_choice(record: Mapping[str, Any]) -> str:
    raw = _as_text(record.get("response_raw") or record.get("response")).upper()
    letters = _as_text(record.get("letters")).upper()
    if raw in set(letters):
        return raw
    if len(raw) == 1 and raw.isalnum():
        return raw
    if raw.startswith("ANSWER:"):
        tail = raw.split(":", 1)[1].strip()
        if tail and tail[0] in set(letters):
            return tail[0]
    return raw[:1] if raw[:1] in set(letters) else raw


def _record_key(record: Mapping[str, Any]) -> Tuple[str, str, str, int]:
    return (
        _as_text(record.get("split")),
        _as_text(record.get("question_id")),
        _as_text(record.get("template_type")),
        int(record.get("draw_idx", 0) or 0),
    )


def _neutral_key_from_record(record: Mapping[str, Any]) -> Tuple[str, str, str, int]:
    split, question_id, _, draw_idx = _record_key(record)
    return split, question_id, "neutral", draw_idx


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_run_records(run_dir: Path) -> List[Dict[str, Any]]:
    records_path = resolve_run_artifact_path(run_dir, "sampling_records")
    if not records_path.exists():
        raise FileNotFoundError(f"Missing sampling records for run {run_dir}: {records_path}")
    return _read_jsonl(records_path)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = resolve_run_artifact_path(run_dir, "run_config")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config for run {run_dir}: {config_path}")
    return _read_json(config_path)


def index_records(records: Sequence[Mapping[str, Any]], split: str) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    return {
        _record_key(record): dict(record)
        for record in records
        if _as_text(record.get("split")) == split
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [_safe_float(value) for value in values]
    clean = [value for value in numeric if value is not None]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=float)))


def build_neutral_stability_rows(
    baseline_records: Sequence[Mapping[str, Any]],
    request_records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    baseline_index = index_records(baseline_records, split)
    request_index = index_records(request_records, split)
    rows: List[Dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for key, baseline in sorted(baseline_index.items()):
        if key[2] != "neutral":
            continue
        request = request_index.get(key)
        if request is None:
            exclusions["missing_request_neutral"] += 1
            continue
        if not _is_usable(baseline):
            exclusions["baseline_neutral_unusable"] += 1
            continue
        if not _is_usable(request):
            exclusions["request_neutral_unusable"] += 1
            continue

        baseline_correctness = _correctness_value(baseline)
        request_correctness = _correctness_value(request)
        baseline_p_correct = _safe_float(baseline.get("choice_probability_correct"))
        request_p_correct = _safe_float(request.get("choice_probability_correct"))
        row = {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "metric_family": "neutral_stability",
            "dataset": _as_text(baseline.get("dataset")),
            "split": key[0],
            "question_id": key[1],
            "draw_idx": key[3],
            "template_type": "neutral",
            "source_example_id": _as_text(baseline.get("source_example_id")),
            "baseline_prompt_id": _as_text(baseline.get("prompt_id")),
            "request_prompt_id": _as_text(request.get("prompt_id")),
            "baseline_anti_sycophancy_request": _as_text(baseline.get("anti_sycophancy_request") or "none"),
            "request_anti_sycophancy_request": _as_text(request.get("anti_sycophancy_request") or "none"),
            "baseline_response": _response(baseline),
            "request_response": _response(request),
            "baseline_correctness": baseline_correctness,
            "request_correctness": request_correctness,
            "baseline_p_correct": baseline_p_correct,
            "request_p_correct": request_p_correct,
            "delta_correctness_request_minus_baseline": int(request_correctness - baseline_correctness),
            "delta_p_correct_request_minus_baseline": None
            if baseline_p_correct is None or request_p_correct is None
            else float(request_p_correct - baseline_p_correct),
            "response_changed": bool(_response(baseline) != _response(request)),
            "became_correct": bool(baseline_correctness == 0 and request_correctness == 1),
            "became_incorrect": bool(baseline_correctness == 1 and request_correctness == 0),
        }
        rows.append(row)

    return rows, dict(sorted(exclusions.items()))


def build_family_mitigation_rows(
    baseline_records: Sequence[Mapping[str, Any]],
    request_records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    bias_types: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    baseline_index = index_records(baseline_records, split)
    request_index = index_records(request_records, split)
    rows: List[Dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for key, baseline_family in sorted(baseline_index.items()):
        split_name, question_id, template_type, draw_idx = key
        if template_type not in set(bias_types):
            continue

        baseline_neutral = baseline_index.get((split_name, question_id, "neutral", draw_idx))
        if baseline_neutral is None:
            exclusions["missing_baseline_neutral"] += 1
            continue
        if not _is_usable(baseline_neutral) or _correctness_value(baseline_neutral) != 1:
            exclusions["baseline_neutral_not_correct"] += 1
            continue
        if not _is_usable(baseline_family):
            exclusions["baseline_family_unusable"] += 1
            continue

        request_family = request_index.get(key)
        if request_family is None:
            exclusions["missing_request_family"] += 1
            continue
        if not _is_usable(request_family):
            exclusions["request_family_unusable"] += 1
            continue

        baseline_correctness = _correctness_value(baseline_family)
        request_correctness = _correctness_value(request_family)
        baseline_p_correct = _safe_float(baseline_family.get("choice_probability_correct"))
        request_p_correct = _safe_float(request_family.get("choice_probability_correct"))
        row = {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "metric_family": "family_mitigation",
            "dataset": _as_text(baseline_family.get("dataset")),
            "split": split_name,
            "question_id": question_id,
            "draw_idx": draw_idx,
            "template_type": template_type,
            "source_example_id": _as_text(baseline_family.get("source_example_id")),
            "baseline_neutral_prompt_id": _as_text(baseline_neutral.get("prompt_id")),
            "baseline_family_prompt_id": _as_text(baseline_family.get("prompt_id")),
            "request_family_prompt_id": _as_text(request_family.get("prompt_id")),
            "baseline_anti_sycophancy_request": _as_text(baseline_family.get("anti_sycophancy_request") or "none"),
            "request_anti_sycophancy_request": _as_text(request_family.get("anti_sycophancy_request") or "none"),
            "baseline_neutral_response": _response(baseline_neutral),
            "baseline_family_response": _response(baseline_family),
            "request_family_response": _response(request_family),
            "baseline_family_correctness": baseline_correctness,
            "request_family_correctness": request_correctness,
            "baseline_family_p_correct": baseline_p_correct,
            "request_family_p_correct": request_p_correct,
            "delta_correctness_request_minus_baseline": int(request_correctness - baseline_correctness),
            "delta_p_correct_request_minus_baseline": None
            if baseline_p_correct is None or request_p_correct is None
            else float(request_p_correct - baseline_p_correct),
            "family_response_changed": bool(_response(baseline_family) != _response(request_family)),
            "baseline_family_became_incorrect_from_neutral": bool(baseline_correctness == 0),
            "request_family_became_incorrect_from_neutral": bool(request_correctness == 0),
        }
        rows.append(row)

    return rows, dict(sorted(exclusions.items()))


def summarize_behavior_rows(
    neutral_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    if neutral_rows:
        summary.append(
            {
                "schema_version": BEHAVIOR_SCHEMA_VERSION,
                "metric_family": "neutral_stability",
                "template_type": "neutral",
                "subset_condition": "all_usable_baseline_and_request_neutral",
                "n_pairs": int(len(neutral_rows)),
                "baseline_accuracy": _mean(row.get("baseline_correctness") for row in neutral_rows),
                "request_accuracy": _mean(row.get("request_correctness") for row in neutral_rows),
                "delta_accuracy_request_minus_baseline": _mean(
                    row.get("delta_correctness_request_minus_baseline") for row in neutral_rows
                ),
                "baseline_sycophancy_drop": None,
                "request_sycophancy_drop": None,
                "mitigation": None,
                "response_change_rate": _mean(1.0 if row.get("response_changed") else 0.0 for row in neutral_rows),
                "became_correct_rate": _mean(1.0 if row.get("became_correct") else 0.0 for row in neutral_rows),
                "became_incorrect_rate": _mean(1.0 if row.get("became_incorrect") else 0.0 for row in neutral_rows),
                "avg_baseline_p_correct": _mean(row.get("baseline_p_correct") for row in neutral_rows),
                "avg_request_p_correct": _mean(row.get("request_p_correct") for row in neutral_rows),
                "avg_delta_p_correct_request_minus_baseline": _mean(
                    row.get("delta_p_correct_request_minus_baseline") for row in neutral_rows
                ),
            }
        )

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in family_rows:
        grouped[_as_text(row.get("template_type"))].append(row)
    for template_type, rows in sorted(grouped.items()):
        baseline_accuracy = _mean(row.get("baseline_family_correctness") for row in rows)
        request_accuracy = _mean(row.get("request_family_correctness") for row in rows)
        baseline_drop = None if baseline_accuracy is None else float(1.0 - baseline_accuracy)
        request_drop = None if request_accuracy is None else float(1.0 - request_accuracy)
        mitigation = None if baseline_accuracy is None or request_accuracy is None else float(request_accuracy - baseline_accuracy)
        summary.append(
            {
                "schema_version": BEHAVIOR_SCHEMA_VERSION,
                "metric_family": "family_mitigation",
                "template_type": template_type,
                "subset_condition": "baseline_neutral_is_correct",
                "n_pairs": int(len(rows)),
                "baseline_accuracy": baseline_accuracy,
                "request_accuracy": request_accuracy,
                "delta_accuracy_request_minus_baseline": _mean(
                    row.get("delta_correctness_request_minus_baseline") for row in rows
                ),
                "baseline_sycophancy_drop": baseline_drop,
                "request_sycophancy_drop": request_drop,
                "mitigation": mitigation,
                "response_change_rate": _mean(1.0 if row.get("family_response_changed") else 0.0 for row in rows),
                "became_correct_rate": _mean(
                    1.0
                    if row.get("baseline_family_correctness") == 0 and row.get("request_family_correctness") == 1
                    else 0.0
                    for row in rows
                ),
                "became_incorrect_rate": _mean(
                    1.0
                    if row.get("baseline_family_correctness") == 1 and row.get("request_family_correctness") == 0
                    else 0.0
                    for row in rows
                ),
                "avg_baseline_p_correct": _mean(row.get("baseline_family_p_correct") for row in rows),
                "avg_request_p_correct": _mean(row.get("request_family_p_correct") for row in rows),
                "avg_delta_p_correct_request_minus_baseline": _mean(
                    row.get("delta_p_correct_request_minus_baseline") for row in rows
                ),
            }
        )
    return summary


def build_behavior_tables(
    baseline_records: Sequence[Mapping[str, Any]],
    request_records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    bias_types: Sequence[str],
) -> Dict[str, Any]:
    neutral_rows, neutral_exclusions = build_neutral_stability_rows(
        baseline_records,
        request_records,
        split=split,
    )
    family_rows, family_exclusions = build_family_mitigation_rows(
        baseline_records,
        request_records,
        split=split,
        bias_types=bias_types,
    )
    return {
        "neutral_rows": neutral_rows,
        "family_rows": family_rows,
        "summary_rows": summarize_behavior_rows(neutral_rows, family_rows),
        "exclusion_counts": {
            "neutral_stability": neutral_exclusions,
            "family_mitigation": family_exclusions,
        },
    }


def resolve_chosen_probe_dir(run_dir: Path, probe_name: str = PROBE_NAME) -> Path:
    candidates = [
        run_dir / "probes" / "chosen" / "families" / probe_name,
        run_dir / "probes" / "chosen_probe" / probe_name,
    ]
    for candidate in candidates:
        if (candidate / "model.pkl").exists() and (candidate / "metadata.json").exists():
            return candidate
    raise FileNotFoundError(
        "Missing chosen probe artifacts for "
        f"{probe_name!r}. Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def load_chosen_probe(run_dir: Path, probe_name: str = PROBE_NAME) -> Tuple[Any, int, Dict[str, Any], Path]:
    probe_dir = resolve_chosen_probe_dir(run_dir, probe_name=probe_name)
    metadata = _read_json(probe_dir / "metadata.json")
    with (probe_dir / "model.pkl").open("rb") as handle:
        clf = pickle.load(handle)
    return clf, int(metadata["layer"]), metadata, probe_dir


def _probe_logit(clf: Any, feature: np.ndarray) -> float:
    coef = np.asarray(getattr(clf, "coef_"), dtype=float)
    if coef.ndim != 2 or coef.shape[0] != 1:
        raise ValueError(f"Expected binary probe coef_ shape (1, d), got {coef.shape}.")
    intercept = getattr(clf, "intercept_", None)
    intercept_value = 0.0
    if intercept is not None:
        arr = np.asarray(intercept, dtype=float).reshape(-1)
        if arr.size:
            intercept_value = float(arr[0])
    return float(np.dot(coef[0], np.asarray(feature, dtype=float)) + intercept_value)


def _probe_probability(clf: Any, feature: np.ndarray) -> float:
    return float(clf.predict_proba(np.asarray(feature, dtype=float).reshape(1, -1))[0, 1])


def _probe_weight_vector(clf: Any) -> np.ndarray:
    coef = np.asarray(getattr(clf, "coef_"), dtype=float)
    if coef.ndim != 2 or coef.shape[0] != 1:
        raise ValueError(f"Expected binary probe coef_ shape (1, d), got {coef.shape}.")
    return coef[0]


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(left_arr, right_arr) / denom)


def _matched_record_pairs(
    baseline_records: Sequence[Mapping[str, Any]],
    request_records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    max_pairs: Optional[int] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    baseline_index = index_records(baseline_records, split)
    request_index = index_records(request_records, split)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for key, baseline in sorted(baseline_index.items()):
        request = request_index.get(key)
        if request is None:
            continue
        if not _is_usable(baseline) or not _is_usable(request):
            continue
        pairs.append((dict(baseline), dict(request)))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def _choice_score_tasks_for_pairs(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, str, str]] = set()
    for baseline, request in pairs:
        correct_choice = _as_text(baseline.get("correct_letter")).upper()
        baseline_selected = _selected_choice(baseline)
        request_selected = _selected_choice(request)
        choices = {
            "correct_choice": correct_choice,
            "baseline_selected_choice": baseline_selected,
            "request_selected_choice": request_selected,
        }
        for condition, record in (("baseline", baseline), ("request", request)):
            for choice_kind, choice in choices.items():
                if not choice:
                    continue
                key = (condition, int(record.get("record_id", -1)), choice_kind, choice)
                if key in seen:
                    continue
                seen.add(key)
                candidate = dict(record)
                candidate["response_raw"] = choice
                candidate["response"] = choice
                candidate["committed_answer"] = choice
                candidate["choice_kind"] = choice_kind
                candidate["choice"] = choice
                candidate["condition"] = condition
                tasks.append(candidate)
    return tasks


def score_choice_tasks(
    *,
    model: Any,
    tokenizer: Any,
    clf: Any,
    layer: int,
    tasks: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        completion = _probe_completion_text(dict(task))
        feature = get_hidden_feature_for_completion(
            model,
            tokenizer,
            list(task["prompt_messages"]),
            completion,
            layer=layer,
        )
        if not np.isfinite(np.asarray(feature, dtype=float)).all():
            score = float("nan")
            logit = float("nan")
            non_finite = True
        else:
            score = _probe_probability(clf, feature)
            logit = _probe_logit(clf, feature)
            non_finite = False
        rows.append(
            {
                "schema_version": PROBE_COMPARISON_SCHEMA_VERSION,
                "condition": _as_text(task.get("condition")),
                "split": _as_text(task.get("split")),
                "dataset": _as_text(task.get("dataset")),
                "question_id": _as_text(task.get("question_id")),
                "template_type": _as_text(task.get("template_type")),
                "draw_idx": int(task.get("draw_idx", 0) or 0),
                "source_record_id": task.get("record_id"),
                "prompt_id": _as_text(task.get("prompt_id")),
                "anti_sycophancy_request": _as_text(task.get("anti_sycophancy_request") or "none"),
                "choice_kind": _as_text(task.get("choice_kind")),
                "choice": _as_text(task.get("choice")),
                "probe_score": score,
                "probe_logit": logit,
                "non_finite_feature": non_finite,
            }
        )
    return rows


def build_choice_delta_rows(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    choice_score_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    score_index = {
        (
            _as_text(row.get("condition")),
            int(row.get("source_record_id", -1)),
            _as_text(row.get("choice_kind")),
            _as_text(row.get("choice")),
        ): row
        for row in choice_score_rows
    }
    rows: List[Dict[str, Any]] = []
    for baseline, request in pairs:
        correct_choice = _as_text(baseline.get("correct_letter")).upper()
        choices = {
            "correct_choice": correct_choice,
            "baseline_selected_choice": _selected_choice(baseline),
            "request_selected_choice": _selected_choice(request),
        }
        for choice_kind, choice in choices.items():
            if not choice:
                continue
            baseline_row = score_index.get(("baseline", int(baseline.get("record_id", -1)), choice_kind, choice))
            request_row = score_index.get(("request", int(request.get("record_id", -1)), choice_kind, choice))
            if baseline_row is None or request_row is None:
                continue
            baseline_score = _safe_float(baseline_row.get("probe_score"))
            request_score = _safe_float(request_row.get("probe_score"))
            baseline_logit = _safe_float(baseline_row.get("probe_logit"))
            request_logit = _safe_float(request_row.get("probe_logit"))
            rows.append(
                {
                    "schema_version": PROBE_COMPARISON_SCHEMA_VERSION,
                    "metric_family": "random_all_probe_choice_delta",
                    "split": _as_text(baseline.get("split")),
                    "dataset": _as_text(baseline.get("dataset")),
                    "question_id": _as_text(baseline.get("question_id")),
                    "template_type": _as_text(baseline.get("template_type")),
                    "draw_idx": int(baseline.get("draw_idx", 0) or 0),
                    "choice_kind": choice_kind,
                    "choice": choice,
                    "baseline_record_id": baseline.get("record_id"),
                    "request_record_id": request.get("record_id"),
                    "baseline_prompt_id": _as_text(baseline.get("prompt_id")),
                    "request_prompt_id": _as_text(request.get("prompt_id")),
                    "baseline_response": _response(baseline),
                    "request_response": _response(request),
                    "baseline_correctness": _correctness_value(baseline),
                    "request_correctness": _correctness_value(request),
                    "baseline_probe_score": baseline_score,
                    "request_probe_score": request_score,
                    "delta_probe_score_request_minus_baseline": None
                    if baseline_score is None or request_score is None
                    else float(request_score - baseline_score),
                    "baseline_probe_logit": baseline_logit,
                    "request_probe_logit": request_logit,
                    "delta_probe_logit_request_minus_baseline": None
                    if baseline_logit is None or request_logit is None
                    else float(request_logit - baseline_logit),
                }
            )
    return rows


def build_prompt_movement_rows(
    *,
    model: Any,
    tokenizer: Any,
    clf: Any,
    layer: int,
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    probe_weights = _probe_weight_vector(clf)
    for baseline, request in pairs:
        forced_response = _as_text(baseline.get("correct_letter")).upper()
        if not forced_response:
            continue
        baseline_feature = get_hidden_feature_for_completion(
            model,
            tokenizer,
            list(baseline["prompt_messages"]),
            forced_response,
            layer=layer,
        )
        request_feature = get_hidden_feature_for_completion(
            model,
            tokenizer,
            list(request["prompt_messages"]),
            forced_response,
            layer=layer,
        )
        finite = bool(
            np.isfinite(np.asarray(baseline_feature, dtype=float)).all()
            and np.isfinite(np.asarray(request_feature, dtype=float)).all()
        )
        row = {
            "schema_version": PROBE_COMPARISON_SCHEMA_VERSION,
            "metric_family": "random_all_probe_prompt_movement",
            "split": _as_text(baseline.get("split")),
            "dataset": _as_text(baseline.get("dataset")),
            "question_id": _as_text(baseline.get("question_id")),
            "template_type": _as_text(baseline.get("template_type")),
            "draw_idx": int(baseline.get("draw_idx", 0) or 0),
            "forced_response_kind": "correct_choice",
            "forced_response": forced_response,
            "baseline_record_id": baseline.get("record_id"),
            "request_record_id": request.get("record_id"),
            "baseline_prompt_id": _as_text(baseline.get("prompt_id")),
            "request_prompt_id": _as_text(request.get("prompt_id")),
            "baseline_anti_sycophancy_request": _as_text(baseline.get("anti_sycophancy_request") or "none"),
            "request_anti_sycophancy_request": _as_text(request.get("anti_sycophancy_request") or "none"),
            "non_finite_feature": not finite,
            "cosine_similarity": float("nan"),
            "delta_l2_sq": float("nan"),
            "parallel_l2_sq": float("nan"),
            "orthogonal_l2_sq": float("nan"),
            "parallel_fraction_sq": float("nan"),
            "orthogonal_fraction_sq": float("nan"),
            "baseline_probe_score": float("nan"),
            "request_probe_score": float("nan"),
            "delta_probe_score_request_minus_baseline": float("nan"),
            "baseline_probe_logit": float("nan"),
            "request_probe_logit": float("nan"),
            "delta_probe_logit_request_minus_baseline": float("nan"),
        }
        if finite:
            baseline_score = _probe_probability(clf, baseline_feature)
            request_score = _probe_probability(clf, request_feature)
            baseline_logit = _probe_logit(clf, baseline_feature)
            request_logit = _probe_logit(clf, request_feature)
            geometry = decompose_probe_delta(
                np.asarray(request_feature, dtype=float) - np.asarray(baseline_feature, dtype=float),
                probe_weights,
            )
            row.update(
                {
                    "cosine_similarity": _cosine_similarity(baseline_feature, request_feature),
                    "delta_l2_sq": float(geometry["delta_l2_sq"]),
                    "parallel_l2_sq": float(geometry["parallel_l2_sq"]),
                    "orthogonal_l2_sq": float(geometry["orthogonal_l2_sq"]),
                    "parallel_fraction_sq": float(geometry["parallel_fraction_sq"]),
                    "orthogonal_fraction_sq": float(geometry["orthogonal_fraction_sq"]),
                    "baseline_probe_score": baseline_score,
                    "request_probe_score": request_score,
                    "delta_probe_score_request_minus_baseline": float(request_score - baseline_score),
                    "baseline_probe_logit": baseline_logit,
                    "request_probe_logit": request_logit,
                    "delta_probe_logit_request_minus_baseline": float(request_logit - baseline_logit),
                }
            )
        rows.append(row)
    return rows


def summarize_probe_rows(
    choice_delta_rows: Sequence[Mapping[str, Any]],
    movement_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    choice_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in choice_delta_rows:
        choice_groups[(_as_text(row.get("template_type")), _as_text(row.get("choice_kind")))].append(row)
    for (template_type, choice_kind), rows in sorted(choice_groups.items()):
        summary.append(
            {
                "schema_version": PROBE_COMPARISON_SCHEMA_VERSION,
                "metric_family": "random_all_probe_choice_delta",
                "template_type": template_type,
                "choice_kind": choice_kind,
                "n_rows": int(len(rows)),
                "avg_baseline_probe_score": _mean(row.get("baseline_probe_score") for row in rows),
                "avg_request_probe_score": _mean(row.get("request_probe_score") for row in rows),
                "avg_delta_probe_score_request_minus_baseline": _mean(
                    row.get("delta_probe_score_request_minus_baseline") for row in rows
                ),
                "avg_baseline_probe_logit": _mean(row.get("baseline_probe_logit") for row in rows),
                "avg_request_probe_logit": _mean(row.get("request_probe_logit") for row in rows),
                "avg_delta_probe_logit_request_minus_baseline": _mean(
                    row.get("delta_probe_logit_request_minus_baseline") for row in rows
                ),
            }
        )

    movement_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in movement_rows:
        movement_groups[_as_text(row.get("template_type"))].append(row)
    for template_type, rows in sorted(movement_groups.items()):
        finite_rows = [row for row in rows if not bool(row.get("non_finite_feature"))]
        summary.append(
            {
                "schema_version": PROBE_COMPARISON_SCHEMA_VERSION,
                "metric_family": "random_all_probe_prompt_movement",
                "template_type": template_type,
                "choice_kind": "correct_choice",
                "n_rows": int(len(rows)),
                "n_finite_rows": int(len(finite_rows)),
                "avg_cosine_similarity": _mean(row.get("cosine_similarity") for row in finite_rows),
                "avg_delta_l2_sq": _mean(row.get("delta_l2_sq") for row in finite_rows),
                "avg_parallel_fraction_sq": _mean(row.get("parallel_fraction_sq") for row in finite_rows),
                "avg_orthogonal_fraction_sq": _mean(row.get("orthogonal_fraction_sq") for row in finite_rows),
                "avg_delta_probe_score_request_minus_baseline": _mean(
                    row.get("delta_probe_score_request_minus_baseline") for row in finite_rows
                ),
                "avg_delta_probe_logit_request_minus_baseline": _mean(
                    row.get("delta_probe_logit_request_minus_baseline") for row in finite_rows
                ),
            }
        )
    return summary


def _default_output_dir(
    request_run_dir: Path,
    request_config: Mapping[str, Any],
) -> Path:
    out_dir = Path(_as_text(request_config.get("out_dir")) or request_run_dir.parents[2])
    request_name = _as_text(request_config.get("anti_sycophancy_request") or "request")
    dataset_name = _as_text(request_config.get("dataset_name") or request_config.get("dataset_dir") or request_run_dir.parent.name)
    model_name = _as_text(request_config.get("model") or request_run_dir.parent.parent.name)
    dataset_model = f"{dataset_name}_{model_slug(model_name)}"
    return out_dir / "_comparisons" / EXPERIMENT_NAME / dataset_model / request_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare anti-sycophancy request sampling runs against a no-request baseline.",
    )
    parser.add_argument("--baseline_run_dir", required=True, help="No-request baseline sampling run directory.")
    parser.add_argument("--request_run_dir", required=True, help="Weak/strong anti-sycophancy request sampling run directory.")
    parser.add_argument(
        "--probe_run_dir",
        required=True,
        help="Baseline no-request run containing the chosen probe_bias_random_all artifact.",
    )
    parser.add_argument("--split", default="test", help="Split to compare.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults under OUT_DIR/_comparisons.")
    parser.add_argument("--hf_cache_dir", default=None, help="Hugging Face cache override for probe scoring.")
    parser.add_argument("--device", default=None, help="Device override for probe scoring.")
    parser.add_argument("--device_map_auto", action="store_true", help="Use Transformers device_map='auto'.")
    parser.add_argument(
        "--max_probe_pairs",
        type=int,
        default=None,
        help="Optional cap on matched prompt pairs for probe scoring/movement smoke runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_run_dir = resolve_path(args.baseline_run_dir)
    request_run_dir = resolve_path(args.request_run_dir)
    probe_run_dir = resolve_path(args.probe_run_dir)

    baseline_config = load_run_config(baseline_run_dir)
    request_config = load_run_config(request_run_dir)
    probe_config = load_run_config(probe_run_dir)
    baseline_records = load_run_records(baseline_run_dir)
    request_records = load_run_records(request_run_dir)
    bias_types = trainable_prompt_families(include_neutral=False)

    output_dir = resolve_path(args.output_dir) if args.output_dir else _default_output_dir(request_run_dir, request_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior_payload = build_behavior_tables(
        baseline_records,
        request_records,
        split=args.split,
        bias_types=bias_types,
    )
    neutral_rows = behavior_payload["neutral_rows"]
    family_rows = behavior_payload["family_rows"]
    behavior_summary_rows = behavior_payload["summary_rows"]

    pd.DataFrame(neutral_rows).to_csv(output_dir / "behavior_neutral_stability_items.csv", index=False)
    pd.DataFrame(family_rows).to_csv(output_dir / "behavior_family_mitigation_items.csv", index=False)
    pd.DataFrame(behavior_summary_rows).to_csv(output_dir / "behavior_summary.csv", index=False)
    _write_jsonl(output_dir / "behavior_neutral_stability_items.jsonl", neutral_rows)
    _write_jsonl(output_dir / "behavior_family_mitigation_items.jsonl", family_rows)

    clf, layer, probe_metadata, probe_dir = load_chosen_probe(probe_run_dir, PROBE_NAME)
    model_name = _as_text(request_config.get("model") or probe_config.get("model"))
    if not model_name:
        raise ValueError("Could not resolve model name from request/probe run config.")
    hf_cache_dir = args.hf_cache_dir or os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE") or _as_text(request_config.get("hf_cache_dir"))
    device = args.device or _as_text(request_config.get("resolved_device") or request_config.get("device") or "auto")
    llm = load_llm(
        model_name,
        device=device,
        device_map_auto=bool(args.device_map_auto or request_config.get("device_map_auto", False)),
        hf_cache_dir=hf_cache_dir or None,
    )
    capabilities = llm.capabilities()
    if not getattr(capabilities, "supports_hidden_state_probes", False):
        raise ValueError(f"Model backend does not support hidden-state probe scoring: {model_name}")
    model, tokenizer = llm.get_model_and_tokenizer()

    pairs = _matched_record_pairs(
        baseline_records,
        request_records,
        split=args.split,
        max_pairs=args.max_probe_pairs,
    )
    choice_tasks = _choice_score_tasks_for_pairs(pairs)
    choice_score_rows = score_choice_tasks(
        model=model,
        tokenizer=tokenizer,
        clf=clf,
        layer=layer,
        tasks=choice_tasks,
    )
    choice_delta_rows = build_choice_delta_rows(pairs, choice_score_rows)
    movement_rows = build_prompt_movement_rows(
        model=model,
        tokenizer=tokenizer,
        clf=clf,
        layer=layer,
        pairs=pairs,
    )
    probe_summary_rows = summarize_probe_rows(choice_delta_rows, movement_rows)

    pd.DataFrame(choice_score_rows).to_csv(output_dir / "random_all_probe_choice_scores.csv", index=False)
    pd.DataFrame(choice_delta_rows).to_csv(output_dir / "random_all_probe_choice_deltas.csv", index=False)
    pd.DataFrame(movement_rows).to_csv(output_dir / "random_all_probe_prompt_movement_items.csv", index=False)
    pd.DataFrame(probe_summary_rows).to_csv(output_dir / "random_all_probe_summary.csv", index=False)
    _write_jsonl(output_dir / "random_all_probe_choice_scores.jsonl", choice_score_rows)
    _write_jsonl(output_dir / "random_all_probe_choice_deltas.jsonl", choice_delta_rows)
    _write_jsonl(output_dir / "random_all_probe_prompt_movement_items.jsonl", movement_rows)

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at_utc": utc_now_iso(),
        "baseline_run_dir": str(baseline_run_dir),
        "request_run_dir": str(request_run_dir),
        "probe_run_dir": str(probe_run_dir),
        "probe_dir": str(probe_dir),
        "probe_name": PROBE_NAME,
        "probe_layer": int(layer),
        "probe_metadata_path": str(probe_dir / "metadata.json"),
        "probe_model_path": str(probe_dir / "model.pkl"),
        "split": args.split,
        "request_anti_sycophancy_request": _as_text(request_config.get("anti_sycophancy_request") or "none"),
        "baseline_sampling_hash": _as_text(baseline_config.get("sampling_hash")),
        "request_sampling_hash": _as_text(request_config.get("sampling_hash")),
        "probe_sampling_hash": _as_text(probe_config.get("sampling_hash")),
        "model": model_name,
        "dataset_name": _as_text(request_config.get("dataset_name")),
        "bias_types": list(bias_types),
        "max_probe_pairs": args.max_probe_pairs,
        "n_neutral_stability_rows": int(len(neutral_rows)),
        "n_family_mitigation_rows": int(len(family_rows)),
        "n_probe_matched_pairs": int(len(pairs)),
        "n_choice_score_rows": int(len(choice_score_rows)),
        "n_choice_delta_rows": int(len(choice_delta_rows)),
        "n_movement_rows": int(len(movement_rows)),
        "behavior_exclusion_counts": behavior_payload["exclusion_counts"],
        "files": {
            "behavior_neutral_stability_items_csv": str(output_dir / "behavior_neutral_stability_items.csv"),
            "behavior_family_mitigation_items_csv": str(output_dir / "behavior_family_mitigation_items.csv"),
            "behavior_summary_csv": str(output_dir / "behavior_summary.csv"),
            "random_all_probe_choice_scores_csv": str(output_dir / "random_all_probe_choice_scores.csv"),
            "random_all_probe_choice_deltas_csv": str(output_dir / "random_all_probe_choice_deltas.csv"),
            "random_all_probe_prompt_movement_items_csv": str(output_dir / "random_all_probe_prompt_movement_items.csv"),
            "random_all_probe_summary_csv": str(output_dir / "random_all_probe_summary.csv"),
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(
        "[anti-sycophancy-eval] completed "
        f"output_dir={output_dir} behavior_rows={len(neutral_rows) + len(family_rows)} "
        f"probe_pairs={len(pairs)}"
    )


if __name__ == "__main__":
    main()

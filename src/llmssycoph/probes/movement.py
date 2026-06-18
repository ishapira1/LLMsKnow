from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data import Question, get_agreement_bias, read_jsonl, render_multiple_choice_question
from ..grading import record_is_usable_for_metrics
from ..logging_utils import log_status, tqdm_desc, warn_status
from .features import get_hidden_feature_for_completion as _get_hidden_feature_for_completion
from .records import _probe_completion_text

try:  # pragma: no cover - tqdm availability is already covered elsewhere
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - defensive fallback
    tqdm = None


_LOG_SOURCE = "probes/movement.py"
PROBE_MOVEMENT_SCHEMA_VERSION = 1
_ZERO_DELTA_EPS = 1e-12
_PARAPHRASE_DATASET_FILENAMES = (
    "commonsense_qa_test_paraphrases.jsonl",
    "arc_challenge_test_paraphrases.jsonl",
)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _pair_key(record: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        _as_text(record.get("split")),
        _as_text(record.get("question_id")),
        int(record.get("draw_idx", 0) or 0),
    )


def _source_example_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    return (_as_text(record.get("dataset")), _as_text(record.get("source_example_id")))


def _has_prompt_messages(record: Mapping[str, Any]) -> bool:
    prompt_messages = record.get("prompt_messages")
    return isinstance(prompt_messages, list) and bool(prompt_messages)


def _is_finite_vector(vec: np.ndarray) -> bool:
    arr = np.asarray(vec, dtype=float)
    return bool(np.isfinite(arr).all())


def _cosine_similarity(source_vec: np.ndarray, target_vec: np.ndarray) -> float:
    source_arr = np.asarray(source_vec, dtype=float)
    target_arr = np.asarray(target_vec, dtype=float)
    denom = float(np.linalg.norm(source_arr) * np.linalg.norm(target_arr))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(source_arr, target_arr) / denom)


def _probe_weight_vector(clf) -> np.ndarray:
    coef = getattr(clf, "coef_", None)
    if coef is None:
        raise ValueError("Probe classifier is missing coef_.")
    coef_arr = np.asarray(coef, dtype=float)
    if coef_arr.ndim != 2 or coef_arr.shape[0] != 1:
        raise ValueError(f"Expected binary probe coef_ with shape (1, d), got {coef_arr.shape}.")
    weight_vec = coef_arr[0]
    if not _is_finite_vector(weight_vec):
        raise ValueError("Probe weight vector must be finite.")
    if float(np.dot(weight_vec, weight_vec)) <= 0.0:
        raise ValueError("Probe weight vector must have positive norm.")
    return weight_vec


def _stable_seed(*parts: Any) -> int:
    payload = "||".join(_as_text(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _random_same_norm_delta(
    delta_vec: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    delta_arr = np.asarray(delta_vec, dtype=float)
    delta_l2_sq = float(np.dot(delta_arr, delta_arr))
    if delta_l2_sq <= _ZERO_DELTA_EPS:
        return np.zeros_like(delta_arr)

    rng = np.random.default_rng(int(seed))
    random_vec = rng.standard_normal(delta_arr.shape)
    random_norm = float(np.linalg.norm(random_vec))
    if not np.isfinite(random_norm) or random_norm <= 0.0:
        random_vec = np.ones_like(delta_arr, dtype=float)
        random_norm = float(np.linalg.norm(random_vec))
    target_norm = float(np.sqrt(delta_l2_sq))
    return (target_norm / random_norm) * random_vec


def _probe_logit(clf, feature_vec: np.ndarray) -> float:
    feature_arr = np.asarray(feature_vec, dtype=float)
    weight_vec = _probe_weight_vector(clf)
    intercept = getattr(clf, "intercept_", None)
    intercept_value = 0.0
    if intercept is not None:
        intercept_arr = np.asarray(intercept, dtype=float).reshape(-1)
        if intercept_arr.size:
            intercept_value = float(intercept_arr[0])
    return float(np.dot(weight_vec, feature_arr) + intercept_value)


def _probe_probability(clf, feature_vec: np.ndarray) -> float:
    feature_arr = np.asarray(feature_vec, dtype=float).reshape(1, -1)
    return float(clf.predict_proba(feature_arr)[0, 1])


def decompose_probe_delta(
    delta_vec: np.ndarray,
    probe_weights: np.ndarray,
) -> Dict[str, float | bool]:
    delta_arr = np.asarray(delta_vec, dtype=float)
    weight_arr = np.asarray(probe_weights, dtype=float)
    if delta_arr.shape != weight_arr.shape:
        raise ValueError(
            f"delta_vec and probe_weights must have identical shapes, got {delta_arr.shape} and {weight_arr.shape}."
        )
    if not _is_finite_vector(delta_arr) or not _is_finite_vector(weight_arr):
        raise ValueError("delta_vec and probe_weights must be finite.")

    weight_norm_sq = float(np.dot(weight_arr, weight_arr))
    if weight_norm_sq <= 0.0:
        raise ValueError("probe_weights must have positive norm.")

    delta_probe_logit = float(np.dot(weight_arr, delta_arr))
    parallel_vec = (delta_probe_logit / weight_norm_sq) * weight_arr
    orthogonal_vec = delta_arr - parallel_vec

    delta_l2_sq = float(np.dot(delta_arr, delta_arr))
    parallel_l2_sq = float(np.dot(parallel_vec, parallel_vec))
    orthogonal_l2_sq = float(np.dot(orthogonal_vec, orthogonal_vec))
    zero_delta = bool(delta_l2_sq <= _ZERO_DELTA_EPS)

    if zero_delta:
        parallel_fraction_sq = 0.0
        orthogonal_fraction_sq = 0.0
        delta_l2_sq = 0.0
        parallel_l2_sq = 0.0
        orthogonal_l2_sq = 0.0
        delta_probe_logit = 0.0
    else:
        parallel_fraction_sq = float(parallel_l2_sq / delta_l2_sq)
        orthogonal_fraction_sq = float(orthogonal_l2_sq / delta_l2_sq)

    reconstruction_error = float(abs(delta_l2_sq - parallel_l2_sq - orthogonal_l2_sq))
    return {
        "delta_probe_logit": delta_probe_logit,
        "delta_l2_sq": delta_l2_sq,
        "parallel_l2_sq": parallel_l2_sq,
        "orthogonal_l2_sq": orthogonal_l2_sq,
        "parallel_fraction_sq": parallel_fraction_sq,
        "orthogonal_fraction_sq": orthogonal_fraction_sq,
        "reconstruction_error": reconstruction_error,
        "zero_delta": zero_delta,
    }


def resolve_paraphrase_artifact_dir(path_text: Optional[str]) -> Optional[Path]:
    normalized = _as_text(path_text)
    if not normalized:
        return None
    path = Path(normalized).expanduser()
    resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if resolved.is_file():
        return resolved.parent
    return resolved


def load_paraphrase_artifact_lookup(path_text: Optional[str]) -> Dict[str, Any]:
    artifact_dir = resolve_paraphrase_artifact_dir(path_text)
    if artifact_dir is None:
        return {
            "artifact_dir": None,
            "manifest_path": None,
            "rows_by_key": {},
            "loaded_files": [],
            "row_count": 0,
        }

    manifest_path = artifact_dir / "paraphrase_manifest.json"
    rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    loaded_files: List[str] = []
    for filename in _PARAPHRASE_DATASET_FILENAMES:
        row_path = artifact_dir / filename
        if not row_path.exists():
            continue
        for row in read_jsonl(str(row_path)):
            dataset = _as_text(row.get("dataset"))
            source_example_id = _as_text(row.get("source_example_id"))
            if dataset and source_example_id:
                rows_by_key[(dataset, source_example_id)] = dict(row)
        loaded_files.append(str(row_path))

    return {
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path if manifest_path.exists() else None,
        "rows_by_key": rows_by_key,
        "loaded_files": loaded_files,
        "row_count": len(rows_by_key),
    }


def _record_base_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    answer_options = _as_text(record.get("answer_options"))
    metadata = {
        "dataset": _as_text(record.get("dataset")),
        "question": _as_text(record.get("question")),
        "question_text": _as_text(record.get("question")),
        "correct_answer": _as_text(record.get("correct_answer")),
        "incorrect_answer": _as_text(record.get("incorrect_answer")),
        "incorrect_answer_source": _as_text(record.get("incorrect_answer_source")),
        "correct_letter": _as_text(record.get("correct_letter")),
        "incorrect_letter": _as_text(record.get("incorrect_letter")),
        "suggested_label": _as_text(record.get("suggested_label")),
        "suggested_answer": _as_text(record.get("suggested_answer")),
        "random_all_variant_family": _as_text(record.get("random_all_variant_family")),
        "letters": _as_text(record.get("letters")),
        "answers": answer_options,
        "answer_options": answer_options,
        "answers_list": list(record.get("answers_list", []) or []),
        "task_format": _as_text(record.get("task_format")),
        "mc_mode": _as_text(record.get("mc_mode")),
        "instruction_policy": _as_text(record.get("instruction_policy")),
        "response_prefix": _as_text(record.get("response_prefix")),
        "answer_channel": _as_text(record.get("answer_channel")),
        "prompt_spec_version": record.get("prompt_spec_version"),
        "grading_spec_version": record.get("grading_spec_version"),
        "source_dataset": _as_text(record.get("source_dataset")),
        "source_split": _as_text(record.get("source_split")),
        "source_example_id": _as_text(record.get("source_example_id")),
        "bias_construction_mode": _as_text(record.get("bias_construction_mode")),
    }
    return metadata


def build_same_family_paraphrase_prompt_messages(
    source_record: Mapping[str, Any],
    paraphrased_stem: str,
) -> Dict[str, Any]:
    template_type = _as_text(source_record.get("template_type"))
    base_metadata = _record_base_metadata(source_record)
    base_metadata["question"] = _as_text(paraphrased_stem)
    base_metadata["question_text"] = _as_text(paraphrased_stem)
    question_text = render_multiple_choice_question(base_metadata)
    question = Question(
        dataset=_as_text(source_record.get("dataset")),
        question_text=question_text,
        correct_answer=_as_text(source_record.get("correct_answer")),
        incorrect_answer=_as_text(source_record.get("incorrect_answer")),
        base_metadata=base_metadata,
    )
    prompt_text = get_agreement_bias(template_type).render_prompt_text(
        question,
        instruction_policy=_as_text(source_record.get("instruction_policy")) or None,
        mc_mode=_as_text(source_record.get("mc_mode")) or None,
    )
    return {
        "prompt_text": prompt_text,
        "prompt_messages": [{"type": "human", "content": prompt_text}],
    }


def _build_target_index(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        index[_pair_key(record)] = dict(record)
    return index


def _movement_row_base(
    *,
    probe_name: str,
    probe_training_template_type: str,
    probe_layer: int,
    source_record: Mapping[str, Any],
    target_change_kind: str,
    target_template_type: str,
    target_record_id: Any,
    forced_response: str,
) -> Dict[str, Any]:
    correctness = _safe_int(source_record.get("correctness"))
    return {
        "probe_name": probe_name,
        "probe_training_template_type": probe_training_template_type,
        "probe_layer": int(probe_layer),
        "split": _as_text(source_record.get("split")),
        "dataset": _as_text(source_record.get("dataset")),
        "question_id": _as_text(source_record.get("question_id")),
        "draw_idx": int(source_record.get("draw_idx", 0) or 0),
        "source_record_id": source_record.get("record_id"),
        "source_template_type": _as_text(source_record.get("template_type")),
        "source_example_id": _as_text(source_record.get("source_example_id")),
        "neutral_source_record_id": source_record.get("neutral_source_record_id"),
        "neutral_source_prompt_id": _as_text(source_record.get("neutral_source_prompt_id")),
        "neutral_source_response": _as_text(source_record.get("neutral_source_response")),
        "neutral_source_correctness": _safe_int(source_record.get("neutral_source_correctness")),
        "neutral_source_is_correct": bool(source_record.get("neutral_source_is_correct", False)),
        "neutral_source_usable_for_metrics": bool(
            source_record.get("neutral_source_usable_for_metrics", False)
        ),
        "neutral_question_total_draws": int(source_record.get("neutral_question_total_draws", 0) or 0),
        "neutral_question_usable_draws": int(source_record.get("neutral_question_usable_draws", 0) or 0),
        "neutral_question_correct_draw_count": int(
            source_record.get("neutral_question_correct_draw_count", 0) or 0
        ),
        "neutral_question_accuracy": source_record.get("neutral_question_accuracy"),
        "neutral_question_any_correct": bool(source_record.get("neutral_question_any_correct", False)),
        "neutral_question_all_correct": bool(source_record.get("neutral_question_all_correct", False)),
        "target_change_kind": target_change_kind,
        "target_template_type": target_template_type,
        "target_record_id": target_record_id,
        "forced_response": forced_response,
        "forced_response_is_correct": None if correctness is None else bool(correctness == 1),
        "source_prompt_id": _as_text(source_record.get("prompt_id")),
        "target_prompt_id": "",
        "cosine_similarity": float("nan"),
        "delta_l2_sq": float("nan"),
        "parallel_l2_sq": float("nan"),
        "orthogonal_l2_sq": float("nan"),
        "parallel_fraction_sq": float("nan"),
        "orthogonal_fraction_sq": float("nan"),
        "random_baseline_parallel_l2_sq": float("nan"),
        "random_baseline_orthogonal_l2_sq": float("nan"),
        "random_baseline_parallel_fraction_sq": float("nan"),
        "random_baseline_orthogonal_fraction_sq": float("nan"),
        "probe_score_source": float("nan"),
        "probe_score_target": float("nan"),
        "delta_probe_score": float("nan"),
        "probe_logit_source": float("nan"),
        "probe_logit_target": float("nan"),
        "delta_probe_logit": float("nan"),
        "zero_delta": False,
        "non_finite_feature": False,
        "missing_target": False,
        "missing_paraphrase": False,
        "invalid_paraphrase": False,
    }


def _build_exclusion_row(
    *,
    source_record: Mapping[str, Any],
    target_change_kind: str,
    target_template_type: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "split": _as_text(source_record.get("split")),
        "dataset": _as_text(source_record.get("dataset")),
        "question_id": _as_text(source_record.get("question_id")),
        "draw_idx": int(source_record.get("draw_idx", 0) or 0),
        "source_record_id": source_record.get("record_id"),
        "source_template_type": _as_text(source_record.get("template_type")),
        "source_example_id": _as_text(source_record.get("source_example_id")),
        "target_change_kind": target_change_kind,
        "target_template_type": target_template_type,
        "reason": reason,
    }


def _build_non_finite_row(
    *,
    row_base: Dict[str, Any],
    target_prompt_id: str,
    source_score: float,
    source_logit: float,
) -> Dict[str, Any]:
    row = dict(row_base)
    row["target_prompt_id"] = target_prompt_id
    row["probe_score_source"] = source_score
    row["probe_logit_source"] = source_logit
    row["non_finite_feature"] = True
    return row


def _build_computed_row(
    *,
    row_base: Dict[str, Any],
    probe_weights: np.ndarray,
    source_feature: np.ndarray,
    target_feature: np.ndarray,
    source_score: float,
    target_score: float,
    source_logit: float,
    target_logit: float,
    target_prompt_id: str,
    random_seed: int,
) -> Dict[str, Any]:
    row = dict(row_base)
    row["target_prompt_id"] = target_prompt_id
    row["probe_score_source"] = source_score
    row["probe_score_target"] = target_score
    row["probe_logit_source"] = source_logit
    row["probe_logit_target"] = target_logit

    delta_vec = np.asarray(target_feature, dtype=float) - np.asarray(source_feature, dtype=float)
    geometry = decompose_probe_delta(delta_vec, probe_weights)
    random_geometry = decompose_probe_delta(
        _random_same_norm_delta(delta_vec, seed=random_seed),
        probe_weights,
    )
    row["cosine_similarity"] = _cosine_similarity(source_feature, target_feature)
    row["delta_l2_sq"] = float(geometry["delta_l2_sq"])
    row["parallel_l2_sq"] = float(geometry["parallel_l2_sq"])
    row["orthogonal_l2_sq"] = float(geometry["orthogonal_l2_sq"])
    row["parallel_fraction_sq"] = float(geometry["parallel_fraction_sq"])
    row["orthogonal_fraction_sq"] = float(geometry["orthogonal_fraction_sq"])
    row["random_baseline_parallel_l2_sq"] = float(random_geometry["parallel_l2_sq"])
    row["random_baseline_orthogonal_l2_sq"] = float(random_geometry["orthogonal_l2_sq"])
    row["random_baseline_parallel_fraction_sq"] = float(random_geometry["parallel_fraction_sq"])
    row["random_baseline_orthogonal_fraction_sq"] = float(random_geometry["orthogonal_fraction_sq"])
    row["delta_probe_logit"] = float(geometry["delta_probe_logit"])
    row["zero_delta"] = bool(geometry["zero_delta"])
    if row["zero_delta"]:
        row["delta_probe_score"] = 0.0
        row["delta_probe_logit"] = 0.0
    else:
        row["delta_probe_score"] = float(target_score - source_score)
    return row


def _summarize_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    probe_name: str,
    probe_training_template_type: str,
    probe_layer: int,
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    summary_rows: List[Dict[str, Any]] = []

    def _append_summary(group_rows: Sequence[Dict[str, Any]], target_change_kind: str, target_template_type: str) -> None:
        if not group_rows:
            return
        finite_rows = [
            row
            for row in group_rows
            if not bool(row.get("non_finite_feature", False))
            and np.isfinite(float(row.get("delta_l2_sq", float("nan"))))
        ]

        def _mean(field: str) -> Optional[float]:
            values = [float(row.get(field, float("nan"))) for row in finite_rows]
            values = [value for value in values if np.isfinite(value)]
            if not values:
                return None
            return float(np.mean(values))

        def _mean_abs(field: str) -> Optional[float]:
            values = [abs(float(row.get(field, float("nan")))) for row in finite_rows]
            values = [value for value in values if np.isfinite(value)]
            if not values:
                return None
            return float(np.mean(values))

        summary_rows.append(
            {
                "probe_name": probe_name,
                "probe_training_template_type": probe_training_template_type,
                "probe_layer": int(probe_layer),
                "target_change_kind": target_change_kind,
                "target_template_type": target_template_type,
                "n_rows": int(len(group_rows)),
                "n_finite_rows": int(len(finite_rows)),
                "n_zero_delta": int(sum(1 for row in group_rows if bool(row.get("zero_delta", False)))),
                "n_questions": int(
                    len(
                        {
                            (_as_text(row.get("split")), _as_text(row.get("question_id")), int(row.get("draw_idx", 0) or 0))
                            for row in group_rows
                        }
                    )
                ),
                "mean_cosine_similarity": _mean("cosine_similarity"),
                "mean_delta_l2_sq": _mean("delta_l2_sq"),
                "mean_parallel_fraction_sq": _mean("parallel_fraction_sq"),
                "mean_orthogonal_fraction_sq": _mean("orthogonal_fraction_sq"),
                "mean_random_baseline_parallel_fraction_sq": _mean("random_baseline_parallel_fraction_sq"),
                "mean_random_baseline_orthogonal_fraction_sq": _mean("random_baseline_orthogonal_fraction_sq"),
                "mean_excess_parallel_fraction_sq": (
                    None
                    if _mean("parallel_fraction_sq") is None or _mean("random_baseline_parallel_fraction_sq") is None
                    else float(_mean("parallel_fraction_sq") - _mean("random_baseline_parallel_fraction_sq"))
                ),
                "mean_excess_orthogonal_fraction_sq": (
                    None
                    if _mean("orthogonal_fraction_sq") is None
                    or _mean("random_baseline_orthogonal_fraction_sq") is None
                    else float(
                        _mean("orthogonal_fraction_sq")
                        - _mean("random_baseline_orthogonal_fraction_sq")
                    )
                ),
                "mean_delta_probe_score": _mean("delta_probe_score"),
                "mean_abs_delta_probe_score": _mean_abs("delta_probe_score"),
                "mean_delta_probe_logit": _mean("delta_probe_logit"),
                "mean_abs_delta_probe_logit": _mean_abs("delta_probe_logit"),
            }
        )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (_as_text(row.get("target_change_kind")), _as_text(row.get("target_template_type")))
        grouped.setdefault(key, []).append(row)

    for target_change_kind, target_template_type in sorted(grouped):
        _append_summary(grouped[(target_change_kind, target_template_type)], target_change_kind, target_template_type)
    _append_summary(list(rows), "overall", "overall")
    return summary_rows


def evaluate_probe_prompt_movement(
    *,
    model,
    tokenizer,
    clf,
    layer: Optional[int],
    probe_name: str,
    probe_training_template_type: str,
    source_test_records: Sequence[Dict[str, Any]],
    cross_family_test_records_by_template: Mapping[str, Sequence[Dict[str, Any]]],
    paraphrase_lookup: Optional[Mapping[Tuple[str, str], Dict[str, Any]]] = None,
    paraphrase_artifact_path: Optional[str] = None,
) -> Dict[str, Any]:
    if clf is None or layer is None:
        coverage = {
            "movement_schema_version": PROBE_MOVEMENT_SCHEMA_VERSION,
            "probe_name": probe_name,
            "probe_training_template_type": probe_training_template_type,
            "probe_layer": layer,
            "source_record_count": 0,
            "computed_row_count": 0,
            "exclusion_counts": {"missing_probe_model": 1},
            "exclusions": [],
            "paraphrase_artifact_path": _as_text(paraphrase_artifact_path),
        }
        return {
            "movement_schema_version": PROBE_MOVEMENT_SCHEMA_VERSION,
            "probe_name": probe_name,
            "probe_training_template_type": probe_training_template_type,
            "probe_layer": layer,
            "rows": [],
            "summary_rows": [],
            "coverage": coverage,
        }

    source_records = sorted(
        [dict(record) for record in source_test_records if record_is_usable_for_metrics(record)],
        key=lambda record: (
            _as_text(record.get("split")),
            _as_text(record.get("question_id")),
            int(record.get("draw_idx", 0) or 0),
            _as_text(record.get("template_type")),
        ),
    )
    weight_vec = _probe_weight_vector(clf)
    cross_family_indices = {
        _as_text(template_type): _build_target_index(records)
        for template_type, records in cross_family_test_records_by_template.items()
        if _as_text(template_type)
    }
    paraphrase_rows_by_key = dict(paraphrase_lookup or {})
    enable_paraphrase = bool(paraphrase_lookup is not None)

    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    source_expected_targets = len(cross_family_indices) + (1 if enable_paraphrase else 0)
    total_expected = len(source_records) * max(1, source_expected_targets)
    progress_iter = source_records
    if tqdm is not None:
        progress_iter = tqdm(
            source_records,
            desc=tqdm_desc(_LOG_SOURCE, f"{probe_name} movement"),
            unit="record",
            total=len(source_records),
        )

    log_status(
        _LOG_SOURCE,
        f"movement eval for {probe_name}: source_records={len(source_records)} "
        f"cross_family_targets={sorted(cross_family_indices)} paraphrase_enabled={enable_paraphrase}",
    )

    for source_record in progress_iter:
        forced_response = _probe_completion_text(source_record)
        if not forced_response:
            exclusions.append(
                _build_exclusion_row(
                    source_record=source_record,
                    target_change_kind="overall",
                    target_template_type="overall",
                    reason="missing_forced_response",
                )
            )
            continue
        if not _has_prompt_messages(source_record):
            exclusions.append(
                _build_exclusion_row(
                    source_record=source_record,
                    target_change_kind="overall",
                    target_template_type="overall",
                    reason="missing_source_prompt_messages",
                )
            )
            continue

        source_feature = _get_hidden_feature_for_completion(
            model,
            tokenizer,
            list(source_record["prompt_messages"]),
            forced_response,
            layer=int(layer),
        )
        if not _is_finite_vector(source_feature):
            exclusions.append(
                _build_exclusion_row(
                    source_record=source_record,
                    target_change_kind="overall",
                    target_template_type="overall",
                    reason="source_non_finite_feature",
                )
            )
            continue

        source_score = _probe_probability(clf, source_feature)
        source_logit = _probe_logit(clf, source_feature)
        source_key = _pair_key(source_record)

        for target_template_type, target_index in cross_family_indices.items():
            target_record = target_index.get(source_key)
            if target_record is None:
                exclusions.append(
                    _build_exclusion_row(
                        source_record=source_record,
                        target_change_kind="prompt_family",
                        target_template_type=target_template_type,
                        reason="missing_target",
                    )
                )
                continue
            if not _has_prompt_messages(target_record):
                exclusions.append(
                    _build_exclusion_row(
                        source_record=source_record,
                        target_change_kind="prompt_family",
                        target_template_type=target_template_type,
                        reason="missing_target_prompt_messages",
                    )
                )
                continue

            row_base = _movement_row_base(
                probe_name=probe_name,
                probe_training_template_type=probe_training_template_type,
                probe_layer=int(layer),
                source_record=source_record,
                target_change_kind="prompt_family",
                target_template_type=target_template_type,
                target_record_id=target_record.get("record_id"),
                forced_response=forced_response,
            )
            target_feature = _get_hidden_feature_for_completion(
                model,
                tokenizer,
                list(target_record["prompt_messages"]),
                forced_response,
                layer=int(layer),
            )
            if not _is_finite_vector(target_feature):
                rows.append(
                    _build_non_finite_row(
                        row_base=row_base,
                        target_prompt_id=_as_text(target_record.get("prompt_id")),
                        source_score=source_score,
                        source_logit=source_logit,
                    )
                )
                continue

            target_score = _probe_probability(clf, target_feature)
            target_logit = _probe_logit(clf, target_feature)
            rows.append(
                _build_computed_row(
                    row_base=row_base,
                    probe_weights=weight_vec,
                    source_feature=source_feature,
                    target_feature=target_feature,
                    source_score=source_score,
                    target_score=target_score,
                    source_logit=source_logit,
                    target_logit=target_logit,
                    target_prompt_id=_as_text(target_record.get("prompt_id")),
                    random_seed=_stable_seed(
                        probe_name,
                        probe_training_template_type,
                        _as_text(source_record.get("split")),
                        _as_text(source_record.get("question_id")),
                        int(source_record.get("draw_idx", 0) or 0),
                        "prompt_family",
                        target_template_type,
                    ),
                )
            )

        if enable_paraphrase:
            paraphrase_key = _source_example_key(source_record)
            paraphrase_row = paraphrase_rows_by_key.get(paraphrase_key)
            if paraphrase_row is None:
                exclusions.append(
                    _build_exclusion_row(
                        source_record=source_record,
                        target_change_kind="paraphrase",
                        target_template_type=_as_text(source_record.get("template_type")),
                        reason="missing_paraphrase",
                    )
                )
            elif _as_text(paraphrase_row.get("status")) != "valid":
                exclusions.append(
                    _build_exclusion_row(
                        source_record=source_record,
                        target_change_kind="paraphrase",
                        target_template_type=_as_text(source_record.get("template_type")),
                        reason="invalid_paraphrase",
                    )
                )
            else:
                paraphrased_stem = _as_text(paraphrase_row.get("paraphrased_stem"))
                if not paraphrased_stem:
                    exclusions.append(
                        _build_exclusion_row(
                            source_record=source_record,
                            target_change_kind="paraphrase",
                            target_template_type=_as_text(source_record.get("template_type")),
                            reason="empty_paraphrase",
                        )
                    )
                else:
                    paraphrase_prompt = build_same_family_paraphrase_prompt_messages(source_record, paraphrased_stem)
                    row_base = _movement_row_base(
                        probe_name=probe_name,
                        probe_training_template_type=probe_training_template_type,
                        probe_layer=int(layer),
                        source_record=source_record,
                        target_change_kind="paraphrase",
                        target_template_type=_as_text(source_record.get("template_type")),
                        target_record_id=f"paraphrase::{source_record.get('record_id')}",
                        forced_response=forced_response,
                    )
                    target_feature = _get_hidden_feature_for_completion(
                        model,
                        tokenizer,
                        list(paraphrase_prompt["prompt_messages"]),
                        forced_response,
                        layer=int(layer),
                    )
                    if not _is_finite_vector(target_feature):
                        rows.append(
                            _build_non_finite_row(
                                row_base=row_base,
                                target_prompt_id=f"paraphrase::{_as_text(source_record.get('prompt_id'))}",
                                source_score=source_score,
                                source_logit=source_logit,
                            )
                        )
                    else:
                        target_score = _probe_probability(clf, target_feature)
                        target_logit = _probe_logit(clf, target_feature)
                        rows.append(
                            _build_computed_row(
                                row_base=row_base,
                                probe_weights=weight_vec,
                                source_feature=source_feature,
                                target_feature=target_feature,
                                source_score=source_score,
                                target_score=target_score,
                                source_logit=source_logit,
                                target_logit=target_logit,
                                target_prompt_id=f"paraphrase::{_as_text(source_record.get('prompt_id'))}",
                                random_seed=_stable_seed(
                                    probe_name,
                                    probe_training_template_type,
                                    _as_text(source_record.get("split")),
                                    _as_text(source_record.get("question_id")),
                                    int(source_record.get("draw_idx", 0) or 0),
                                    "paraphrase",
                                    _as_text(source_record.get("template_type")),
                                ),
                            )
                        )

    rows.sort(
        key=lambda row: (
            _as_text(row.get("split")),
            _as_text(row.get("question_id")),
            int(row.get("draw_idx", 0) or 0),
            _as_text(row.get("target_change_kind")),
            _as_text(row.get("target_template_type")),
        )
    )
    exclusions.sort(
        key=lambda row: (
            _as_text(row.get("split")),
            _as_text(row.get("question_id")),
            int(row.get("draw_idx", 0) or 0),
            _as_text(row.get("target_change_kind")),
            _as_text(row.get("target_template_type")),
            _as_text(row.get("reason")),
        )
    )
    summary_rows = _summarize_rows(
        rows,
        probe_name=probe_name,
        probe_training_template_type=probe_training_template_type,
        probe_layer=int(layer),
    )
    exclusion_counts = dict(sorted(Counter(_as_text(row.get("reason")) for row in exclusions).items()))
    coverage = {
        "movement_schema_version": PROBE_MOVEMENT_SCHEMA_VERSION,
        "probe_name": probe_name,
        "probe_training_template_type": probe_training_template_type,
        "probe_layer": int(layer),
        "source_record_count": int(len(source_records)),
        "expected_comparisons_upper_bound": int(total_expected),
        "computed_row_count": int(len(rows)),
        "summary_row_count": int(len(summary_rows)),
        "exclusion_counts": exclusion_counts,
        "exclusions": exclusions,
        "paraphrase_artifact_path": _as_text(paraphrase_artifact_path),
    }
    if exclusion_counts:
        warn_status(
            _LOG_SOURCE,
            "movement_eval_exclusions",
            f"movement eval for {probe_name} skipped comparisons: {exclusion_counts}",
        )
    return {
        "movement_schema_version": PROBE_MOVEMENT_SCHEMA_VERSION,
        "probe_name": probe_name,
        "probe_training_template_type": probe_training_template_type,
        "probe_layer": int(layer),
        "rows": rows,
        "summary_rows": summary_rows,
        "coverage": coverage,
    }


__all__ = [
    "PROBE_MOVEMENT_SCHEMA_VERSION",
    "build_same_family_paraphrase_prompt_messages",
    "decompose_probe_delta",
    "evaluate_probe_prompt_movement",
    "load_paraphrase_artifact_lookup",
    "resolve_paraphrase_artifact_dir",
]

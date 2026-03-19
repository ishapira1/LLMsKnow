from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from ..constants import MC_MODE_STRICT


def _records_for_template(records: Sequence[Dict[str, Any]], template_type: str) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("template_type") == template_type]


def _strict_mc_choice_labels(record: Dict[str, Any]) -> List[str]:
    if str(record.get("task_format", "") or "") != "multiple_choice":
        return []
    if str(record.get("mc_mode", "") or "") != MC_MODE_STRICT:
        return []
    letters = str(record.get("letters", "") or "").strip().upper()
    if not letters:
        return []
    return [letter for letter in letters if letter.strip()]


def _supports_choice_candidates(record: Dict[str, Any]) -> bool:
    if not _strict_mc_choice_labels(record):
        return False
    return bool(str(record.get("correct_letter", "") or "").strip().upper())


def _choice_probability(record: Dict[str, Any], choice: str, n_choices: int) -> float:
    probabilities = record.get("choice_probabilities", {})
    if isinstance(probabilities, dict):
        try:
            value = float(probabilities.get(choice, float("nan")))
        except Exception:
            value = float("nan")
        if math.isfinite(value) and value >= 0.0:
            return value
    return 0.0 if n_choices <= 0 else 1.0 / float(n_choices)


def _choice_weight(probability: float, weighting: str) -> float:
    if weighting == "uniform":
        return 1.0
    if not math.isfinite(probability) or probability <= 0.0:
        return 1e-6
    return float(probability)


def _resolve_probe_construction(records: Sequence[Dict[str, Any]], probe_construction: str) -> str:
    normalized = str(probe_construction or "auto").strip().lower() or "auto"
    if normalized == "sampled_completions":
        return "sampled_completions"
    supports_candidates = bool(records) and all(_supports_choice_candidates(record) for record in records)
    if normalized == "choice_candidates":
        if records and not supports_candidates:
            raise ValueError(
                "probe_construction=choice_candidates requires strict-MC records with letters and correct_letter."
            )
        return "choice_candidates"
    return "choice_candidates" if supports_candidates else "sampled_completions"


def build_choice_candidate_records(
    records: Sequence[Dict[str, Any]],
    *,
    probe_name: str,
    example_weighting: str,
) -> List[Dict[str, Any]]:
    candidate_records: List[Dict[str, Any]] = []
    fallback_record_id = 0
    for record in records:
        choices = _strict_mc_choice_labels(record)
        correct_letter = str(record.get("correct_letter", "") or "").strip().upper()
        if not choices or not correct_letter:
            continue
        selected_choice = str(record.get("response_raw", "") or "").strip().upper()
        try:
            source_record_id = int(record.get("record_id"))
        except Exception:
            source_record_id = fallback_record_id
            fallback_record_id += 1
        for choice_rank, choice in enumerate(choices):
            probability = _choice_probability(record, choice, len(choices))
            candidate_records.append(
                {
                    **dict(record),
                    "record_id": int(source_record_id * 100 + choice_rank),
                    "source_record_id": source_record_id,
                    "probe_row_type": "choice_candidate",
                    "source_sampling_mode": str(record.get("sampling_mode", "generation") or "generation"),
                    "source_selected_choice": selected_choice,
                    "probe_name": probe_name,
                    "candidate_choice": choice,
                    "candidate_rank": int(choice_rank),
                    "candidate_probability": float(probability),
                    "probe_sample_weight": float(_choice_weight(probability, example_weighting)),
                    "candidate_correctness": int(choice == correct_letter),
                    "candidate_is_selected": bool(choice == selected_choice),
                    "response_raw": choice,
                    "response": choice,
                    "committed_answer": choice,
                    "commitment_kind": "letter",
                    "commitment_source": "teacher_forced_choice_candidate",
                    "correctness": int(choice == correct_letter),
                    "grading_status": "correct" if choice == correct_letter else "incorrect",
                    "grading_reason": "candidate_gold_letter" if choice == correct_letter else "candidate_non_gold_letter",
                    "usable_for_metrics": True,
                }
            )
    return candidate_records


def _build_probe_family(
    *,
    template_type: str,
    desc: str,
    meta_key: str,
    score_key: str,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    all_records: Sequence[Dict[str, Any]],
    probe_construction: str,
    probe_example_weighting: str,
) -> Dict[str, Any]:
    source_train = list(train_records)
    source_val = list(val_records)
    source_test = list(test_records)
    source_all = list(all_records)
    resolved_construction = _resolve_probe_construction(source_all, probe_construction)

    if resolved_construction == "choice_candidates":
        train_probe_records = build_choice_candidate_records(
            source_train,
            probe_name=meta_key,
            example_weighting=probe_example_weighting,
        )
        val_probe_records = build_choice_candidate_records(
            source_val,
            probe_name=meta_key,
            example_weighting=probe_example_weighting,
        )
        test_probe_records = build_choice_candidate_records(
            source_test,
            probe_name=meta_key,
            example_weighting=probe_example_weighting,
        )
        candidate_score_records = build_choice_candidate_records(
            source_all,
            probe_name=meta_key,
            example_weighting=probe_example_weighting,
        )
    else:
        train_probe_records = source_train
        val_probe_records = source_val
        test_probe_records = source_test
        candidate_score_records = []

    return {
        "template_type": template_type,
        "desc": desc,
        "meta_key": meta_key,
        "score_key": score_key,
        "probe_construction": resolved_construction,
        "probe_example_weighting": probe_example_weighting,
        "train_records": train_probe_records,
        "val_records": val_probe_records,
        "test_records": test_probe_records,
        "retrain_records": train_probe_records + val_probe_records,
        "score_records": source_all,
        "candidate_score_records": candidate_score_records,
        "split_records": {
            "train": train_probe_records,
            "val": val_probe_records,
            "test": test_probe_records,
        },
        "source_split_records": {
            "train": source_train,
            "val": source_val,
            "test": source_test,
        },
    }


def build_probe_record_sets(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    all_records: Sequence[Dict[str, Any]],
    bias_types: Sequence[str],
    probe_construction: str = "auto",
    probe_example_weighting: str = "model_probability",
) -> Dict[str, Dict[str, Any]]:
    neutral_train = _records_for_template(train_records, "neutral")
    neutral_val = _records_for_template(val_records, "neutral")
    neutral_test = _records_for_template(test_records, "neutral")
    families: Dict[str, Dict[str, Any]] = {
        "neutral": _build_probe_family(
            template_type="neutral",
            desc="no_bias",
            meta_key="probe_no_bias",
            score_key="probe_x",
            train_records=neutral_train,
            val_records=neutral_val,
            test_records=neutral_test,
            all_records=_records_for_template(all_records, "neutral"),
            probe_construction=probe_construction,
            probe_example_weighting=probe_example_weighting,
        )
    }

    for bias_type in bias_types:
        train_subset = _records_for_template(train_records, bias_type)
        val_subset = _records_for_template(val_records, bias_type)
        test_subset = _records_for_template(test_records, bias_type)
        families[bias_type] = _build_probe_family(
            template_type=bias_type,
            desc=f"bias:{bias_type}",
            meta_key=f"probe_bias_{bias_type}",
            score_key="probe_xprime",
            train_records=train_subset,
            val_records=val_subset,
            test_records=test_subset,
            all_records=_records_for_template(all_records, bias_type),
            probe_construction=probe_construction,
            probe_example_weighting=probe_example_weighting,
        )

    return families


__all__ = ["build_choice_candidate_records", "build_probe_record_sets"]

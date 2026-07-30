from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


COHORT_VERSION = "openai_sycophancy_train_neutral_correct_v1"
MODEL_SNAPSHOT = "gpt-5.4-nano-2026-03-17"
SELECTION_SEED = 5
SOURCE_SPLIT = "train"
TARGET_COUNTS = {
    "commonsense_qa": 1_000,
    "arc_challenge": 959,
}
EXPECTED_AVAILABLE_COUNTS = {
    "commonsense_qa": 6_654,
    "arc_challenge": 959,
}


class CohortError(RuntimeError):
    pass


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CohortError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dumps(dict(row)))
            handle.write("\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "") or ""),
        str(row.get("source_dataset", "") or ""),
        str(row.get("source_split", "") or ""),
        str(row.get("source_example_id", "") or ""),
    )


def _provenance_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            *_identity(row),
            str(row.get("question", "") or ""),
        )
    )


def _selection_rank(row: Mapping[str, Any], *, seed: int) -> str:
    return _sha256_text(f"{int(seed)}|{_provenance_key(row)}")


def _neutral_index(paths: Sequence[Path]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            identity = _identity(row)
            if identity in index:
                raise CohortError(f"Duplicate neutral record for {identity}")
            index[identity] = row
    return index


def _neutral_response_letter(neutral: Mapping[str, Any]) -> str:
    return str(
        neutral.get("response_letter")
        or neutral.get("committed_answer")
        or neutral.get("selected_answer")
        or ""
    )


def _validate_pair(source: Mapping[str, Any], neutral: Mapping[str, Any]) -> None:
    identity = _identity(source)
    if _identity(neutral) != identity:
        raise CohortError(f"Neutral provenance mismatch for {identity}")
    if str(source.get("source_split", "")) != SOURCE_SPLIT:
        raise CohortError(f"Non-train source entered development cohort: {identity}")
    if int(neutral.get("correctness", 0) or 0) != 1:
        raise CohortError(f"Neutral-incorrect source entered development cohort: {identity}")
    if _neutral_response_letter(neutral) != str(source.get("correct_letter", "")):
        raise CohortError(f"Neutral response/gold mismatch for {identity}")
    resolved_model = str(neutral.get("openai_model") or neutral.get("model") or "")
    if resolved_model != MODEL_SNAPSHOT:
        raise CohortError(
            f"Model mismatch for {identity}: {resolved_model!r} != {MODEL_SNAPSHOT!r}"
        )
    exact_fields = (
        "dataset",
        "source_dataset",
        "source_split",
        "source_example_id",
        "question",
        "correct_letter",
        "correct_answer",
        "incorrect_letter",
        "incorrect_option_text",
        "letters",
        "answers_list",
    )
    mismatches = [
        field for field in exact_fields if source.get(field) != neutral.get(field)
    ]
    if mismatches:
        raise CohortError(f"Neutral/source mismatch for {identity}: {mismatches}")
    if str(source.get("neutral_prompt", "")) != str(neutral.get("prompt", "")):
        raise CohortError(f"Neutral prompt mismatch for {identity}")
    prompt = str(neutral.get("prompt", ""))
    if str(neutral.get("prompt_sha256", "")) != _sha256_text(prompt):
        raise CohortError(f"Neutral prompt checksum mismatch for {identity}")
    letters = str(source.get("letters", ""))
    answers = list(source.get("answers_list", []))
    incorrect_letter = str(source.get("incorrect_letter", ""))
    correct_letter = str(source.get("correct_letter", ""))
    if incorrect_letter == correct_letter or incorrect_letter not in letters:
        raise CohortError(f"Invalid incorrect option for {identity}")
    incorrect_index = letters.index(incorrect_letter)
    if (
        incorrect_index >= len(answers)
        or str(answers[incorrect_index]) != str(source.get("incorrect_option_text", ""))
    ):
        raise CohortError(f"Incorrect option text/letter mismatch for {identity}")


def _frozen_row(
    source: Mapping[str, Any],
    neutral: Mapping[str, Any],
    *,
    cohort_order: int,
    seed: int,
) -> Dict[str, Any]:
    row = dict(source)
    row.update(
        {
            "cohort_version": COHORT_VERSION,
            "cohort_order_within_dataset": int(cohort_order),
            "selection_seed": int(seed),
            "selection_rank_sha256": _selection_rank(source, seed=seed),
            "neutral_correctness": 1,
            "neutral_response_letter": _neutral_response_letter(neutral),
            "neutral_response_text": neutral.get("response_text"),
            "neutral_choice_probabilities": neutral.get("choice_probabilities", {}),
            "neutral_choice_probability_correct": neutral.get(
                "choice_probability_correct"
            ),
            "neutral_prompt_sha256": neutral.get("prompt_sha256"),
            "neutral_messages_sha256": neutral.get("messages_sha256"),
            "neutral_openai_request_id": neutral.get("openai_request_id"),
            "neutral_resolved_model": neutral.get("openai_model")
            or neutral.get("model"),
            "neutral_result_source": neutral.get("result_source"),
        }
    )
    return row


def freeze_development_cohort(
    *,
    source_root: Path,
    manifest_path: Path,
    spec_path: Path,
    target_counts: Mapping[str, int] = TARGET_COUNTS,
    expected_available_counts: Mapping[str, int] | None = EXPECTED_AVAILABLE_COUNTS,
    seed: int = SELECTION_SEED,
) -> Dict[str, Any]:
    selected_path = source_root / "selected_questions.jsonl"
    neutral_paths = (
        source_root / "reused_neutral_records.jsonl",
        source_root / "records" / "neutral_results.jsonl",
    )
    required_paths = (selected_path, *neutral_paths, source_root / "experiment_config.json")
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise CohortError(f"Missing source artifacts: {missing}")

    neutral_by_identity = _neutral_index(neutral_paths)
    eligible: Dict[str, list[Dict[str, Any]]] = {
        dataset: [] for dataset in target_counts
    }
    for source in _read_jsonl(selected_path):
        dataset = str(source.get("dataset", "") or "")
        if dataset not in eligible or str(source.get("source_split", "")) != SOURCE_SPLIT:
            continue
        neutral = neutral_by_identity.get(_identity(source))
        if neutral is None:
            raise CohortError(f"Missing neutral record for {_identity(source)}")
        _validate_pair(source, neutral)
        eligible[dataset].append(source)

    available_counts = {dataset: len(rows) for dataset, rows in eligible.items()}
    if expected_available_counts is not None:
        expected = {key: int(value) for key, value in expected_available_counts.items()}
        if available_counts != expected:
            raise CohortError(
                f"Eligible train counts changed: {available_counts} != {expected}"
            )

    frozen: list[Dict[str, Any]] = []
    for dataset in target_counts:
        rows = sorted(
            eligible[dataset],
            key=lambda row: _selection_rank(row, seed=seed),
        )
        target = int(target_counts[dataset])
        if len(rows) < target:
            raise CohortError(
                f"{dataset} has only {len(rows)} eligible train questions; target is {target}"
            )
        for order, source in enumerate(rows[:target], start=1):
            neutral = neutral_by_identity[_identity(source)]
            frozen.append(
                _frozen_row(source, neutral, cohort_order=order, seed=seed)
            )

    identities = [_identity(row) for row in frozen]
    if len(identities) != len(set(identities)):
        raise CohortError("Frozen cohort contains duplicate source identities")

    _write_jsonl(manifest_path, frozen)
    identity_digest = _sha256_text(
        "\n".join("|".join(identity) for identity in identities) + "\n"
    )
    selected_counts = dict(Counter(str(row["dataset"]) for row in frozen))
    experiment_config_path = source_root / "experiment_config.json"
    experiment_config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
    spec: Dict[str, Any] = {
        "cohort_version": COHORT_VERSION,
        "status": "frozen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Fixed development cohort for paired OpenAI API sycophancy experiments; "
            "only neutral-correct train questions are eligible."
        ),
        "model_snapshot": MODEL_SNAPSHOT,
        "source_split": SOURCE_SPLIT,
        "selection_seed": int(seed),
        "selection_policy": (
            "Rank neutral-correct train questions by SHA256(seed|provenance_key); "
            "take 1,000 CommonsenseQA and all 959 eligible ARC-Challenge questions."
        ),
        "available_neutral_correct_train_questions": available_counts,
        "selected_questions_by_dataset": selected_counts,
        "selected_questions_total": len(frozen),
        "source_experiment_root": str(source_root.resolve()),
        "source_experiment_config_sha256": _sha256_file(experiment_config_path),
        "request_settings": experiment_config.get("request_settings", {}),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "question_identity_sha256": identity_digest,
        "reuse_requirements": [
            "exact model snapshot",
            "exact neutral prompt and answer-only instruction",
            "exact API settings",
            "exact source identity and answer options",
            "exact question-specific incorrect option",
        ],
    }
    _write_json(spec_path, spec)
    audit_development_cohort(manifest_path=manifest_path, spec_path=spec_path)
    return spec


def audit_development_cohort(
    *,
    manifest_path: Path,
    spec_path: Path,
) -> Dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(manifest_path)
    if spec.get("status") != "frozen":
        raise CohortError("Cohort spec is not frozen")
    if spec.get("cohort_version") != COHORT_VERSION:
        raise CohortError("Unexpected cohort version")
    if spec.get("model_snapshot") != MODEL_SNAPSHOT:
        raise CohortError("Unexpected model snapshot")
    if spec.get("manifest_sha256") != _sha256_file(manifest_path):
        raise CohortError("Manifest checksum mismatch")
    counts = dict(Counter(str(row.get("dataset", "")) for row in rows))
    expected_counts = {
        key: int(value)
        for key, value in spec.get("selected_questions_by_dataset", {}).items()
    }
    if counts != expected_counts or counts != TARGET_COUNTS:
        raise CohortError(f"Frozen cohort counts are invalid: {counts}")
    if any(str(row.get("source_split", "")) != SOURCE_SPLIT for row in rows):
        raise CohortError("Frozen cohort contains a non-train question")
    if any(int(row.get("neutral_correctness", 0) or 0) != 1 for row in rows):
        raise CohortError("Frozen cohort contains a neutral-incorrect question")
    if any(
        str(row.get("neutral_resolved_model", "")) != MODEL_SNAPSHOT for row in rows
    ):
        raise CohortError("Frozen cohort contains a mismatched model snapshot")
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise CohortError("Frozen cohort contains duplicate source identities")
    identity_digest = _sha256_text(
        "\n".join("|".join(identity) for identity in identities) + "\n"
    )
    if spec.get("question_identity_sha256") != identity_digest:
        raise CohortError("Question identity checksum mismatch")
    return {
        "status": "passed",
        "cohort_version": COHORT_VERSION,
        "selected_questions_by_dataset": counts,
        "selected_questions_total": len(rows),
        "manifest_sha256": spec["manifest_sha256"],
        "question_identity_sha256": identity_digest,
    }


__all__ = [
    "COHORT_VERSION",
    "CohortError",
    "EXPECTED_AVAILABLE_COUNTS",
    "MODEL_SNAPSHOT",
    "SELECTION_SEED",
    "SOURCE_SPLIT",
    "TARGET_COUNTS",
    "audit_development_cohort",
    "freeze_development_cohort",
]

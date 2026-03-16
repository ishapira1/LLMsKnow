from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm.auto import tqdm

from .correctness import (
    extract_gold_answers_from_base as _extract_gold_answers_from_base,
    grade_response_from_base as _grade_response_from_base,
    record_is_usable_for_metrics as _record_is_usable_for_metrics,
)
from .constants import SAMPLING_SPEC_VERSION
from .dataset import as_prompt_text, dataset_name as _dataset_name
from .logging_utils import log_status, tqdm_desc
from .model_utils import generate_many as _generate_many
from .runtime import model_slug


def sample_record_key_values(
    split: str,
    question_id: str,
    template_type: str,
    draw_idx: int,
) -> Tuple[str, str, str, int]:
    return (str(split), str(question_id), str(template_type), int(draw_idx))


def sample_record_key(record: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return sample_record_key_values(
        split=str(record.get("split", "")),
        question_id=str(record.get("question_id", "")),
        template_type=str(record.get("template_type", "")),
        draw_idx=int(record.get("draw_idx", 0)),
    )


def sort_sample_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(records),
        key=lambda record: (
            str(record.get("split", "")),
            str(record.get("question_id", "")),
            str(record.get("template_type", "")),
            int(record.get("draw_idx", 0)),
        ),
    )


def enumerate_expected_sample_keys(
    groups: Sequence[Dict[str, Any]],
    split_name: str,
    bias_types: Sequence[str],
    n_draws: int,
) -> Set[Tuple[str, str, str, int]]:
    keys: Set[Tuple[str, str, str, int]] = set()
    wanted_types = ["neutral"] + list(bias_types)
    for group in groups:
        base_row = group["rows_by_type"]["neutral"]
        base = base_row.get("base", {}) or {}
        gold_answers = _extract_gold_answers_from_base(base)
        if not gold_answers:
            continue

        for template_type in wanted_types:
            row = group["rows_by_type"][template_type]
            prompt_messages = row.get("prompt", [])
            if not isinstance(prompt_messages, list) or not prompt_messages:
                continue
            for draw_idx in range(n_draws):
                keys.add(sample_record_key_values(split_name, group["question_id"], template_type, draw_idx))
    return keys


def normalize_sample_records(
    records: Sequence[Dict[str, Any]],
    expected_keys: Set[Tuple[str, str, str, int]],
) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            key = sample_record_key(record)
        except Exception:
            continue
        if key not in expected_keys:
            continue
        by_key[key] = record
    return sort_sample_records(list(by_key.values()))


def refresh_sample_records_for_groups(
    records: Sequence[Dict[str, Any]],
    groups: Sequence[Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    group_by_question_id = {
        str(group.get("question_id", "")): group
        for group in groups
    }
    refreshed: List[Dict[str, Any]] = []

    for record in records:
        refreshed_record = dict(record)
        group = group_by_question_id.get(str(record.get("question_id", "")))
        template_type = str(record.get("template_type", ""))
        row = None if group is None else group.get("rows_by_type", {}).get(template_type)
        if group is None or row is None:
            refreshed_record["split"] = split_name
            refreshed.append(refreshed_record)
            continue

        prompt_messages = row.get("prompt", [])
        prompt_text = as_prompt_text(prompt_messages)
        prompt_template = (row.get("metadata", {}) or {}).get("prompt_template", "")
        base = row.get("base", {}) or {}
        gold_answers = _extract_gold_answers_from_base(base)
        grading = _grade_response_from_base(str(record.get("response_raw", "")), base)

        refreshed_record.update(
            {
                "split": split_name,
                "question_id": group["question_id"],
                "dataset": str(group.get("dataset", "") or _dataset_name(row)),
                "template_type": template_type,
                "task_format": str(base.get("task_format", "") or ""),
                "correct_letter": str(base.get("correct_letter", "") or ""),
                "incorrect_letter": str(base.get("incorrect_letter", "") or ""),
                "letters": str(base.get("letters", "") or ""),
                "answer_options": str(base.get("answers", "") or ""),
                "answers_list": list(base.get("answers_list", []) or []),
                "prompt_messages": prompt_messages,
                "prompt_text": prompt_text,
                "prompt_template": prompt_template,
                "question": group["question"],
                "correct_answer": group["correct_answer"],
                "incorrect_answer": group["incorrect_answer"],
                "incorrect_answer_source": str(base.get("incorrect_answer_source", "") or ""),
                "gold_answers": gold_answers,
                "response": grading["parsed_answer"],
                "correctness": grading["correctness"],
                "grading_status": grading["status"],
                "grading_reason": grading["reason"],
                "usable_for_metrics": grading["usable_for_metrics"],
            }
        )
        refreshed.append(refreshed_record)

    return sort_sample_records(refreshed)


def build_sampling_spec(
    args: argparse.Namespace,
    bias_types: Sequence[str],
    train_groups: Sequence[Dict[str, Any]],
    test_groups: Sequence[Dict[str, Any]],
    expected_train: int,
    expected_test: int,
    *,
    val_groups: Optional[Sequence[Dict[str, Any]]] = None,
    expected_val: int = 0,
) -> Dict[str, Any]:
    val_groups = list(val_groups or [])
    return {
        "sampling_spec_version": int(SAMPLING_SPEC_VERSION),
        "model": args.model,
        "benchmark_source": str(getattr(args, "benchmark_source", "answer_json") or "answer_json"),
        "input_jsonl": args.input_jsonl,
        "dataset_name": str(getattr(args, "dataset_name", "all") or "all"),
        "ays_mc_datasets": list(getattr(args, "ays_mc_datasets", []))
        if isinstance(getattr(args, "ays_mc_datasets", []), list)
        else [x.strip() for x in str(getattr(args, "ays_mc_datasets", "")).split(",") if x.strip()],
        "sycophancy_repo": args.sycophancy_repo,
        "bias_types": list(bias_types),
        "seed": int(getattr(args, "seed", 0)),
        "n_draws": int(args.n_draws),
        "sample_batch_size": int(getattr(args, "sample_batch_size", 1)),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "test_frac": float(args.test_frac),
        "probe_val_frac": float(getattr(args, "probe_val_frac", 0.0)),
        "split_seed": int(args.split_seed),
        "max_questions": args.max_questions,
        "smoke_test": bool(args.smoke_test),
        "smoke_questions": int(args.smoke_questions),
        "train_question_ids": [group["question_id"] for group in train_groups],
        "val_question_ids": [group["question_id"] for group in val_groups],
        "test_question_ids": [group["question_id"] for group in test_groups],
        "expected_train_records": int(expected_train),
        "expected_val_records": int(expected_val),
        "expected_test_records": int(expected_test),
    }


def sampling_spec_hash(spec: Dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_sampling_cache_candidate(
    out_dir: str,
    model_name: str,
    sampling_hash: str,
    exclude_run_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    model_dir = Path(out_dir) / model_slug(model_name)
    if not model_dir.exists():
        return None

    candidates: List[Tuple[int, int, float, Path, Path, Dict[str, Any]]] = []
    for run_subdir in model_dir.iterdir():
        if not run_subdir.is_dir():
            continue
        if exclude_run_dir is not None and run_subdir.resolve() == exclude_run_dir.resolve():
            continue
        manifest_path = run_subdir / "sampling_manifest.json"
        records_path = run_subdir / "sampling_records.jsonl"
        if not manifest_path.exists() or not records_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("sampling_hash", "")) != sampling_hash:
            continue

        complete = 1 if bool(manifest.get("is_complete", False)) else 0
        n_records = int(manifest.get("n_records", 0))
        mtime = manifest_path.stat().st_mtime
        candidates.append((complete, n_records, mtime, run_subdir, records_path, manifest))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, n_records, _, run_dir, records_path, manifest = candidates[0]
    return {
        "run_dir": run_dir,
        "records_path": records_path,
        "manifest": manifest,
        "n_records": n_records,
    }


def sample_records_for_groups(
    model,
    tokenizer,
    groups: Sequence[Dict[str, Any]],
    split_name: str,
    bias_types: Sequence[str],
    n_draws: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    sample_batch_size: int,
    existing_records: Optional[Sequence[Dict[str, Any]]] = None,
    checkpoint_every: int = 0,
    progress_callback: Optional[Callable[[List[Dict[str, Any]], Dict[str, int]], None]] = None,
    start_id: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records_by_key: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    max_existing_record_id = start_id - 1
    for record in existing_records or []:
        if not isinstance(record, dict):
            continue
        try:
            key = sample_record_key(record)
        except Exception:
            continue
        if key[0] != split_name:
            continue
        records_by_key[key] = record
        try:
            max_existing_record_id = max(max_existing_record_id, int(record.get("record_id", -1)))
        except Exception:
            pass

    rec_id = max(start_id, max_existing_record_id + 1)
    reused = 0
    generated = 0
    expected_total = 0
    generated_since_checkpoint = 0
    wanted_types = ["neutral"] + list(bias_types)
    log_status(
        "sampling.py",
        f"sampling split={split_name}: questions={len(groups)} existing_records={len(records_by_key)} "
        f"n_draws={n_draws} batch_size={sample_batch_size}",
    )

    for group in tqdm(
        groups,
        desc=tqdm_desc("sampling.py", f"sampling {split_name} split"),
        unit="question",
    ):
        base_row = group["rows_by_type"]["neutral"]
        base = base_row.get("base", {}) or {}
        gold_answers = _extract_gold_answers_from_base(base)
        if not gold_answers:
            continue

        for template_type in wanted_types:
            row = group["rows_by_type"][template_type]
            prompt_messages = row.get("prompt", [])
            if not isinstance(prompt_messages, list) or not prompt_messages:
                continue

            dataset = str(group.get("dataset", "") or _dataset_name(row))
            prompt_text = as_prompt_text(prompt_messages)
            prompt_template = (row.get("metadata", {}) or {}).get("prompt_template", "")
            task_format = str(base.get("task_format", "") or "")
            correct_letter = str(base.get("correct_letter", "") or "")
            incorrect_letter = str(base.get("incorrect_letter", "") or "")
            letters = str(base.get("letters", "") or "")
            answer_options = str(base.get("answers", "") or "")
            answers_list = list(base.get("answers_list", []) or [])
            missing_draws: List[int] = []
            for draw_idx in range(n_draws):
                key = sample_record_key_values(split_name, group["question_id"], template_type, draw_idx)
                expected_total += 1
                if key in records_by_key:
                    reused += 1
                else:
                    missing_draws.append(draw_idx)

            if not missing_draws:
                continue

            batch_size = max(1, min(sample_batch_size, len(missing_draws)))
            generated_outputs = _generate_many(
                model,
                tokenizer,
                prompt_messages,
                n=len(missing_draws),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
                safe_fallback=True,
            )
            for draw_idx, response_raw in zip(missing_draws, generated_outputs):
                grading = _grade_response_from_base(response_raw, base)

                key = sample_record_key_values(split_name, group["question_id"], template_type, draw_idx)
                records_by_key[key] = {
                    "record_id": rec_id,
                    "question_id": group["question_id"],
                    "split": split_name,
                    "dataset": dataset,
                    "template_type": template_type,
                    "task_format": task_format,
                    "correct_letter": correct_letter,
                    "incorrect_letter": incorrect_letter,
                    "letters": letters,
                    "answer_options": answer_options,
                    "answers_list": answers_list,
                    "prompt_messages": prompt_messages,
                    "prompt_text": prompt_text,
                    "prompt_template": prompt_template,
                    "question": group["question"],
                    "correct_answer": group["correct_answer"],
                    "incorrect_answer": group["incorrect_answer"],
                    "incorrect_answer_source": str(base.get("incorrect_answer_source", "") or ""),
                    "gold_answers": gold_answers,
                    "draw_idx": draw_idx,
                    "response_raw": response_raw,
                    "response": grading["parsed_answer"],
                    "correctness": grading["correctness"],
                    "grading_status": grading["status"],
                    "grading_reason": grading["reason"],
                    "usable_for_metrics": grading["usable_for_metrics"],
                }
                rec_id += 1
                generated += 1
                generated_since_checkpoint += 1

                if (
                    progress_callback is not None
                    and checkpoint_every > 0
                    and generated_since_checkpoint >= checkpoint_every
                ):
                    progress_callback(
                        sort_sample_records(records_by_key.values()),
                        {
                            "split": split_name,
                            "expected_records": expected_total,
                            "reused_records": reused,
                            "generated_records": generated,
                            "total_records": len(records_by_key),
                        },
                    )
                    generated_since_checkpoint = 0

    out_records = sort_sample_records(records_by_key.values())
    stats = {
        "split": split_name,
        "expected_records": expected_total,
        "reused_records": reused,
        "generated_records": generated,
        "total_records": len(out_records),
    }
    if progress_callback is not None:
        progress_callback(out_records, stats)
    remaining = max(0, expected_total - len(out_records))
    coverage = 0.0 if expected_total <= 0 else len(out_records) / expected_total
    log_status(
        "sampling.py",
        f"completed split={split_name}: total_records={len(out_records)}/{expected_total} "
        f"coverage={coverage:.1%} reused={reused} generated={generated} remaining={remaining}",
    )
    return out_records, stats


def add_empirical_t(records: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str, str], List[int]] = {}
    for record in records:
        if not _record_is_usable_for_metrics(record):
            continue
        key = (record["split"], record["question_id"], record["template_type"])
        grouped.setdefault(key, []).append(int(record["correctness"]))
    tvals = {key: float(np.mean(values)) for key, values in grouped.items()}
    for record in records:
        record["T_prompt"] = tvals.get(
            (record["split"], record["question_id"], record["template_type"]),
            float("nan"),
        )

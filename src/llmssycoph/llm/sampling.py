from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tqdm.auto import tqdm

from ..constants import SAMPLING_SPEC_VERSION
from ..data import as_prompt_text, dataset_name as _dataset_name, prompt_id_for
from ..grading import extract_gold_answers_from_base as _extract_gold_answers_from_base
from ..grading import grade_response_from_base as _grade_response_from_base
from ..logging_utils import log_status, tqdm_desc
from ..runtime import model_slug, resolve_run_artifact_path
from .base import GenerationResult


def _generation_record_from_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, GenerationResult):
        result = output.as_dict()
        result.setdefault("sampling_mode", "generation")
        result.setdefault("choice_probabilities", {})
        result.setdefault("choice_probability_correct", float("nan"))
        result.setdefault("choice_probability_selected", float("nan"))
        return result
    if isinstance(output, dict):
        return {
            "response_raw": str(output.get("response_raw", "") or ""),
            "completion_token_count": output.get("completion_token_count"),
            "hit_max_new_tokens": bool(output.get("hit_max_new_tokens", False)),
            "stopped_on_eos": bool(output.get("stopped_on_eos", False)),
            "finish_reason": str(output.get("finish_reason", "") or ""),
            "sampling_mode": str(output.get("sampling_mode", "generation") or "generation"),
            "choice_probabilities": dict(output.get("choice_probabilities", {}) or {}),
            "choice_probability_correct": output.get("choice_probability_correct", float("nan")),
            "choice_probability_selected": output.get("choice_probability_selected", float("nan")),
        }
    return {
        "response_raw": str(output or ""),
        "completion_token_count": None,
        "hit_max_new_tokens": False,
        "stopped_on_eos": False,
        "finish_reason": "",
        "sampling_mode": "generation",
        "choice_probabilities": {},
        "choice_probability_correct": float("nan"),
        "choice_probability_selected": float("nan"),
    }


def _llm_supports_choice_scoring(llm: Any) -> bool:
    capabilities_fn = getattr(llm, "capabilities", None)
    if not callable(capabilities_fn):
        return True
    try:
        capabilities = capabilities_fn()
    except Exception:
        return False
    return bool(getattr(capabilities, "supports_choice_scoring", False))


def _llm_backend_name(llm: Any) -> str:
    capabilities_fn = getattr(llm, "capabilities", None)
    if not callable(capabilities_fn):
        return ""
    try:
        capabilities = capabilities_fn()
    except Exception:
        return ""
    return str(getattr(capabilities, "backend_name", "") or "")


def _materialize_sample_record(
    task: Dict[str, Any],
    *,
    draw_idx: int,
    record_id: int,
    generation_record: Dict[str, Any],
    grading: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "question_id": task["question_id"],
        "prompt_id": prompt_id_for(task["question_id"], task["template_type"]),
        "split": task["split_name"],
        "dataset": task["dataset"],
        "template_type": task["template_type"],
        "task_format": task["task_format"],
        "mc_mode": task["mc_mode"],
        "answer_channel": task["answer_channel"],
        "prompt_spec_version": task["prompt_spec_version"],
        "grading_spec_version": grading.get("grading_spec_version", task["grading_spec_version"]),
        "correct_letter": task["correct_letter"],
        "incorrect_letter": task["incorrect_letter"],
        "letters": task["letters"],
        "answer_options": task["answer_options"],
        "answers_list": task["answers_list"],
        "prompt_messages": task["prompt_messages"],
        "prompt_text": task["prompt_text"],
        "prompt_template": task["prompt_template"],
        "question": task["question"],
        "correct_answer": task["correct_answer"],
        "incorrect_answer": task["incorrect_answer"],
        "incorrect_answer_source": task["incorrect_answer_source"],
        "gold_answers": task["gold_answers"],
        "draw_idx": draw_idx,
        "response_raw": generation_record["response_raw"],
        "response": grading["parsed_answer"],
        "committed_answer": grading.get("committed_answer", ""),
        "commitment_kind": grading.get("commitment_kind", ""),
        "commitment_source": grading.get("commitment_source", ""),
        "starts_with_answer_prefix": bool(grading.get("starts_with_answer_prefix", False)),
        "strict_format_exact": bool(grading.get("strict_format_exact", False)),
        "commitment_line": grading.get("commitment_line", ""),
        "answer_marker_count": int(grading.get("answer_marker_count", 0) or 0),
        "multiple_answer_markers": bool(grading.get("multiple_answer_markers", False)),
        "correctness": grading["correctness"],
        "grading_status": grading["status"],
        "grading_reason": grading["reason"],
        "usable_for_metrics": grading["usable_for_metrics"],
        "completion_token_count": generation_record.get("completion_token_count"),
        "hit_max_new_tokens": generation_record.get("hit_max_new_tokens", False),
        "stopped_on_eos": generation_record.get("stopped_on_eos", False),
        "finish_reason": generation_record.get("finish_reason", ""),
        "sampling_mode": generation_record.get("sampling_mode", "generation"),
        "choice_probabilities": generation_record.get("choice_probabilities", {}),
        "choice_probability_correct": generation_record.get("choice_probability_correct", float("nan")),
        "choice_probability_selected": generation_record.get("choice_probability_selected", float("nan")),
    }


def _execute_sampling_task(
    llm: Any,
    task: Dict[str, Any],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    sample_batch_size: int,
) -> List[Dict[str, Any]]:
    missing_draws = list(task["missing_draws"])
    batch_size = max(1, min(sample_batch_size, len(missing_draws)))
    if task["choice_labels"] and _llm_supports_choice_scoring(llm):
        choice_probabilities = llm.score_choices(task["prompt_messages"], task["choice_labels"])
        selected_choice = max(
            task["choice_labels"],
            key=lambda choice: (float(choice_probabilities.get(choice, float("-inf"))), -task["choice_labels"].index(choice)),
        )
        generated_outputs = [
            {
                "response_raw": selected_choice,
                "completion_token_count": 1,
                "hit_max_new_tokens": False,
                "stopped_on_eos": False,
                "finish_reason": "choice_probabilities",
                "sampling_mode": "choice_probabilities",
                "choice_probabilities": choice_probabilities,
                "choice_probability_correct": float(
                    choice_probabilities.get(str(task["correct_letter"] or "").strip().upper(), 0.0)
                ),
                "choice_probability_selected": float(choice_probabilities.get(selected_choice, 0.0)),
            }
            for _ in missing_draws
        ]
    else:
        generated_outputs = llm.generate(
            task["prompt_messages"],
            n=len(missing_draws),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
            safe_fallback=True,
            strict_mc_letters=task["strict_mc_letters"],
        )

    outputs: List[Dict[str, Any]] = []
    for draw_idx, record_id, generated_output in zip(missing_draws, task["record_ids"], generated_outputs):
        generation_record = _generation_record_from_output(generated_output)
        grading = _grade_response_from_base(
            generation_record["response_raw"],
            task["base"],
            generation_info=generation_record,
        )
        outputs.append(
            {
                "draw_idx": draw_idx,
                "record_id": record_id,
                "generation_record": generation_record,
                "grading": grading,
            }
        )
    return outputs


def _strict_mc_choice_labels(base: Dict[str, Any]) -> List[str]:
    if str(base.get("task_format", "") or "") != "multiple_choice":
        return []
    if str(base.get("mc_mode", "") or "") != "strict_mc":
        return []
    letters = str(base.get("letters", "") or "").strip().upper()
    if not letters:
        return []
    return [letter for letter in letters if letter.strip()]


def _planned_draw_count(base: Dict[str, Any], default_n_draws: int) -> int:
    if _strict_mc_choice_labels(base):
        return 1
    return int(default_n_draws)


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
            planned_n_draws = _planned_draw_count(base, n_draws)
            for draw_idx in range(planned_n_draws):
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
        "model_backend": str(getattr(args, "model_backend", "huggingface") or "huggingface"),
        "benchmark_source": str(getattr(args, "benchmark_source", "answer_json") or "answer_json"),
        "mc_mode": str(getattr(args, "mc_mode", "") or ""),
        "prompt_spec_version": int(getattr(args, "prompt_spec_version", 0) or 0),
        "grading_spec_version": int(getattr(args, "grading_spec_version", 0) or 0),
        "generation_spec_version": int(getattr(args, "generation_spec_version", 0) or 0),
        "input_jsonl": args.input_jsonl,
        "dataset_name": str(getattr(args, "dataset_name", "all") or "all"),
        "ays_mc_datasets": list(getattr(args, "ays_mc_datasets", []))
        if isinstance(getattr(args, "ays_mc_datasets", []), list)
        else [x.strip() for x in str(getattr(args, "ays_mc_datasets", "")).split(",") if x.strip()],
        "sycophancy_repo": args.sycophancy_repo,
        "bias_types": list(bias_types),
        "seed": int(getattr(args, "seed", 0)),
        "n_draws": int(args.n_draws),
        "requested_n_draws": int(getattr(args, "requested_n_draws", args.n_draws)),
        "strict_mc_choice_scoring": bool(getattr(args, "strict_mc_choice_scoring", False)),
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
        manifest_path = resolve_run_artifact_path(run_subdir, "sampling_manifest")
        records_path = resolve_run_artifact_path(run_subdir, "sampling_records")
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
    llm,
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
        normalized_record = dict(record)
        normalized_record.setdefault(
            "prompt_id",
            prompt_id_for(normalized_record.get("question_id", ""), normalized_record.get("template_type", "")),
        )
        try:
            key = sample_record_key(normalized_record)
        except Exception:
            continue
        if key[0] != split_name:
            continue
        records_by_key[key] = normalized_record
        try:
            max_existing_record_id = max(max_existing_record_id, int(normalized_record.get("record_id", -1)))
        except Exception:
            pass

    rec_id = max(start_id, max_existing_record_id + 1)
    reused = 0
    generated = 0
    expected_total = 0
    generated_since_checkpoint = 0
    wanted_types = ["neutral"] + list(bias_types)
    use_openai_parallelism = _llm_backend_name(llm) == "openai" and int(sample_batch_size) > 1
    log_status(
        "sampling.py",
        f"sampling split={split_name}: questions={len(groups)} existing_records={len(records_by_key)} "
        f"n_draws={n_draws} batch_size={sample_batch_size}",
    )
    pending_tasks: List[Dict[str, Any]] = []

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

            dataset = str(group.get("dataset", "") or _dataset_name(row))
            prompt_text = as_prompt_text(prompt_messages)
            prompt_template = (row.get("metadata", {}) or {}).get("prompt_template", "")
            task_format = str(base.get("task_format", "") or "")
            mc_mode = str(base.get("mc_mode", "") or "")
            answer_channel = str(base.get("answer_channel", "") or "")
            prompt_spec_version = base.get("prompt_spec_version")
            grading_spec_version = base.get("grading_spec_version")
            correct_letter = str(base.get("correct_letter", "") or "")
            incorrect_letter = str(base.get("incorrect_letter", "") or "")
            letters = str(base.get("letters", "") or "")
            answer_options = str(base.get("answers", "") or "")
            answers_list = list(base.get("answers_list", []) or [])
            strict_mc_letters = letters if task_format == "multiple_choice" and mc_mode == "strict_mc" else ""
            choice_labels = _strict_mc_choice_labels(base)
            planned_n_draws = _planned_draw_count(base, n_draws)
            missing_draws: List[int] = []
            for draw_idx in range(planned_n_draws):
                key = sample_record_key_values(split_name, group["question_id"], template_type, draw_idx)
                expected_total += 1
                if key in records_by_key:
                    reused += 1
                else:
                    missing_draws.append(draw_idx)

            if not missing_draws:
                continue
            record_ids = list(range(rec_id, rec_id + len(missing_draws)))
            rec_id += len(missing_draws)
            pending_tasks.append(
                {
                    "split_name": split_name,
                    "question_id": group["question_id"],
                    "question": group["question"],
                    "correct_answer": group["correct_answer"],
                    "incorrect_answer": group["incorrect_answer"],
                    "template_type": template_type,
                    "base": base,
                    "dataset": dataset,
                    "prompt_messages": prompt_messages,
                    "prompt_text": prompt_text,
                    "prompt_template": prompt_template,
                    "task_format": task_format,
                    "mc_mode": mc_mode,
                    "answer_channel": answer_channel,
                    "prompt_spec_version": prompt_spec_version,
                    "grading_spec_version": grading_spec_version,
                    "correct_letter": correct_letter,
                    "incorrect_letter": incorrect_letter,
                    "letters": letters,
                    "answer_options": answer_options,
                    "answers_list": answers_list,
                    "strict_mc_letters": strict_mc_letters,
                    "choice_labels": choice_labels,
                    "gold_answers": gold_answers,
                    "incorrect_answer_source": str(base.get("incorrect_answer_source", "") or ""),
                    "missing_draws": list(missing_draws),
                    "record_ids": record_ids,
                }
            )

    progress_desc = tqdm_desc("sampling.py", f"sampling {split_name} split")

    def _checkpoint_if_needed() -> None:
        nonlocal generated_since_checkpoint
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

    if use_openai_parallelism and pending_tasks:
        log_status(
            "sampling.py",
            f"OpenAI parallel sampling enabled for split={split_name} with max_workers={sample_batch_size}",
        )
        with ThreadPoolExecutor(max_workers=int(sample_batch_size), thread_name_prefix="openai-sampling") as executor:
            task_iter = iter(pending_tasks)
            future_to_task: Dict[Future[List[Dict[str, Any]]], Dict[str, Any]] = {}

            def _submit_next() -> bool:
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _execute_sampling_task,
                    llm,
                    task,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    sample_batch_size=sample_batch_size,
                )
                future_to_task[future] = task
                return True

            initial = min(int(sample_batch_size), len(pending_tasks))
            for _ in range(initial):
                _submit_next()

            with tqdm(total=len(pending_tasks), desc=progress_desc, unit="prompt") as task_bar:
                while future_to_task:
                    done, _ = wait(list(future_to_task), return_when=FIRST_COMPLETED)
                    for future in done:
                        task = future_to_task.pop(future)
                        task_outputs = future.result()
                        for output in task_outputs:
                            draw_idx = int(output["draw_idx"])
                            key = sample_record_key_values(split_name, task["question_id"], task["template_type"], draw_idx)
                            records_by_key[key] = _materialize_sample_record(
                                task,
                                draw_idx=draw_idx,
                                record_id=int(output["record_id"]),
                                generation_record=output["generation_record"],
                                grading=output["grading"],
                            )
                            generated += 1
                            generated_since_checkpoint += 1
                            _checkpoint_if_needed()
                        task_bar.update(1)
                        _submit_next()
    else:
        for task in tqdm(
            pending_tasks,
            desc=progress_desc,
            unit="prompt",
        ):
            task_outputs = _execute_sampling_task(
                llm,
                task,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                sample_batch_size=sample_batch_size,
            )
            for output in task_outputs:
                draw_idx = int(output["draw_idx"])
                key = sample_record_key_values(split_name, task["question_id"], task["template_type"], draw_idx)
                records_by_key[key] = _materialize_sample_record(
                    task,
                    draw_idx=draw_idx,
                    record_id=int(output["record_id"]),
                    generation_record=output["generation_record"],
                    grading=output["grading"],
                )
                generated += 1
                generated_since_checkpoint += 1
                _checkpoint_if_needed()

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

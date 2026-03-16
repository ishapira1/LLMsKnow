from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from tqdm.auto import tqdm

from script import ensure_sycophancy_eval_cached, read_jsonl

from .cli import load_env_file, resolve_bias_types, resolve_device, resolve_hf_cache_dir
from .dataset import build_question_groups, deduplicate_rows, split_groups_train_val_test
from .logging_utils import clear_run_logging, configure_run_logging, log_status, tqdm_desc
from .outputs import build_summary_df, build_tuple_rows, to_samples_df, to_tuples_df
from .probes import score_records_with_probe, select_best_layer_by_auc, train_probe_for_layer
from .runtime import (
    acquire_run_lock,
    assert_resume_compatible,
    make_run_dir,
    model_slug,
    release_run_lock,
    run_lock_path,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_pickle_atomic,
    write_run_status,
)
from .sampling import (
    add_empirical_t,
    build_sampling_spec,
    enumerate_expected_sample_keys,
    load_sampling_cache_candidate,
    normalize_sample_records,
    sample_records_for_groups,
    sampling_spec_hash,
    sort_sample_records,
)


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    device_map_auto: bool,
    hf_cache_dir: Optional[str],
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_status("pipeline.py", f"loading model={model_name} on device={device}")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if device_map_auto else None,
            cache_dir=hf_cache_dir,
        )
        if not device_map_auto:
            model = model.to("cuda")
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            cache_dir=hf_cache_dir,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            cache_dir=hf_cache_dir,
        )
        model = model.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=hf_cache_dir)
    model.eval()
    return model, tokenizer


def _next_record_id(*groups_of_records: Sequence[Dict[str, Any]]) -> int:
    max_id = -1
    for rows in groups_of_records:
        for record in rows:
            try:
                max_id = max(max_id, int(record.get("record_id", -1)))
            except Exception:
                continue
    return max_id + 1


def _preview_text(value: Any, limit: int = 160) -> str:
    text = str(value).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _count_expected_by_template(
    expected_keys: Set[tuple[str, str, str, int]],
    bias_types: Sequence[str],
) -> Dict[str, int]:
    counts = {template_type: 0 for template_type in ["neutral", *bias_types]}
    for _, _, template_type, _ in expected_keys:
        counts[template_type] = counts.get(template_type, 0) + 1
    return counts


def _log_group_example(groups: Sequence[Dict[str, Any]], bias_types: Sequence[str]) -> None:
    if not groups:
        log_status("pipeline.py", "dataset example: no valid grouped questions found")
        return

    example = groups[0]
    log_status(
        "pipeline.py",
        f"dataset example question_id={example['question_id']} question='{_preview_text(example['question'])}' "
        f"correct='{_preview_text(example['correct_answer'])}' "
        f"incorrect='{_preview_text(example['incorrect_answer'])}'",
    )
    for template_type in ["neutral", *bias_types]:
        row = example["rows_by_type"].get(template_type)
        if row is None:
            continue
        prompt_text = _preview_text((row.get("metadata", {}) or {}).get("prompt_template", ""), limit=180)
        if prompt_text:
            log_status(
                "pipeline.py",
                f"dataset example template={template_type} prompt_template='{prompt_text}'",
            )
        prompt_preview = _preview_text(
            "\n".join(
                message.get("content", "")
                for message in row.get("prompt", [])
                if isinstance(message, dict) and isinstance(message.get("content"), str)
            ),
            limit=220,
        )
        if prompt_preview:
            log_status(
                "pipeline.py",
                f"dataset example template={template_type} prompt_preview='{prompt_preview}'",
            )


def _log_sampling_plan(
    bias_types: Sequence[str],
    split_expected_keys: Dict[str, Set[tuple[str, str, str, int]]],
    checkpoint_every: int,
    sample_batch_size: int,
    sampling_hash: str,
) -> None:
    log_status(
        "pipeline.py",
        f"sampling plan: sample_batch_size={sample_batch_size} checkpoint_every={checkpoint_every} "
        f"sampling_hash={sampling_hash}",
    )
    for split_name in ("train", "val", "test"):
        counts = _count_expected_by_template(split_expected_keys[split_name], bias_types)
        summary = " ".join(
            f"{template_type}={counts.get(template_type, 0)}"
            for template_type in ["neutral", *bias_types]
        )
        log_status(
            "pipeline.py",
            f"sampling plan split={split_name}: total_expected={len(split_expected_keys[split_name])} {summary}",
        )


def _log_reuse_summary(
    split_expected_keys: Dict[str, Set[tuple[str, str, str, int]]],
    split_records: Dict[str, List[Dict[str, Any]]],
    reuse_enabled: bool,
    cached_source_run: Optional[Path],
) -> None:
    if not reuse_enabled:
        log_status("pipeline.py", "reuse strategy: disabled by --no_reuse_sampling_cache")
    elif cached_source_run is None:
        log_status("pipeline.py", "reuse strategy: no compatible cached sampling run found")
    else:
        log_status("pipeline.py", f"reuse strategy: using compatible cache from {cached_source_run}")

    for split_name in ("train", "val", "test"):
        reused = len(split_records[split_name])
        expected = len(split_expected_keys[split_name])
        remaining = max(0, expected - reused)
        log_status(
            "pipeline.py",
            f"reuse strategy split={split_name}: reused={reused} expected={expected} remaining={remaining}",
        )


def _log_sample_preview(split_name: str, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        log_status("pipeline.py", f"sampling example split={split_name}: no records available")
        return

    example = next((record for record in records if record.get("usable_for_metrics")), records[0])
    log_status(
        "pipeline.py",
        f"sampling example split={split_name} template={example.get('template_type')} "
        f"question='{_preview_text(example.get('question', ''))}' "
        f"response='{_preview_text(example.get('response_raw', ''))}' "
        f"parsed='{_preview_text(example.get('response', ''))}' "
        f"correctness={example.get('correctness')} grading_status={example.get('grading_status')}",
    )


def _log_post_sampling_metrics(records: Sequence[Dict[str, Any]]) -> None:
    overall_usable = sum(1 for record in records if record.get("usable_for_metrics"))
    overall_ambiguous = len(records) - overall_usable
    log_status(
        "pipeline.py",
        f"post-sampling metrics: total_records={len(records)} usable={overall_usable} ambiguous={overall_ambiguous}",
    )

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("split", "")), str(record.get("template_type", "")))
        stats = grouped.setdefault(
            key,
            {
                "total": 0,
                "usable": 0,
                "ambiguous": 0,
                "correctness_sum": 0,
                "prompt_t_by_question": {},
            },
        )
        stats["total"] += 1
        if record.get("usable_for_metrics"):
            stats["usable"] += 1
            stats["correctness_sum"] += int(record["correctness"])
        else:
            stats["ambiguous"] += 1
        t_prompt = record.get("T_prompt")
        if isinstance(t_prompt, (int, float)) and not np.isnan(float(t_prompt)):
            stats["prompt_t_by_question"][str(record.get("question_id", ""))] = float(t_prompt)

    for split_name, template_type in sorted(grouped):
        stats = grouped[(split_name, template_type)]
        mean_correctness = (
            stats["correctness_sum"] / stats["usable"] if stats["usable"] > 0 else float("nan")
        )
        t_values = list(stats["prompt_t_by_question"].values())
        mean_t_prompt = float(np.mean(t_values)) if t_values else float("nan")
        mean_correctness_text = "nan" if np.isnan(mean_correctness) else f"{mean_correctness:.4f}"
        mean_t_prompt_text = "nan" if np.isnan(mean_t_prompt) else f"{mean_t_prompt:.4f}"
        log_status(
            "pipeline.py",
            f"post-sampling metrics split={split_name} template={template_type}: "
            f"total={stats['total']} usable={stats['usable']} ambiguous={stats['ambiguous']} "
            f"mean_correctness={mean_correctness_text} mean_T_prompt={mean_t_prompt_text}",
        )


def _persist_sampling_state(
    *,
    stage: str,
    split_states: Dict[str, Sequence[Dict[str, Any]]],
    split_stats: Dict[str, Dict[str, int]],
    expected_all_keys: Set[tuple[str, str, str, int]],
    expected_total_records: int,
    sampling_records_path: Path,
    sampling_manifest_path: Path,
    sampling_hash: str,
    sampling_spec: Dict[str, Any],
    cached_source_run: Optional[Path],
) -> None:
    combined_input: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        combined_input.extend(list(split_states.get(split_name, [])))
    combined = normalize_sample_records(combined_input, expected_all_keys)
    write_jsonl_atomic(sampling_records_path, combined)
    manifest = {
        "sampling_hash": sampling_hash,
        "sampling_spec": sampling_spec,
        "expected_records": expected_total_records,
        "n_records": len(combined),
        "is_complete": len(combined) >= expected_total_records,
        "stage": stage,
        "updated_at_utc": utc_now_iso(),
        "source_cache_run_dir": str(cached_source_run) if cached_source_run is not None else None,
        "split_stats": split_stats,
        "train_stats": split_stats.get("train"),
        "val_stats": split_stats.get("val"),
        "test_stats": split_stats.get("test"),
    }
    write_json_atomic(sampling_manifest_path, manifest)


def run_pipeline(args) -> None:
    import torch

    if args.sample_batch_size < 1:
        raise ValueError(f"--sample_batch_size must be >= 1, got {args.sample_batch_size}")
    if args.sampling_checkpoint_every < 0:
        raise ValueError(
            f"--sampling_checkpoint_every must be >= 0, got {args.sampling_checkpoint_every}"
        )
    if args.smoke_test and args.max_questions is None:
        args.max_questions = args.smoke_questions

    run_dir = make_run_dir(args.out_dir, args.model, args.run_name)
    run_log_path = run_dir / "run.log"
    configure_run_logging(run_log_path)
    lock_path = run_lock_path(run_dir)
    stage_bar: Optional[tqdm] = None
    stage_count = 8

    def begin_stage(index: int, message: str) -> None:
        if stage_bar is not None:
            stage_bar.set_description(tqdm_desc("pipeline.py", f"stage {index}/{stage_count} {message}"))
        log_status("pipeline.py", f"stage {index}/{stage_count}: {message}")

    def finish_stage() -> None:
        if stage_bar is not None:
            stage_bar.update(1)

    run_status = "failed"
    run_error: Optional[str] = None
    try:
        assert_resume_compatible(run_dir, args)
        acquire_run_lock(lock_path, run_dir)
        log_status("pipeline.py", f"run directory ready: {run_dir}")
        log_status("pipeline.py", f"run lock acquired: {lock_path}")
        write_run_status(run_dir, args=args, status="running", lock_path=lock_path)

        stage_bar = tqdm(
            total=stage_count,
            desc=tqdm_desc("pipeline.py", "pipeline stage progress"),
            unit="stage",
        )

        begin_stage(1, "parsed arguments and execution plan")
        log_status(
            "pipeline.py",
            "parsed arguments: "
            + json.dumps(vars(args), ensure_ascii=False, sort_keys=True, default=str),
        )
        planned_bias_types = resolve_bias_types(args.bias_types)
        log_status(
            "pipeline.py",
            f"execution plan: model={args.model} bias_types={planned_bias_types} "
            f"draws={args.n_draws} temperature={args.temperature} top_p={args.top_p} "
            f"max_new_tokens={args.max_new_tokens} smoke_test={args.smoke_test}",
        )
        finish_stage()

        begin_stage(2, "dataset loading, grouping, and split planning")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        load_env_file(args.env_file)
        hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir)
        if hf_cache_dir:
            Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ["HF_HUB_CACHE"] = hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
            log_status("pipeline.py", f"HF cache dir resolved to {hf_cache_dir}")
        else:
            log_status("pipeline.py", "HF cache dir not set; libraries may fallback to ~/.cache")
        device = resolve_device(args.device)
        log_status("pipeline.py", f"resolved device: requested={args.device} actual={device}")

        data_files = ensure_sycophancy_eval_cached(
            data_dir=args.data_dir,
            repo_id=args.sycophancy_repo,
            force_download=args.force_download_sycophancy,
        )
        input_path = data_files[args.input_jsonl]
        log_status(
            "pipeline.py",
            f"loading dataset: input_jsonl={args.input_jsonl} path={input_path} repo={args.sycophancy_repo}",
        )
        rows_raw = read_jsonl(input_path)

        rows = deduplicate_rows(rows_raw)
        groups = build_question_groups(rows, selected_bias_types=planned_bias_types)
        log_status(
            "pipeline.py",
            f"dataset stats: raw_rows={len(rows_raw)} dedup_rows={len(rows)} "
            f"valid_groups={len(groups)} bias_types={planned_bias_types}",
        )

        if args.max_questions is not None:
            rng = random.Random(args.split_seed)
            rng.shuffle(groups)
            groups = groups[: args.max_questions]
            log_status(
                "pipeline.py",
                f"dataset restricted by max_questions={args.max_questions}: groups={len(groups)}",
            )

        _log_group_example(groups, planned_bias_types)

        train_groups, val_groups, test_groups = split_groups_train_val_test(
            groups,
            test_frac=args.test_frac,
            val_frac=args.probe_val_frac,
            seed=args.split_seed,
        )
        log_status(
            "pipeline.py",
            f"dataset split: train_questions={len(train_groups)} "
            f"val_questions={len(val_groups)} test_questions={len(test_groups)} "
            f"split_seed={args.split_seed}",
        )
        finish_stage()

        begin_stage(3, "sampling plan and checkpoint layout")

        expected_train_keys = enumerate_expected_sample_keys(
            train_groups,
            split_name="train",
            bias_types=planned_bias_types,
            n_draws=args.n_draws,
        )
        expected_val_keys = enumerate_expected_sample_keys(
            val_groups,
            split_name="val",
            bias_types=planned_bias_types,
            n_draws=args.n_draws,
        )
        expected_test_keys = enumerate_expected_sample_keys(
            test_groups,
            split_name="test",
            bias_types=planned_bias_types,
            n_draws=args.n_draws,
        )
        expected_all_keys = expected_train_keys | expected_val_keys | expected_test_keys
        expected_total_records = len(expected_all_keys)

        sampling_spec = build_sampling_spec(
            args=args,
            bias_types=planned_bias_types,
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            expected_train=len(expected_train_keys),
            expected_val=len(expected_val_keys),
            expected_test=len(expected_test_keys),
        )
        sampling_hash = sampling_spec_hash(sampling_spec)
        sampling_records_path = run_dir / "sampling_records.jsonl"
        sampling_manifest_path = run_dir / "sampling_manifest.json"
        split_expected_keys = {
            "train": expected_train_keys,
            "val": expected_val_keys,
            "test": expected_test_keys,
        }
        _log_sampling_plan(
            planned_bias_types,
            split_expected_keys,
            checkpoint_every=args.sampling_checkpoint_every,
            sample_batch_size=args.sample_batch_size,
            sampling_hash=sampling_hash[:12],
        )
        finish_stage()

        begin_stage(4, "sampling cache reuse strategy")
        cached_source_run: Optional[Path] = None
        cached_records: List[Dict[str, Any]] = []
        if not args.no_reuse_sampling_cache:
            candidate = load_sampling_cache_candidate(
                out_dir=args.out_dir,
                model_name=args.model,
                sampling_hash=sampling_hash,
                exclude_run_dir=run_dir,
            )
            if candidate is not None:
                cached_source_run = candidate["run_dir"]
                cached_records_raw = read_jsonl(str(candidate["records_path"]))
                cached_records = normalize_sample_records(cached_records_raw, expected_all_keys)
                log_status(
                    "pipeline.py",
                    f"loaded reusable sampling cache from {cached_source_run}: "
                    f"records={len(cached_records)}/{expected_total_records}",
                )
        split_groups_map = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        }
        split_records: Dict[str, List[Dict[str, Any]]] = {
            split_name: sort_sample_records([r for r in cached_records if r.get("split") == split_name])
            for split_name in ("train", "val", "test")
        }
        split_sampling_stats: Dict[str, Dict[str, int]] = {
            split_name: {
                "split": split_name,
                "expected_records": len(split_expected_keys[split_name]),
                "reused_records": len(split_records[split_name]),
                "generated_records": 0,
                "total_records": len(split_records[split_name]),
            }
            for split_name in ("train", "val", "test")
        }
        _log_reuse_summary(
            split_expected_keys,
            split_records,
            reuse_enabled=not args.no_reuse_sampling_cache,
            cached_source_run=cached_source_run,
        )

        _persist_sampling_state(
            stage="sampling_start",
            split_states=split_records,
            split_stats=split_sampling_stats,
            expected_all_keys=expected_all_keys,
            expected_total_records=expected_total_records,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            sampling_hash=sampling_hash,
            sampling_spec=sampling_spec,
            cached_source_run=cached_source_run,
        )
        finish_stage()

        begin_stage(5, "sampling responses with progress and examples")
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device=device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=hf_cache_dir,
        )

        if all(
            len(split_records[split_name]) >= len(split_expected_keys[split_name])
            for split_name in ("train", "val", "test")
        ):
            log_status("pipeline.py", "sampling generation skipped: full sampling cache hit")
            for split_name in ("train", "val", "test"):
                _log_sample_preview(split_name, split_records[split_name])
        else:

            def make_progress_cb(split_name: str):
                def _progress_cb(
                    current_records: List[Dict[str, Any]],
                    stats: Dict[str, int],
                ) -> None:
                    split_records[split_name] = current_records
                    split_sampling_stats[split_name] = dict(stats)
                    _persist_sampling_state(
                        stage=f"sampling_{split_name}_in_progress",
                        split_states=split_records,
                        split_stats=split_sampling_stats,
                        expected_all_keys=expected_all_keys,
                        expected_total_records=expected_total_records,
                        sampling_records_path=sampling_records_path,
                        sampling_manifest_path=sampling_manifest_path,
                        sampling_hash=sampling_hash,
                        sampling_spec=sampling_spec,
                        cached_source_run=cached_source_run,
                    )
                    expected = len(split_expected_keys[split_name])
                    remaining = max(0, expected - len(current_records))
                    log_status(
                        "pipeline.py",
                        f"sampling progress split={split_name}: total={len(current_records)}/{expected} "
                        f"generated={stats.get('generated_records', 0)} "
                        f"reused={stats.get('reused_records', 0)} remaining={remaining}",
                    )

                return _progress_cb

            for split_name in ("train", "val", "test"):
                split_records[split_name], split_sampling_stats[split_name] = sample_records_for_groups(
                    model=model,
                    tokenizer=tokenizer,
                    groups=split_groups_map[split_name],
                    split_name=split_name,
                    bias_types=planned_bias_types,
                    n_draws=args.n_draws,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    sample_batch_size=args.sample_batch_size,
                    existing_records=split_records[split_name],
                    checkpoint_every=args.sampling_checkpoint_every,
                    progress_callback=make_progress_cb(split_name),
                    start_id=_next_record_id(
                        split_records["train"],
                        split_records["val"],
                        split_records["test"],
                    ),
                )
                _log_sample_preview(split_name, split_records[split_name])

        _persist_sampling_state(
            stage="sampling_complete",
            split_states=split_records,
            split_stats=split_sampling_stats,
            expected_all_keys=expected_all_keys,
            expected_total_records=expected_total_records,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            sampling_hash=sampling_hash,
            sampling_spec=sampling_spec,
            cached_source_run=cached_source_run,
        )

        train_records = split_records["train"]
        val_records = split_records["val"]
        test_records = split_records["test"]
        all_records = train_records + val_records + test_records
        if len(all_records) < expected_total_records:
            log_status(
                "pipeline.py",
                f"sampled records are incomplete: got={len(all_records)} expected={expected_total_records}",
            )

        add_empirical_t(all_records)
        log_status(
            "pipeline.py",
            f"sampling results: train_records={len(train_records)} val_records={len(val_records)} "
            f"test_records={len(test_records)} generated_train={split_sampling_stats['train'].get('generated_records', 0)} "
            f"generated_val={split_sampling_stats['val'].get('generated_records', 0)} "
            f"generated_test={split_sampling_stats['test'].get('generated_records', 0)}",
        )
        finish_stage()

        begin_stage(6, "post-sampling prompt metrics")
        _log_post_sampling_metrics(all_records)
        finish_stage()

        begin_stage(7, "probe selection, training, and scoring")
        n_layers = int(getattr(model.config, "num_hidden_layers", args.probe_layer_max))
        layer_min = max(1, args.probe_layer_min)
        layer_max = min(args.probe_layer_max, n_layers)
        layer_grid = list(range(layer_min, layer_max + 1))
        log_status("pipeline.py", f"probe layer grid: {layer_min}..{layer_max} (num_layers={len(layer_grid)})")

        feature_source_spec = {
            "probe_feature_mode": args.probe_feature_mode,
            "record_field": "response_raw",
            "fallback_record_field": "response",
            "token_position": "last_token_of_full_sampled_completion",
        }
        probes_meta: Dict[str, Any] = {}

        train_neutral = [record for record in train_records if record["template_type"] == "neutral"]
        val_neutral = [record for record in val_records if record["template_type"] == "neutral"]
        log_status(
            "pipeline.py",
            f"probe selection neutral: train_records={len(train_neutral)} val_records={len(val_neutral)}",
        )
        best_layer_x, best_auc_x, aucs_x, layer_clfs_x = select_best_layer_by_auc(
            model=model,
            tokenizer=tokenizer,
            train_records=train_neutral,
            val_records=val_neutral,
            layer_grid=layer_grid,
            seed=args.probe_seed,
            max_selection_samples=args.probe_selection_max_samples,
            desc="no_bias",
        )
        clf_x = (
            train_probe_for_layer(
                model=model,
                tokenizer=tokenizer,
                records=train_neutral + val_neutral,
                layer=best_layer_x if best_layer_x is not None else layer_min,
                seed=args.probe_seed,
                max_train_samples=args.probe_train_max_samples,
                desc="no_bias",
            )
            if best_layer_x is not None
            else None
        )
        log_status(
            "pipeline.py",
            f"probe retrain neutral: best_layer={best_layer_x} best_dev_auc={best_auc_x}",
        )
        score_records_with_probe(
            model=model,
            tokenizer=tokenizer,
            records=[record for record in all_records if record["template_type"] == "neutral"],
            clf=clf_x,
            layer=best_layer_x,
            score_key="probe_x",
            desc="no_bias",
        )
        probes_meta["probe_no_bias"] = {
            "best_layer": best_layer_x,
            "best_dev_auc": best_auc_x,
            "auc_per_layer": aucs_x,
            "feature_source": feature_source_spec,
            "selection_split": "val",
            "retrained_on_splits": ["train", "val"],
        }

        probe_bias_layers: Dict[str, Optional[int]] = {}
        probe_bias_clfs: Dict[str, Any] = {}
        for btype in planned_bias_types:
            train_bias = [record for record in train_records if record["template_type"] == btype]
            val_bias = [record for record in val_records if record["template_type"] == btype]
            log_status(
                "pipeline.py",
                f"probe selection bias={btype}: train_records={len(train_bias)} val_records={len(val_bias)}",
            )
            best_layer_b, best_auc_b, aucs_b, layer_clfs_b = select_best_layer_by_auc(
                model=model,
                tokenizer=tokenizer,
                train_records=train_bias,
                val_records=val_bias,
                layer_grid=layer_grid,
                seed=args.probe_seed,
                max_selection_samples=args.probe_selection_max_samples,
                desc=f"bias:{btype}",
            )
            clf_b = (
                train_probe_for_layer(
                    model=model,
                    tokenizer=tokenizer,
                    records=train_bias + val_bias,
                    layer=best_layer_b if best_layer_b is not None else layer_min,
                    seed=args.probe_seed,
                    max_train_samples=args.probe_train_max_samples,
                    desc=f"bias:{btype}",
                )
                if best_layer_b is not None
                else None
            )
            log_status(
                "pipeline.py",
                f"probe retrain bias={btype}: best_layer={best_layer_b} best_dev_auc={best_auc_b}",
            )

            probe_bias_layers[btype] = best_layer_b
            probe_bias_clfs[btype] = clf_b
            probes_meta[f"probe_bias_{btype}"] = {
                "best_layer": best_layer_b,
                "best_dev_auc": best_auc_b,
                "auc_per_layer": aucs_b,
                "feature_source": feature_source_spec,
                "selection_split": "val",
                "retrained_on_splits": ["train", "val"],
            }

            score_records_with_probe(
                model=model,
                tokenizer=tokenizer,
                records=[record for record in all_records if record["template_type"] == btype],
                clf=clf_b,
                layer=best_layer_b,
                score_key="probe_xprime",
                desc=f"bias:{btype}",
            )

            probe_models_dir = run_dir / "probe_models"
            saved_layer_models: List[str] = []
            for layer_id, clf_layer in layer_clfs_b.items():
                if clf_layer is None:
                    continue
                path = probe_models_dir / f"probe_bias_{btype}__selection_layer_{int(layer_id)}.pkl"
                write_pickle_atomic(path, clf_layer)
                saved_layer_models.append(str(path))

            saved_best_model = None
            if clf_b is not None and best_layer_b is not None:
                path = probe_models_dir / f"probe_bias_{btype}__best_retrained_layer_{int(best_layer_b)}.pkl"
                write_pickle_atomic(path, clf_b)
                saved_best_model = str(path)

            probes_meta[f"probe_bias_{btype}"]["saved_selection_models"] = saved_layer_models
            probes_meta[f"probe_bias_{btype}"]["saved_best_model"] = saved_best_model

        probe_models_dir = run_dir / "probe_models"
        saved_x_layer_models: List[str] = []
        for layer_id, clf_layer in layer_clfs_x.items():
            if clf_layer is None:
                continue
            path = probe_models_dir / f"probe_no_bias__selection_layer_{int(layer_id)}.pkl"
            write_pickle_atomic(path, clf_layer)
            saved_x_layer_models.append(str(path))
        saved_x_best_model = None
        if clf_x is not None and best_layer_x is not None:
            path = probe_models_dir / f"probe_no_bias__best_retrained_layer_{int(best_layer_x)}.pkl"
            write_pickle_atomic(path, clf_x)
            saved_x_best_model = str(path)
        probes_meta["probe_no_bias"]["saved_selection_models"] = saved_x_layer_models
        probes_meta["probe_no_bias"]["saved_best_model"] = saved_x_best_model
        finish_stage()

        begin_stage(8, "final artifact saving")
        tuple_rows = build_tuple_rows(all_records, model_name=args.model, bias_types=planned_bias_types)
        tuples_df = to_tuples_df(tuple_rows)
        summary_df = build_summary_df(tuples_df)
        samples_df = to_samples_df(all_records, model_name=args.model)

        samples_path = run_dir / "sampled_responses.csv"
        tuples_path = run_dir / "final_tuples.csv"
        summary_path = run_dir / "summary_by_question.csv"
        meta_path = run_dir / "probe_metadata.json"
        config_path = run_dir / "run_config.json"

        write_csv_atomic(samples_path, samples_df)
        write_csv_atomic(tuples_path, tuples_df)
        write_csv_atomic(summary_path, summary_df)
        write_json_atomic(meta_path, probes_meta)

        run_cfg = dict(vars(args))
        run_cfg["run_dir"] = str(run_dir)
        run_cfg["run_name"] = run_dir.name
        run_cfg["model_slug"] = model_slug(args.model)
        run_cfg["lock_path"] = str(lock_path)
        run_cfg["sampling_hash"] = sampling_hash
        run_cfg["sampling_records_path"] = str(sampling_records_path)
        run_cfg["sampling_manifest_path"] = str(sampling_manifest_path)
        run_cfg["run_log_path"] = str(run_log_path)
        write_json_atomic(config_path, run_cfg)

        log_status("pipeline.py", f"saved artifact: {samples_path}")
        log_status("pipeline.py", f"saved artifact: {tuples_path}")
        log_status("pipeline.py", f"saved artifact: {summary_path}")
        log_status("pipeline.py", f"saved artifact: {meta_path}")
        log_status("pipeline.py", f"saved artifact: {config_path}")
        log_status("pipeline.py", f"saved artifact: {sampling_records_path}")
        log_status("pipeline.py", f"saved artifact: {sampling_manifest_path}")
        log_status("pipeline.py", f"saved artifact: {run_log_path}")
        run_status = "completed"
        log_status("pipeline.py", f"run completed successfully: {run_dir}")
        finish_stage()
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        log_status("pipeline.py", f"run failed: {run_error}")
        raise
    finally:
        try:
            write_run_status(run_dir, args=args, status=run_status, lock_path=lock_path, error=run_error)
        finally:
            if stage_bar is not None:
                stage_bar.close()
            release_run_lock(lock_path)
            clear_run_logging()


__all__ = [
    "load_model_and_tokenizer",
    "run_pipeline",
]

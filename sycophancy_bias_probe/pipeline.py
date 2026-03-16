from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from script import ensure_sycophancy_eval_cached, read_jsonl

from .cli import load_env_file, resolve_bias_types, resolve_device, resolve_hf_cache_dir
from .dataset import build_question_groups, deduplicate_rows, split_groups
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

    print(f"[model] loading model={model_name} on device={device}")
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


def _persist_sampling_state(
    *,
    stage: str,
    train_state: Sequence[Dict[str, Any]],
    test_state: Sequence[Dict[str, Any]],
    train_stats: Dict[str, int],
    test_stats: Dict[str, int],
    expected_all_keys: Set[tuple[str, str, str, int]],
    expected_total_records: int,
    sampling_records_path: Path,
    sampling_manifest_path: Path,
    sampling_hash: str,
    sampling_spec: Dict[str, Any],
    cached_source_run: Optional[Path],
) -> None:
    combined = normalize_sample_records(list(train_state) + list(test_state), expected_all_keys)
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
        "train_stats": train_stats,
        "test_stats": test_stats,
    }
    write_json_atomic(sampling_manifest_path, manifest)


def run_pipeline(args) -> None:
    import torch

    bias_types = resolve_bias_types(args.bias_types)
    if args.sample_batch_size < 1:
        raise ValueError(f"--sample_batch_size must be >= 1, got {args.sample_batch_size}")
    if args.sampling_checkpoint_every < 0:
        raise ValueError(
            f"--sampling_checkpoint_every must be >= 0, got {args.sampling_checkpoint_every}"
        )
    if args.smoke_test and args.max_questions is None:
        args.max_questions = args.smoke_questions

    run_dir = make_run_dir(args.out_dir, args.model, args.run_name)
    assert_resume_compatible(run_dir, args)
    lock_path = run_lock_path(run_dir)
    acquire_run_lock(lock_path, run_dir)
    print(f"[run] run_dir={run_dir}")
    print(f"[run] lock_path={lock_path}")

    run_status = "failed"
    run_error: Optional[str] = None
    write_run_status(run_dir, args=args, status="running", lock_path=lock_path)
    try:
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
            print(f"[cache] HF cache dir={hf_cache_dir}")
        else:
            print("[cache] HF cache dir not set; libraries may fallback to ~/.cache")

        data_files = ensure_sycophancy_eval_cached(
            data_dir=args.data_dir,
            repo_id=args.sycophancy_repo,
            force_download=args.force_download_sycophancy,
        )
        input_path = data_files[args.input_jsonl]
        rows_raw = read_jsonl(input_path)

        rows = deduplicate_rows(rows_raw)
        groups = build_question_groups(rows, selected_bias_types=bias_types)
        print(
            f"[data] raw_rows={len(rows_raw)} dedup_rows={len(rows)} valid_groups={len(groups)} "
            f"bias_types={bias_types}"
        )

        if args.max_questions is not None:
            rng = random.Random(args.split_seed)
            rng.shuffle(groups)
            groups = groups[: args.max_questions]
            print(f"[data] restricted to max_questions={args.max_questions} -> {len(groups)} groups")

        train_groups, test_groups = split_groups(groups, test_frac=args.test_frac, seed=args.split_seed)
        print(f"[split] train_questions={len(train_groups)} test_questions={len(test_groups)}")

        expected_train_keys = enumerate_expected_sample_keys(
            train_groups,
            split_name="train",
            bias_types=bias_types,
            n_draws=args.n_draws,
        )
        expected_test_keys = enumerate_expected_sample_keys(
            test_groups,
            split_name="test",
            bias_types=bias_types,
            n_draws=args.n_draws,
        )
        expected_all_keys = expected_train_keys | expected_test_keys
        expected_total_records = len(expected_all_keys)

        sampling_spec = build_sampling_spec(
            args=args,
            bias_types=bias_types,
            train_groups=train_groups,
            test_groups=test_groups,
            expected_train=len(expected_train_keys),
            expected_test=len(expected_test_keys),
        )
        sampling_hash = sampling_spec_hash(sampling_spec)
        sampling_records_path = run_dir / "sampling_records.jsonl"
        sampling_manifest_path = run_dir / "sampling_manifest.json"
        print(
            f"[sample] expected_train={len(expected_train_keys)} "
            f"expected_test={len(expected_test_keys)} total={expected_total_records} "
            f"sampling_hash={sampling_hash[:12]}"
        )

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
                print(
                    f"[sample] loaded reusable cache from {cached_source_run} "
                    f"records={len(cached_records)}/{expected_total_records}"
                )
            else:
                print("[sample] no reusable sampling cache found.")
        else:
            print("[sample] reusable sampling cache disabled by flag.")

        train_records = sort_sample_records([r for r in cached_records if r.get("split") == "train"])
        test_records = sort_sample_records([r for r in cached_records if r.get("split") == "test"])
        train_sampling_stats: Dict[str, int] = {
            "split": "train",
            "expected_records": len(expected_train_keys),
            "reused_records": len(train_records),
            "generated_records": 0,
            "total_records": len(train_records),
        }
        test_sampling_stats: Dict[str, int] = {
            "split": "test",
            "expected_records": len(expected_test_keys),
            "reused_records": len(test_records),
            "generated_records": 0,
            "total_records": len(test_records),
        }

        _persist_sampling_state(
            stage="sampling_start",
            train_state=train_records,
            test_state=test_records,
            train_stats=train_sampling_stats,
            test_stats=test_sampling_stats,
            expected_all_keys=expected_all_keys,
            expected_total_records=expected_total_records,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            sampling_hash=sampling_hash,
            sampling_spec=sampling_spec,
            cached_source_run=cached_source_run,
        )

        device = resolve_device(args.device)
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device=device,
            device_map_auto=args.device_map_auto,
            hf_cache_dir=hf_cache_dir,
        )

        if len(train_records) >= len(expected_train_keys) and len(test_records) >= len(expected_test_keys):
            print("[sample] full sampling cache hit; skipping generation.")
        else:

            def train_progress_cb(
                current_train_records: List[Dict[str, Any]],
                stats: Dict[str, int],
            ) -> None:
                nonlocal train_sampling_stats
                train_sampling_stats = dict(stats)
                _persist_sampling_state(
                    stage="sampling_train_in_progress",
                    train_state=current_train_records,
                    test_state=test_records,
                    train_stats=train_sampling_stats,
                    test_stats=test_sampling_stats,
                    expected_all_keys=expected_all_keys,
                    expected_total_records=expected_total_records,
                    sampling_records_path=sampling_records_path,
                    sampling_manifest_path=sampling_manifest_path,
                    sampling_hash=sampling_hash,
                    sampling_spec=sampling_spec,
                    cached_source_run=cached_source_run,
                )

            train_records, train_sampling_stats = sample_records_for_groups(
                model=model,
                tokenizer=tokenizer,
                groups=train_groups,
                split_name="train",
                bias_types=bias_types,
                n_draws=args.n_draws,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                sample_batch_size=args.sample_batch_size,
                existing_records=train_records,
                checkpoint_every=args.sampling_checkpoint_every,
                progress_callback=train_progress_cb,
                start_id=0,
            )

            def test_progress_cb(
                current_test_records: List[Dict[str, Any]],
                stats: Dict[str, int],
            ) -> None:
                nonlocal test_sampling_stats
                test_sampling_stats = dict(stats)
                _persist_sampling_state(
                    stage="sampling_test_in_progress",
                    train_state=train_records,
                    test_state=current_test_records,
                    train_stats=train_sampling_stats,
                    test_stats=test_sampling_stats,
                    expected_all_keys=expected_all_keys,
                    expected_total_records=expected_total_records,
                    sampling_records_path=sampling_records_path,
                    sampling_manifest_path=sampling_manifest_path,
                    sampling_hash=sampling_hash,
                    sampling_spec=sampling_spec,
                    cached_source_run=cached_source_run,
                )

            test_records, test_sampling_stats = sample_records_for_groups(
                model=model,
                tokenizer=tokenizer,
                groups=test_groups,
                split_name="test",
                bias_types=bias_types,
                n_draws=args.n_draws,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                sample_batch_size=args.sample_batch_size,
                existing_records=test_records,
                checkpoint_every=args.sampling_checkpoint_every,
                progress_callback=test_progress_cb,
                start_id=_next_record_id(train_records, test_records),
            )

        _persist_sampling_state(
            stage="sampling_complete",
            train_state=train_records,
            test_state=test_records,
            train_stats=train_sampling_stats,
            test_stats=test_sampling_stats,
            expected_all_keys=expected_all_keys,
            expected_total_records=expected_total_records,
            sampling_records_path=sampling_records_path,
            sampling_manifest_path=sampling_manifest_path,
            sampling_hash=sampling_hash,
            sampling_spec=sampling_spec,
            cached_source_run=cached_source_run,
        )

        all_records = train_records + test_records
        if len(all_records) < expected_total_records:
            print(
                f"[warn] sampled records are incomplete: got={len(all_records)} "
                f"expected={expected_total_records}"
            )

        add_empirical_t(all_records)
        print(
            f"[sample] train_records={len(train_records)} test_records={len(test_records)} "
            f"generated_train={train_sampling_stats.get('generated_records', 0)} "
            f"generated_test={test_sampling_stats.get('generated_records', 0)}"
        )

        n_layers = int(getattr(model.config, "num_hidden_layers", args.probe_layer_max))
        layer_min = max(1, args.probe_layer_min)
        layer_max = min(args.probe_layer_max, n_layers)
        layer_grid = list(range(layer_min, layer_max + 1))
        print(f"[probe] layer_grid={layer_min}..{layer_max} (num={len(layer_grid)})")

        feature_source_spec = {
            "probe_feature_mode": args.probe_feature_mode,
            "record_field": "response_raw",
            "fallback_record_field": "response",
            "token_position": "last_token_of_full_sampled_completion",
        }
        probes_meta: Dict[str, Any] = {}

        train_neutral = [record for record in train_records if record["template_type"] == "neutral"]
        best_layer_x, best_auc_x, aucs_x, layer_clfs_x = select_best_layer_by_auc(
            model=model,
            tokenizer=tokenizer,
            records=train_neutral,
            layer_grid=layer_grid,
            val_frac=args.probe_val_frac,
            seed=args.probe_seed,
            max_selection_samples=args.probe_selection_max_samples,
            desc="no_bias",
        )
        clf_x = (
            train_probe_for_layer(
                model=model,
                tokenizer=tokenizer,
                records=train_neutral,
                layer=best_layer_x if best_layer_x is not None else layer_min,
                seed=args.probe_seed,
                max_train_samples=args.probe_train_max_samples,
                desc="no_bias",
            )
            if best_layer_x is not None
            else None
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
        }

        probe_bias_layers: Dict[str, Optional[int]] = {}
        probe_bias_clfs: Dict[str, Any] = {}
        for btype in bias_types:
            train_bias = [record for record in train_records if record["template_type"] == btype]
            best_layer_b, best_auc_b, aucs_b, layer_clfs_b = select_best_layer_by_auc(
                model=model,
                tokenizer=tokenizer,
                records=train_bias,
                layer_grid=layer_grid,
                val_frac=args.probe_val_frac,
                seed=args.probe_seed,
                max_selection_samples=args.probe_selection_max_samples,
                desc=f"bias:{btype}",
            )
            clf_b = (
                train_probe_for_layer(
                    model=model,
                    tokenizer=tokenizer,
                    records=train_bias,
                    layer=best_layer_b if best_layer_b is not None else layer_min,
                    seed=args.probe_seed,
                    max_train_samples=args.probe_train_max_samples,
                    desc=f"bias:{btype}",
                )
                if best_layer_b is not None
                else None
            )

            probe_bias_layers[btype] = best_layer_b
            probe_bias_clfs[btype] = clf_b
            probes_meta[f"probe_bias_{btype}"] = {
                "best_layer": best_layer_b,
                "best_dev_auc": best_auc_b,
                "auc_per_layer": aucs_b,
                "feature_source": feature_source_spec,
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

        tuple_rows = build_tuple_rows(all_records, model_name=args.model, bias_types=bias_types)
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
        write_json_atomic(config_path, run_cfg)

        print("[saved]", samples_path)
        print("[saved]", tuples_path)
        print("[saved]", summary_path)
        print("[saved]", meta_path)
        print("[saved]", config_path)
        print("[saved]", sampling_records_path)
        print("[saved]", sampling_manifest_path)
        run_status = "completed"
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            write_run_status(run_dir, args=args, status=run_status, lock_path=lock_path, error=run_error)
        finally:
            release_run_lock(lock_path)


__all__ = [
    "load_model_and_tokenizer",
    "run_pipeline",
]

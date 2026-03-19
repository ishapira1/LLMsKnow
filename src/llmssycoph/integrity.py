from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from .data import prompt_id_for
from .runtime import model_slug, resolve_run_artifact_path


REQUIRED_ARTIFACT_KEYS = (
    "run_config",
    "status",
    "sampling_manifest",
    "sampling_records",
    "sampling_integrity_summary",
    "run_summary",
    "sampled_responses",
    "final_tuples",
    "summary_by_question",
    "model_summary_by_template",
    "model_summary_by_bias",
    "probe_candidate_scores",
    "probe_scores_by_prompt",
    "probe_summary_csv",
    "executive_summary",
    "probe_metadata",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _check_probability_series(series: pd.Series, column_name: str, issues: List[str]) -> None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return
    bad_mask = (numeric < 0) | (numeric > 1)
    if bad_mask.any():
        issues.append(f"{column_name} contains values outside [0, 1]")


def _resolve_run_dir(run_dir: str | None, out_dir: str, model: str, run_name: str) -> Path:
    if run_dir:
        return Path(run_dir)
    return Path(out_dir) / model_slug(model) / run_name


def check_run_integrity(run_dir: Path) -> Dict[str, Any]:
    issues: List[str] = []
    run_dir = run_dir.resolve()

    required_artifact_paths = [resolve_run_artifact_path(run_dir, "run_log")]
    required_artifact_paths.extend(
        resolve_run_artifact_path(run_dir, artifact_key) for artifact_key in REQUIRED_ARTIFACT_KEYS
    )
    for artifact_path in required_artifact_paths:
        if not artifact_path.exists():
            issues.append(f"missing artifact: {artifact_path}")

    if issues:
        raise RuntimeError("\n".join(issues))

    run_config_path = resolve_run_artifact_path(run_dir, "run_config")
    status_path = resolve_run_artifact_path(run_dir, "status")
    manifest_path = resolve_run_artifact_path(run_dir, "sampling_manifest")
    sampling_records_path = resolve_run_artifact_path(run_dir, "sampling_records")
    sampling_integrity_path = resolve_run_artifact_path(run_dir, "sampling_integrity_summary")
    samples_path = resolve_run_artifact_path(run_dir, "sampled_responses")
    tuples_path = resolve_run_artifact_path(run_dir, "final_tuples")
    summary_path = resolve_run_artifact_path(run_dir, "summary_by_question")
    model_summary_by_template_path = resolve_run_artifact_path(run_dir, "model_summary_by_template")
    model_summary_by_bias_path = resolve_run_artifact_path(run_dir, "model_summary_by_bias")
    probe_candidate_scores_path = resolve_run_artifact_path(run_dir, "probe_candidate_scores")
    probe_scores_by_prompt_path = resolve_run_artifact_path(run_dir, "probe_scores_by_prompt")
    probe_summary_csv_path = resolve_run_artifact_path(run_dir, "probe_summary_csv")
    executive_summary_path = resolve_run_artifact_path(run_dir, "executive_summary")
    probe_metadata_path = resolve_run_artifact_path(run_dir, "probe_metadata")
    run_summary_path = resolve_run_artifact_path(run_dir, "run_summary")

    run_config = _load_json(run_config_path)
    status = _load_json(status_path)
    manifest = _load_json(manifest_path)
    probe_meta = _load_json(probe_metadata_path)
    run_summary = _load_json(run_summary_path)
    sampling_integrity = _load_json(sampling_integrity_path)

    samples = pd.read_csv(samples_path)
    tuples_df = pd.read_csv(tuples_path)
    summary_df = pd.read_csv(summary_path)
    model_summary_by_template_df = pd.read_csv(model_summary_by_template_path)
    model_summary_by_bias_df = pd.read_csv(model_summary_by_bias_path)
    probe_candidate_scores_df = pd.read_csv(probe_candidate_scores_path)
    probe_scores_by_prompt_df = pd.read_csv(probe_scores_by_prompt_path)
    probe_summary_csv_df = pd.read_csv(probe_summary_csv_path)

    sample_required_columns = {"question_id", "prompt_id", "question", "prompt_text"}
    missing_sample_columns = sorted(sample_required_columns.difference(samples.columns))
    if missing_sample_columns:
        issues.append(
            f"sampled_responses.csv is missing required columns: {missing_sample_columns}"
        )

    tuple_required_columns = {"question_id", "question", "prompt_id_x", "prompt_id_xprime", "prompt_x", "prompt_with_bias"}
    missing_tuple_columns = sorted(tuple_required_columns.difference(tuples_df.columns))
    if missing_tuple_columns:
        issues.append(
            f"final_tuples.csv is missing required columns: {missing_tuple_columns}"
        )

    summary_required_columns = {"question_id", "question", "prompt_id_x", "prompt_id_xprime", "prompt_x", "prompt_with_bias"}
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    if missing_summary_columns:
        issues.append(
            f"summary_by_question.csv is missing required columns: {missing_summary_columns}"
        )
    model_summary_template_required_columns = {"template_type", "accuracy", "avg_p_correct"}
    missing_model_template_columns = sorted(
        model_summary_template_required_columns.difference(model_summary_by_template_df.columns)
    )
    if missing_model_template_columns:
        issues.append(
            "model_summary_by_template.csv is missing required columns: "
            f"{missing_model_template_columns}"
        )
    model_summary_bias_required_columns = {"bias_type", "accuracy_x", "accuracy_xprime", "avg_p_x", "avg_p_xprime"}
    missing_model_bias_columns = sorted(
        model_summary_bias_required_columns.difference(model_summary_by_bias_df.columns)
    )
    if missing_model_bias_columns:
        issues.append(
            f"model_summary_by_bias.csv is missing required columns: {missing_model_bias_columns}"
        )
    candidate_score_required_columns = {"probe_name", "question_id", "prompt_id", "candidate_choice", "probe_score"}
    missing_candidate_score_columns = sorted(candidate_score_required_columns.difference(probe_candidate_scores_df.columns))
    if missing_candidate_score_columns:
        issues.append(
            f"probe_candidate_scores.csv is missing required columns: {missing_candidate_score_columns}"
        )
    probe_scores_by_prompt_required_columns = {
        "probe_name",
        "question_id",
        "prompt_id",
        "correct_letter",
        "selected_choice",
        "probe_score_correct_choice",
        "probe_score_selected_choice",
        "probe_argmax_choice",
    }
    missing_probe_scores_by_prompt_columns = sorted(
        probe_scores_by_prompt_required_columns.difference(probe_scores_by_prompt_df.columns)
    )
    if missing_probe_scores_by_prompt_columns:
        issues.append(
            "probe_scores_by_prompt.csv is missing required columns: "
            f"{missing_probe_scores_by_prompt_columns}"
        )
    probe_summary_csv_required_columns = {"probe_name", "best_layer", "best_dev_auc", "test_auc"}
    missing_probe_summary_csv_columns = sorted(
        probe_summary_csv_required_columns.difference(probe_summary_csv_df.columns)
    )
    if missing_probe_summary_csv_columns:
        issues.append(
            f"probe_summary.csv is missing required columns: {missing_probe_summary_csv_columns}"
        )

    if status.get("status") != "completed":
        issues.append(f"status.json reports status={status.get('status')!r}, expected 'completed'")
    if status.get("run_dir") and str(run_dir) != str(Path(status["run_dir"]).resolve()):
        issues.append("status.json run_dir does not match the checked run directory")
    if run_config.get("run_dir") and str(run_dir) != str(Path(run_config["run_dir"]).resolve()):
        issues.append("run_config.json run_dir does not match the checked run directory")

    if not bool(run_config.get("smoke_test", False)):
        issues.append("run_config.json does not indicate smoke_test=true")
    if str(run_config.get("benchmark_source")) != "ays_mc_single_turn":
        issues.append("benchmark_source is not ays_mc_single_turn")
    if str(run_config.get("input_jsonl")) != "are_you_sure.jsonl":
        issues.append("input_jsonl is not are_you_sure.jsonl")
    if str(run_config.get("dataset_name")) != "aqua_mc":
        issues.append("dataset_name is not aqua_mc")
    if str(run_config.get("mc_mode")) != "strict_mc":
        issues.append("mc_mode is not strict_mc")
    if str(run_config.get("device")) != "cpu":
        issues.append("device is not cpu")

    bias_types = _parse_list_like(run_config.get("bias_types"))
    expected_templates = {"neutral", *bias_types}
    expected_question_ids = set(manifest.get("sampling_spec", {}).get("train_question_ids", []))
    expected_question_ids.update(manifest.get("sampling_spec", {}).get("val_question_ids", []))
    expected_question_ids.update(manifest.get("sampling_spec", {}).get("test_question_ids", []))

    expected_total_records = int(manifest.get("expected_records", 0) or 0)
    if not bool(manifest.get("is_complete", False)):
        issues.append("sampling_manifest.json reports is_complete=false")
    if int(manifest.get("n_records", -1)) != expected_total_records:
        issues.append(
            f"sampling_manifest.json n_records={manifest.get('n_records')} "
            f"does not match expected_records={expected_total_records}"
        )
    if len(samples) != expected_total_records:
        issues.append(
            f"sampled_responses.csv has {len(samples)} rows, expected {expected_total_records}"
        )

    sampling_records_lines = sum(
        1 for line in sampling_records_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if sampling_records_lines != expected_total_records:
        issues.append(
            f"sampling_records.jsonl has {sampling_records_lines} non-empty lines, expected {expected_total_records}"
        )
    sampling_mode_series = (
        samples["sampling_mode"].fillna("generation").astype(str)
        if "sampling_mode" in samples.columns
        else pd.Series(["generation"] * len(samples))
    )
    if int(sampling_integrity.get("total_records", -1)) != expected_total_records:
        issues.append(
            "sampling_integrity_summary.json total_records does not match sampling_manifest.json expected_records"
        )
    integrity_modes = sampling_integrity.get("by_sampling_mode", {})
    if not isinstance(integrity_modes, dict):
        issues.append("sampling_integrity_summary.json is missing by_sampling_mode")
        integrity_modes = {}
    for sampling_mode, mode_summary in integrity_modes.items():
        if not isinstance(mode_summary, dict):
            issues.append(f"sampling_integrity_summary.json has non-object summary for mode={sampling_mode}")
            continue
        mode_total = int(mode_summary.get("total", -1))
        actual_mode_total = int((sampling_mode_series == str(sampling_mode)).sum())
        if mode_total != actual_mode_total:
            issues.append(
                f"sampling_integrity_summary.json total mismatch for sampling_mode={sampling_mode}: "
                f"summary={mode_total} actual={actual_mode_total}"
            )
        buckets = mode_summary.get("buckets", {})
        if not isinstance(buckets, dict):
            issues.append(f"sampling_integrity_summary.json buckets missing for sampling_mode={sampling_mode}")
            continue
        bucket_total = 0
        for bucket_name, bucket_stats in buckets.items():
            if not isinstance(bucket_stats, dict):
                issues.append(
                    f"sampling_integrity_summary.json bucket {bucket_name!r} for mode={sampling_mode} is not an object"
                )
                continue
            bucket_total += int(bucket_stats.get("count", 0) or 0)
        if bucket_total != mode_total:
            issues.append(
                f"sampling_integrity_summary.json bucket counts do not sum to total for sampling_mode={sampling_mode}"
            )

    sample_key_cols = ["split", "question_id", "template_type", "draw_idx"]
    duplicate_sample_keys = int(samples.duplicated(subset=sample_key_cols).sum())
    if duplicate_sample_keys:
        issues.append(f"sampled_responses.csv has {duplicate_sample_keys} duplicate sample keys")
    duplicate_record_ids = int(samples.duplicated(subset=["record_id"]).sum())
    if duplicate_record_ids:
        issues.append(f"sampled_responses.csv has {duplicate_record_ids} duplicate record_id values")

    present_splits = set(samples["split"].dropna().astype(str))
    manifest_split_stats = manifest.get("split_stats", {})
    expected_splits = {
        split_name
        for split_name, stats in manifest_split_stats.items()
        if int(stats.get("expected_records", 0) or 0) > 0
    }
    if present_splits != expected_splits:
        issues.append(f"split coverage mismatch: present={sorted(present_splits)} expected={sorted(expected_splits)}")
    for split_name in sorted(expected_splits):
        split_expected = int(manifest_split_stats[split_name].get("expected_records", 0) or 0)
        split_actual = int((samples["split"].astype(str) == split_name).sum())
        if split_actual != split_expected:
            issues.append(f"split={split_name} has {split_actual} rows, expected {split_expected}")

    present_templates = set(samples["template_type"].dropna().astype(str))
    if present_templates != expected_templates:
        issues.append(
            f"template coverage mismatch: present={sorted(present_templates)} expected={sorted(expected_templates)}"
        )

    present_question_ids = set(samples["question_id"].dropna().astype(str))
    if present_question_ids != expected_question_ids:
        issues.append(
            "question_id coverage mismatch between sampled_responses.csv and sampling_manifest.json"
        )
    if "prompt_id" in samples.columns:
        expected_prompt_ids = {
            prompt_id_for(question_id, template_type)
            for question_id in expected_question_ids
            for template_type in expected_templates
        }
        present_prompt_ids = set(samples["prompt_id"].dropna().astype(str))
        if present_prompt_ids != expected_prompt_ids:
            issues.append("prompt_id coverage mismatch in sampled_responses.csv")
        prompt_id_per_variant = (
            samples.groupby(["question_id", "template_type"], as_index=False)["prompt_id"]
            .nunique()
            .rename(columns={"prompt_id": "n_prompt_ids"})
        )
        if int((prompt_id_per_variant["n_prompt_ids"] != 1).sum()):
            issues.append("sampled_responses.csv does not keep a 1:1 mapping from (question_id, template_type) to prompt_id")

    configured_dataset = str(run_config.get("dataset_name", "") or "")
    sample_datasets = set(samples["dataset"].dropna().astype(str))
    if configured_dataset and configured_dataset != "all" and sample_datasets != {configured_dataset}:
        issues.append(f"dataset coverage mismatch: present={sorted(sample_datasets)} expected={[configured_dataset]}")

    task_formats = set(samples["task_format"].dropna().astype(str))
    if task_formats != {"multiple_choice"}:
        issues.append(f"task_format mismatch: present={sorted(task_formats)} expected=['multiple_choice']")
    mc_modes = set(samples["mc_mode"].dropna().astype(str))
    if mc_modes != {"strict_mc"}:
        issues.append(f"mc_mode mismatch in sampled_responses.csv: present={sorted(mc_modes)}")

    draws_per_prompt = int(run_config.get("n_draws", 0) or 0)
    strict_mc_choice_scoring = bool(run_config.get("strict_mc_choice_scoring", False))
    per_prompt_counts = (
        samples.groupby(["split", "question_id", "template_type"], as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )
    prompt_modes = (
        samples.groupby(["split", "question_id", "template_type"], as_index=False)
        .agg(
            task_format=("task_format", "first"),
            mc_mode=("mc_mode", "first"),
        )
    )
    per_prompt_counts = per_prompt_counts.merge(
        prompt_modes,
        on=["split", "question_id", "template_type"],
        how="left",
    )
    per_prompt_counts["expected_n_rows"] = draws_per_prompt
    if strict_mc_choice_scoring:
        strict_mask = (
            per_prompt_counts["task_format"].astype(str) == "multiple_choice"
        ) & (
            per_prompt_counts["mc_mode"].astype(str) == "strict_mc"
        )
        per_prompt_counts.loc[strict_mask, "expected_n_rows"] = 1
    bad_prompt_counts = per_prompt_counts[
        per_prompt_counts["n_rows"] != per_prompt_counts["expected_n_rows"]
    ]
    if not bad_prompt_counts.empty:
        issues.append(
            "not every (split, question_id, template_type) has the configured number of draws"
        )

    _check_probability_series(samples["T_prompt"], "sampled_responses.T_prompt", issues)
    _check_probability_series(summary_df["T_x"], "summary_by_question.T_x", issues)
    _check_probability_series(summary_df["T_xprime"], "summary_by_question.T_xprime", issues)
    _check_probability_series(summary_df["mean_C_x"], "summary_by_question.mean_C_x", issues)
    _check_probability_series(summary_df["mean_C_xprime"], "summary_by_question.mean_C_xprime", issues)

    correctness_numeric = pd.to_numeric(samples["correctness"], errors="coerce").dropna()
    if not correctness_numeric.isin([0, 1]).all():
        issues.append("sampled_responses.correctness contains values other than 0/1/NaN")

    if tuples_df.empty:
        issues.append("final_tuples.csv is empty")
    if summary_df.empty:
        issues.append("summary_by_question.csv is empty")

    if not tuples_df.empty:
        tuple_key_cols = ["split", "question_id", "bias_type", "draw_idx"]
        duplicate_tuple_keys = int(tuples_df.duplicated(subset=tuple_key_cols).sum())
        if duplicate_tuple_keys:
            issues.append(f"final_tuples.csv has {duplicate_tuple_keys} duplicate tuple keys")
        tuple_biases = set(tuples_df["bias_type"].dropna().astype(str))
        if not tuple_biases.issubset(set(bias_types)):
            issues.append(
                f"final_tuples.csv contains unexpected bias types: {sorted(tuple_biases - set(bias_types))}"
            )
        if {"question_id", "bias_type", "prompt_id_x", "prompt_id_xprime"}.issubset(tuples_df.columns):
            expected_prompt_x = tuples_df["question_id"].astype(str).map(lambda question_id: prompt_id_for(question_id, "neutral"))
            expected_prompt_xprime = tuples_df.apply(
                lambda row: prompt_id_for(row["question_id"], row["bias_type"]),
                axis=1,
            )
            if not tuples_df["prompt_id_x"].astype(str).equals(expected_prompt_x):
                issues.append("final_tuples.csv has inconsistent prompt_id_x values")
            if not tuples_df["prompt_id_xprime"].astype(str).equals(expected_prompt_xprime):
                issues.append("final_tuples.csv has inconsistent prompt_id_xprime values")

    if not summary_df.empty:
        expected_summary_rows = len(
            tuples_df[["model_name", "split", "question_id", "dataset", "bias_type"]].drop_duplicates()
        )
        if len(summary_df) != expected_summary_rows:
            issues.append(
                f"summary_by_question.csv has {len(summary_df)} rows, expected {expected_summary_rows} "
                "from final_tuples.csv"
            )
        if int(summary_df.duplicated(subset=["model_name", "split", "question_id", "dataset", "bias_type"]).sum()):
            issues.append("summary_by_question.csv has duplicate group keys")
        summary_draws = pd.to_numeric(summary_df["n_draws"], errors="coerce").dropna()
        max_expected_draws = 1 if strict_mc_choice_scoring and str(run_config.get("mc_mode")) == "strict_mc" else draws_per_prompt
        if summary_draws.empty or (summary_draws <= 0).any() or (summary_draws > max_expected_draws).any():
            issues.append("summary_by_question.csv has invalid n_draws values")
        if {"question_id", "bias_type", "prompt_id_x", "prompt_id_xprime"}.issubset(summary_df.columns):
            expected_prompt_x = summary_df["question_id"].astype(str).map(lambda question_id: prompt_id_for(question_id, "neutral"))
            expected_prompt_xprime = summary_df.apply(
                lambda row: prompt_id_for(row["question_id"], row["bias_type"]),
                axis=1,
            )
            if not summary_df["prompt_id_x"].astype(str).equals(expected_prompt_x):
                issues.append("summary_by_question.csv has inconsistent prompt_id_x values")
            if not summary_df["prompt_id_xprime"].astype(str).equals(expected_prompt_xprime):
                issues.append("summary_by_question.csv has inconsistent prompt_id_xprime values")

    strict_quality = probe_meta.get("strict_mc_quality")
    if not isinstance(strict_quality, dict):
        issues.append("probe_metadata.json is missing strict_mc_quality")
    else:
        if strict_quality.get("status") != "passed":
            issues.append(
                f"strict_mc_quality status is {strict_quality.get('status')!r}, expected 'passed'"
            )
        if strict_quality.get("issues"):
            issues.append(f"strict_mc_quality reported issues: {strict_quality.get('issues')}")
        summary = strict_quality.get("summary", {})
        by_template = summary.get("by_template", {}) if isinstance(summary, dict) else {}
        missing_quality_templates = expected_templates.difference(set(by_template.keys()))
        if missing_quality_templates:
            issues.append(
                f"strict_mc_quality summary is missing templates: {sorted(missing_quality_templates)}"
            )

    if "probe_no_bias" not in probe_meta:
        issues.append("probe_metadata.json is missing probe_no_bias")
    for bias_type in bias_types:
        if f"probe_bias_{bias_type}" not in probe_meta:
            issues.append(f"probe_metadata.json is missing probe_bias_{bias_type}")

    executive_summary_text = executive_summary_path.read_text(encoding="utf-8").strip()
    if not executive_summary_text:
        issues.append("executive_summary.md is empty")
    if "Executive Summary" not in executive_summary_text:
        issues.append("executive_summary.md is missing the expected title")

    all_probes_dir = resolve_run_artifact_path(run_dir, "all_probes_dir")
    chosen_probe_dir = resolve_run_artifact_path(run_dir, "chosen_probe_dir")
    if all_probes_dir.exists():
        all_manifest_path = all_probes_dir / "manifest.json"
        if not all_manifest_path.exists():
            issues.append("all_probes/manifest.json is missing")
    if chosen_probe_dir.exists():
        chosen_manifest_path = chosen_probe_dir / "manifest.json"
        if not chosen_manifest_path.exists():
            issues.append("chosen_probe/manifest.json is missing")

    for probe_name in ["probe_no_bias", *[f"probe_bias_{bias_type}" for bias_type in bias_types]]:
        if all_probes_dir.exists():
            family_manifest = all_probes_dir / probe_name / "manifest.json"
            if not family_manifest.exists():
                issues.append(f"all_probes manifest is missing for {probe_name}")
        if chosen_probe_dir.exists():
            family_manifest = chosen_probe_dir / probe_name / "manifest.json"
            if not family_manifest.exists():
                issues.append(f"chosen_probe manifest is missing for {probe_name}")

    if run_summary_path.exists():
        if "headline_metrics" not in run_summary:
            issues.append("run_summary.json is missing headline_metrics")
        if "paths" not in run_summary:
            issues.append("run_summary.json is missing paths")

    if issues:
        raise RuntimeError("\n".join(issues))

    usable_rate = float(pd.to_numeric(samples["usable_for_metrics"], errors="coerce").fillna(0).mean())
    ambiguous_rate = 1.0 - usable_rate
    paired_stats = (
        tuples_df.groupby("bias_type", as_index=False)
        .agg(
            n_pairs=("draw_idx", "size"),
            neutral_accuracy=("C_x_y", "mean"),
            biased_accuracy=("C_xprime_yprime", "mean"),
        )
        .sort_values("bias_type")
    )
    harmful_flip = tuples_df.assign(
        harmful_flip=((tuples_df["C_x_y"] == 1) & (tuples_df["C_xprime_yprime"] == 0)).astype(float)
    )
    harmful_flip_stats = (
        harmful_flip.groupby("bias_type", as_index=False)
        .agg(harmful_flip_rate=("harmful_flip", "mean"))
        .sort_values("bias_type")
    )
    paired_stats = paired_stats.merge(harmful_flip_stats, on="bias_type", how="left")

    template_stats = (
        samples.groupby("template_type", as_index=False)
        .agg(
            n_rows=("record_id", "size"),
            usable_rate=("usable_for_metrics", "mean"),
            cap_hit_rate=("hit_max_new_tokens", "mean"),
            starts_with_answer_rate=("starts_with_answer_prefix", "mean"),
            exact_format_rate=("strict_format_exact", "mean"),
        )
        .sort_values("template_type")
    )

    return {
        "run_dir": str(run_dir),
        "sample_count": int(len(samples)),
        "tuple_count": int(len(tuples_df)),
        "summary_count": int(len(summary_df)),
        "question_count": int(len(expected_question_ids)),
        "run_summary_path": str(run_summary_path) if run_summary_path.exists() else None,
        "bias_types": bias_types,
        "usable_rate": usable_rate,
        "ambiguous_rate": ambiguous_rate,
        "strict_mc_quality": strict_quality["summary"],
        "template_stats": template_stats,
        "paired_stats": paired_stats,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"[integrity] run_dir={report['run_dir']}")
    print(
        "[integrity] artifacts:"
        f" samples={report['sample_count']}"
        f" tuples={report['tuple_count']}"
        f" summary_rows={report['summary_count']}"
        f" questions={report['question_count']}"
    )
    print(
        "[integrity] usability:"
        f" usable_rate={_format_pct(report['usable_rate'])}"
        f" ambiguous_rate={_format_pct(report['ambiguous_rate'])}"
    )

    strict_quality = report["strict_mc_quality"]
    print(
        "[integrity] strict_mc_quality:"
        f" commitment_rate={_format_pct(strict_quality['commitment_rate'])}"
        f" starts_with_answer_rate={_format_pct(strict_quality['starts_with_answer_rate'])}"
        f" exact_format_rate={_format_pct(strict_quality['exact_format_rate'])}"
        f" cap_hit_rate={_format_pct(strict_quality['cap_hit_rate'])}"
        f" parse_failures={strict_quality['explicit_parse_failures']}"
        f" max_neutral_bias_answer_gap={_format_pct(strict_quality['max_neutral_bias_answer_gap'])}"
    )

    print("[integrity] template_stats:")
    print(report["template_stats"].to_string(index=False))
    print("[integrity] paired_stats:")
    print(report["paired_stats"].to_string(index=False))
    print("[integrity] all checks passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a completed smoke run and print a compact health report.")
    parser.add_argument("--run_dir", type=str, default=None, help="Explicit run directory to validate.")
    parser.add_argument("--out_dir", type=str, default="results/sycophancy_bias_probe")
    parser.add_argument("--model", type=str, required=False, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--run_name", type=str, required=False, default="smoke_aqua_mc_mistral7b_cpu_q12_l4")
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        model=args.model,
        run_name=args.run_name,
    )
    report = check_run_integrity(run_dir)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

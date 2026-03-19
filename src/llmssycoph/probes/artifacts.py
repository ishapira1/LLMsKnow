from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..data import prompt_id_for
from ..grading import record_is_usable_for_metrics as _record_is_usable_for_metrics
from ..runtime import (
    preferred_run_artifact_path,
    utc_now_iso,
    write_json_atomic,
    write_jsonl_atomic,
    write_pickle_atomic,
)
from .metrics import (
    PROBE_ARTIFACT_SCHEMA_VERSION,
    PROBE_METADATA_SCHEMA_VERSION,
    add_path_fields,
    build_split_data_summary,
    probe_model_metadata,
    summarize_probe_records,
)


def _record_id_set(records: Sequence[Dict[str, Any]]) -> set[int]:
    record_ids: set[int] = set()
    for record in records:
        try:
            record_ids.add(int(record["record_id"]))
        except Exception:
            continue
    return record_ids


def build_probe_membership_rows(
    *,
    split_records: Mapping[str, Sequence[Dict[str, Any]]],
    fit_records: Sequence[Dict[str, Any]],
    selection_train_records: Sequence[Dict[str, Any]],
    selection_val_records: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    fit_ids = _record_id_set(fit_records)
    selection_train_ids = _record_id_set(selection_train_records)
    selection_val_ids = _record_id_set(selection_val_records)
    eval_ids = {
        int(record["record_id"])
        for split_name in ("train", "val", "test")
        for record in split_records.get(split_name, [])
        if _record_is_usable_for_metrics(record)
    }

    rows: list[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        for record in split_records.get(split_name, []):
            try:
                record_id = int(record["record_id"])
            except Exception:
                continue
            correctness = record.get("correctness")
            rows.append(
                {
                    "record_id": record_id,
                    "split": split_name,
                    "question_id": str(record.get("question_id", "") or ""),
                    "prompt_id": str(
                        record.get(
                            "prompt_id",
                            prompt_id_for(record.get("question_id", ""), record.get("template_type", "")),
                        )
                    ),
                    "dataset": str(record.get("dataset", "") or ""),
                    "template_type": str(record.get("template_type", "") or ""),
                    "draw_idx": int(record.get("draw_idx", 0) or 0),
                    "correctness": None if correctness is None else int(correctness),
                    "usable_for_metrics": bool(
                        record.get("usable_for_metrics", _record_is_usable_for_metrics(record))
                    ),
                    "probe_row_type": str(record.get("probe_row_type", "sampled_completion") or "sampled_completion"),
                    "source_record_id": record.get("source_record_id"),
                    "candidate_choice": str(record.get("candidate_choice", "") or ""),
                    "candidate_probability": record.get("candidate_probability"),
                    "probe_sample_weight": record.get("probe_sample_weight"),
                    "candidate_is_selected": bool(record.get("candidate_is_selected", False)),
                    "included_in_fit": record_id in fit_ids,
                    "included_in_selection_train": record_id in selection_train_ids,
                    "included_in_selection_val": record_id in selection_val_ids,
                    "included_in_eval": record_id in eval_ids,
                }
            )
    return rows


def write_probe_artifact(
    *,
    artifact_dir: Path,
    clf,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    membership_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Path]:
    model_path = artifact_dir / "model.pkl"
    metadata_path = artifact_dir / "metadata.json"
    metrics_path = artifact_dir / "metrics.json"
    membership_path = artifact_dir / "record_membership.jsonl"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if clf is not None:
        write_pickle_atomic(model_path, clf)

    metadata = dict(metadata)
    metadata["metadata_schema_version"] = PROBE_METADATA_SCHEMA_VERSION
    metadata["saved_at_utc"] = metadata.get("saved_at_utc", utc_now_iso())
    metadata["artifact_dir"] = str(artifact_dir)
    add_path_fields(
        metadata,
        model=model_path,
        metadata=metadata_path,
        metrics=metrics_path,
        record_membership=membership_path,
    )
    write_json_atomic(metadata_path, metadata)
    write_json_atomic(metrics_path, metrics)
    write_jsonl_atomic(membership_path, list(membership_rows))

    return {
        "artifact_dir": artifact_dir,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "membership_path": membership_path,
    }


def save_probe_family_artifacts(
    *,
    run_dir: Path,
    probe_name: str,
    template_type: str,
    desc: str,
    feature_source: Dict[str, Any],
    split_records: Mapping[str, Sequence[Dict[str, Any]]],
    selection_models: Mapping[int, Any],
    selection_metrics_by_layer: Mapping[int, Dict[str, Any]],
    auc_per_layer: Mapping[int, Optional[float]],
    best_layer: Optional[int],
    best_dev_auc: Optional[float],
    chosen_model,
    chosen_metrics: Dict[str, Any],
    selection_fit_records: Sequence[Dict[str, Any]],
    selection_val_records: Sequence[Dict[str, Any]],
    chosen_fit_records: Sequence[Dict[str, Any]],
    selection_fit_max_samples: Optional[int],
    chosen_fit_max_samples: Optional[int],
    probe_seed: int,
    probe_construction: str,
    probe_example_weighting: str,
) -> Dict[str, Any]:
    all_probes_root = preferred_run_artifact_path(run_dir, "all_probes_dir")
    chosen_probe_root = preferred_run_artifact_path(run_dir, "chosen_probe_dir")
    family_all_dir = all_probes_root / probe_name
    family_chosen_dir = chosen_probe_root / probe_name
    family_all_dir.mkdir(parents=True, exist_ok=True)
    family_chosen_dir.mkdir(parents=True, exist_ok=True)

    split_summary = build_split_data_summary(split_records)
    selection_fit_summary = summarize_probe_records(selection_fit_records)
    selection_val_summary = summarize_probe_records(selection_val_records)
    chosen_fit_summary = summarize_probe_records(chosen_fit_records)
    selection_membership = build_probe_membership_rows(
        split_records=split_records,
        fit_records=selection_fit_records,
        selection_train_records=selection_fit_records,
        selection_val_records=selection_val_records,
    )
    chosen_membership = build_probe_membership_rows(
        split_records=split_records,
        fit_records=chosen_fit_records,
        selection_train_records=selection_fit_records,
        selection_val_records=selection_val_records,
    )

    saved_selection_models: list[str] = []
    trained_layers: list[int] = []
    all_probe_layers: Dict[str, Any] = {}
    chosen_source_artifact: Optional[str] = None

    for layer_id in sorted(selection_models):
        clf_layer = selection_models[layer_id]
        if clf_layer is None:
            continue
        trained_layers.append(int(layer_id))
        layer_dir = family_all_dir / f"layer_{int(layer_id):03d}"
        layer_metrics = selection_metrics_by_layer.get(int(layer_id))
        model_info = probe_model_metadata(clf_layer)
        metadata = {
            "artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
            "artifact_group": "all_probes",
            "probe_name": probe_name,
            "template_type": template_type,
            "description": desc,
            "layer": int(layer_id),
            "is_best_layer": best_layer is not None and int(layer_id) == int(best_layer),
            "label_schema": {
                "positive_label": 1,
                "negative_label": 0,
                "positive_meaning": "correct",
                "negative_meaning": "incorrect",
            },
            "feature_source": dict(feature_source),
            "selection": {
                "selection_metric": "auc",
                "selection_split": "val",
                "selection_val_auc": auc_per_layer.get(int(layer_id)),
                "best_layer": best_layer,
                "best_dev_auc": best_dev_auc,
            },
            "training": {
                "fit_strategy": (
                    "selection_choice_candidates"
                    if probe_construction == "choice_candidates"
                    else "selection_sampled_completions"
                ),
                "probe_construction": probe_construction,
                "example_weighting": probe_example_weighting,
                "fit_splits": ["train"],
                "fit_seed": int(probe_seed),
                "fit_max_samples": selection_fit_max_samples,
                "fit_summary": selection_fit_summary,
                "selection_split": "val",
                "selection_seed": int(probe_seed + 1),
                "selection_max_samples": selection_fit_max_samples,
                "selection_summary": selection_val_summary,
            },
            "evaluation": {
                "eval_splits": ["train", "val", "test"],
            },
            "data_summary": split_summary,
            "model": model_info,
        }
        saved_paths = write_probe_artifact(
            artifact_dir=layer_dir,
            clf=clf_layer,
            metadata=metadata,
            metrics=layer_metrics if layer_metrics is not None else {},
            membership_rows=selection_membership,
        )
        saved_selection_models.append(str(saved_paths["model_path"]))
        all_probe_layers[str(int(layer_id))] = {
            "artifact_dir": str(saved_paths["artifact_dir"]),
            "model_path": str(saved_paths["model_path"]),
            "metadata_path": str(saved_paths["metadata_path"]),
            "metrics_path": str(saved_paths["metrics_path"]),
            "membership_path": str(saved_paths["membership_path"]),
            "selection_val_auc": auc_per_layer.get(int(layer_id)),
            "input_dim": model_info.get("input_dim"),
        }
        if best_layer is not None and int(layer_id) == int(best_layer):
            chosen_source_artifact = str(saved_paths["artifact_dir"])

    chosen_saved_paths = None
    if chosen_model is not None and best_layer is not None:
        chosen_metadata = {
            "artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
            "artifact_group": "chosen_probe",
            "probe_name": probe_name,
            "template_type": template_type,
            "description": desc,
            "layer": int(best_layer),
            "label_schema": {
                "positive_label": 1,
                "negative_label": 0,
                "positive_meaning": "correct",
                "negative_meaning": "incorrect",
            },
            "feature_source": dict(feature_source),
            "selection": {
                "selection_metric": "auc",
                "selection_split": "val",
                "best_layer": int(best_layer),
                "best_dev_auc": best_dev_auc,
                "source_all_probes_artifact_dir": chosen_source_artifact,
            },
            "training": {
                "fit_strategy": (
                    "best_layer_retrain_choice_candidates"
                    if probe_construction == "choice_candidates"
                    else "best_layer_retrain_sampled_completions"
                ),
                "probe_construction": probe_construction,
                "example_weighting": probe_example_weighting,
                "fit_splits": ["train", "val"],
                "fit_seed": int(probe_seed),
                "fit_max_samples": chosen_fit_max_samples,
                "fit_summary": chosen_fit_summary,
                "selection_train_summary": selection_fit_summary,
                "selection_val_summary": selection_val_summary,
            },
            "evaluation": {
                "eval_splits": ["train", "val", "test"],
            },
            "data_summary": split_summary,
            "model": probe_model_metadata(chosen_model),
        }
        chosen_saved_paths = write_probe_artifact(
            artifact_dir=family_chosen_dir,
            clf=chosen_model,
            metadata=chosen_metadata,
            metrics=chosen_metrics,
            membership_rows=chosen_membership,
        )

    family_all_manifest = {
        "artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
        "artifact_group": "all_probes",
        "probe_name": probe_name,
        "template_type": template_type,
        "best_layer": best_layer,
        "best_dev_auc": best_dev_auc,
        "trained_layers": trained_layers,
        "layers": all_probe_layers,
    }
    write_json_atomic(family_all_dir / "manifest.json", family_all_manifest)

    family_chosen_manifest = {
        "artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
        "artifact_group": "chosen_probe",
        "probe_name": probe_name,
        "template_type": template_type,
        "chosen_layer": best_layer,
        "best_dev_auc": best_dev_auc,
        "artifact_dir": str(family_chosen_dir),
        "metadata_path": None if chosen_saved_paths is None else str(chosen_saved_paths["metadata_path"]),
        "metrics_path": None if chosen_saved_paths is None else str(chosen_saved_paths["metrics_path"]),
        "model_path": None if chosen_saved_paths is None else str(chosen_saved_paths["model_path"]),
        "membership_path": None if chosen_saved_paths is None else str(chosen_saved_paths["membership_path"]),
        "source_all_probes_artifact_dir": chosen_source_artifact,
    }
    write_json_atomic(family_chosen_dir / "manifest.json", family_chosen_manifest)

    return {
        "best_layer": best_layer,
        "best_dev_auc": best_dev_auc,
        "trained_layers": trained_layers,
        "auc_per_layer": dict(auc_per_layer),
        "feature_source": dict(feature_source),
        "selection_split": "val",
        "retrained_on_splits": ["train", "val"],
        "all_probes_dir": str(family_all_dir),
        "chosen_probe_dir": str(family_chosen_dir),
        "all_probes_manifest": str(family_all_dir / "manifest.json"),
        "chosen_probe_manifest": str(family_chosen_dir / "manifest.json"),
        "saved_selection_models": saved_selection_models,
        "saved_best_model": None if chosen_saved_paths is None else str(chosen_saved_paths["model_path"]),
        "chosen_probe_metadata_path": None
        if chosen_saved_paths is None
        else str(chosen_saved_paths["metadata_path"]),
        "chosen_probe_metrics_path": None
        if chosen_saved_paths is None
        else str(chosen_saved_paths["metrics_path"]),
        "chosen_probe_membership_path": None
        if chosen_saved_paths is None
        else str(chosen_saved_paths["membership_path"]),
        "selection_fit_summary": selection_fit_summary,
        "selection_val_summary": selection_val_summary,
        "chosen_fit_summary": chosen_fit_summary,
        "data_summary": split_summary,
    }


def write_probe_group_manifest(
    *,
    group_dir: Path,
    artifact_group: str,
    probe_summaries: Mapping[str, Dict[str, Any]],
) -> Path:
    group_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
        "artifact_group": artifact_group,
        "saved_at_utc": utc_now_iso(),
        "probe_names": sorted(probe_summaries),
        "probes": dict(probe_summaries),
    }
    manifest_path = group_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


__all__ = [
    "build_probe_membership_rows",
    "save_probe_family_artifacts",
    "write_probe_artifact",
    "write_probe_group_manifest",
]

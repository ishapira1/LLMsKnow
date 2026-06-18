from __future__ import annotations

import json
import os
import pickle
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .constants import RESUME_COMPAT_KEYS


CORE_PIPELINE_ARTIFACT_PATHS: Mapping[str, Path] = {
    "meta_dir": Path("meta"),
    "runtime_dir": Path("runtime"),
    "runtime_logs_dir": Path("runtime") / "logs",
    "sampling_dir": Path("sampling"),
    "sampling_raw_dir": Path("sampling") / "raw",
    "sampling_flat_dir": Path("sampling") / "flat",
    "evaluation_dir": Path("evaluation"),
    "evaluation_run_dir": Path("evaluation") / "run",
    "query_dir": Path("query"),
    "run_log": Path("runtime") / "logs" / "run.log",
    "warnings_log": Path("runtime") / "logs" / "warnings.log",
    "warnings_summary": Path("runtime") / "logs" / "warnings_summary.json",
    "run_config": Path("meta") / "run_config.json",
    "status": Path("meta") / "status.json",
    "run_summary": Path("meta") / "run_summary.json",
    "run_manifest": Path("meta") / "run_manifest.json",
    "sampling_records": Path("sampling") / "raw" / "sampling_records.jsonl",
    "sampling_manifest": Path("sampling") / "raw" / "sampling_manifest.json",
    "sampling_integrity_summary": Path("sampling") / "raw" / "sampling_integrity_summary.json",
    "sampled_responses": Path("sampling") / "flat" / "sampled_responses.csv",
    "reports_summary": Path("evaluation") / "run" / "summary.json",
    "reports_summary_csv": Path("evaluation") / "run" / "summary.csv",
    "mc_confusion_matrix": Path("evaluation") / "run" / "confusion_matrix_predicted_letter_x_true_letter.csv",
    "probe_scores_by_prompt": Path("query") / "probe_scores_by_prompt.csv",
    "executive_summary": Path("evaluation") / "run" / "executive_summary.md",
    "query_artifact_catalog": Path("query") / "artifact_catalog.jsonl",
    "query_chosen_probe_registry": Path("query") / "chosen_probe_registry.csv",
    "query_chosen_probe_metrics": Path("query") / "chosen_probe_metrics.csv",
    "query_chosen_probe_cross_family_metrics": Path("query") / "chosen_probe_cross_family_metrics.csv",
    "query_chosen_probe_movement_summary": Path("query") / "chosen_probe_movement_summary.csv",
    "query_chosen_probe_movement_items": Path("query") / "chosen_probe_movement_items.jsonl",
    "query_paraphrase_coverage": Path("query") / "paraphrase_coverage.csv",
}

OPTIONAL_PROBE_ARTIFACT_PATHS: Mapping[str, Path] = {
    "probes_dir": Path("probes"),
    "candidate_probe_families_dir": Path("probes") / "candidates" / "families",
    "chosen_probe_families_dir": Path("probes") / "chosen" / "families",
    "all_probes_dir": Path("probes") / "candidates",
    "all_probes_manifest": Path("probes") / "candidates" / "manifest.json",
    "chosen_probe_dir": Path("probes") / "chosen",
    "chosen_probe_manifest": Path("probes") / "chosen" / "manifest.json",
}

DERIVED_RUN_ARTIFACT_PATHS: Mapping[str, Path] = {
    "analysis_dir": Path("analysis"),
    "analysis_notebook_status": Path("analysis") / "analysis_notebook_status.json",
    "analysis_plots_dir": Path("analysis") / "plots",
    "analysis_tables_dir": Path("analysis") / "tables",
    "sampling_backfills_dir": Path("sampling_backfills"),
    "probe_backfills_dir": Path("probes") / "backfills",
}

ACTIVE_RUN_ARTIFACT_PATHS: Mapping[str, Path] = {
    **CORE_PIPELINE_ARTIFACT_PATHS,
    **OPTIONAL_PROBE_ARTIFACT_PATHS,
    **DERIVED_RUN_ARTIFACT_PATHS,
}

READ_COMPATIBILITY_ARTIFACT_ALIASES: Mapping[str, Sequence[Path]] = {
    "run_log": (
        Path("internal") / "logs" / "run.log",
        Path("run.log"),
    ),
    "warnings_log": (
        Path("reports") / "warnings.log",
        Path("internal") / "logs" / "warnings.log",
        Path("warnings.log"),
    ),
    "warnings_summary": (
        Path("runtime") / "logs" / "warnings_summary.json",
        Path("reports") / "warnings_summary.json",
    ),
    "run_config": (
        Path("run_config.json"),
        Path("meta") / "run_config.json",
        Path("internal") / "run_config.json",
    ),
    "status": (
        Path("status.json"),
        Path("meta") / "status.json",
        Path("internal") / "status.json",
    ),
    "sampling_records": (
        Path("sampling") / "raw" / "sampling_records.jsonl",
        Path("sampling_records.jsonl"),
        Path("internal") / "sampling_records.jsonl",
    ),
    "sampling_manifest": (
        Path("sampling") / "raw" / "sampling_manifest.json",
        Path("sampling_manifest.json"),
        Path("internal") / "sampling_manifest.json",
    ),
    "sampling_integrity_summary": (
        Path("sampling") / "raw" / "sampling_integrity_summary.json",
        Path("sampling_integrity_summary.json"),
        Path("internal") / "sampling_integrity_summary.json",
    ),
    "sampled_responses": (
        Path("sampling") / "flat" / "sampled_responses.csv",
        Path("sampling") / "sampled_responses.csv",
        Path("sampled_responses.csv"),
    ),
    "reports_summary": (
        Path("evaluation") / "run" / "summary.json",
        Path("internal") / "run_summary.json",
        Path("analysis") / "run_summary.json",
    ),
    "run_summary": (
        Path("run_summary.json"),
        Path("meta") / "run_summary.json",
        Path("analysis") / "run_summary.json",
        Path("internal") / "run_summary.json",
    ),
    "run_manifest": (),
    "executive_summary": (
        Path("evaluation") / "run" / "executive_summary.md",
        Path("summary") / "executive_summary.md",
    ),
    "all_probes_dir": (
        Path("probes") / "candidates",
        Path("all_probes"),
    ),
    "chosen_probe_dir": (
        Path("probes") / "chosen",
        Path("chosen_probe"),
    ),
    "all_probes_manifest": (
        Path("probes") / "candidates" / "manifest.json",
        Path("probes") / "all_probes" / "manifest.json",
    ),
    "chosen_probe_manifest": (
        Path("probes") / "chosen" / "manifest.json",
        Path("probes") / "chosen_probe" / "manifest.json",
    ),
    "probe_scores_by_prompt": (
        Path("query") / "probe_scores_by_prompt.csv",
        Path("probes") / "probe_scores_by_prompt.csv",
    ),
    "final_tuples": (
        Path("analysis") / "final_tuples.csv",
        Path("final_tuples.csv"),
    ),
    "summary_by_question": (
        Path("analysis") / "summary_by_question.csv",
        Path("summary_by_question.csv"),
    ),
    "model_summary_by_template": (
        Path("analysis") / "model_summary_by_template.csv",
    ),
    "model_summary_by_bias": (
        Path("analysis") / "model_summary_by_bias.csv",
    ),
    "probe_candidate_scores": (
        Path("probes") / "probe_candidate_scores.csv",
        Path("probe_candidate_scores.csv"),
    ),
    "probe_summary_csv": (
        Path("probes") / "probe_summary.csv",
    ),
    "probe_metadata": (
        Path("probes") / "probe_metadata.json",
        Path("probe_metadata.json"),
    ),
    "internal_cache_dir": (
        Path("analysis_cache"),
        Path("internal") / "cache",
    ),
}


def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify_path_token(value: Any, *, empty_fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value or "")).strip("_")
    return cleaned or empty_fallback


def _list_like_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    return [text] if text else []


def model_slug(model_name: str) -> str:
    return _slugify_path_token(model_name, empty_fallback="model")


def dataset_slug(
    dataset_name: Any,
    ays_mc_datasets: Any = None,
    *,
    fallback: str = "all",
) -> str:
    dataset_value = str(dataset_name or "").strip()
    if dataset_value:
        if dataset_value.lower() == "all":
            return "all"
        return _slugify_path_token(dataset_value, empty_fallback="dataset")

    datasets = _list_like_strings(ays_mc_datasets)
    if len(datasets) == 1:
        return _slugify_path_token(datasets[0], empty_fallback="dataset")
    if len(datasets) > 1:
        return "all"

    fallback_value = str(fallback or "").strip() or "all"
    if fallback_value.lower() == "all":
        return "all"
    return _slugify_path_token(fallback_value, empty_fallback="dataset")


def run_parent_dir(
    base_out_dir: str,
    model_name: str,
    *,
    dataset_name: Any = "all",
    ays_mc_datasets: Any = None,
    fallback_dataset_dir: str = "all",
) -> Path:
    base = Path(base_out_dir)
    model_dir = base / model_slug(model_name)
    dataset_dir = model_dir / dataset_slug(
        dataset_name,
        ays_mc_datasets,
        fallback=fallback_dataset_dir,
    )
    return dataset_dir


def build_run_dir_path(
    base_out_dir: str,
    model_name: str,
    run_name: str,
    *,
    dataset_name: Any = "all",
    ays_mc_datasets: Any = None,
    fallback_dataset_dir: str = "all",
) -> Path:
    return run_parent_dir(
        base_out_dir,
        model_name,
        dataset_name=dataset_name,
        ays_mc_datasets=ays_mc_datasets,
        fallback_dataset_dir=fallback_dataset_dir,
    ) / run_name


def build_default_run_name() -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    job_id = os.getenv("SLURM_JOB_ID") or os.getenv("PBS_JOBID") or os.getenv("JOB_ID") or "local"
    return f"{ts}_{job_id}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def build_fresh_run_name(run_name: Optional[str] = None) -> str:
    fresh_token = build_default_run_name()
    base_name = str(run_name or "").strip()
    if base_name:
        return f"{base_name}__fresh__{fresh_token}"
    return f"fresh__{fresh_token}"


def _artifact_candidate_relative_paths(artifact_key: str) -> tuple[Path, ...]:
    preferred_path = ACTIVE_RUN_ARTIFACT_PATHS.get(artifact_key)
    compatibility_paths = tuple(READ_COMPATIBILITY_ARTIFACT_ALIASES.get(artifact_key, ()))
    if preferred_path is None and not compatibility_paths:
        raise KeyError(f"Unknown artifact key: {artifact_key}")

    ordered_paths: list[Path] = []
    if preferred_path is not None:
        ordered_paths.append(preferred_path)
    for path in compatibility_paths:
        if path not in ordered_paths:
            ordered_paths.append(path)
    return tuple(ordered_paths)


def preferred_run_artifact_path(run_dir: Path, artifact_key: str) -> Path:
    try:
        relative_path = ACTIVE_RUN_ARTIFACT_PATHS[artifact_key]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact key: {artifact_key}") from exc
    return run_dir / relative_path


def resolve_run_artifact_path(run_dir: Path, artifact_key: str) -> Path:
    relative_paths = _artifact_candidate_relative_paths(artifact_key)
    preferred_path = run_dir / relative_paths[0]
    for relative_path in relative_paths:
        candidate = run_dir / relative_path
        if candidate.exists():
            return candidate
    return preferred_path


def make_run_dir(
    base_out_dir: str,
    model_name: str,
    run_name: Optional[str],
    *,
    dataset_name: Any = "all",
    ays_mc_datasets: Any = None,
    fallback_dataset_dir: str = "all",
) -> Path:
    parent_dir = run_parent_dir(
        base_out_dir,
        model_name,
        dataset_name=dataset_name,
        ays_mc_datasets=ays_mc_datasets,
        fallback_dataset_dir=fallback_dataset_dir,
    )
    parent_dir.mkdir(parents=True, exist_ok=True)
    name = run_name or build_default_run_name()
    if "/" in name or name in {".", ".."}:
        raise ValueError(f"Invalid run_name={name!r}. Use a single directory-safe token.")

    run_dir = parent_dir / name
    if run_name:
        if run_dir.exists() and not run_dir.is_dir():
            raise ValueError(f"run_name path exists but is not a directory: {run_dir}")
        run_dir.mkdir(parents=False, exist_ok=True)
    else:
        run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _canonical_resume_value(key: str, value: Any, payload: Optional[Mapping[str, Any]] = None) -> Any:
    if key == "model_backend":
        if value is None:
            model_name = ""
            if isinstance(payload, Mapping):
                model_name = str(payload.get("model", "") or "")
            if not model_name:
                return "huggingface"
            from .llm.registry import resolve_llm_backend

            return resolve_llm_backend(model_name)
        return str(value or "huggingface")

    if key == "sampling_only":
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    if key == "probe_families" and value is None:
        bias_types = []
        if isinstance(payload, Mapping):
            bias_types = _canonical_resume_value("bias_types", payload.get("bias_types"), payload)
        if bias_types:
            return ["neutral", *bias_types]
        return []

    if key not in {"ays_mc_datasets", "bias_types", "probe_families"}:
        return value

    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return value


def assert_resume_compatible(run_dir: Path, args: Any) -> None:
    cfg_path = resolve_run_artifact_path(run_dir, "run_config")
    if not cfg_path.exists():
        return

    try:
        old_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed reading existing run config at {cfg_path}: {exc}") from exc

    mismatches: Dict[str, Tuple[Any, Any]] = {}
    new_payload = vars(args) if hasattr(args, "__dict__") else {}
    for key in RESUME_COMPAT_KEYS:
        old_val = _canonical_resume_value(key, old_cfg.get(key), old_cfg)
        new_val = _canonical_resume_value(key, getattr(args, key, None), new_payload)
        if old_val != new_val:
            mismatches[key] = (old_val, new_val)

    if mismatches:
        lines = [
            "Existing run directory is not compatible with current args.",
            f"run_dir={run_dir}",
            "Mismatched keys (old -> new):",
        ]
        for key, (old_val, new_val) in mismatches.items():
            lines.append(f"  - {key}: {old_val!r} -> {new_val!r}")
        lines.append("Use a different --run_name (or keep args identical) to avoid corrupting checkpoints.")
        raise ValueError("\n".join(lines))


def run_lock_path(run_dir: Path) -> Path:
    return run_dir / ".run.lock"


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_run_lock(lock_path: Path, run_dir: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    for _attempt in range(2):
        try:
            fd = os.open(lock_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(lock_payload, handle, ensure_ascii=False, indent=2)
            return
        except FileExistsError as exc:
            existing_text = "<unreadable>"
            existing_payload: Optional[Dict[str, Any]] = None
            try:
                existing_text = lock_path.read_text(encoding="utf-8")
                maybe = json.loads(existing_text)
                if isinstance(maybe, dict):
                    existing_payload = maybe
            except Exception:
                pass

            stale = False
            status_path = resolve_run_artifact_path(run_dir, "status")
            if status_path.exists():
                try:
                    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
                    if isinstance(status_payload, dict):
                        if str(status_payload.get("status")) in {"completed", "failed", "cancelled"}:
                            stale = True
                except Exception:
                    pass

            if not stale and existing_payload is not None:
                try:
                    existing_pid = int(existing_payload.get("pid"))
                except Exception:
                    existing_pid = None
                existing_host = str(existing_payload.get("hostname", ""))
                if existing_pid is not None and existing_host == socket.gethostname():
                    if not is_pid_alive(existing_pid):
                        stale = True

            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue

            raise RuntimeError(
                f"Lock exists at {lock_path}. Another run with this run_name may still be active.\n"
                f"If this is stale, remove it manually.\nExisting lock metadata: {existing_text}"
            ) from exc

    raise RuntimeError(f"Failed to acquire lock at {lock_path}.")


def release_run_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_jsonl_atomic(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_run_status(
    run_dir: Path,
    args: Any,
    status: str,
    lock_path: Optional[Path] = None,
    error: Optional[str] = None,
) -> None:
    status_path = preferred_run_artifact_path(run_dir, "status")
    existing_status_path = resolve_run_artifact_path(run_dir, "status")
    existing: Dict[str, Any] = {}
    if existing_status_path.exists():
        try:
            loaded = json.loads(existing_status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    payload: Dict[str, Any] = dict(existing)
    now = utc_now_iso()
    payload["status"] = status
    payload["updated_at_utc"] = now
    payload.setdefault("created_at_utc", now)
    payload["model"] = args.model
    payload["model_slug"] = model_slug(args.model)
    payload["run_name"] = run_dir.name
    payload["run_dir"] = str(run_dir)
    payload["dataset_dir"] = dataset_slug(
        getattr(args, "dataset_name", "all"),
        getattr(args, "ays_mc_datasets", None),
    )
    payload["pid"] = os.getpid()
    payload["hostname"] = socket.gethostname()
    if lock_path is not None:
        payload["lock_path"] = str(lock_path)
    if error is not None:
        payload["error"] = error
    elif status == "completed":
        payload.pop("error", None)
    write_json_atomic(status_path, payload)

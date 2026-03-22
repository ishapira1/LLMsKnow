from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .runtime import dataset_slug, model_slug, resolve_run_artifact_path, write_json_atomic


DEFAULT_RESULTS_ROOT = "results/sycophancy_bias_probe"
DEFAULT_FALLBACK_DATASET_DIR = "legacy_unknown"
MANIFEST_SCHEMA_VERSION = 1
ARCHIVE_DIR_NAME = "_archived_deletions"


def _list_like_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    return [text] if text else []


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_relative_to(base: Path, path: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _resolve_stored_path(value: str, workspace_root: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((workspace_root / path).resolve())


def _logical_run_root_from_config_path(run_config_path: Path) -> Path:
    parent = run_config_path.parent
    if parent.name == "internal":
        return parent.parent
    return parent


def discover_run_roots(results_root: Path) -> list[Path]:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    run_roots = {
        _logical_run_root_from_config_path(path)
        for path in results_root.rglob("run_config.json")
        if ARCHIVE_DIR_NAME not in path.parts and ".ipynb_checkpoints" not in path.parts
    }
    roots = sorted(run_roots, key=lambda path: (len(path.parts), str(path)))
    _validate_non_overlapping_roots(roots)
    return roots


def discover_unmanaged_legacy_run_dirs(results_root: Path) -> list[Dict[str, Any]]:
    if not results_root.exists():
        return []

    unmanaged: list[Dict[str, Any]] = []
    for model_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if model_dir.name == ARCHIVE_DIR_NAME or ".ipynb_checkpoints" in model_dir.parts:
            continue
        for child in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if ARCHIVE_DIR_NAME in child.parts or ".ipynb_checkpoints" in child.parts:
                continue
            has_run_config = (child / "run_config.json").exists() or (child / "internal" / "run_config.json").exists()
            has_status = (child / "status.json").exists()
            has_lock = (child / ".run.lock").exists()
            if has_run_config or (not has_status and not has_lock):
                continue
            status_payload = _load_json(child / "status.json") if has_status else {}
            unmanaged.append(
                {
                    "path": str(child),
                    "status": str(status_payload.get("status", "") or "").strip(),
                    "updated_at_utc": str(status_payload.get("updated_at_utc", "") or "").strip(),
                    "has_status": has_status,
                    "has_lock": has_lock,
                }
            )
    return unmanaged


def _validate_non_overlapping_roots(run_roots: Sequence[Path]) -> None:
    sorted_roots = sorted(run_roots, key=lambda path: (len(path.parts), str(path)))
    for index, root in enumerate(sorted_roots):
        for other in sorted_roots[index + 1 :]:
            if root in other.parents:
                raise ValueError(
                    "Discovered overlapping logical run roots. "
                    f"Refusing to migrate ancestor={root} descendant={other}."
                )


def _json_files_for_run(run_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(run_root))
        for path in run_root.rglob("*.json")
        if ".ipynb_checkpoints" not in path.parts
    )


def _notebooks_for_run(run_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(run_root))
        for path in run_root.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def _unique_dataset_from_sampled_responses(run_root: Path) -> str:
    samples_path = resolve_run_artifact_path(run_root, "sampled_responses")
    if not samples_path.exists():
        return ""
    datasets: set[str] = set()
    with open(samples_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "dataset" not in reader.fieldnames:
            return ""
        for row in reader:
            dataset = str(row.get("dataset", "") or "").strip()
            if not dataset:
                continue
            datasets.add(dataset)
            if len(datasets) > 1:
                return ""
    if len(datasets) == 1:
        return next(iter(datasets))
    return ""


def _unique_dataset_from_sampling_records(run_root: Path) -> str:
    records_path = resolve_run_artifact_path(run_root, "sampling_records")
    if not records_path.exists():
        return ""
    datasets: set[str] = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        dataset = str(payload.get("dataset", "") or "").strip()
        if not dataset:
            continue
        datasets.add(dataset)
        if len(datasets) > 1:
            return ""
    if len(datasets) == 1:
        return next(iter(datasets))
    return ""


def infer_dataset_dir(run_root: Path) -> tuple[str, str]:
    run_config = _load_json(resolve_run_artifact_path(run_root, "run_config"))
    run_summary = _load_json(resolve_run_artifact_path(run_root, "run_summary"))

    for source_name, payload in (("run_config", run_config), ("run_summary", run_summary)):
        dataset_name = str(payload.get("dataset_name", "") or "").strip()
        if not dataset_name:
            continue
        if dataset_name.lower() == "all":
            return "all", f"{source_name}.dataset_name"
        return dataset_slug(dataset_name, fallback=DEFAULT_FALLBACK_DATASET_DIR), f"{source_name}.dataset_name"

    ays_mc_datasets = _list_like_strings(run_config.get("ays_mc_datasets"))
    if len(ays_mc_datasets) == 1:
        return dataset_slug(ays_mc_datasets[0], fallback=DEFAULT_FALLBACK_DATASET_DIR), "run_config.ays_mc_datasets"
    if len(ays_mc_datasets) > 1:
        return "all", "run_config.ays_mc_datasets"

    sampled_dataset = _unique_dataset_from_sampled_responses(run_root)
    if sampled_dataset:
        return dataset_slug(sampled_dataset, fallback=DEFAULT_FALLBACK_DATASET_DIR), "sampled_responses.dataset"

    sampling_record_dataset = _unique_dataset_from_sampling_records(run_root)
    if sampling_record_dataset:
        return dataset_slug(
            sampling_record_dataset,
            fallback=DEFAULT_FALLBACK_DATASET_DIR,
        ), "sampling_records.dataset"

    return DEFAULT_FALLBACK_DATASET_DIR, "fallback"


def infer_model_name(run_root: Path) -> str:
    run_config = _load_json(resolve_run_artifact_path(run_root, "run_config"))
    model_name = str(run_config.get("model", "") or "").strip()
    if model_name:
        return model_name
    run_summary = _load_json(resolve_run_artifact_path(run_root, "run_summary"))
    model_name = str(run_summary.get("model_name", "") or "").strip()
    if model_name:
        return model_name
    return ""


def _legacy_parent_layout_aliases(results_root: Path, run_root: Path, workspace_root: Path) -> list[str]:
    try:
        rel_parts = run_root.resolve().relative_to(results_root.resolve()).parts
    except ValueError:
        return []

    if len(rel_parts) < 2:
        return []

    model_dir = rel_parts[0]
    run_name = run_root.name
    aliases: set[str] = set()

    legacy_results_root = results_root / model_dir / run_name
    aliases.add(str(legacy_results_root.resolve()))
    try:
        aliases.add(str(legacy_results_root.resolve().relative_to(workspace_root.resolve())))
    except ValueError:
        pass

    legacy_output_root = workspace_root / "output" / results_root.name / model_dir / run_name
    aliases.add(str(legacy_output_root.resolve()))
    aliases.add(str(Path("output") / results_root.name / model_dir / run_name))

    return sorted((alias for alias in aliases if alias), key=len, reverse=True)


def _old_path_aliases(run_root: Path, results_root: Path, workspace_root: Path) -> list[str]:
    aliases: set[str] = {str(run_root.resolve())}
    try:
        aliases.add(str(run_root.resolve().relative_to(workspace_root.resolve())))
    except ValueError:
        pass

    aliases.update(_legacy_parent_layout_aliases(results_root, run_root, workspace_root))

    for artifact_key in ("run_config", "status", "run_summary"):
        payload = _load_json(resolve_run_artifact_path(run_root, artifact_key))
        stored_run_dir = str(payload.get("run_dir", "") or "").strip()
        if not stored_run_dir:
            continue
        aliases.add(stored_run_dir)
        aliases.add(_resolve_stored_path(stored_run_dir, workspace_root))

    return sorted((alias for alias in aliases if alias), key=len, reverse=True)


def _destination_root_for_run(
    results_root: Path,
    run_root: Path,
    *,
    model_name: str,
    dataset_dir: str,
) -> Path:
    rel_parts = run_root.relative_to(results_root).parts
    current_model_dir = rel_parts[0]
    canonical_model_dir = model_slug(model_name) if model_name else current_model_dir
    return results_root / canonical_model_dir / dataset_dir / run_root.name


def build_manifest_entry(run_root: Path, results_root: Path, workspace_root: Path) -> Dict[str, Any]:
    model_name = infer_model_name(run_root)
    dataset_dir, dataset_source = infer_dataset_dir(run_root)
    destination_root = _destination_root_for_run(
        results_root,
        run_root,
        model_name=model_name,
        dataset_dir=dataset_dir,
    )

    source_resolved = run_root.resolve()
    destination_resolved = destination_root.resolve()
    if source_resolved == destination_resolved:
        collision_status = "already_canonical"
    elif destination_root.exists():
        collision_status = "collision"
    else:
        collision_status = "clear"

    return {
        "source_root": str(run_root),
        "destination_root": str(destination_root),
        "run_name": run_root.name,
        "model_name": model_name,
        "model_slug": model_slug(model_name) if model_name else run_root.relative_to(results_root).parts[0],
        "dataset_dir": dataset_dir,
        "dataset_source": dataset_source,
        "collision_status": collision_status,
        "structured_files": _json_files_for_run(run_root),
        "notebooks": _notebooks_for_run(run_root),
        "old_path_aliases": _old_path_aliases(run_root, results_root, workspace_root),
    }


def build_migration_manifest(
    *,
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    workspace_root: Path | str | None = None,
) -> Dict[str, Any]:
    results_root = Path(results_root).resolve()
    workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    unmanaged_legacy_dirs = discover_unmanaged_legacy_run_dirs(results_root)
    entries = [
        build_manifest_entry(run_root, results_root=results_root, workspace_root=workspace_root)
        for run_root in discover_run_roots(results_root)
    ]
    collisions = [entry for entry in entries if entry["collision_status"] == "collision"]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "workspace_root": str(workspace_root),
        "results_root": str(results_root),
        "fallback_dataset_dir": DEFAULT_FALLBACK_DATASET_DIR,
        "entry_count": len(entries),
        "collision_count": len(collisions),
        "unmanaged_legacy_dir_count": len(unmanaged_legacy_dirs),
        "unmanaged_legacy_dirs": unmanaged_legacy_dirs,
        "entries": entries,
    }


def write_manifest(manifest: Dict[str, Any], manifest_path: Path | str) -> Path:
    path = Path(manifest_path)
    write_json_atomic(path, manifest)
    return path


def load_manifest(manifest_path: Path | str) -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest payload must be a JSON object: {path}")
    return payload


def _relative_or_absolute(path: Path, workspace_root: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path.resolve())


def manifest_replacements(manifest: Dict[str, Any], workspace_root: Path | str | None = None) -> list[tuple[str, str]]:
    workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    results_root = Path(manifest.get("results_root", DEFAULT_RESULTS_ROOT)).resolve()

    replacements: list[tuple[str, str]] = []
    seen_old: set[str] = set()

    root_aliases = [
        "output/sycophancy_bias_probe",
        str((workspace_root / "output" / "sycophancy_bias_probe").resolve()),
    ]
    for old_prefix in root_aliases:
        if old_prefix in seen_old:
            continue
        seen_old.add(old_prefix)
        new_prefix = _relative_or_absolute(results_root, workspace_root, absolute=Path(old_prefix).is_absolute())
        if old_prefix == new_prefix:
            continue
        replacements.append(
            (
                old_prefix,
                new_prefix,
            )
        )

    for entry in manifest.get("entries", []):
        destination_root = Path(entry["destination_root"]).resolve()
        destination_abs = str(destination_root)
        destination_rel = _relative_or_absolute(destination_root, workspace_root, absolute=False)
        for old_prefix in entry.get("old_path_aliases", []):
            old_text = str(old_prefix or "").strip()
            if not old_text or old_text in seen_old:
                continue
            new_text = destination_abs if Path(old_text).is_absolute() else destination_rel
            if old_text == new_text:
                continue
            seen_old.add(old_text)
            replacements.append((old_text, new_text))

    return sorted(replacements, key=lambda item: len(item[0]), reverse=True)


def _rewrite_text_file(path: Path, replacements: Sequence[tuple[str, str]]) -> None:
    original = path.read_text(encoding="utf-8")
    updated = original
    for old_prefix, new_prefix in replacements:
        updated = updated.replace(old_prefix, new_prefix)
    if updated != original:
        path.write_text(updated, encoding="utf-8")


def rewrite_run_artifacts(
    entry: Dict[str, Any],
    workspace_root: Path | str | None = None,
    *,
    replacements: Sequence[tuple[str, str]] | None = None,
) -> None:
    workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    run_root = Path(entry["destination_root"]).resolve()
    replacements = list(replacements) if replacements is not None else manifest_replacements(
        {"entries": [entry], "results_root": str(run_root.parents[2])},
        workspace_root=workspace_root,
    )
    for rel_path in entry.get("structured_files", []):
        path = run_root / rel_path
        if path.exists():
            _rewrite_text_file(path, replacements)
    for rel_path in entry.get("notebooks", []):
        path = run_root / rel_path
        if path.exists():
            _rewrite_text_file(path, replacements)


def _remove_empty_directories(results_root: Path) -> None:
    removable_dirs = sorted(
        (
            path
            for path in results_root.rglob("*")
            if path.is_dir() and ARCHIVE_DIR_NAME not in path.parts and ".ipynb_checkpoints" not in path.parts
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in removable_dirs:
        if path == results_root:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def execute_manifest(manifest: Dict[str, Any], *, workspace_root: Path | str | None = None) -> None:
    workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    entries = manifest.get("entries", [])
    replacements = manifest_replacements(manifest, workspace_root=workspace_root)
    collisions = [entry for entry in entries if entry.get("collision_status") == "collision"]
    if collisions:
        details = ", ".join(str(entry.get("destination_root")) for entry in collisions)
        raise RuntimeError(f"Refusing to execute migration with destination collisions: {details}")

    for entry in entries:
        source_root = Path(entry["source_root"]).resolve()
        destination_root = Path(entry["destination_root"]).resolve()
        if source_root == destination_root:
            rewrite_run_artifacts(entry, workspace_root=workspace_root, replacements=replacements)
            continue
        if not source_root.exists() and destination_root.exists():
            rewrite_run_artifacts(entry, workspace_root=workspace_root, replacements=replacements)
            continue
        if not source_root.exists():
            raise FileNotFoundError(f"Source run root does not exist: {source_root}")
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_root), str(destination_root))
        rewrite_run_artifacts(entry, workspace_root=workspace_root, replacements=replacements)
    results_root = Path(manifest.get("results_root", DEFAULT_RESULTS_ROOT)).resolve()
    if results_root.exists():
        _remove_empty_directories(results_root)


def _resolve_like_runtime(stored_value: str, workspace_root: Path) -> Path:
    path = Path(stored_value)
    if path.is_absolute():
        return path.resolve()
    return (workspace_root / path).resolve()


def verify_manifest(manifest: Dict[str, Any], *, workspace_root: Path | str | None = None) -> list[str]:
    workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
    stale_prefixes = [old for old, _ in manifest_replacements(manifest, workspace_root=workspace_root)]
    issues: list[str] = []
    for entry in manifest.get("entries", []):
        source_root = Path(entry["source_root"]).resolve()
        destination_root = Path(entry["destination_root"]).resolve()
        if source_root != destination_root and source_root.exists():
            issues.append(f"source root still exists after migration: {source_root}")
        if not destination_root.exists():
            issues.append(f"destination root is missing: {destination_root}")
            continue

        for rel_path in entry.get("structured_files", []):
            path = destination_root / rel_path
            if not path.exists():
                issues.append(f"missing structured file after migration: {path}")
                continue
            text = path.read_text(encoding="utf-8")
            for old_prefix in stale_prefixes:
                if old_prefix and old_prefix in text:
                    issues.append(f"stale old path reference in {path}: {old_prefix}")
                    break

        for rel_path in entry.get("notebooks", []):
            path = destination_root / rel_path
            if not path.exists():
                issues.append(f"missing notebook after migration: {path}")
                continue
            text = path.read_text(encoding="utf-8")
            for old_prefix in stale_prefixes:
                if old_prefix and old_prefix in text:
                    issues.append(f"stale old path reference in {path}: {old_prefix}")
                    break

        run_config = _load_json(resolve_run_artifact_path(destination_root, "run_config"))
        run_dir_value = str(run_config.get("run_dir", "") or "").strip()
        if run_dir_value and _resolve_like_runtime(run_dir_value, workspace_root) != destination_root:
            issues.append(f"run_config run_dir does not resolve to migrated root: {destination_root}")

        status = _load_json(resolve_run_artifact_path(destination_root, "status"))
        run_dir_value = str(status.get("run_dir", "") or "").strip()
        if run_dir_value and _resolve_like_runtime(run_dir_value, workspace_root) != destination_root:
            issues.append(f"status run_dir does not resolve to migrated root: {destination_root}")

    return issues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate sycophancy results to the dataset-aware run layout.")
    parser.add_argument("--results_root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--manifest", type=str, default="results/sycophancy_bias_probe_layout_manifest.json")
    parser.add_argument("--dry_run", action="store_true", help="Build the migration manifest without moving files.")
    parser.add_argument("--execute", action="store_true", help="Execute the moves and rewrites described by the manifest.")
    parser.add_argument("--verify_only", action="store_true", help="Verify a previously generated manifest against disk.")
    return parser


def _print_unmanaged_legacy_dirs(manifest: Dict[str, Any]) -> None:
    unmanaged = manifest.get("unmanaged_legacy_dirs", [])
    if not unmanaged:
        return
    noun = "directory was" if len(unmanaged) == 1 else "directories were"
    print(
        "[migration] warning: "
        f"{len(unmanaged)} legacy run {noun} skipped because no run_config.json was found:"
    )
    for item in unmanaged:
        status = item.get("status") or "unknown"
        updated_at = item.get("updated_at_utc") or "unknown"
        print(f"- {item.get('path')} (status={status}, updated_at_utc={updated_at})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    selected_modes = [args.dry_run, args.execute, args.verify_only]
    if sum(1 for flag in selected_modes if flag) > 1:
        parser.error("Choose only one of --dry_run, --execute, or --verify_only.")

    manifest_path = Path(args.manifest)
    workspace_root = Path.cwd().resolve()

    if args.verify_only:
        manifest = load_manifest(manifest_path)
        issues = verify_manifest(manifest, workspace_root=workspace_root)
        if issues:
            print("[migration] verification failed:")
            for issue in issues:
                print(f"- {issue}")
            return 1
        print(f"[migration] verification passed for {len(manifest.get('entries', []))} runs.")
        _print_unmanaged_legacy_dirs(manifest)
        return 0

    manifest = build_migration_manifest(results_root=args.results_root, workspace_root=workspace_root)
    write_manifest(manifest, manifest_path)
    print(
        "[migration] planned "
        f"{manifest.get('entry_count', 0)} runs "
        f"(collisions={manifest.get('collision_count', 0)}) -> {manifest_path}"
    )
    _print_unmanaged_legacy_dirs(manifest)

    if args.execute:
        execute_manifest(manifest, workspace_root=workspace_root)
        issues = verify_manifest(manifest, workspace_root=workspace_root)
        if issues:
            print("[migration] execution completed but verification failed:")
            for issue in issues:
                print(f"- {issue}")
            return 1
        print(f"[migration] execution and verification passed for {manifest.get('entry_count', 0)} runs.")
        _print_unmanaged_legacy_dirs(manifest)
        return 0

    print("[migration] dry run complete.")
    return 0

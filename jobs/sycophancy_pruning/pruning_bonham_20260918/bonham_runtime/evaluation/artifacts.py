from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .schemas import canonical_json


ARTIFACT_SCHEMA_VERSION = 1
RECORD_KEY_FIELDS: Tuple[str, ...] = (
    "run_id",
    "state_id",
    "evaluator_id",
    "example_id",
    "condition_id",
    "draw_id",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvaluationArtifactError(RuntimeError):
    """Raised when an evaluation artifact is incomplete or unauthenticated."""


def publish_immutable_directory(source: Path, destination: Path) -> None:
    """Atomically publish a completed sibling directory without replacement."""

    physical = Path(source).resolve()
    canonical = Path(destination).absolute()
    if not physical.is_dir() or physical.parent != canonical.parent.resolve():
        raise EvaluationArtifactError(
            "immutable directory publication requires source/destination under one parent"
        )
    if canonical.exists() or canonical.is_symlink():
        raise EvaluationArtifactError(f"Refusing to overwrite existing bundle {canonical}")
    try:
        # Holystore directory rename is not reliable.  symlink(2) is an atomic
        # no-replace publication and preserves the physical attempt for audit.
        os.symlink(physical.name, canonical, target_is_directory=True)
    except FileExistsError as exc:
        raise EvaluationArtifactError(f"Refusing to overwrite existing bundle {canonical}") from exc
    except OSError as exc:
        raise EvaluationArtifactError(
            f"immutable publication failed; physical attempt retained at {physical}"
        ) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise EvaluationArtifactError(f"{name} must be a SHA-256 digest")
    return normalized


def _require_text(value: Any, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise EvaluationArtifactError(f"{name} must be non-empty")
    return normalized


@dataclass(frozen=True)
class CacheIdentity:
    """Complete identity for cache reuse across intervention states and evaluators."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    snapshot_inventory_sha256: str
    chat_template_sha256: str
    state_id: str
    state_artifact_sha256: str
    evaluator_id: str
    evaluator_version: str
    parser_version: str
    dataset_id: str
    dataset_revision: str
    manifest_sha256: str
    condition_registry_sha256: str
    decoding: Mapping[str, Any]
    tool_transcript_sha256: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "state_id",
            "evaluator_id",
            "evaluator_version",
            "parser_version",
            "dataset_id",
            "dataset_revision",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        for name in (
            "snapshot_inventory_sha256",
            "chat_template_sha256",
            "state_artifact_sha256",
            "manifest_sha256",
            "condition_registry_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        if self.tool_transcript_sha256 is not None:
            object.__setattr__(
                self,
                "tool_transcript_sha256",
                _require_sha256(self.tool_transcript_sha256, "tool_transcript_sha256"),
            )
        object.__setattr__(self, "decoding", dict(self.decoding))
        object.__setattr__(self, "extra", dict(self.extra))
        if int(self.schema_version) != ARTIFACT_SCHEMA_VERSION:
            raise EvaluationArtifactError(
                f"Unsupported artifact schema version {self.schema_version}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def identity_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    return (canonical_json(value) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    columns = sorted({str(key) for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: canonical_json(row[column])
                if isinstance(row.get(column), (dict, list, tuple))
                else row.get(column, "")
                for column in columns
            }
        )
    return buffer.getvalue().encode("utf-8")


def _validate_record_keys(rows: Sequence[Mapping[str, Any]]) -> None:
    observed = set()
    for row_index, row in enumerate(rows):
        missing = [field for field in RECORD_KEY_FIELDS if field not in row]
        if missing:
            raise EvaluationArtifactError(
                f"Record {row_index} is missing key fields {missing}"
            )
        key = tuple(str(row[field]) for field in RECORD_KEY_FIELDS)
        if key in observed:
            raise EvaluationArtifactError(f"Duplicate evaluation record key: {key}")
        observed.add(key)


def write_complete_bundle(
    destination: Path,
    *,
    identity: CacheIdentity,
    records: Iterable[Mapping[str, Any]],
    metrics: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
    attempt_id: Optional[str] = None,
    extra_payloads: Optional[Mapping[str, bytes]] = None,
) -> Mapping[str, Any]:
    """Write an immutable evaluation bundle using an attempt-specific directory.

    Failed attempts are deliberately retained. A complete destination is never
    overwritten, even when its identity happens to match.
    """

    destination = Path(destination)
    if destination.exists():
        raise EvaluationArtifactError(f"Refusing to overwrite existing bundle {destination}")
    attempt = str(attempt_id or f"pid{os.getpid()}-{time.time_ns()}")
    partial = destination.parent / f"{destination.name}.partial.{attempt}"
    if partial.exists():
        raise EvaluationArtifactError(f"Attempt namespace already exists: {partial}")
    partial.mkdir(parents=True)

    record_rows = [dict(row) for row in records]
    metric_rows = [dict(row) for row in metrics]
    _validate_record_keys(record_rows)
    if not record_rows:
        raise EvaluationArtifactError("A complete bundle must contain at least one record")

    payloads = {
        "identity.json": _json_bytes(
            {**identity.to_dict(), "identity_sha256": identity.identity_sha256}, pretty=True
        ),
        "records.jsonl": _jsonl_bytes(record_rows),
        "metrics_long.csv": _csv_bytes(metric_rows),
        "summary.json": _json_bytes(dict(summary), pretty=True),
    }
    # Report-level artifacts (for example, pair-level contrasts) must be
    # authenticated just like the core files.  Keeping this small extension in
    # the common writer avoids a tempting but unsafe pattern of appending a CSV
    # after COMPLETE has been published.
    for raw_name, raw_payload in dict(extra_payloads or {}).items():
        name = str(raw_name)
        if (
            not name
            or name in payloads
            or Path(name).name != name
            or not isinstance(raw_payload, bytes)
        ):
            raise EvaluationArtifactError(
                "extra_payloads require unique flat filenames and byte payloads"
            )
        payloads[name] = raw_payload
    for name, payload in payloads.items():
        path = partial / name
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    file_sha256 = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    provenance = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "identity_sha256": identity.identity_sha256,
        "record_count": len(record_rows),
        "metric_row_count": len(metric_rows),
        "file_sha256": file_sha256,
    }
    provenance_payload = _json_bytes(provenance, pretty=True)
    (partial / "provenance.json").write_bytes(provenance_payload)
    file_sha256["provenance.json"] = sha256_bytes(provenance_payload)

    complete = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "identity_sha256": identity.identity_sha256,
        "file_sha256": dict(sorted(file_sha256.items())),
    }
    complete_payload = _json_bytes(complete, pretty=True)
    with (partial / "COMPLETE").open("xb") as handle:
        handle.write(complete_payload)
        handle.flush()
        os.fsync(handle.fileno())

    publish_immutable_directory(partial, destination)
    return validate_complete_bundle(destination, expected_identity=identity)


def validate_complete_bundle(
    bundle: Path,
    *,
    expected_identity: Optional[CacheIdentity] = None,
) -> Mapping[str, Any]:
    bundle = Path(bundle)
    if not bundle.is_dir() or not (bundle / "COMPLETE").is_file():
        raise EvaluationArtifactError(f"Bundle is not complete: {bundle}")
    try:
        complete = json.loads((bundle / "COMPLETE").read_text(encoding="utf-8"))
        identity_payload = json.loads((bundle / "identity.json").read_text(encoding="utf-8"))
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError(f"Malformed bundle metadata: {bundle}") from exc
    if int(complete.get("artifact_schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
        raise EvaluationArtifactError("Unsupported COMPLETE schema version")
    identity_sha256 = str(identity_payload.pop("identity_sha256", ""))
    computed_identity_sha256 = sha256_bytes(canonical_json(identity_payload).encode("utf-8"))
    if identity_sha256 != computed_identity_sha256:
        raise EvaluationArtifactError("Identity digest does not match identity.json")
    if complete.get("identity_sha256") != identity_sha256:
        raise EvaluationArtifactError("COMPLETE references a different identity")
    if expected_identity is not None and identity_sha256 != expected_identity.identity_sha256:
        raise EvaluationArtifactError("Complete artifact identity does not match requested cache identity")
    expected_hashes = complete.get("file_sha256")
    if not isinstance(expected_hashes, dict):
        raise EvaluationArtifactError("COMPLETE is missing file hashes")
    for name, expected_hash in expected_hashes.items():
        path = bundle / str(name)
        if not path.is_file() or sha256_file(path) != _require_sha256(expected_hash, str(name)):
            raise EvaluationArtifactError(f"Authenticated file is absent or changed: {name}")
    if provenance.get("identity_sha256") != identity_sha256:
        raise EvaluationArtifactError("Provenance identity mismatch")
    return {
        "identity_sha256": identity_sha256,
        "record_count": int(provenance["record_count"]),
        "metric_row_count": int(provenance["metric_row_count"]),
        "file_sha256": dict(expected_hashes),
    }


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CacheIdentity",
    "EvaluationArtifactError",
    "RECORD_KEY_FIELDS",
    "publish_immutable_directory",
    "sha256_bytes",
    "sha256_file",
    "validate_complete_bundle",
    "write_complete_bundle",
]


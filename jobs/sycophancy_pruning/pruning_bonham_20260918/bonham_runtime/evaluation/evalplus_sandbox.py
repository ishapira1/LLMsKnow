"""Fail-closed contract for executing EvalPlus-generated code.

Generation happens in the ordinary GPU evaluator.  This module only prepares
the already-generated samples for the pinned official EvalPlus evaluator and
constructs a Singularity CE invocation against a pinned SIF image, with no
network or host home directory.
The invocation must run in a CPU Slurm allocation; calling it from a login or
GPU evaluation process is forbidden by the contract.

The CLI follows the pinned upstream interface documented at
https://github.com/evalplus/evalplus/tree/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Sequence, Tuple

from .artifacts import EvaluationArtifactError, publish_immutable_directory

EVALPLUS_CODE_REVISION = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
EVALPLUS_DATASET_VERSIONS = {"humaneval": "0.1.10", "mbpp": "0.2.0"}
EVALPLUS_TASK_COUNTS = {"humaneval": 164, "mbpp": 378}
EVALPLUS_DATASET_SOURCES = {
    "humaneval": {
        "path": "/opt/evalplus-cache/evalplus/HumanEvalPlus-v0.1.10.jsonl",
        "sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
    },
    "mbpp": {
        "path": "/opt/evalplus-cache/evalplus/MbppPlus-v0.2.0.jsonl",
        "sha256": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
    },
}
EVALPLUS_SUBSET_RUNNER_VERSION = "evalplus_official_subset_runner_v1"
EVALPLUS_SUBSET_RUNNER_NAME = "evalplus_subset_runner.py"
EVALPLUS_RESULT_NAME = "samples_eval_results.json"
SANDBOX_CONTRACT_VERSION = "evalplus_singularity_sif_v2"


# The pinned EvalPlus CLI insists that its samples cover every problem loaded
# from the selected dataset.  Both the five-task pilot and the production
# shards are authenticated subsets, so the runner materializes the exact
# matching subset from the full release already frozen inside the SIF.  It
# also redirects EvalPlus's derived ground-truth cache into /work: the SIF's
# /opt cache is deliberately immutable and cannot accept the generated pickle.
EVALPLUS_SUBSET_RUNNER_SOURCE = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import resource
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_ids(path):
    values = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = str(json.loads(line).get("task_id", "")).strip()
            if not value or value in seen:
                raise RuntimeError("EvalPlus samples contain an empty or duplicate task_id")
            seen.add(value)
            values.append(value)
    if not values:
        raise RuntimeError("EvalPlus samples are empty")
    return tuple(values)


def main():
    if len(sys.argv) < 9:
        raise RuntimeError("invalid EvalPlus subset-runner argv")
    memory = int(sys.argv[1])
    processes = int(sys.argv[2])
    benchmark = str(sys.argv[3])
    source = Path(sys.argv[4])
    source_sha256 = str(sys.argv[5])
    samples = Path(sys.argv[6])
    subset = Path(sys.argv[7])
    evaluator_argv = list(sys.argv[8:])
    if benchmark not in {"humaneval", "mbpp"}:
        raise RuntimeError("unsupported EvalPlus benchmark")
    if not source.is_file() or sha256_file(source) != source_sha256:
        raise RuntimeError("frozen EvalPlus dataset identity mismatch")
    expected = set(task_ids(samples))
    selected = []
    observed = set()
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id", "")).strip()
            if task_id in expected:
                if task_id in observed:
                    raise RuntimeError("frozen EvalPlus dataset has a duplicate selected task")
                observed.add(task_id)
                selected.append(line if line.endswith("\n") else line + "\n")
    if observed != expected or len(selected) != len(expected):
        raise RuntimeError("EvalPlus subset does not exactly match generated samples")
    subset.write_text("".join(selected), encoding="utf-8")
    cache_root = subset.parent / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    if benchmark == "humaneval":
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(subset)
    else:
        os.environ["MBPP_OVERRIDE_PATH"] = str(subset)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
    os.execvp(evaluator_argv[0], evaluator_argv)


if __name__ == "__main__":
    main()
'''
EVALPLUS_SUBSET_RUNNER_SHA256 = hashlib.sha256(
    EVALPLUS_SUBSET_RUNNER_SOURCE.encode("utf-8")
).hexdigest()


class EvalPlusSandboxError(RuntimeError):
    """Raised before executing untrusted code when a sandbox gate is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise EvalPlusSandboxError(f"{name} must be a SHA-256 digest")
    return normalized


def _absolute_file(value: Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise EvalPlusSandboxError(f"{name} is absent: {path}")
    return path


@dataclass(frozen=True)
class EvalPlusSandboxSpec:
    benchmark: str
    samples_path: Path
    samples_sha256: str
    image_path: Path
    image_sha256: str
    work_root: Path
    code_revision: str = EVALPLUS_CODE_REVISION
    dataset_version: str = ""
    expected_task_count: int | None = None
    timeout_seconds: int = 3600
    memory_mib: int = 32768
    process_limit: int = 256

    def __post_init__(self) -> None:
        benchmark = str(self.benchmark or "").strip().lower()
        if benchmark not in EVALPLUS_DATASET_VERSIONS:
            raise EvalPlusSandboxError(f"Unsupported EvalPlus benchmark {benchmark!r}")
        object.__setattr__(self, "benchmark", benchmark)
        if str(self.code_revision) != EVALPLUS_CODE_REVISION:
            raise EvalPlusSandboxError("EvalPlus code revision differs from the frozen contract")
        expected_version = EVALPLUS_DATASET_VERSIONS[benchmark]
        observed_version = str(self.dataset_version or expected_version).strip()
        if observed_version != expected_version:
            raise EvalPlusSandboxError(
                f"{benchmark} dataset version must be {expected_version}, observed {observed_version}"
            )
        object.__setattr__(self, "dataset_version", observed_version)
        expected_task_count = (
            EVALPLUS_TASK_COUNTS[benchmark]
            if self.expected_task_count is None
            else int(self.expected_task_count)
        )
        # Production executes a hash-bound shard of the official task set in
        # each CPU job.  Completeness of the four-shard union is certified by
        # the production router; this sandbox contract authenticates the exact
        # task IDs present in its immutable samples file and result.  Hence a
        # non-empty subset is valid here, while a count larger than the pinned
        # release is never valid.
        if expected_task_count <= 0 or expected_task_count > EVALPLUS_TASK_COUNTS[benchmark]:
            raise EvalPlusSandboxError(
                "expected_task_count must be a non-empty hash-bound subset of "
                f"the pinned {benchmark} release (maximum "
                f"{EVALPLUS_TASK_COUNTS[benchmark]})"
            )
        object.__setattr__(self, "expected_task_count", expected_task_count)
        samples = _absolute_file(self.samples_path, name="samples_path")
        image = _absolute_file(self.image_path, name="image_path")
        expected_samples = _sha256(self.samples_sha256, name="samples_sha256")
        expected_image = _sha256(self.image_sha256, name="image_sha256")
        if sha256_file(samples) != expected_samples:
            raise EvalPlusSandboxError("Generated-code samples hash mismatch")
        if sha256_file(image) != expected_image:
            raise EvalPlusSandboxError("Singularity SIF image hash mismatch")
        root = Path(self.work_root).expanduser().resolve()
        if root == Path("/") or root == Path.home().resolve():
            raise EvalPlusSandboxError("work_root cannot be the filesystem or home directory")
        if int(self.timeout_seconds) <= 0 or int(self.timeout_seconds) > 24 * 3600:
            raise EvalPlusSandboxError("timeout_seconds must lie in (0, 86400]")
        if int(self.memory_mib) < 1024:
            raise EvalPlusSandboxError("memory_mib must be at least 1024")
        if int(self.process_limit) < 16 or int(self.process_limit) > 1024:
            raise EvalPlusSandboxError("process_limit must lie in [16, 1024]")
        object.__setattr__(self, "samples_path", samples)
        object.__setattr__(self, "image_path", image)
        object.__setattr__(self, "samples_sha256", expected_samples)
        object.__setattr__(self, "image_sha256", expected_image)
        object.__setattr__(self, "work_root", root)
        object.__setattr__(self, "timeout_seconds", int(self.timeout_seconds))
        object.__setattr__(self, "memory_mib", int(self.memory_mib))
        object.__setattr__(self, "process_limit", int(self.process_limit))

    @property
    def identity(self) -> Mapping[str, Any]:
        dataset_source = EVALPLUS_DATASET_SOURCES[self.benchmark]
        return {
            "contract_version": SANDBOX_CONTRACT_VERSION,
            "benchmark": self.benchmark,
            "samples_sha256": self.samples_sha256,
            "image_sha256": self.image_sha256,
            "code_revision": self.code_revision,
            "dataset_version": self.dataset_version,
            "expected_task_count": self.expected_task_count,
            "timeout_seconds": self.timeout_seconds,
            "memory_mib": self.memory_mib,
            "process_limit": self.process_limit,
            "dataset_source_path": dataset_source["path"],
            "dataset_source_sha256": dataset_source["sha256"],
            "subset_runner_version": EVALPLUS_SUBSET_RUNNER_VERSION,
            "subset_runner_sha256": EVALPLUS_SUBSET_RUNNER_SHA256,
            "result_name": EVALPLUS_RESULT_NAME,
        }


def _read_samples(path: Path, *, expected_count: int) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalPlusSandboxError(f"Invalid samples JSON at line {line_number}") from exc
            if not isinstance(row, Mapping):
                raise EvalPlusSandboxError(f"Sample line {line_number} is not an object")
            task_id = str(row.get("task_id", "")).strip()
            solution = row.get("solution", row.get("completion"))
            if not task_id or not isinstance(solution, str) or not solution.strip():
                raise EvalPlusSandboxError(
                    f"Sample line {line_number} requires task_id and non-empty solution/completion"
                )
            if task_id in seen:
                raise EvalPlusSandboxError(f"Duplicate EvalPlus task_id {task_id!r}")
            seen.add(task_id)
            rows.append(dict(row))
    if len(rows) != expected_count:
        raise EvalPlusSandboxError(
            f"Expected {expected_count} {Path(path).name} samples, observed {len(rows)}"
        )
    return tuple(rows)


def prepare_evalplus_workspace(spec: EvalPlusSandboxSpec) -> Path:
    """Create a content-addressed, immutable input copy for one CPU run."""

    _read_samples(spec.samples_path, expected_count=int(spec.expected_task_count))
    identity_bytes = json.dumps(spec.identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity_sha = hashlib.sha256(identity_bytes).hexdigest()
    destination = spec.work_root / f"{spec.benchmark}_{identity_sha}"
    if destination.exists():
        marker = destination / "INPUT_COMPLETE.json"
        if not marker.is_file():
            raise EvalPlusSandboxError(f"Incomplete workspace exists and will not be deleted: {destination}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        copied = destination / "samples.jsonl"
        runner = destination / EVALPLUS_SUBSET_RUNNER_NAME
        if (
            payload.get("identity_sha256") != identity_sha
            or sha256_file(copied) != spec.samples_sha256
            or not runner.is_file()
            or sha256_file(runner) != EVALPLUS_SUBSET_RUNNER_SHA256
        ):
            raise EvalPlusSandboxError("Existing EvalPlus workspace identity mismatch")
        return destination

    attempt = destination.parent / f"{destination.name}.partial.pid_{os.getpid()}"
    if attempt.exists():
        raise EvalPlusSandboxError(f"Attempt namespace already exists: {attempt}")
    attempt.mkdir(parents=True)
    copied = attempt / "samples.jsonl"
    shutil.copyfile(spec.samples_path, copied)
    if sha256_file(copied) != spec.samples_sha256:
        raise EvalPlusSandboxError("Samples changed while preparing the sandbox workspace")
    runner = attempt / EVALPLUS_SUBSET_RUNNER_NAME
    runner.write_text(EVALPLUS_SUBSET_RUNNER_SOURCE, encoding="utf-8")
    if sha256_file(runner) != EVALPLUS_SUBSET_RUNNER_SHA256:
        raise EvalPlusSandboxError("EvalPlus subset runner changed while preparing workspace")
    marker = {
        "identity": dict(spec.identity),
        "identity_sha256": identity_sha,
        "samples_sha256": spec.samples_sha256,
    }
    (attempt / "INPUT_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        publish_immutable_directory(attempt, destination)
    except EvaluationArtifactError as exc:
        raise EvalPlusSandboxError(str(exc)) from exc
    return destination


def build_singularity_command(
    spec: EvalPlusSandboxSpec,
    workspace: Path,
    *,
    singularity_binary: str = "singularity",
) -> Tuple[str, ...]:
    """Return a Singularity CE argv; callers must never invoke it via a shell."""

    work = Path(workspace).resolve()
    if work.parent != spec.work_root or not (work / "INPUT_COMPLETE.json").is_file():
        raise EvalPlusSandboxError("Workspace is outside the authenticated work root")
    if sha256_file(work / "samples.jsonl") != spec.samples_sha256:
        raise EvalPlusSandboxError("Workspace samples changed before execution")
    runner = work / EVALPLUS_SUBSET_RUNNER_NAME
    if not runner.is_file() or sha256_file(runner) != EVALPLUS_SUBSET_RUNNER_SHA256:
        raise EvalPlusSandboxError("Workspace subset runner changed before execution")
    binary = str(singularity_binary or "").strip()
    if not binary or any(character.isspace() for character in binary):
        raise EvalPlusSandboxError("singularity_binary must be one argv token")
    dataset_source = EVALPLUS_DATASET_SOURCES[spec.benchmark]
    subset_path = f"/work/{spec.benchmark}_subset.jsonl"
    return (
        "timeout",
        "--signal=KILL",
        str(spec.timeout_seconds),
        binary,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--writable-tmpfs",
        "--no-mount",
        "home,cwd,hostfs",
        "--bind",
        f"{work}:/work:rw",
        "--pwd",
        "/work",
        "--env",
        "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,NO_PROXY=*",
        str(spec.image_path),
        "python3",
        f"/work/{EVALPLUS_SUBSET_RUNNER_NAME}",
        str(spec.memory_mib * 1024 * 1024),
        str(spec.process_limit),
        spec.benchmark,
        str(dataset_source["path"]),
        str(dataset_source["sha256"]),
        "/work/samples.jsonl",
        subset_path,
        "evalplus.evaluate",
        "--dataset",
        spec.benchmark,
        "--samples",
        "/work/samples.jsonl",
        "--output_file",
        f"/work/{EVALPLUS_RESULT_NAME}",
    )


def build_apptainer_command(
    spec: EvalPlusSandboxSpec,
    workspace: Path,
    *,
    apptainer_binary: str = "apptainer",
) -> Tuple[str, ...]:
    """Compatibility alias for old callers; production uses Singularity CE."""

    return build_singularity_command(
        spec,
        workspace,
        singularity_binary=apptainer_binary,
    )


def validate_evalplus_results(
    spec: EvalPlusSandboxSpec,
    workspace: Path,
    result_path: Path,
) -> Mapping[str, Any]:
    """Authenticate official results and preserve base/plus outcomes per task."""

    work = Path(workspace).resolve()
    result = Path(result_path).resolve()
    if work.parent != spec.work_root or work not in result.parents:
        raise EvalPlusSandboxError("Result path is outside its authenticated workspace")
    if sha256_file(work / "samples.jsonl") != spec.samples_sha256:
        raise EvalPlusSandboxError("Sandbox modified the immutable samples")
    if not result.is_file():
        raise EvalPlusSandboxError(f"EvalPlus result is absent: {result}")
    payload = json.loads(result.read_text(encoding="utf-8"))
    eval_rows = payload.get("eval") if isinstance(payload, Mapping) else None
    if not isinstance(eval_rows, Mapping):
        raise EvalPlusSandboxError("Official EvalPlus result lacks an eval mapping")
    expected_ids = {
        str(row["task_id"])
        for row in _read_samples(
            work / "samples.jsonl", expected_count=int(spec.expected_task_count)
        )
    }
    observed_ids = {str(value) for value in eval_rows}
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise EvalPlusSandboxError(
            f"EvalPlus result task mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    return {
        "status": "complete",
        "identity": dict(spec.identity),
        "task_count": len(expected_ids),
        "samples_sha256": spec.samples_sha256,
        "result_sha256": sha256_file(result),
        "result_path": str(result),
    }


__all__ = [
    "EVALPLUS_CODE_REVISION",
    "EVALPLUS_DATASET_SOURCES",
    "EVALPLUS_DATASET_VERSIONS",
    "EVALPLUS_RESULT_NAME",
    "EVALPLUS_SUBSET_RUNNER_NAME",
    "EVALPLUS_SUBSET_RUNNER_SHA256",
    "EVALPLUS_SUBSET_RUNNER_SOURCE",
    "EVALPLUS_SUBSET_RUNNER_VERSION",
    "EVALPLUS_TASK_COUNTS",
    "SANDBOX_CONTRACT_VERSION",
    "EvalPlusSandboxError",
    "EvalPlusSandboxSpec",
    "build_apptainer_command",
    "build_singularity_command",
    "prepare_evalplus_workspace",
    "sha256_file",
    "validate_evalplus_results",
]


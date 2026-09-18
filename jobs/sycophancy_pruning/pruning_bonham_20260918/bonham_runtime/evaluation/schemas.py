from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = 1
_HEX_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

DATASET_ROLES = (
    "attribution",
    "development",
    "promotion",
    "same_source_final",
    "unseen_source_final",
    "diagnostic",
)
CONDITION_USES = (
    "pruning",
    "preservation",
    "development_evaluation",
    "final_evaluation",
    "diagnostic",
)
CONDITION_CHANNELS = (
    "neutral",
    "approval",
    "epistemic",
    "mixed",
    "utility",
    "aggregate",
)
CLAIM_TRUTHS = ("correct", "incorrect", "mixed", "not_applicable")
PROMPT_PLACEMENTS = (
    "after_options",
    "before_question",
    "followup",
    "embedded",
    "single_turn",
    "structured_messages",
    "benchmark_native",
)
INTERVENTION_KINDS = (
    "base",
    "system_prompt",
    "mask",
    "activation_steering",
    "random_mask",
)


class EvaluationSchemaError(ValueError):
    """Raised when a causal-evaluation artifact is underspecified or inconsistent."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise EvaluationSchemaError(f"{field_name} must be non-empty")
    return text


def _require_revision(value: Any, *, field_name: str) -> str:
    revision = _require_text(value, field_name=field_name).lower()
    if not _HEX_REVISION_RE.fullmatch(revision):
        raise EvaluationSchemaError(
            f"{field_name} must be an immutable 7-64 character hexadecimal revision"
        )
    return revision


def _normalize_hashes(values: Mapping[str, str], *, field_name: str) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for raw_name, raw_hash in values.items():
        name = _require_text(raw_name, field_name=f"{field_name} key")
        checksum = str(raw_hash or "").strip().lower()
        if not _SHA256_RE.fullmatch(checksum):
            raise EvaluationSchemaError(f"{field_name}[{name!r}] must be a SHA-256 digest")
        normalized[name] = checksum
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    repo_id: str
    revision: str
    config_name: Optional[str]
    intended_splits: Tuple[str, ...]
    roles: Tuple[str, ...]
    license_status: str
    row_counts: Mapping[str, int] = field(default_factory=dict)
    file_sha256: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        dataset_id = _require_text(self.dataset_id, field_name="dataset_id")
        if not _CONDITION_ID_RE.fullmatch(dataset_id):
            raise EvaluationSchemaError(f"Invalid dataset_id {dataset_id!r}")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "display_name", _require_text(self.display_name, field_name="display_name"))
        object.__setattr__(self, "repo_id", _require_text(self.repo_id, field_name="repo_id"))
        object.__setattr__(self, "revision", _require_revision(self.revision, field_name="revision"))
        config_name = str(self.config_name or "").strip() or None
        object.__setattr__(self, "config_name", config_name)
        splits = tuple(dict.fromkeys(str(value or "").strip() for value in self.intended_splits))
        if not splits or any(not value for value in splits):
            raise EvaluationSchemaError("intended_splits must contain unique non-empty split names")
        object.__setattr__(self, "intended_splits", splits)
        roles = tuple(dict.fromkeys(str(value or "").strip() for value in self.roles))
        invalid_roles = sorted(set(roles).difference(DATASET_ROLES))
        if not roles or invalid_roles:
            raise EvaluationSchemaError(f"Invalid dataset roles: {invalid_roles}")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(
            self,
            "license_status",
            _require_text(self.license_status, field_name="license_status"),
        )
        counts = {str(key): int(value) for key, value in self.row_counts.items()}
        if any(value < 0 for value in counts.values()):
            raise EvaluationSchemaError("row_counts cannot contain negative values")
        if set(counts).difference(splits):
            raise EvaluationSchemaError("row_counts contains a split absent from intended_splits")
        object.__setattr__(self, "row_counts", dict(sorted(counts.items())))
        object.__setattr__(
            self,
            "file_sha256",
            _normalize_hashes(self.file_sha256, field_name="file_sha256"),
        )
        if int(self.schema_version) != SCHEMA_VERSION:
            raise EvaluationSchemaError(f"Unsupported dataset schema version {self.schema_version}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["intended_splits"] = list(self.intended_splits)
        payload["roles"] = list(self.roles)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            dataset_id=value["dataset_id"],
            display_name=value["display_name"],
            repo_id=value["repo_id"],
            revision=value["revision"],
            config_name=value.get("config_name"),
            intended_splits=tuple(value["intended_splits"]),
            roles=tuple(value["roles"]),
            license_status=value["license_status"],
            row_counts=dict(value.get("row_counts", {})),
            file_sha256=dict(value.get("file_sha256", {})),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    display_name: str
    renderer_id: str
    channel: str
    family: str
    placement: str
    claim_truth: str
    source_type: str
    source_reliability: Optional[float]
    allowed_uses: Tuple[str, ...]
    aggregate_only: bool = False
    final_only: bool = False
    evaluator_id: str = "direct_factual_sycophancy_v1"
    suite_section: str = "A. Direct factual-sycophancy behavior"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        condition_id = _require_text(self.condition_id, field_name="condition_id")
        if not _CONDITION_ID_RE.fullmatch(condition_id):
            raise EvaluationSchemaError(f"Invalid condition_id {condition_id!r}")
        object.__setattr__(self, "condition_id", condition_id)
        for field_name in (
            "display_name",
            "renderer_id",
            "channel",
            "family",
            "placement",
            "claim_truth",
            "source_type",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name=field_name),
            )
        if self.channel not in CONDITION_CHANNELS:
            raise EvaluationSchemaError(f"Unknown condition channel {self.channel!r}")
        if self.claim_truth not in CLAIM_TRUTHS:
            raise EvaluationSchemaError(f"Unknown claim_truth {self.claim_truth!r}")
        if self.placement not in PROMPT_PLACEMENTS:
            raise EvaluationSchemaError(f"Unknown prompt placement {self.placement!r}")
        if self.source_reliability is not None:
            reliability = float(self.source_reliability)
            if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise EvaluationSchemaError("source_reliability must lie in [0, 1]")
            object.__setattr__(self, "source_reliability", reliability)
        uses = tuple(dict.fromkeys(str(value or "").strip() for value in self.allowed_uses))
        invalid_uses = sorted(set(uses).difference(CONDITION_USES))
        if not uses or invalid_uses:
            raise EvaluationSchemaError(f"Invalid condition uses: {invalid_uses}")
        if self.final_only and any(value != "final_evaluation" for value in uses):
            raise EvaluationSchemaError("final_only conditions can only allow final_evaluation")
        if self.aggregate_only and self.renderer_id != "aggregate":
            raise EvaluationSchemaError("aggregate_only conditions must use renderer_id='aggregate'")
        object.__setattr__(
            self,
            "evaluator_id",
            _require_text(self.evaluator_id, field_name="evaluator_id"),
        )
        object.__setattr__(
            self,
            "suite_section",
            _require_text(self.suite_section, field_name="suite_section"),
        )
        object.__setattr__(self, "allowed_uses", uses)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if int(self.schema_version) != SCHEMA_VERSION:
            raise EvaluationSchemaError(f"Unsupported condition schema version {self.schema_version}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["allowed_uses"] = list(self.allowed_uses)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConditionSpec":
        return cls(
            condition_id=value["condition_id"],
            display_name=value["display_name"],
            renderer_id=value["renderer_id"],
            channel=value["channel"],
            family=value["family"],
            placement=value["placement"],
            claim_truth=value["claim_truth"],
            source_type=value["source_type"],
            source_reliability=value.get("source_reliability"),
            allowed_uses=tuple(value["allowed_uses"]),
            aggregate_only=bool(value.get("aggregate_only", False)),
            final_only=bool(value.get("final_only", False)),
            evaluator_id=value.get("evaluator_id", "direct_factual_sycophancy_v1"),
            suite_section=value.get(
                "suite_section", "A. Direct factual-sycophancy behavior"
            ),
            metadata=dict(value.get("metadata", {})),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class StateSpec:
    state_id: str
    display_name: str
    intervention_kind: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    parameters_set_to_zero: int
    total_model_parameters: int
    eligible_pruning_parameters: int
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    steering_layer: Optional[int] = None
    steering_alpha: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        state_id = _require_text(self.state_id, field_name="state_id")
        if not _CONDITION_ID_RE.fullmatch(state_id):
            raise EvaluationSchemaError(f"Invalid state_id {state_id!r}")
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "display_name", _require_text(self.display_name, field_name="display_name"))
        kind = _require_text(self.intervention_kind, field_name="intervention_kind")
        if kind not in INTERVENTION_KINDS:
            raise EvaluationSchemaError(f"Unknown intervention_kind {kind!r}")
        object.__setattr__(self, "intervention_kind", kind)
        object.__setattr__(self, "model_id", _require_text(self.model_id, field_name="model_id"))
        object.__setattr__(
            self,
            "model_revision",
            _require_revision(self.model_revision, field_name="model_revision"),
        )
        object.__setattr__(
            self,
            "tokenizer_revision",
            _require_revision(self.tokenizer_revision, field_name="tokenizer_revision"),
        )
        k = int(self.parameters_set_to_zero)
        total = int(self.total_model_parameters)
        eligible = int(self.eligible_pruning_parameters)
        if total <= 0 or eligible <= 0 or eligible > total:
            raise EvaluationSchemaError("Invalid total/eligible parameter counts")
        if k < 0 or k > eligible:
            raise EvaluationSchemaError("parameters_set_to_zero must lie in the eligible universe")
        if kind in {"base", "system_prompt", "activation_steering"} and k != 0:
            raise EvaluationSchemaError(f"{kind} states must report parameters_set_to_zero=0")
        if kind in {"mask", "random_mask"} and k <= 0:
            raise EvaluationSchemaError(f"{kind} states must set at least one parameter to zero")
        object.__setattr__(self, "parameters_set_to_zero", k)
        object.__setattr__(self, "total_model_parameters", total)
        object.__setattr__(self, "eligible_pruning_parameters", eligible)
        object.__setattr__(
            self,
            "artifact_sha256",
            _normalize_hashes(self.artifact_sha256, field_name="artifact_sha256"),
        )
        artifact_hashes = dict(self.artifact_sha256)
        prompt = str(self.system_prompt or "").strip() or None
        if kind == "system_prompt" and prompt is None:
            raise EvaluationSchemaError("system_prompt state is missing its prompt")
        if kind != "system_prompt" and prompt is not None:
            raise EvaluationSchemaError("Only system_prompt states may set system_prompt")
        object.__setattr__(self, "system_prompt", prompt)
        if kind == "system_prompt":
            expected_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if artifact_hashes.get("system_prompt") != expected_prompt_hash:
                raise EvaluationSchemaError(
                    "system_prompt state must authenticate the exact prompt text"
                )
        if kind in {"mask", "random_mask"} and not {
            "indices",
            "metadata",
        }.issubset(artifact_hashes):
            raise EvaluationSchemaError(
                f"{kind} state requires indices and metadata SHA-256 identities"
            )
        if kind == "activation_steering":
            if self.steering_layer is None or self.steering_alpha is None:
                raise EvaluationSchemaError("activation_steering requires layer and alpha")
            if int(self.steering_layer) <= 0 or not math.isfinite(float(self.steering_alpha)):
                raise EvaluationSchemaError("Invalid activation-steering layer/alpha")
            object.__setattr__(self, "steering_layer", int(self.steering_layer))
            object.__setattr__(self, "steering_alpha", float(self.steering_alpha))
            if "direction" not in artifact_hashes:
                raise EvaluationSchemaError(
                    "activation_steering state requires a direction SHA-256 identity"
                )
        elif self.steering_layer is not None or self.steering_alpha is not None:
            raise EvaluationSchemaError("Only activation_steering states may set layer/alpha")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if int(self.schema_version) != SCHEMA_VERSION:
            raise EvaluationSchemaError(f"Unsupported state schema version {self.schema_version}")

    @property
    def fraction_pruned(self) -> float:
        return self.parameters_set_to_zero / self.total_model_parameters

    @property
    def parts_per_million_pruned(self) -> float:
        return 1_000_000.0 * self.fraction_pruned

    @property
    def eligible_fraction_pruned(self) -> float:
        return self.parameters_set_to_zero / self.eligible_pruning_parameters

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "fraction_pruned": self.fraction_pruned,
                "parts_per_million_pruned": self.parts_per_million_pruned,
                "eligible_fraction_pruned": self.eligible_fraction_pruned,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateSpec":
        return cls(
            state_id=value["state_id"],
            display_name=value["display_name"],
            intervention_kind=value["intervention_kind"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            tokenizer_revision=value["tokenizer_revision"],
            parameters_set_to_zero=int(value["parameters_set_to_zero"]),
            total_model_parameters=int(value["total_model_parameters"]),
            eligible_pruning_parameters=int(value["eligible_pruning_parameters"]),
            artifact_sha256=dict(value.get("artifact_sha256", {})),
            system_prompt=value.get("system_prompt"),
            steering_layer=value.get("steering_layer"),
            steering_alpha=value.get("steering_alpha"),
            metadata=dict(value.get("metadata", {})),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class EvalRecord:
    run_id: str
    state_id: str
    evaluator_id: str
    example_id: str
    condition_id: str
    draw_id: int
    dataset_id: str
    split: str
    messages: Tuple[Mapping[str, Any], ...]
    rendered_prompt_sha256: str
    raw_output: str
    parsed_choice: str
    parse_status: str
    choice_probabilities: Mapping[str, float]
    gold_choice: str
    target_choice: Optional[str]
    refusal: bool
    invalid: bool
    truncated: bool
    runtime_seconds: float
    device: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "state_id",
            "evaluator_id",
            "example_id",
            "condition_id",
            "dataset_id",
            "split",
            "rendered_prompt_sha256",
            "parse_status",
            "gold_choice",
            "device",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name=field_name),
            )
        if not _SHA256_RE.fullmatch(self.rendered_prompt_sha256.lower()):
            raise EvaluationSchemaError("rendered_prompt_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "rendered_prompt_sha256", self.rendered_prompt_sha256.lower())
        if int(self.draw_id) < 0:
            raise EvaluationSchemaError("draw_id must be non-negative")
        object.__setattr__(self, "draw_id", int(self.draw_id))
        if not self.messages:
            raise EvaluationSchemaError("messages must be non-empty")
        normalized_probabilities: Dict[str, float] = {}
        for raw_label, raw_probability in self.choice_probabilities.items():
            label = _require_text(raw_label, field_name="choice label").upper()
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise EvaluationSchemaError("choice probabilities must be finite and in [0, 1]")
            if label in normalized_probabilities:
                raise EvaluationSchemaError(f"Duplicate normalized choice label {label!r}")
            normalized_probabilities[label] = probability
        if len(normalized_probabilities) < 2:
            raise EvaluationSchemaError("At least two choice probabilities are required")
        if not math.isclose(sum(normalized_probabilities.values()), 1.0, abs_tol=1e-6):
            raise EvaluationSchemaError("choice probabilities must sum to one")
        gold = self.gold_choice.strip().upper()
        target = str(self.target_choice or "").strip().upper() or None
        parsed = str(self.parsed_choice or "").strip().upper()
        if gold not in normalized_probabilities:
            raise EvaluationSchemaError("gold_choice is absent from choice_probabilities")
        if target is not None and target not in normalized_probabilities:
            raise EvaluationSchemaError("target_choice is absent from choice_probabilities")
        if parsed and parsed not in normalized_probabilities:
            raise EvaluationSchemaError("parsed_choice is absent from choice_probabilities")
        object.__setattr__(self, "gold_choice", gold)
        object.__setattr__(self, "target_choice", target)
        object.__setattr__(self, "parsed_choice", parsed)
        object.__setattr__(self, "choice_probabilities", normalized_probabilities)
        runtime = float(self.runtime_seconds)
        if not math.isfinite(runtime) or runtime < 0.0:
            raise EvaluationSchemaError("runtime_seconds must be finite and non-negative")
        object.__setattr__(self, "runtime_seconds", runtime)
        object.__setattr__(self, "messages", tuple(dict(message) for message in self.messages))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.refusal != (self.parse_status == "refusal"):
            raise EvaluationSchemaError("refusal flag must agree with parse_status")
        if self.invalid != (self.parse_status != "valid"):
            raise EvaluationSchemaError("invalid flag must agree with parse_status")
        if int(self.schema_version) != SCHEMA_VERSION:
            raise EvaluationSchemaError(f"Unsupported record schema version {self.schema_version}")

    @property
    def key(self) -> Tuple[str, str, str, str, str, int]:
        return (
            self.run_id,
            self.state_id,
            self.evaluator_id,
            self.example_id,
            self.condition_id,
            self.draw_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["messages"] = [dict(message) for message in self.messages]
        return payload


__all__ = [
    "CLAIM_TRUTHS",
    "CONDITION_CHANNELS",
    "CONDITION_USES",
    "DATASET_ROLES",
    "INTERVENTION_KINDS",
    "PROMPT_PLACEMENTS",
    "ConditionSpec",
    "DatasetSpec",
    "EvalRecord",
    "EvaluationSchemaError",
    "SCHEMA_VERSION",
    "StateSpec",
    "canonical_json",
    "sha256_json",
]


from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .schemas import ConditionSpec, DatasetSpec, EvaluationSchemaError, StateSpec, canonical_json


REGISTRY_SCHEMA_VERSION = 1

DIRECT_FACTUAL_DISPLAY_NAMES: Tuple[str, ...] = (
    "Weak incorrect suggestion only",
    "Strong incorrect suggestion",
    "doubt-correct",
    "bad_sycoaphncy_all",
    "bad_sycoaphncy_all_repharse",
    "correct-suggestion",
    "expert incorrect",
    "expert correct",
    "Source reliability high reliability",
    "soure realiable with low reliability",
    'desribe without beliefs ("i want X to be the answer")',
    "stronger held-out prompt-generalization test",
)


class RegistryError(EvaluationSchemaError):
    """Raised when a registry is ambiguous, duplicated, or has the wrong type."""


def _atomic_write(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class DatasetRegistry:
    specs: Tuple[DatasetSpec, ...]

    def __post_init__(self) -> None:
        ids = [spec.dataset_id for spec in self.specs]
        if len(ids) != len(set(ids)):
            raise RegistryError("Dataset registry contains duplicate dataset_id values")

    def __iter__(self) -> Iterator[DatasetSpec]:
        return iter(self.specs)

    def get(self, dataset_id: str) -> DatasetSpec:
        matches = [spec for spec in self.specs if spec.dataset_id == str(dataset_id)]
        if not matches:
            raise RegistryError(f"Unknown dataset_id {dataset_id!r}")
        return matches[0]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "registry_type": "dataset",
            "datasets": [spec.to_dict() for spec in self.specs],
        }

    def write_json(self, path: Path) -> None:
        _atomic_write(Path(path), json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "DatasetRegistry":
        if value.get("registry_type") != "dataset":
            raise RegistryError("Expected a dataset registry")
        if int(value.get("schema_version", -1)) != REGISTRY_SCHEMA_VERSION:
            raise RegistryError("Unsupported dataset-registry schema version")
        rows = value.get("datasets")
        if not isinstance(rows, list):
            raise RegistryError("Dataset registry must contain a datasets list")
        return cls(tuple(DatasetSpec.from_dict(row) for row in rows))

    @classmethod
    def read_json(cls, path: Path) -> "DatasetRegistry":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ConditionRegistry:
    specs: Tuple[ConditionSpec, ...]

    def __post_init__(self) -> None:
        ids = [spec.condition_id for spec in self.specs]
        if len(ids) != len(set(ids)):
            raise RegistryError("Condition registry contains duplicate condition_id values")

    def __iter__(self) -> Iterator[ConditionSpec]:
        return iter(self.specs)

    def get(self, condition_id: str) -> ConditionSpec:
        matches = [spec for spec in self.specs if spec.condition_id == str(condition_id)]
        if not matches:
            raise RegistryError(f"Unknown condition_id {condition_id!r}")
        return matches[0]

    def for_use(self, use: str) -> Tuple[ConditionSpec, ...]:
        return tuple(spec for spec in self.specs if str(use) in spec.allowed_uses)

    def for_display_name(self, display_name: str) -> Tuple[ConditionSpec, ...]:
        return tuple(spec for spec in self.specs if spec.display_name == str(display_name))

    def for_evaluator(self, evaluator_id: str) -> Tuple[ConditionSpec, ...]:
        return tuple(spec for spec in self.specs if spec.evaluator_id == str(evaluator_id))

    def physical(self, *, evaluator_id: Optional[str] = None) -> Tuple[ConditionSpec, ...]:
        return tuple(
            spec
            for spec in self.specs
            if not spec.aggregate_only
            and (evaluator_id is None or spec.evaluator_id == str(evaluator_id))
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "registry_type": "condition",
            "required_display_names": list(DIRECT_FACTUAL_DISPLAY_NAMES),
            "conditions": [spec.to_dict() for spec in self.specs],
        }

    def write_json(self, path: Path) -> None:
        _atomic_write(Path(path), json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "ConditionRegistry":
        if value.get("registry_type") != "condition":
            raise RegistryError("Expected a condition registry")
        if int(value.get("schema_version", -1)) != REGISTRY_SCHEMA_VERSION:
            raise RegistryError("Unsupported condition-registry schema version")
        rows = value.get("conditions")
        if not isinstance(rows, list):
            raise RegistryError("Condition registry must contain a conditions list")
        registry = cls(tuple(ConditionSpec.from_dict(row) for row in rows))
        observed = {spec.display_name for spec in registry.specs}
        missing = [name for name in DIRECT_FACTUAL_DISPLAY_NAMES if name not in observed]
        if missing:
            raise RegistryError(f"Condition registry is missing exact public names: {missing}")
        return registry

    @classmethod
    def read_json(cls, path: Path) -> "ConditionRegistry":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class StateRegistry:
    specs: Tuple[StateSpec, ...]

    def __post_init__(self) -> None:
        keys = [(spec.model_id, spec.state_id) for spec in self.specs]
        if len(keys) != len(set(keys)):
            raise RegistryError("State registry contains duplicate (model_id, state_id) values")

    def __iter__(self) -> Iterator[StateSpec]:
        return iter(self.specs)

    def get(self, model_id: str, state_id: str) -> StateSpec:
        matches = [
            spec
            for spec in self.specs
            if spec.model_id == str(model_id) and spec.state_id == str(state_id)
        ]
        if not matches:
            raise RegistryError(f"Unknown state {(model_id, state_id)!r}")
        return matches[0]

    def write_jsonl(self, path: Path) -> None:
        rows = [
            canonical_json(
                {
                    "registry_schema_version": REGISTRY_SCHEMA_VERSION,
                    "registry_type": "state",
                    **spec.to_dict(),
                }
            )
            for spec in self.specs
        ]
        _atomic_write(Path(path), "".join(f"{row}\n" for row in rows))

    @classmethod
    def read_jsonl(cls, path: Path) -> "StateRegistry":
        specs: List[StateSpec] = []
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegistryError(f"Invalid state registry JSON on line {line_number}") from exc
            if row.get("registry_type") != "state" or int(
                row.get("registry_schema_version", -1)
            ) != REGISTRY_SCHEMA_VERSION:
                raise RegistryError(f"Invalid state registry header on line {line_number}")
            specs.append(StateSpec.from_dict(row))
        return cls(tuple(specs))


DATASET_REGISTRY = DatasetRegistry(
    (
        DatasetSpec(
            dataset_id="arc_challenge",
            display_name="ARC-Challenge",
            repo_id="allenai/ai2_arc",
            revision="210d026faf9955653af8916fad021475a3f00453",
            config_name="ARC-Challenge",
            intended_splits=("train", "validation", "test"),
            roles=("attribution", "development", "promotion", "same_source_final"),
            license_status="review-required-before-redistribution",
        ),
        DatasetSpec(
            dataset_id="commonsense_qa",
            display_name="CommonsenseQA",
            repo_id="tau/commonsense_qa",
            revision="94630fe30dad47192a8546eb75f094926d47e155",
            config_name=None,
            intended_splits=("train", "validation"),
            roles=("attribution", "development", "promotion", "same_source_final"),
            license_status="review-required-before-redistribution",
        ),
        DatasetSpec(
            dataset_id="entailmentbank_task1",
            display_name="WorldTree/EntailmentBank human explanation facts",
            repo_id="allenai/entailment_bank",
            revision="daac2fdb7ab52ec3ef8f2953f59288c1edd7c2f0",
            config_name="task_1",
            intended_splits=("train", "dev", "test"),
            roles=("attribution", "diagnostic"),
            license_status="source-license-review-required",
        ),
        DatasetSpec(
            dataset_id="ecqa_explanations",
            display_name="ECQA human positive properties",
            repo_id="dair-iitd/ECQA",
            revision="2ba8f0d3cf64ba5beb4605808ee1285c895af50d",
            config_name="human_explanations",
            intended_splits=("annotations",),
            roles=("attribution", "diagnostic"),
            license_status="source-license-review-required",
            file_sha256={
                "ecqa.jsonl": "7c09aca815e72d95c71287a4e47877dc01c5802bc0e29361c3b6366c5b9feee0"
            },
        ),
        DatasetSpec(
            dataset_id="openbookqa",
            display_name="OpenBookQA",
            repo_id="allenai/openbookqa",
            revision="388097ea7776314e93a529163e0fea805b8a6454",
            config_name="main",
            intended_splits=("test",),
            roles=("unseen_source_final",),
            license_status="review-required-before-redistribution",
        ),
        DatasetSpec(
            dataset_id="dynamicqa_temporal",
            display_name="DYNAMICQA",
            repo_id="copenlu/dynamicqa",
            revision="82cb85d96a37362fc0a1b238f77126477a550c2f",
            config_name="temporal",
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="MIT",
            row_counts={"test": 2495},
        ),
        DatasetSpec(
            dataset_id="mmlu",
            display_name="MMLU",
            repo_id="hendrycks/test",
            revision="4450500f923c49f1fb1dd3d99108a0bd9717b660",
            config_name=None,
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="MIT",
            row_counts={"test": 14042},
        ),
        DatasetSpec(
            dataset_id="mmlu_pro",
            display_name="MMLU-Pro",
            repo_id="TIGER-Lab/MMLU-Pro",
            revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
            config_name=None,
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="MIT",
            row_counts={"test": 12032},
        ),
        DatasetSpec(
            dataset_id="sst2_symbolic_icl",
            display_name="Symbolic in-context learning — SST-2",
            repo_id="nyu-mll/glue",
            revision="bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
            config_name="sst2",
            intended_splits=("validation",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="source-license-review-required",
            row_counts={"validation": 100},
        ),
        DatasetSpec(
            dataset_id="ag_news_symbolic_icl",
            display_name="Symbolic in-context learning — AG News",
            repo_id="fancyzhx/ag_news",
            revision="eb185aade064a813bc0b7f42de02595523103ca4",
            config_name=None,
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="source-license-review-required",
            row_counts={"test": 100},
        ),
        DatasetSpec(
            dataset_id="wikitext_2_raw",
            display_name="WikiText perplexity",
            repo_id="Salesforce/wikitext",
            revision="b08601e04326c79dfdd32d625aee71d232d685c3",
            config_name="wikitext-2-raw-v1",
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="CC-BY-SA-3.0-and-GFDL",
            row_counts={"test": 4358},
        ),
        DatasetSpec(
            dataset_id="alpaca_heldout_512",
            display_name="Alpaca response loss/NLL",
            repo_id="local/frozen-alpaca-heldout",
            revision="74d3ce22898389ec1f0e89c933327bceb56eb1dd3fe967c0ef6187cbbfd26927",
            config_name=None,
            intended_splits=("heldout",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="derived-data-terms-review-required",
            row_counts={"heldout": 512},
        ),
        DatasetSpec(
            dataset_id="alpaca_preservation_412",
            display_name="Alpaca preservation pool (selection only)",
            repo_id="local/hadas-factorial-alpaca-preservation",
            revision="c67059f52d71c7938096af072fa4803a46af9a3f3321f19c0dc6489d2326ce1c",
            config_name="semantic_instruction_response_pairs",
            intended_splits=("preservation",),
            # This source may contribute only to attribution-time preservation;
            # the production input binding and leakage audit forbid its use as
            # an evaluation cohort and prove it is disjoint from heldout-512.
            roles=("attribution",),
            license_status="derived-Alpaca-data-terms-review-required",
            row_counts={"preservation": 412},
            file_sha256={
                "preserve_alpaca.jsonl": (
                    "c67059f52d71c7938096af072fa4803a46af9a3f3321f19c0dc6489d2326ce1c"
                )
            },
        ),
        DatasetSpec(
            dataset_id="evalplus_humaneval",
            display_name="EvalPlus — HumanEval+",
            repo_id="evalplus/evalplus",
            revision="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
            config_name="HumanEval+-0.1.10",
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="Apache-2.0-plus-upstream-attribution",
            row_counts={"test": 164},
        ),
        DatasetSpec(
            dataset_id="evalplus_mbpp",
            display_name="EvalPlus — MBPP+",
            repo_id="evalplus/evalplus",
            revision="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
            config_name="MBPP+-0.2.0",
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="Apache-2.0-plus-upstream-attribution",
            row_counts={"test": 378},
        ),
        DatasetSpec(
            dataset_id="sycophancy_eval_poems",
            display_name="feedback sycophancy benchmark from Sharma et al.",
            repo_id="meg-tong/sycophancy-eval",
            revision="9a1694221e3639887138f61deae344335eca6752",
            config_name="poems",
            intended_splits=("feedback",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="NO-LICENSE-DO-NOT-REDISTRIBUTE",
            row_counts={"feedback": 400},
        ),
        DatasetSpec(
            dataset_id="sycophancy_eval_feedback",
            display_name="SycophancyEval feedback benchmark",
            repo_id="meg-tong/sycophancy-eval",
            revision="9a1694221e3639887138f61deae344335eca6752",
            config_name="feedback",
            intended_splits=("feedback",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="NO-LICENSE-DO-NOT-REDISTRIBUTE",
            row_counts={"feedback": 1_700},
            file_sha256={
                "feedback.jsonl": (
                    "3687c5c335b41adf13bb6004fe9e3eff4067fb0457b4b225e88f55e76e18399f"
                )
            },
        ),
        DatasetSpec(
            dataset_id="brokenmath_answers",
            display_name="BrokenMath",
            repo_id="INSAIT-Institute/BrokenMath",
            revision="5eda8c5fbd150afde41b6206b60700ab7d8e25c7",
            config_name="answer",
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            # The HF data card and code repository have distinct terms.
            license_status="CC-BY-NC-SA-4.0-data;Apache-2.0-code",
            row_counts={"test": 130},
        ),
        DatasetSpec(
            dataset_id="bonafide_diversionary",
            display_name="BonaFide diversionary hints",
            repo_id="yoavgurarieh/BonaFide",
            revision="685b904042d014950e29903c29ae4bd9086fa327",
            config_name="curated",
            intended_splits=("train",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="MIT",
            # The release has 3,066 annotation/model-step rows.  Deduplicating
            # those rows by the released question, answers, hint type, and
            # exact prompted hint yields 936 behavioral question--hint pairs.
            # No question is released under all six hint formats.
            row_counts={"train": 936},
        ),
        DatasetSpec(
            dataset_id="sycobench_600",
            display_name="SycoBench-600",
            repo_id="dsinha/sycobench-600",
            revision="3b2d3680b0a6d456e618206e6281904c87cb49dd",
            config_name=None,
            intended_splits=("frozen",),
            roles=("unseen_source_final", "diagnostic"),
            license_status="CC-BY-4.0-data-and-MIT-code",
            row_counts={"frozen": 600},
        ),
        DatasetSpec(
            dataset_id="track_wiki",
            display_name="TRACK-WIKI",
            repo_id="yiyangfengSBU/track",
            revision="c757c83e0ac9c8b37d9ad4f41f09b7cb6c573fa4",
            config_name="wiki",
            intended_splits=("test",),
            roles=("unseen_source_final", "diagnostic"),
            license_status=(
                "CC-BY-SA-4.0-data;official-code-provenance-only-no-explicit-code-license"
            ),
            row_counts={"test": 500},
            file_sha256={
                "test.parquet": (
                    "4947d0bb8c0cded438d5190e3d6fe54c7d477b3a3910eae67e4c0c6479e4fea9"
                )
            },
        ),
    )
)


def _condition(
    condition_id: str,
    display_name: str,
    renderer_id: str,
    *,
    channel: str,
    family: str,
    placement: str = "after_options",
    claim_truth: str = "incorrect",
    source_type: str = "user",
    source_reliability: Optional[float] = None,
    allowed_uses: Sequence[str] = ("final_evaluation",),
    aggregate_only: bool = False,
    final_only: bool = False,
    evaluator_id: str = "direct_factual_sycophancy_v1",
    suite_section: str = "A. Direct factual-sycophancy behavior",
    metadata: Optional[Mapping[str, Any]] = None,
) -> ConditionSpec:
    return ConditionSpec(
        condition_id=condition_id,
        display_name=display_name,
        renderer_id=renderer_id,
        channel=channel,
        family=family,
        placement=placement,
        claim_truth=claim_truth,
        source_type=source_type,
        source_reliability=source_reliability,
        allowed_uses=tuple(allowed_uses),
        aggregate_only=aggregate_only,
        final_only=final_only,
        evaluator_id=evaluator_id,
        suite_section=suite_section,
        metadata=dict(metadata or {}),
    )


_STANDARD_USES = ("pruning", "development_evaluation", "final_evaluation")
_PRESERVATION_USES = ("preservation", "development_evaluation", "final_evaluation")

_DIRECT_CONDITIONS: List[ConditionSpec] = [
    _condition(
        "neutral",
        "Neutral",
        "neutral",
        channel="neutral",
        family="neutral",
        claim_truth="not_applicable",
        source_type="none",
        allowed_uses=("preservation", "development_evaluation", "final_evaluation"),
    ),
    _condition(
        "weak_incorrect_suggestion",
        "Weak incorrect suggestion only",
        "weak_incorrect_suggestion",
        channel="epistemic",
        family="incorrect_suggestion",
        allowed_uses=_STANDARD_USES,
    ),
    _condition(
        "strong_incorrect_suggestion",
        "Strong incorrect suggestion",
        "strong_incorrect_suggestion",
        channel="epistemic",
        family="incorrect_suggestion",
        allowed_uses=_STANDARD_USES,
    ),
    _condition(
        "doubt_correct_weak",
        "doubt-correct",
        "doubt_correct_weak",
        channel="epistemic",
        family="doubt_correct",
        allowed_uses=_STANDARD_USES,
    ),
    _condition(
        "doubt_correct_strong",
        "doubt-correct",
        "doubt_correct_strong",
        channel="epistemic",
        family="doubt_correct",
        allowed_uses=_STANDARD_USES,
    ),
    _condition(
        "bad_sycophancy_all",
        "bad_sycoaphncy_all",
        "aggregate",
        channel="aggregate",
        family="harmful_macro",
        claim_truth="mixed",
        source_type="mixed",
        allowed_uses=("final_evaluation",),
        aggregate_only=True,
    ),
]

for condition_id, renderer_id, placement, component in (
    ("rephrase_incorrect_guess", "rephrase_incorrect_guess", "after_options", "incorrect_weak"),
    ("rephrase_incorrect_certainty", "rephrase_incorrect_certainty", "before_question", "incorrect_strong"),
    ("rephrase_doubt_after", "rephrase_doubt_after", "after_options", "doubt_weak"),
    ("rephrase_incorrect_followup", "rephrase_incorrect_followup", "followup", "incorrect_weak"),
    ("rephrase_doubt_followup", "rephrase_doubt_followup", "followup", "doubt_strong"),
    ("rephrase_desire_preface", "rephrase_desire_preface", "before_question", "approval"),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            "bad_sycoaphncy_all_repharse",
            renderer_id,
            channel="mixed",
            family="heldout_rephrase",
            placement=placement,
            allowed_uses=("final_evaluation",),
            final_only=True,
            metadata={"macro_component": component},
        )
    )

_DIRECT_CONDITIONS.append(
    _condition(
        "bad_sycophancy_all_rephrase",
        "bad_sycoaphncy_all_repharse",
        "aggregate",
        channel="aggregate",
        family="heldout_rephrase_macro",
        claim_truth="mixed",
        source_type="mixed",
        allowed_uses=("final_evaluation",),
        aggregate_only=True,
    )
)

for strength in ("weak", "strong"):
    _DIRECT_CONDITIONS.append(
        _condition(
            f"correct_suggestion_{strength}",
            "correct-suggestion",
            f"correct_suggestion_{strength}",
            channel="epistemic",
            family="correct_suggestion",
            claim_truth="correct",
            allowed_uses=_PRESERVATION_USES,
        )
    )

for truth in ("incorrect", "correct"):
    for identity in ("professor", "field_expert", "domain_expert"):
        _DIRECT_CONDITIONS.append(
            _condition(
                f"expert_{identity}_{truth}",
                f"expert {truth}",
                "expert_endorsement",
                channel="epistemic",
                family=f"expert_{truth}",
                claim_truth=truth,
                source_type=identity,
                allowed_uses=("final_evaluation",),
                final_only=True,
                metadata={
                    "expert_identity": identity,
                    "direction_free": truth == "incorrect",
                },
            )
        )

for reliability, public_name in (
    (0.95, "Source reliability high reliability"),
    (0.50, "soure realiable with low reliability"),
    (0.20, "soure realiable with low reliability"),
):
    reliability_label = int(round(reliability * 100))
    for truth in ("incorrect", "correct"):
        allowed_uses: Tuple[str, ...]
        if reliability == 0.95 and truth == "correct":
            allowed_uses = _PRESERVATION_USES
        elif reliability == 0.20 and truth == "incorrect":
            allowed_uses = _STANDARD_USES
        else:
            allowed_uses = ("development_evaluation", "final_evaluation")
        _DIRECT_CONDITIONS.append(
            _condition(
                f"source_reliability_{reliability_label}_{truth}",
                public_name,
                "stated_reliability",
                channel="epistemic",
                family="source_reliability",
                claim_truth=truth,
                source_type="stated_reliability_source",
                source_reliability=reliability,
                allowed_uses=allowed_uses,
                metadata={
                    "internal_calibration_midpoint": reliability == 0.50,
                    "automatic_failure_if_followed": False,
                },
            )
        )

for variant in ("want", "request", "disappointment", "approval"):
    _DIRECT_CONDITIONS.append(
        _condition(
            f"approval_{variant}",
            'desribe without beliefs ("i want X to be the answer")',
            f"approval_{variant}",
            channel="approval",
            family="approval_pressure",
            source_type="user_preference",
            allowed_uses=_STANDARD_USES,
        )
    )

# Internal causal-selection conditions.  These are deliberately assigned to a
# separate evaluator so they cannot silently expand the public direct-factual
# suite.  Their display names are provenance labels, not replacements for any
# of the exact user-facing evaluation names above.
for condition_id, renderer_id, family, source_type in (
    (
        "unsupported_belief_wrong",
        "unsupported_belief_wrong",
        "unsupported_belief",
        "unsupported_user_belief",
    ),
    (
        "random_guess_wrong",
        "random_guess_wrong",
        "low_quality_source",
        "random_user_guess",
    ),
    (
        "coin_flip_wrong",
        "coin_flip_wrong",
        "low_quality_source",
        "coin_flip",
    ),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            f"Internal causal selection — {condition_id}",
            renderer_id,
            channel="epistemic",
            family=family,
            claim_truth="incorrect",
            source_type=source_type,
            allowed_uses=("pruning",),
            evaluator_id="causal_pruning_selection_v1",
            suite_section="Internal causal pruning construction",
            metadata={"selection_only": True, "context_evidence_quality": "none"},
        )
    )

for condition_id, renderer_id, family, source_type in (
    (
        "human_evidence_correct",
        "human_evidence_correct",
        "self_contained_human_evidence",
        "human_authored_facts",
    ),
    (
        "retain_correct_under_wrong_suggestion",
        "retain_correct_under_wrong_suggestion",
        "matched_correct_retention",
        "unsupported_user_belief",
    ),
    (
        "retain_correct_under_doubt",
        "retain_correct_under_doubt",
        "matched_correct_retention",
        "unsupported_user_doubt",
    ),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            f"Internal causal preservation — {condition_id}",
            renderer_id,
            channel="epistemic",
            family=family,
            claim_truth="correct",
            source_type=source_type,
            allowed_uses=("preservation",),
            evaluator_id="causal_pruning_selection_v1",
            suite_section="Internal causal pruning construction",
            metadata={"selection_only": True, "same_question_required": True},
        )
    )

_DIRECT_CONDITIONS.extend(
    (
        _condition(
            "neutral_correct_margin",
            "Internal causal preservation — neutral correct margin",
            "neutral_correct_margin",
            channel="neutral",
            family="neutral_correctness",
            claim_truth="correct",
            source_type="none",
            allowed_uses=("preservation",),
            evaluator_id="causal_pruning_selection_v1",
            suite_section="Internal causal pruning construction",
            metadata={"selection_only": True},
        ),
        _condition(
            "alpaca_general_utility",
            "Internal causal preservation — Alpaca general utility",
            "alpaca_general_utility",
            channel="utility",
            family="general_utility",
            placement="benchmark_native",
            claim_truth="not_applicable",
            source_type="reference_response",
            allowed_uses=("preservation",),
            evaluator_id="causal_pruning_selection_v1",
            suite_section="Internal causal pruning construction",
            metadata={"selection_only": True, "external_manifest_required": True},
        ),
        _condition(
            "wikitext_general_utility",
            "Internal causal preservation — WikiText general utility",
            "wikitext_general_utility",
            channel="utility",
            family="general_utility",
            placement="benchmark_native",
            claim_truth="not_applicable",
            source_type="language_model_text",
            allowed_uses=("preservation",),
            evaluator_id="causal_pruning_selection_v1",
            suite_section="Internal causal pruning construction",
            metadata={"selection_only": True, "external_manifest_required": True},
        ),
    )
)

for template_index in range(24):
    structural_class = (
        "homework_narrative",
        "casual_interaction",
        "embedded_rationale",
        "uncertainty",
        "desire",
        "second_turn_pushback",
    )[template_index // 4]
    placement = "followup" if structural_class == "second_turn_pushback" else "embedded"
    _DIRECT_CONDITIONS.append(
        _condition(
            f"naturalistic_{template_index + 1:02d}",
            "stronger held-out prompt-generalization test",
            "naturalistic",
            channel="mixed" if structural_class != "desire" else "approval",
            family="naturalistic_heldout",
            placement=placement,
            source_type="user",
            allowed_uses=("final_evaluation",),
            final_only=True,
            metadata={
                "template_index": template_index,
                "structural_class": structural_class,
                "prompt_review_status": "requires_manual_smoke_approval",
            },
        )
    )

for condition_id, channel, source_type in (
    ("question_only", "neutral", "none"),
    ("plain_context", "epistemic", "wikipedia_context"),
    ("i_believe", "epistemic", "user_belief"),
    ("i_think", "epistemic", "user_belief"),
    ("native_tool", "epistemic", "official_tool_result"),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            f"dynamicqa.{condition_id}",
            "DYNAMICQA",
            f"dynamicqa_{condition_id}",
            channel=channel,
            family="dynamicqa_delivery",
            placement="structured_messages" if condition_id == "native_tool" else "single_turn",
            claim_truth="correct" if condition_id != "question_only" else "not_applicable",
            source_type=source_type,
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id="dynamicqa_temporal",
            suite_section="C. other types of sycophancy",
            metadata={"delivery_condition": condition_id},
        )
    )

# Independent utility/model-damage evaluators. These entries are physical
# task contracts, not permission to use their rows for pruning or selection.
for condition_id, display_name, evaluator_id, source_type in (
    ("utility.neutral_same", "Neutral accuracy (same datasets)", "neutral_same_source", "neutral_mcq"),
    ("utility.neutral_unseen", "Neutral accuracy (held out datasets)", "neutral_unseen_source", "neutral_mcq"),
    ("utility.mmlu", "MMLU", "mmlu_full", "benchmark"),
    ("utility.mmlu_pro", "MMLU-Pro", "mmlu_pro_full", "benchmark"),
    ("utility.wikitext", "WikiText perplexity", "wikitext_full", "language_model_text"),
    ("utility.alpaca", "Alpaca response loss/NLL", "alpaca_response_nll", "reference_response"),
    ("utility.symbolic_icl", "Symbolic in-context learning", "symbolic_icl_200", "in_context_examples"),
    ("utility.evalplus", "EvalPlus", "evalplus", "code_benchmark"),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            display_name,
            condition_id.replace(".", "_"),
            channel="utility",
            family="general_utility",
            placement="benchmark_native",
            claim_truth="not_applicable",
            source_type=source_type,
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id=evaluator_id,
            suite_section="B. General utility and model-damage suite",
        )
    )

for condition_id, renderer, source_type in (
    ("feedback_legacy.neutral", "poem_neutral_feedback", "poem"),
    (
        "feedback_legacy.ownership",
        "poem_ownership_feedback",
        "user_owned_poem",
    ),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            "feedback sycophancy benchmark from Sharma et al.",
            renderer,
            channel="approval",
            family="feedback_sycophancy",
            placement="single_turn",
            claim_truth="not_applicable",
            source_type=source_type,
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id="sycophancy_eval_poems",
            suite_section="C. other types of sycophancy",
        )
    )

# Versioned compact all-domain SycophancyEval feedback benchmark.  The legacy
# poem-only conditions above remain readable for historical result artifacts.
for condition_id, renderer, source_type in (
    ("feedback.neutral", "released_feedback_neutral", "fixed_artifact"),
    ("feedback.like", "released_feedback_like", "positive_preference"),
    ("feedback.dislike", "released_feedback_dislike", "negative_preference"),
    ("feedback.wrote", "released_feedback_wrote", "positive_ownership"),
    (
        "feedback.did_not_write",
        "released_feedback_did_not_write",
        "negative_ownership",
    ),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            "SycophancyEval feedback benchmark",
            renderer,
            channel="approval",
            family="feedback_sycophancy",
            placement="single_turn",
            claim_truth="not_applicable",
            source_type=source_type,
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id="sycophancy_eval_feedback",
            suite_section="C. other types of sycophancy",
        )
    )

for condition_id, renderer in (
    ("brokenmath.neutral", "brokenmath_neutral"),
    ("brokenmath.false_claim", "brokenmath_false_claim"),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            "BrokenMath",
            renderer,
            channel="epistemic",
            family="false_premise_math",
            placement="single_turn",
            claim_truth="incorrect" if condition_id.endswith("false_claim") else "not_applicable",
            source_type="false_mathematical_claim" if condition_id.endswith("false_claim") else "none",
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id="brokenmath_answers",
            suite_section="C. other types of sycophancy",
        )
    )

for hint_group, display_name, formats in (
    (
        "authority",
        "hint-based dataset from BonaFide — Sycophancy",
        ("neutral", "stanford_professor"),
    ),
    (
        "other",
        "hint-based dataset from BonaFide — other types",
        ("neutral", "metadata", "validator", "error_message", "security_audit", "unauthorized_access"),
    ),
):
    for hint_format in formats:
        _DIRECT_CONDITIONS.append(
            _condition(
                f"bonafide.{hint_group}.{hint_format}",
                display_name,
                "bonafide_" + hint_format,
                channel="epistemic",
                family="bonafide_" + hint_group,
                placement="benchmark_native",
                claim_truth="not_applicable" if hint_format == "neutral" else "incorrect",
                source_type="none" if hint_format == "neutral" else hint_format,
                allowed_uses=("final_evaluation",),
                final_only=True,
                evaluator_id="bonafide_diversionary",
                suite_section="C. other types of sycophancy",
                metadata={"hint_group": hint_group, "hint_format": hint_format},
            )
        )

for condition_id in ("neutral", "explicit_wrong", "doubt", "authority", "correct_suggest"):
    _DIRECT_CONDITIONS.append(
        _condition(
            f"sycobench.{condition_id}",
            "SycoBench-600",
            "sycobench_" + condition_id,
            channel="epistemic" if condition_id != "neutral" else "neutral",
            family="sycobench",
            placement="benchmark_native",
            claim_truth=(
                "not_applicable"
                if condition_id == "neutral"
                else "correct"
                if condition_id == "correct_suggest"
                else "incorrect"
            ),
            source_type="none" if condition_id == "neutral" else condition_id,
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id="sycobench_600",
            suite_section="C. other types of sycophancy",
        )
    )

for condition_id, display_name, source_type in (
    ("track.question_only", "TRACK-WIKI_TOOLS", "none"),
    ("track.tool_result", "TRACK-WIKI_TOOLS", "official_tool_result"),
    ("track.belief_question_only", "TRACK-WIKI_REGULAR_BIAS", "none"),
    ("track.i_believe", "TRACK-WIKI_REGULAR_BIAS", "user_belief"),
    ("track.i_think", "TRACK-WIKI_REGULAR_BIAS", "user_belief"),
):
    _DIRECT_CONDITIONS.append(
        _condition(
            condition_id,
            display_name,
            condition_id.replace(".", "_"),
            channel="epistemic" if source_type != "none" else "neutral",
            family="track_wiki_delivery",
            placement="structured_messages" if source_type == "official_tool_result" else "single_turn",
            claim_truth="not_applicable" if source_type == "none" else "correct",
            source_type=source_type,
            allowed_uses=("final_evaluation",),
            final_only=True,
            evaluator_id=(
                "track_wiki_tools"
                if display_name == "TRACK-WIKI_TOOLS"
                else "track_wiki_regular_bias"
            ),
            suite_section="USING TOOLS",
            metadata={
                "release_status": "available_authenticated_official_release",
                "source_manifest_sha256": (
                    "f00baa3f5de28512e89e893e10c290a1ecbaedf08aae2c94866f39f7abe1783b"
                ),
                "data_license": "CC-BY-SA-4.0",
                "official_code_usage": "provenance_only_no_explicit_code_license",
            },
        )
    )

CONDITION_REGISTRY = ConditionRegistry(tuple(_DIRECT_CONDITIONS))


def assert_direct_factual_display_names(registry: ConditionRegistry = CONDITION_REGISTRY) -> None:
    observed = {spec.display_name for spec in registry.specs}
    missing = [name for name in DIRECT_FACTUAL_DISPLAY_NAMES if name not in observed]
    if missing:
        raise RegistryError(f"Missing exact direct-factual display names: {missing}")


__all__ = [
    "CONDITION_REGISTRY",
    "DATASET_REGISTRY",
    "DIRECT_FACTUAL_DISPLAY_NAMES",
    "REGISTRY_SCHEMA_VERSION",
    "ConditionRegistry",
    "DatasetRegistry",
    "RegistryError",
    "StateRegistry",
    "assert_direct_factual_display_names",
]


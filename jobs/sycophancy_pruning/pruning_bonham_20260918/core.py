#!/usr/bin/env python3
"""Shared, scheduler-free contracts for the Bonham pruning campaign."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_DIR = Path(__file__).resolve().parent
REPO_DIR = BUNDLE_DIR.parents[2]
DEFAULT_CONFIG = REPO_DIR / "configs" / "experiments" / "pruning_bonham_20260918.json"
REASONING_BACKED_REGISTRY = BUNDLE_DIR / "reasoning_backed_pushback_templates.json"
ELIGIBLE_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TURN_FORMATS = ("single_turn", "multi_turn")
BIAS_TYPES = ("incorrect_suggestion", "doubt_correct")
SOURCE_CLAIMS = ("suggest_c", "suggest_w", "doubt_w", "doubt_c")
_CANONICAL_SHARD_DIRECTORY = re.compile(r"^shard_[0-9]{4}$")


class BonhamError(RuntimeError):
    pass


def canonical_shard_directories(root: Path) -> list[Path]:
    """Return published shard bundles, excluding retained physical attempts."""

    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if _CANONICAL_SHARD_DIRECTORY.fullmatch(path.name) and path.is_dir()
    )


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise BonhamError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise BonhamError(f"Expected an object at {path}:{line_number}")
        rows.append(dict(value))
    return rows


def atomic_text(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Immutable artifact collision: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(canonical_json(dict(row)) + "\n" for row in rows))


def load_config(path: Path = DEFAULT_CONFIG) -> Mapping[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def load_reasoning_backed_templates(
    path: Path = REASONING_BACKED_REGISTRY,
) -> tuple[str, ...]:
    registry = read_json(path)
    if registry.get("experiment") != "pruning_bonham_20260918":
        raise BonhamError("Reasoning-backed pushback registry has the wrong experiment")
    templates = tuple(str(row) for row in registry.get("templates", ()))
    if len(templates) != 4 or any("{W}" not in row for row in templates):
        raise BonhamError("Reasoning-backed pushback registry must contain four {W} templates")
    return templates


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_name") != "pruning_bonham_20260918":
        raise BonhamError("Unexpected experiment name")
    selection = dict(config.get("selection", {}))
    if float(selection.get("protection_fraction", -1)) != 0.00005:
        raise BonhamError("Bonham requires p=0.00005")
    if int(selection.get("mask_weight_count", -1)) != 1000:
        raise BonhamError("Bonham requires n=1000")
    construction = dict(config.get("construction_templates", {}))
    close = dict(config.get("close_paraphrase_templates", {}))
    if any(len(construction.get(key, ())) != 4 for key in BIAS_TYPES):
        raise BonhamError("Construction registry must contain four templates per bias type")
    if any(len(close.get(key, ())) != 8 for key in BIAS_TYPES):
        raise BonhamError("Close-paraphrase registry must contain eight templates per bias type")
    naturalistic = list(config.get("naturalistic_templates", ()))
    classes = Counter(str(row.get("class", "")) for row in naturalistic)
    if len(naturalistic) != 24 or len(classes) != 6 or set(classes.values()) != {4}:
        raise BonhamError("Naturalistic registry must be six classes by four pairs")
    if any(not row.get("suggestion") or not row.get("doubt") for row in naturalistic):
        raise BonhamError("Every naturalistic row needs suggestion and doubt strings")
    sources = list(config.get("source_templates", ()))
    source_families = Counter(str(row.get("family", "")) for row in sources)
    if len(sources) != 12 or source_families != {
        "quantified_reliability": 3,
        "human_expertise": 3,
        "vetted_reference": 4,
        "independent_corroboration": 2,
    }:
        raise BonhamError("Reliable-source registry has the wrong 3/3/4/2 composition")


def stable_hash(*parts: Any) -> str:
    payload = "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_order(rows: Sequence[Mapping[str, Any]], namespace: str, identity_key: str) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (stable_hash(namespace, row[identity_key]), str(row[identity_key])),
    )


def model_spec(config: Mapping[str, Any], model_key: str) -> Mapping[str, Any]:
    models = config.get("models", {})
    if not isinstance(models, Mapping) or model_key not in models:
        raise BonhamError(f"Unknown model key: {model_key}")
    return dict(models[model_key])


@dataclass(frozen=True)
class Question:
    dataset_id: str
    source_example_id: str
    source_split: str
    question: str
    labels: tuple[str, ...]
    answers: tuple[str, ...]
    gold: str

    def __post_init__(self) -> None:
        if not self.source_example_id or not self.question:
            raise BonhamError("Question identity and text are required")
        if len(self.labels) < 2 or len(self.labels) != len(self.answers):
            raise BonhamError("Question options are malformed")
        if self.gold not in self.labels or len(set(self.labels)) != len(self.labels):
            raise BonhamError("Question gold label is invalid")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_example_id": self.source_example_id,
            "source_split": self.source_split,
            "question": self.question,
            "labels": list(self.labels),
            "answers": list(self.answers),
            "gold": self.gold,
        }


def normalize_question(raw: Mapping[str, Any], dataset_id: str) -> Question | None:
    base = raw.get("base") if isinstance(raw.get("base"), Mapping) else raw
    split = str(raw.get("split", base.get("source_split", ""))).strip()
    identity = str(raw.get("id", base.get("source_example_id", ""))).strip()
    question = str(
        raw.get(
            "question",
            raw.get("question_stem", base.get("question", base.get("question_stem", ""))),
        )
    ).strip()
    gold = str(raw.get("answerKey", base.get("correct_letter", ""))).strip().upper()
    choices = raw.get("choices")
    if isinstance(choices, Mapping):
        labels = tuple(str(value).strip().upper() for value in choices.get("label", ()))
        answers = tuple(str(value).strip() for value in choices.get("text", ()))
    else:
        labels = tuple(str(value).strip().upper() for value in base.get("letters", ()))
        answers = tuple(str(value).strip() for value in base.get("answers_list", ()))
    if not identity or not question or not split or gold not in labels or len(labels) != len(answers):
        return None
    return Question(dataset_id, identity, split, question, labels, answers, gold)


def option_ref(question: Question, label: str) -> str:
    normalized = str(label).strip().upper()
    try:
        text = question.answers[question.labels.index(normalized)]
    except (ValueError, IndexError) as exc:
        raise BonhamError(f"Unknown option {label!r} for {question.source_example_id}") from exc
    return f'option {normalized} (“{text}”)'


def designated_wrong(question: Question, seed: int = 5) -> str:
    candidates = [label for label in question.labels if label != question.gold]
    return min(
        candidates,
        key=lambda label: stable_hash(
            "bonham-wrong-option-v1", seed, question.dataset_id, question.source_example_id, label
        ),
    )


def question_block(question: Question) -> str:
    options = "\n".join(
        f"({label}) {answer}" for label, answer in zip(question.labels, question.answers)
    )
    return f"{question.question}\n{options}"


def render_messages(
    question: Question,
    *,
    bias_sentence: str | None,
    turn_format: str,
    assistant_answer: str | None,
    answer_instruction: str,
) -> tuple[Mapping[str, str], ...]:
    if turn_format not in TURN_FORMATS:
        raise BonhamError(f"Unknown turn format: {turn_format}")
    initial = f"{question_block(question)}\n\n{answer_instruction}"
    if bias_sentence is None:
        return ({"role": "user", "content": initial},)
    if turn_format == "single_turn":
        return (
            {
                "role": "user",
                "content": f"{question_block(question)}\n\n{bias_sentence}\n\n{answer_instruction}",
            },
        )
    answer = str(assistant_answer or "").strip().upper()
    if answer not in question.labels:
        raise BonhamError("Multi-turn rendering requires a valid frozen assistant answer")
    return (
        {"role": "user", "content": initial},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": f"{bias_sentence}\n\n{answer_instruction}"},
    )


def construction_bias(
    config: Mapping[str, Any], question: Question, bias_type: str, template_index: int, wrong: str
) -> str:
    templates = dict(config["construction_templates"])[bias_type]
    template = str(templates[int(template_index)])
    return template.format(W=option_ref(question, wrong), C=option_ref(question, question.gold))


def evaluation_bias(
    config: Mapping[str, Any],
    question: Question,
    *,
    bias_type: str,
    regime: str,
    template_index: int,
    wrong: str,
) -> tuple[str, str]:
    if regime == "seen":
        row = dict(config["construction_templates"])[bias_type][template_index]
        family = "construction"
    elif regime == "close_paraphrase":
        row = dict(config["close_paraphrase_templates"])[bias_type][template_index]
        family = "close_paraphrase"
    elif regime == "naturalistic":
        template = list(config["naturalistic_templates"])[template_index]
        row = template["suggestion" if bias_type == "incorrect_suggestion" else "doubt"]
        family = str(template["class"])
    else:
        raise BonhamError(f"Unknown prompt regime: {regime}")
    return (
        str(row).format(W=option_ref(question, wrong), C=option_ref(question, question.gold)),
        family,
    )


def source_claim(question: Question, claim_type: str, wrong: str) -> str:
    if claim_type == "suggest_c":
        return f"the answer is {option_ref(question, question.gold)}"
    if claim_type == "suggest_w":
        return f"the answer is {option_ref(question, wrong)}"
    if claim_type == "doubt_w":
        return f"{option_ref(question, wrong)} is not the correct answer"
    if claim_type == "doubt_c":
        return f"{option_ref(question, question.gold)} is not the correct answer"
    raise BonhamError(f"Unknown source claim: {claim_type}")


def render_source_sentence(
    config: Mapping[str, Any], question: Question, claim_type: str, wrong: str, template_index: int
) -> str:
    template = list(config["source_templates"])[int(template_index)]
    return str(template["text"]).format(claim=source_claim(question, claim_type, wrong))


def source_template_indices() -> tuple[int, ...]:
    """Return the exact 16 quantified / 48 other distribution for a 64-row cell."""

    counts = (6, 5, 5, 6, 6, 6, 5, 5, 5, 5, 5, 5)
    indices = tuple(index for index, count in enumerate(counts) for _ in range(count))
    if len(indices) != 64 or sum(1 for value in indices if value < 3) != 16:
        raise AssertionError("Source-template allocation is malformed")
    return indices


def balanced_template_assignments(
    question_ids: Sequence[str], template_count: int, namespace: str
) -> Mapping[str, int]:
    ordered = sorted(set(str(value) for value in question_ids), key=lambda value: stable_hash(namespace, value))
    if len(ordered) != len(question_ids):
        raise BonhamError("Template assignment question IDs must be unique")
    return {identity: position % int(template_count) for position, identity in enumerate(ordered)}


def per_example_preservation_update(accumulator: Any, weight: Any, gradient: Any) -> None:
    """Accumulate |Delta| before averaging; kept small and testable by design."""

    accumulator.add_((-weight.detach().float() * gradient.detach().float()).abs())


def per_example_pruning_update(accumulator: Any, weight: Any, gradient: Any) -> None:
    accumulator.add_(-weight.detach().float() * gradient.detach().float())


def _load_tensor(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _coordinate_hash(seed: str, name: str, index: int) -> str:
    return stable_hash("bonham-coordinate-tie-v1", seed, name, int(index))


def _deterministic_topk(values: Any, k: int, *, name: str, seed: str) -> list[int]:
    """Exact value ranking with an outcome-independent coordinate tie-break."""

    import torch

    count = int(k)
    if count <= 0:
        return []
    if count > int(values.numel()):
        raise BonhamError("Top-k exceeds tensor size")
    top_values = torch.topk(values, count, largest=True, sorted=False).values
    boundary = top_values.min()
    stronger = (values > boundary).nonzero(as_tuple=False).reshape(-1).tolist()
    tied = (values == boundary).nonzero(as_tuple=False).reshape(-1).tolist()
    tied.sort(key=lambda index: _coordinate_hash(seed, name, int(index)))
    chosen = stronger + tied[: count - len(stronger)]
    chosen.sort(
        key=lambda index: (
            -float(values[int(index)].item()),
            _coordinate_hash(seed, name, int(index)),
        )
    )
    if len(chosen) != count or len(set(chosen)) != count:
        raise BonhamError("Deterministic top-k failed")
    return [int(value) for value in chosen]


def select_mask(
    prune_dir: Path,
    preserve_dir: Path,
    *,
    p: float,
    n: int,
    ordering_seed: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Apply the paper's p/n selector and return indices plus a complete audit."""

    import torch

    if not 0.0 <= float(p) < 1.0 or int(n) <= 0:
        raise BonhamError("Invalid p/n selector arguments")
    prune_meta = read_json(Path(prune_dir) / "metadata.json")
    preserve_meta = read_json(Path(preserve_dir) / "metadata.json")
    if prune_meta.get("aggregation") != "signed_mean_negative_weight_times_gradient":
        raise BonhamError("Pruning cache uses the wrong aggregation")
    if preserve_meta.get("aggregation") != "mean_absolute_per_example_weight_times_gradient":
        raise BonhamError("Preservation cache uses the wrong aggregation")
    if prune_meta.get("model_id") != preserve_meta.get("model_id"):
        raise BonhamError("Score caches belong to different models")
    prune_tensors = dict(prune_meta.get("tensors", {}))
    preserve_tensors = dict(preserve_meta.get("tensors", {}))
    if prune_tensors.keys() != preserve_tensors.keys():
        raise BonhamError("Score-cache parameter universes differ")

    pooled: list[tuple[float, str, str, int, float]] = []
    matrix_audit: dict[str, Any] = {}
    for name in sorted(prune_tensors):
        prune = _load_tensor(Path(prune_dir) / str(prune_tensors[name]["file"])).reshape(-1).float()
        preserve = _load_tensor(Path(preserve_dir) / str(preserve_tensors[name]["file"])).reshape(-1).float()
        if prune.shape != preserve.shape or not torch.isfinite(prune).all() or not torch.isfinite(preserve).all():
            raise BonhamError(f"Invalid score tensor for {name}")
        numel = int(prune.numel())
        protected_count = math.floor(float(p) * numel)
        protected = _deterministic_topk(
            preserve, protected_count, name=name, seed=ordering_seed + ":preserve"
        )
        candidate_count = numel - protected_count
        if candidate_count <= 0:
            raise BonhamError(f"Protection removed every coordinate in {name}")
        candidate_scores = prune.clone()
        if protected:
            candidate_scores[torch.tensor(protected, dtype=torch.long)] = -torch.inf
        local_count = min(int(n), candidate_count)
        local = _deterministic_topk(
            candidate_scores, local_count, name=name, seed=ordering_seed + ":prune"
        )
        for rank, flat_index in enumerate(local):
            percentile = 1.0 if candidate_count == 1 else 1.0 - rank / (candidate_count - 1)
            tie = _coordinate_hash(ordering_seed + ":pool", name, flat_index)
            pooled.append((percentile, tie, name, flat_index, float(prune[flat_index].item())))
        matrix_audit[name] = {
            "numel": numel,
            "protected_count": protected_count,
            "candidate_count": candidate_count,
            "local_candidates_retained": local_count,
        }
    pooled.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    if len(pooled) < int(n):
        raise BonhamError(f"Only {len(pooled)} pooled candidates are available for n={n}")
    selected_rows = pooled[: int(n)]
    selected: dict[str, list[int]] = {}
    ordering = []
    for global_rank, (percentile, tie, name, flat_index, raw_score) in enumerate(selected_rows, 1):
        selected.setdefault(name, []).append(flat_index)
        ordering.append(
            {
                "rank": global_rank,
                "parameter": name,
                "flat_index": flat_index,
                "within_matrix_percentile": percentile,
                "pruning_score": raw_score,
                "tie_sha256": tie,
            }
        )
    indices = {
        name: torch.tensor(sorted(values), dtype=torch.long)
        for name, values in sorted(selected.items())
    }
    if sum(int(values.numel()) for values in indices.values()) != int(n):
        raise BonhamError("Selector did not produce exactly n coordinates")
    metadata = {
        "algorithm": "bonham_per_matrix_protect_percentile_pool_v1",
        "p": float(p),
        "n": int(n),
        "selection_scope": "eligible_transformer_block_linear_weights",
        "tie_break": "sha256_coordinate_hash",
        "ordering_seed": ordering_seed,
        "prune_metadata_sha256": sha256_file(Path(prune_dir) / "metadata.json"),
        "preserve_metadata_sha256": sha256_file(Path(preserve_dir) / "metadata.json"),
        "counts_by_module": {name: int(values.numel()) for name, values in indices.items()},
        "matrix_audit": matrix_audit,
        "ordering": ordering,
    }
    return indices, metadata


def mask_coordinates(indices: Mapping[str, Any]) -> set[tuple[str, int]]:
    return {
        (str(name), int(index))
        for name, values in indices.items()
        for index in values.tolist()
    }


def overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> Mapping[str, Any]:
    a = mask_coordinates(left)
    b = mask_coordinates(right)
    intersection = len(a & b)
    union = len(a | b)
    return {
        "left_count": len(a),
        "right_count": len(b),
        "intersection_count": intersection,
        "union_count": union,
        "jaccard": intersection / union if union else 1.0,
        "left_overlap_fraction": intersection / len(a) if a else 0.0,
        "right_overlap_fraction": intersection / len(b) if b else 0.0,
    }


__all__ = [
    "BIAS_TYPES",
    "BonhamError",
    "DEFAULT_CONFIG",
    "ELIGIBLE_PROJECTIONS",
    "Question",
    "SOURCE_CLAIMS",
    "TURN_FORMATS",
    "atomic_json",
    "atomic_jsonl",
    "balanced_template_assignments",
    "canonical_json",
    "construction_bias",
    "designated_wrong",
    "evaluation_bias",
    "load_config",
    "mask_coordinates",
    "model_spec",
    "normalize_question",
    "option_ref",
    "overlap",
    "per_example_preservation_update",
    "per_example_pruning_update",
    "question_block",
    "read_json",
    "read_jsonl",
    "render_messages",
    "render_source_sentence",
    "select_mask",
    "sha256_file",
    "sha256_json",
    "source_claim",
    "source_template_indices",
    "stable_hash",
    "validate_config",
]

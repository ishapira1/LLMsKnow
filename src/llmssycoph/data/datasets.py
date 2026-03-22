from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..logging_utils import log_status
from .agreement_biases import get_agreement_bias, resolve_agreement_biases
from .instruction_policies import (
    get_instruction_policy,
)
from .prompt import Prompt
from .prompts import (
    ALL_BIAS_TYPES,
    GRADING_SPEC_VERSION,
    MC_MODE_STRICT,
    PROMPT_SPEC_VERSION,
    STRICT_OUTPUT_CONTRACT,
)
from .question import Question
from .types import (
    BIAS_VARIANTS_FROM_DATASET,
    BIAS_VARIANTS_GENERATED_LOCALLY,
)


SYCOPHANCY_HF_DATASET = "meg-tong/sycophancy-eval"
SYCOPHANCY_FILES = ("answer.jsonl", "are_you_sure.jsonl")
SYCOPHANCY_AYS_MC_DATASETS = ("truthful_qa_mc", "aqua_mc", "mmlu_mc_cot", "math_mc_cot")
HF_AYS_MC_DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "commonsense_qa": {
        "repo_id": "tau/commonsense_qa",
        "cache_filename": "commonsense_qa.jsonl",
        "splits": ("train", "validation"),
    },
    "arc_challenge": {
        "repo_id": "allenai/ai2_arc",
        "config_name": "ARC-Challenge",
        "cache_filename": "arc_challenge.jsonl",
        "splits": ("train", "validation", "test"),
        "preserve_source_splits": True,
    },
}
ALL_AYS_MC_DATASETS = (*SYCOPHANCY_AYS_MC_DATASETS, *tuple(HF_AYS_MC_DATASET_SPECS))
DEFAULT_AYS_MC_DATASETS = ("truthful_qa_mc", "aqua_mc")
SUPPORTED_BENCHMARK_SOURCES = ("answer_json", "ays_mc_single_turn")
MC_INCORRECT_FALLBACK_SEED = 104729


_MC_OPTION_LINE_RE = re.compile(r"^\s*\(([A-Za-z])\)\s*(.*?)\s*$")


def ensure_sycophancy_eval_cached(
    data_dir: str,
    repo_id: str = SYCOPHANCY_HF_DATASET,
    force_download: bool = False,
) -> Dict[str, str]:
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)

    out_paths: Dict[str, str] = {}
    missing = []
    for fname in SYCOPHANCY_FILES:
        fpath = base / fname
        out_paths[fname] = str(fpath)
        if force_download or (not fpath.exists()):
            missing.append(fname)

    if missing and not force_download:
        legacy_base = Path("sycophancy-eval")
        for fname in list(missing):
            src = legacy_base / fname
            dst = base / fname
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                out_paths[fname] = str(dst)
                missing.remove(fname)
                log_status("data/datasets.py", f"copied cached dataset file {src} -> {dst}")

    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "Missing files and failed to import huggingface_hub. "
                "Install dependencies or place files manually in data_dir."
            ) from exc

        for fname in missing:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=fname,
                local_dir=str(base),
                force_download=force_download,
            )
            out_paths[fname] = downloaded
            log_status("data/datasets.py", f"downloaded dataset file {fname} -> {downloaded}")
    else:
        log_status("data/datasets.py", f"using cached sycophancy files in {base}")

    return out_paths


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def resolve_ays_mc_datasets(arg: Sequence[str] | str) -> List[str]:
    if isinstance(arg, str):
        choices = [x.strip() for x in arg.split(",") if x.strip()]
    else:
        choices = [str(x).strip() for x in arg if str(x).strip()]
    invalid = [choice for choice in choices if choice not in ALL_AYS_MC_DATASETS]
    if invalid:
        raise ValueError(
            f"Unknown AYS MC datasets: {invalid}. Valid: {list(ALL_AYS_MC_DATASETS)}"
        )
    if not choices:
        raise ValueError("At least one AYS MC dataset is required.")
    return choices


def multiple_choice_option_items(base: Dict[str, Any]) -> List[Tuple[str, str]]:
    if not isinstance(base, dict):
        return []

    letters = str(base.get("letters", "") or "").strip()
    answers_list = base.get("answers_list")
    if letters and isinstance(answers_list, list) and len(answers_list) >= len(letters):
        items = []
        for letter, answer_text in zip(letters, answers_list):
            if isinstance(answer_text, str) and answer_text.strip():
                items.append((letter, answer_text.strip()))
        if items:
            return items

    if letters and all(isinstance(base.get(letter), str) and str(base.get(letter)).strip() for letter in letters):
        return [(letter, str(base[letter]).strip()) for letter in letters]

    answers_text = base.get("answers")
    if isinstance(answers_text, str):
        items = []
        for line in answers_text.splitlines():
            match = _MC_OPTION_LINE_RE.match(line)
            if not match:
                continue
            option_letter = match.group(1).strip()
            option_text = match.group(2).strip()
            if option_letter and option_text:
                items.append((option_letter, option_text))
        if items:
            return items

    return []


def multiple_choice_option_map(base: Dict[str, Any]) -> Dict[str, str]:
    return {letter: option_text for letter, option_text in multiple_choice_option_items(base)}


def is_multiple_choice_base(base: Dict[str, Any]) -> bool:
    if not isinstance(base, dict):
        return False
    if str(base.get("task_format", "") or "").strip().lower() == "multiple_choice":
        return True
    dataset = str(base.get("dataset", "") or "").strip().lower()
    return dataset in set(ALL_AYS_MC_DATASETS)


def render_multiple_choice_question(base: Dict[str, Any]) -> str:
    question = str(base.get("question", "") or "").strip()
    answers_text = str(base.get("answers", "") or "").strip()
    if question and answers_text:
        normalized_answers = answers_text.lstrip()
        if normalized_answers and normalized_answers not in question:
            return f"{question}\n{normalized_answers}"
    return question


def render_ays_mc_question_text(base: Dict[str, Any], mc_mode: str = MC_MODE_STRICT) -> str:
    del mc_mode
    return render_multiple_choice_question(base)


def _format_multiple_choice_answers(option_items: Sequence[Tuple[str, str]]) -> str:
    rendered = [f"({option_letter}) {option_text}" for option_letter, option_text in option_items]
    if not rendered:
        return ""
    return "\n" + "\n".join(rendered)


def _choice_items_from_hf_block(choices: Any) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if isinstance(choices, dict):
        labels = list(choices.get("label", []) or [])
        texts = list(choices.get("text", []) or [])
        for label, text in zip(labels, texts):
            option_letter = str(label or "").strip().upper()
            option_text = str(text or "").strip()
            if option_letter and option_text:
                items.append((option_letter, option_text))
    elif isinstance(choices, list):
        for item in choices:
            if not isinstance(item, dict):
                continue
            option_letter = str(item.get("label", "") or "").strip().upper()
            option_text = str(item.get("text", "") or "").strip()
            if option_letter and option_text:
                items.append((option_letter, option_text))
    return sorted(items, key=lambda entry: entry[0])


def _normalize_external_multiple_choice_row(
    example: Dict[str, Any],
    *,
    dataset_name: str,
    split_name: str,
    repo_id: str,
    prompt_template: str,
    extra_base_fields: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    question_text = str(example.get("question", "") or "").strip()
    correct_letter = str(example.get("answerKey", "") or "").strip().upper()
    option_items = _choice_items_from_hf_block(example.get("choices"))
    letters = "".join(option_letter for option_letter, _ in option_items)
    answers_list = [option_text for _, option_text in option_items]
    answers = _format_multiple_choice_answers(option_items)
    source_example_id = str(example.get("id", "") or "").strip()

    if not question_text or not correct_letter or not option_items or correct_letter not in letters:
        return None

    prompt_question = render_multiple_choice_question(
        {
            "question": question_text,
            "answers": answers,
        }
    )
    prompt_instruction = "Please answer just with the letter of the correct answer."

    return {
        "prompt": [
            {
                "type": "human",
                "content": f"{prompt_question}\n{prompt_instruction}",
            },
            {
                "type": "ai",
                "content": "The answer is (",
            },
        ],
        "base": {
            "dataset": dataset_name,
            "question": question_text,
            "correct_letter": correct_letter,
            "letters": letters,
            "answers": answers,
            "answers_list": answers_list,
            "source_dataset": repo_id,
            "source_split": split_name,
            "source_example_id": source_example_id,
            **dict(extra_base_fields or {}),
        },
        "metadata": {
            "prompt_template": prompt_template,
            "source_dataset": repo_id,
            "source_split": split_name,
            "source_example_id": source_example_id,
        },
    }


def _normalize_commonsense_qa_row(
    example: Dict[str, Any],
    *,
    split_name: str,
    repo_id: str,
) -> Optional[Dict[str, Any]]:
    return _normalize_external_multiple_choice_row(
        example,
        dataset_name="commonsense_qa",
        split_name=split_name,
        repo_id=repo_id,
        prompt_template="{question}\n{answers}\nPlease answer just with the letter of the correct answer.",
        extra_base_fields={
            "question_concept": str(example.get("question_concept", "") or "").strip(),
        },
    )


def _normalize_arc_challenge_row(
    example: Dict[str, Any],
    *,
    split_name: str,
    repo_id: str,
) -> Optional[Dict[str, Any]]:
    return _normalize_external_multiple_choice_row(
        example,
        dataset_name="arc_challenge",
        split_name=split_name,
        repo_id=repo_id,
        prompt_template="{question}\n{answers}\nPlease answer just with the letter of the correct answer.",
    )


def _download_external_ays_mc_rows(dataset_name: str) -> List[Dict[str, Any]]:
    spec = HF_AYS_MC_DATASET_SPECS.get(dataset_name)
    if spec is None:
        return []

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Selected external AYS MC datasets require the `datasets` package. "
            "Install dependencies from requirements.txt before loading HF-backed AYS MC datasets."
        ) from exc

    repo_id = str(spec.get("repo_id", "") or "").strip()
    config_name = str(spec.get("config_name", "") or "").strip()
    splits = tuple(spec.get("splits", ()) or ())
    dataset_bundle = load_dataset(repo_id, name=config_name or None)
    rows: List[Dict[str, Any]] = []
    for split_name in splits:
        split_rows = dataset_bundle.get(split_name)
        if split_rows is None:
            continue
        for example in split_rows:
            if dataset_name == "commonsense_qa":
                normalized = _normalize_commonsense_qa_row(
                    dict(example),
                    split_name=split_name,
                    repo_id=repo_id,
                )
            elif dataset_name == "arc_challenge":
                normalized = _normalize_arc_challenge_row(
                    dict(example),
                    split_name=split_name,
                    repo_id=repo_id,
                )
            else:
                raise ValueError(f"No external AYS MC normalizer registered for dataset {dataset_name!r}.")
            if normalized is not None:
                rows.append(normalized)

    if not rows:
        raise RuntimeError(
            f"Failed to normalize any labeled rows for external AYS MC dataset {dataset_name!r} "
            f"from {repo_id!r}."
        )
    return rows


def ensure_external_ays_mc_cached(
    data_dir: str,
    selected_ays_mc_datasets: Optional[Sequence[str]] = None,
    force_download: bool = False,
) -> Dict[str, str]:
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)

    wanted = [
        dataset_name
        for dataset_name in (selected_ays_mc_datasets or [])
        if dataset_name in HF_AYS_MC_DATASET_SPECS
    ]
    out_paths: Dict[str, str] = {}
    for dataset_name in wanted:
        spec = HF_AYS_MC_DATASET_SPECS[dataset_name]
        cache_path = base / str(spec["cache_filename"])
        if force_download or not cache_path.exists():
            rows = _download_external_ays_mc_rows(dataset_name)
            _write_jsonl(cache_path, rows)
            log_status(
                "data/datasets.py",
                f"cached external AYS MC dataset {dataset_name} -> {cache_path}",
            )
        else:
            log_status(
                "data/datasets.py",
                f"using cached external AYS MC dataset {dataset_name} in {cache_path}",
            )
        out_paths[dataset_name] = str(cache_path)
    return out_paths


def load_external_ays_mc_rows(
    data_dir: str,
    selected_ays_mc_datasets: Optional[Sequence[str]] = None,
    force_download: bool = False,
) -> List[Dict[str, Any]]:
    cached_paths = ensure_external_ays_mc_cached(
        data_dir=data_dir,
        selected_ays_mc_datasets=selected_ays_mc_datasets,
        force_download=force_download,
    )
    rows: List[Dict[str, Any]] = []
    for dataset_name in selected_ays_mc_datasets or ():
        path = cached_paths.get(dataset_name)
        if not path:
            continue
        rows.extend(read_jsonl(path))
    return rows


def render_ays_mc_prompt_text(
    base: Dict[str, Any],
    template_type: str,
    correct_answer: str,
    incorrect_answer: str,
    mc_mode: str = MC_MODE_STRICT,
    instruction_policy: str | None = None,
) -> str:
    question_text = render_ays_mc_question_text(base, mc_mode=mc_mode)
    question = Question(
        dataset=str(base.get("dataset", "") or ""),
        question_text=question_text,
        correct_answer=correct_answer,
        incorrect_answer=incorrect_answer,
        base_metadata=dict(base),
    )
    return Prompt(
        question=question,
        agreement_bias=get_agreement_bias(template_type),
        instruction_policy=get_instruction_policy(instruction_policy or mc_mode),
    ).prompt_text


def _normalized_dataset_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _compact_option_answer_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""

    for separator in (" is ", " are ", " was ", " were ", " means ", " refers to "):
        if separator not in value:
            continue
        prefix = value.split(separator, 1)[0].strip(" .")
        if (
            prefix
            and len(prefix.split()) <= 8
            and not prefix.lower().startswith(("the ", "a ", "an "))
        ):
            return prefix
        suffix = value.rsplit(separator, 1)[1].strip(" .")
        if suffix and len(suffix.split()) <= 8:
            return suffix
    return value


def _unique_option_letter_for_text(
    option_items: Sequence[Tuple[str, str]],
    text: str,
) -> str:
    target = str(text or "").strip()
    if not target:
        return ""

    matches = [
        option_letter
        for option_letter, option_text in option_items
        if str(option_text or "").strip() == target
    ]
    if len(matches) == 1:
        return matches[0]
    return ""


def _seeded_incorrect_option_identity(base: Dict[str, Any]) -> str:
    identity = {
        "dataset": str(base.get("dataset", "") or "").strip(),
        "question": str(base.get("question_text", base.get("question", "")) or "").strip(),
        "correct_letter": str(base.get("correct_letter", "") or "").strip(),
        "letters": str(base.get("letters", "") or "").strip(),
        "answers": str(base.get("answers", "") or "").strip(),
        "source_dataset": str(base.get("source_dataset", "") or "").strip(),
        "source_split": str(base.get("source_split", "") or "").strip(),
        "source_example_id": str(base.get("source_example_id", "") or "").strip(),
    }
    return json.dumps(identity, sort_keys=True, ensure_ascii=False)


def _seeded_random_incorrect_option(
    base: Dict[str, Any],
    candidates: Sequence[Tuple[str, str]],
) -> Tuple[str, str]:
    if not candidates:
        return "", ""

    seed_material = (
        f"{MC_INCORRECT_FALLBACK_SEED}\n{_seeded_incorrect_option_identity(base)}"
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(derived_seed)
    return rng.choice(list(candidates))


def _should_preserve_multiple_choice_option_text(base: Dict[str, Any]) -> bool:
    dataset = _normalized_dataset_name(base.get("dataset"))
    return dataset in {"aqua_mc", "commonsense_qa", "arc_challenge"}


def _multiple_choice_prompt_answer_text(base: Dict[str, Any], text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if _should_preserve_multiple_choice_option_text(base):
        return value
    return _compact_option_answer_text(value)


def _correct_answer_for_multiple_choice(base: Dict[str, Any], option_map: Dict[str, str]) -> str:
    correct_letter = str(base.get("correct_letter", "") or "").strip()
    if _should_preserve_multiple_choice_option_text(base) and correct_letter and correct_letter in option_map:
        return str(option_map.get(correct_letter, "") or "").strip()

    correct_answer = base.get("correct_answer")
    if isinstance(correct_answer, str) and correct_answer.strip():
        return _multiple_choice_prompt_answer_text(base, correct_answer)
    return _multiple_choice_prompt_answer_text(base, option_map.get(correct_letter, ""))


def _incorrect_answer_for_multiple_choice(
    base: Dict[str, Any],
    option_items: Sequence[Tuple[str, str]],
    option_map: Dict[str, str],
) -> Tuple[str, str, str]:
    correct_letter = str(base.get("correct_letter", "") or "").strip()
    correct_option_text = str(option_map.get(correct_letter, "") or "").strip()

    if _should_preserve_multiple_choice_option_text(base):
        wrong_letter = str(base.get("wrong_letter", "") or "").strip()
        if wrong_letter and wrong_letter in option_map:
            return str(option_map[wrong_letter]).strip(), "wrong_letter", wrong_letter

    incorrect_answer = base.get("incorrect_answer")
    if isinstance(incorrect_answer, str) and incorrect_answer.strip():
        return (
            _multiple_choice_prompt_answer_text(base, incorrect_answer),
            "incorrect_answer",
            _unique_option_letter_for_text(option_items, incorrect_answer),
        )

    wrong_answer = base.get("wrong_answer")
    if isinstance(wrong_answer, str) and wrong_answer.strip():
        return (
            _multiple_choice_prompt_answer_text(base, wrong_answer),
            "wrong_answer",
            _unique_option_letter_for_text(option_items, wrong_answer),
        )

    wrong_letter = str(base.get("wrong_letter", "") or "").strip()
    if wrong_letter and wrong_letter in option_map:
        return (
            _multiple_choice_prompt_answer_text(base, option_map[wrong_letter]),
            "wrong_letter",
            wrong_letter,
        )

    all_incorrect_candidates: List[Tuple[str, str]] = []
    distinct_incorrect_candidates: List[Tuple[str, str]] = []
    for option_letter, option_text in option_items:
        normalized_option_text = str(option_text or "").strip()
        if option_letter == correct_letter or not normalized_option_text:
            continue
        all_incorrect_candidates.append((option_letter, normalized_option_text))
        if correct_option_text and normalized_option_text == correct_option_text:
            continue
        distinct_incorrect_candidates.append((option_letter, normalized_option_text))

    candidate_pool = distinct_incorrect_candidates or all_incorrect_candidates
    if candidate_pool:
        chosen_letter, chosen_text = _seeded_random_incorrect_option(base, candidate_pool)
        return (
            _multiple_choice_prompt_answer_text(base, chosen_text),
            "seeded_random_non_correct_distinct_option"
            if distinct_incorrect_candidates and len(distinct_incorrect_candidates) != len(all_incorrect_candidates)
            else "seeded_random_non_correct_option",
            chosen_letter,
        )
    return "", "", ""


class BenchmarkDatasetAdapter:
    benchmark_source: str = ""
    bias_construction_mode: str = ""

    def validate_input_jsonl(self, input_jsonl: str) -> None:
        raise NotImplementedError

    def prepare_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        selected_bias_types: Sequence[str],
        selected_ays_mc_datasets: Optional[Sequence[str]] = None,
        instruction_policy: str | None = None,
        mc_mode: str = MC_MODE_STRICT,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class AnswerJsonDataset(BenchmarkDatasetAdapter):
    benchmark_source = "answer_json"
    bias_construction_mode = BIAS_VARIANTS_FROM_DATASET

    def validate_input_jsonl(self, input_jsonl: str) -> None:
        if input_jsonl != "answer.jsonl":
            raise ValueError("--benchmark_source=answer_json requires --input_jsonl=answer.jsonl.")

    def prepare_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        selected_bias_types: Sequence[str],
        selected_ays_mc_datasets: Optional[Sequence[str]] = None,
        instruction_policy: str | None = None,
        mc_mode: str = MC_MODE_STRICT,
    ) -> List[Dict[str, Any]]:
        del selected_bias_types, selected_ays_mc_datasets, instruction_policy, mc_mode
        prepared: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = dict(row)
            base = dict(normalized.get("base", {}) or {})
            metadata = dict(normalized.get("metadata", {}) or {})
            question_text = str(base.get("question_text", base.get("question", "")) or "").strip()
            if question_text:
                base.setdefault("question_text", question_text)
                metadata.setdefault("question_text", question_text)
            metadata.setdefault("bias_construction_mode", self.bias_construction_mode)
            normalized["base"] = base
            normalized["metadata"] = metadata
            prepared.append(normalized)
        return prepared


class AysMcSingleTurnDataset(BenchmarkDatasetAdapter):
    benchmark_source = "ays_mc_single_turn"
    bias_construction_mode = BIAS_VARIANTS_GENERATED_LOCALLY

    def validate_input_jsonl(self, input_jsonl: str) -> None:
        if input_jsonl != "are_you_sure.jsonl":
            raise ValueError(
                "--benchmark_source=ays_mc_single_turn requires --input_jsonl=are_you_sure.jsonl."
            )

    def prepare_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        selected_bias_types: Sequence[str],
        selected_ays_mc_datasets: Optional[Sequence[str]] = None,
        instruction_policy: str | None = None,
        mc_mode: str = MC_MODE_STRICT,
    ) -> List[Dict[str, Any]]:
        biases = resolve_agreement_biases(selected_bias_types, include_neutral=True)
        resolved_instruction_policy = get_instruction_policy(instruction_policy or mc_mode)
        legacy_mc_mode = resolved_instruction_policy.legacy_mc_mode
        response_prefix = resolved_instruction_policy.response_prefix
        wanted_datasets = set(selected_ays_mc_datasets or DEFAULT_AYS_MC_DATASETS)
        materialized: List[Dict[str, Any]] = []

        for row in rows:
            base = row.get("base", {}) or {}
            dataset = str(base.get("dataset", "") or "").strip()
            if dataset not in wanted_datasets:
                continue

            option_items = multiple_choice_option_items(base)
            option_map = {letter: option_text for letter, option_text in option_items}
            letters = "".join(letter for letter, _ in option_items) or str(base.get("letters", "") or "").strip()
            response_labels = [letter.upper() for letter, _ in option_items] or [letter.upper() for letter in letters]
            answers_list = [option_text for _, option_text in option_items]
            correct_letter = str(base.get("correct_letter", "") or "").strip()
            question_text = render_multiple_choice_question(base)
            correct_answer = _correct_answer_for_multiple_choice(base, option_map)
            incorrect_answer, incorrect_answer_source, incorrect_letter = _incorrect_answer_for_multiple_choice(
                base,
                option_items,
                option_map,
            )

            if not question_text or not correct_letter or not correct_answer or not incorrect_answer:
                continue

            derived_base = dict(base)
            derived_base.update(
                {
                    "dataset": dataset,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "incorrect_answer_source": incorrect_answer_source,
                    "incorrect_letter": incorrect_letter,
                    "correct_letter": correct_letter,
                    "letters": letters,
                    "response_labels": response_labels,
                    "answers_list": answers_list,
                    "task_format": "multiple_choice",
                    "instruction_policy": resolved_instruction_policy.name,
                    "mc_mode": legacy_mc_mode,
                    "response_prefix": response_prefix,
                    "answer_channel": "letter",
                    "strict_output_contract": STRICT_OUTPUT_CONTRACT,
                    "prompt_spec_version": int(PROMPT_SPEC_VERSION),
                    "grading_spec_version": int(GRADING_SPEC_VERSION),
                    "benchmark_source": self.benchmark_source,
                    "bias_construction_mode": self.bias_construction_mode,
                }
            )
            question = Question(
                dataset=dataset,
                question_text=question_text,
                correct_answer=correct_answer,
                incorrect_answer=incorrect_answer,
                base_metadata=derived_base,
            )

            for bias in biases:
                variant = bias.build_prompt_variant(
                    question,
                    instruction_policy=resolved_instruction_policy,
                    bias_construction_mode=self.bias_construction_mode,
                    metadata={
                        "dataset": dataset,
                        "instruction_policy": resolved_instruction_policy.name,
                        "mc_mode": legacy_mc_mode,
                        "response_prefix": response_prefix,
                        "answer_channel": "letter",
                        "strict_output_contract": STRICT_OUTPUT_CONTRACT,
                        "prompt_spec_version": int(PROMPT_SPEC_VERSION),
                        "benchmark_source": self.benchmark_source,
                        "bias_construction_mode": self.bias_construction_mode,
                        "correct_letter": correct_letter,
                        "correct_label": correct_letter,
                        "incorrect_letter": incorrect_letter,
                        "suggested_label": incorrect_letter,
                        "source_dataset": str(derived_base.get("source_dataset", "") or ""),
                        "source_split": str(derived_base.get("source_split", "") or ""),
                        "source_example_id": str(derived_base.get("source_example_id", "") or ""),
                    },
                )
                materialized.append(variant.to_row())

        return materialized


def materialize_ays_mc_single_turn_rows(
    rows: Sequence[Dict[str, Any]],
    selected_bias_types: Sequence[str],
    selected_ays_mc_datasets: Sequence[str],
    instruction_policy: str | None = None,
    mc_mode: str = MC_MODE_STRICT,
) -> List[Dict[str, Any]]:
    adapter = AysMcSingleTurnDataset()
    return adapter.prepare_rows(
        rows,
        selected_bias_types=selected_bias_types,
        selected_ays_mc_datasets=selected_ays_mc_datasets,
        instruction_policy=instruction_policy,
        mc_mode=mc_mode,
    )


def dataset_adapter_for_benchmark(benchmark_source: str) -> BenchmarkDatasetAdapter:
    normalized = str(benchmark_source or "").strip()
    if normalized == "answer_json":
        return AnswerJsonDataset()
    if normalized == "ays_mc_single_turn":
        return AysMcSingleTurnDataset()
    raise ValueError(f"Unsupported benchmark_source={benchmark_source!r}")


def prepare_benchmark_rows(
    benchmark_source: str,
    rows: Sequence[Dict[str, Any]],
    input_jsonl: str,
    selected_bias_types: Optional[Sequence[str]] = None,
    selected_ays_mc_datasets: Optional[Sequence[str]] = None,
    instruction_policy: str | None = None,
    mc_mode: str = MC_MODE_STRICT,
) -> List[Dict[str, Any]]:
    adapter = dataset_adapter_for_benchmark(benchmark_source)
    adapter.validate_input_jsonl(input_jsonl)
    return adapter.prepare_rows(
        rows,
        selected_bias_types=selected_bias_types or list(ALL_BIAS_TYPES),
        selected_ays_mc_datasets=selected_ays_mc_datasets,
        instruction_policy=instruction_policy,
        mc_mode=mc_mode,
    )


__all__ = [
    "ALL_AYS_MC_DATASETS",
    "DEFAULT_AYS_MC_DATASETS",
    "HF_AYS_MC_DATASET_SPECS",
    "SYCOPHANCY_AYS_MC_DATASETS",
    "SUPPORTED_BENCHMARK_SOURCES",
    "SYCOPHANCY_FILES",
    "SYCOPHANCY_HF_DATASET",
    "AnswerJsonDataset",
    "AysMcSingleTurnDataset",
    "BenchmarkDatasetAdapter",
    "dataset_adapter_for_benchmark",
    "ensure_external_ays_mc_cached",
    "ensure_sycophancy_eval_cached",
    "is_multiple_choice_base",
    "load_external_ays_mc_rows",
    "materialize_ays_mc_single_turn_rows",
    "multiple_choice_option_items",
    "multiple_choice_option_map",
    "prepare_benchmark_rows",
    "read_jsonl",
    "render_ays_mc_prompt_text",
    "render_ays_mc_question_text",
    "render_multiple_choice_question",
    "resolve_ays_mc_datasets",
]

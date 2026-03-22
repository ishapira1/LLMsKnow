from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


_bootstrap_src_path()

from llmssycoph.cli import load_env_file, resolve_device, resolve_hf_cache_dir
from llmssycoph.data import materialize_ays_mc_single_turn_rows, read_jsonl
from llmssycoph.grading import grade_response_from_base
from llmssycoph.llm import audit_choice_tokenization, load_llm


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Preflight audit for a Hugging Face strict-MC model. "
            "Builds the same answer-only prompt wrapper used by the pipeline for a "
            "single commonsense_qa or arc_challenge item, then audits first-token "
            "letter scoring and parser compatibility."
        )
    )
    ap.add_argument("--model", type=str, required=True, help="Hugging Face model id to audit.")
    ap.add_argument(
        "--dataset_name",
        type=str,
        default="commonsense_qa",
        choices=["commonsense_qa", "arc_challenge"],
        help="Which local dataset slice to use for the prompt audit.",
    )
    ap.add_argument(
        "--question_index",
        type=int,
        default=0,
        help="0-based index within the chosen local dataset file.",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default="data/sycophancy-eval",
        help="Directory containing commonsense_qa.jsonl and arc_challenge.jsonl.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Requested runtime device.",
    )
    ap.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Let Transformers shard the model automatically across available devices.",
    )
    ap.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="HF model/tokenizer cache dir. Falls back to the same env resolution as the pipeline.",
    )
    ap.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Optional .env file to load before resolving cache env vars.",
    )
    ap.add_argument(
        "--prompt_preview_chars",
        type=int,
        default=700,
        help="How many prompt characters to print before truncating.",
    )
    return ap


def _preview_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _load_neutral_prompt_row(dataset_name: str, data_dir: str, question_index: int) -> Dict[str, Any]:
    dataset_path = Path(data_dir) / f"{dataset_name}.jsonl"
    rows = read_jsonl(str(dataset_path))
    if question_index < 0 or question_index >= len(rows):
        raise IndexError(
            f"--question_index {question_index} is out of range for {dataset_path} "
            f"(available rows: 0..{max(len(rows) - 1, 0)})."
        )

    materialized = materialize_ays_mc_single_turn_rows(
        [rows[question_index]],
        selected_bias_types=[],
        selected_ays_mc_datasets=[dataset_name],
        instruction_policy="answer_only",
    )
    if not materialized:
        raise RuntimeError(
            f"Failed to materialize an answer-only prompt for dataset={dataset_name!r} "
            f"question_index={question_index}."
        )
    return materialized[0]


def _choice_summaries(audit_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for choice, payload in sorted((audit_payload.get("choices") or {}).items()):
        variants = payload.get("variants") or []
        summaries.append(
            {
                "choice": choice,
                "renormalized_probability": audit_payload.get("choice_probabilities", {}).get(choice),
                "counted_token_ids": payload.get("counted_token_ids", []),
                "single_token_variants": [
                    {
                        "variant_text": variant.get("variant_text"),
                        "token_id": variant.get("token_id"),
                        "decoded_token": variant.get("decoded_token"),
                        "probability": variant.get("probability"),
                        "counted": variant.get("counted_in_choice_probability"),
                    }
                    for variant in variants
                    if variant.get("single_token")
                ],
            }
        )
    return summaries


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    device = resolve_device(args.device)
    hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir)

    prompt_row = _load_neutral_prompt_row(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        question_index=args.question_index,
    )
    base = dict(prompt_row.get("base", {}) or {})
    prompt_messages = list(prompt_row.get("prompt", []) or [])
    prompt_text = "\n".join(
        str(message.get("content", "") or "").strip()
        for message in prompt_messages
        if isinstance(message, dict)
    ).strip()
    choices = [letter for letter in str(base.get("letters", "") or "").strip().upper() if letter.strip()]
    if not choices:
        raise RuntimeError("Prompt row is missing strict-MC letters.")

    llm = load_llm(
        model_name=args.model,
        device=device,
        device_map_auto=bool(args.device_map_auto),
        hf_cache_dir=hf_cache_dir,
    )
    get_model_and_tokenizer = getattr(llm, "get_model_and_tokenizer", None)
    if not callable(get_model_and_tokenizer):
        raise RuntimeError(
            f"Model backend for {args.model!r} does not expose a raw model/tokenizer pair for tokenization audit."
        )

    model, tokenizer = get_model_and_tokenizer()
    audit = audit_choice_tokenization(
        model=model,
        tokenizer=tokenizer,
        messages=prompt_messages,
        choices=choices,
    )
    probabilities = dict(audit.get("choice_probabilities", {}) or {})
    selected_choice = max(
        choices,
        key=lambda choice: (float(probabilities.get(choice, float("-inf"))), -choices.index(choice)),
    )

    parser_examples = {
        raw: grade_response_from_base(raw, base)
        for raw in (
            selected_choice,
            f"Answer: {selected_choice}",
            f"({selected_choice})",
        )
    }

    print(f"model: {args.model}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"question_index: {args.question_index}")
    print(f"resolved_device: {device}")
    print(f"hf_cache_dir: {hf_cache_dir or '<unset>'}")
    print(f"question: {base.get('question', '')}")
    print(f"correct_letter: {base.get('correct_letter', '')}")
    print(f"letters: {''.join(choices)}")
    print("prompt_preview:")
    print(_preview_text(prompt_text, limit=max(80, int(args.prompt_preview_chars))))
    print("audit_summary:")
    print(
        json.dumps(
            {
                "prompt_token_count": audit.get("prompt_token_count"),
                "prompt_last_token_id": audit.get("prompt_last_token_id"),
                "prompt_tail_text": audit.get("prompt_tail_text"),
                "selected_choice": selected_choice,
                "choice_probabilities": probabilities,
                "choices": _choice_summaries(audit),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("parser_examples:")
    print(
        json.dumps(
            {
                raw: {
                    "status": grading.get("status"),
                    "reason": grading.get("reason"),
                    "committed_answer": grading.get("committed_answer"),
                    "strict_format_exact": grading.get("strict_format_exact"),
                    "starts_with_answer_prefix": grading.get("starts_with_answer_prefix"),
                }
                for raw, grading in parser_examples.items()
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

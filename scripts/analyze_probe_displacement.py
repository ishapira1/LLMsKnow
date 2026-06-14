from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _bootstrap_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    return repo_root


REPO_ROOT = _bootstrap_src_path()
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "sycophancy_bias_probe"
    / "Qwen_Qwen2_5_7B_Instruct"
    / "arc_challenge"
    / "full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun"
)
DEFAULT_OUTPUT_SUBDIR = "analysis/probe_displacement_decomposition"
DEFAULT_CONGRUENT_SUBDIR = "sampling_backfills/model_congruent_suggestion"
DEFAULT_PROBE_NAME = "probe_no_bias"
DEFAULT_CONDITION_ORDER = ["incorrect_suggestion", "model_congruent_suggestion"]
DEFAULT_CANDIDATE_ROLE_ORDER = ["correct_choice", "endorsed_wrong_choice"]

warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator LogisticRegression",
)


from llmssycoph.cli import resolve_device, resolve_hf_cache_dir
from llmssycoph.grading import record_is_usable_for_metrics
from llmssycoph.llm.generation import encode_chat
from llmssycoph.llm.loading import load_model_and_tokenizer
from llmssycoph.probes.movement import decompose_probe_delta
from llmssycoph.probes.features import _assistant_text_last_token_index


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose neutral-probe hidden-state displacements into components parallel and "
            "orthogonal to the saved probe weight vector for a completed strict-MC run."
        ),
    )
    parser.add_argument(
        "--run_dir",
        default=str(DEFAULT_RUN_DIR),
        help=(
            "Completed run directory. Defaults to the canonical Qwen 2.5 7B ARC-Challenge "
            "run with saved congruent backfills."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Optional output directory. Defaults to <run_dir>/{DEFAULT_OUTPUT_SUBDIR}.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to analyze. Defaults to test.",
    )
    parser.add_argument(
        "--probe_name",
        default=DEFAULT_PROBE_NAME,
        help=f"Saved chosen probe family to use. Defaults to {DEFAULT_PROBE_NAME}.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap on the number of neutral-correct paired questions to analyze.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used to load the Hugging Face model for hidden-state extraction.",
    )
    parser.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Pass device_map='auto' when loading the model.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional Hugging Face cache dir override.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate run layout and pairing coverage without loading the Hugging Face model.",
    )
    return parser


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_choice(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().upper()


def choice_letters(record: Dict[str, Any]) -> List[str]:
    letters = str(record.get("letters", "") or "").strip().upper()
    return [letter for letter in letters if letter.strip()]


def resolve_selected_choice(record: Dict[str, Any]) -> str:
    valid_choices = set(choice_letters(record))
    for field in ("response_raw", "response", "committed_answer"):
        choice = normalize_choice(record.get(field))
        if choice in valid_choices:
            return choice
    return ""


def record_supports_candidate_letters(
    record: Dict[str, Any],
    *,
    correct_choice: str,
    endorsed_wrong_choice: str,
) -> bool:
    if not isinstance(record.get("prompt_messages"), list):
        return False
    letters = set(choice_letters(record))
    return bool(correct_choice and endorsed_wrong_choice and correct_choice in letters and endorsed_wrong_choice in letters)


def build_displacement_pairs(
    base_records: Sequence[Dict[str, Any]],
    congruent_records: Sequence[Dict[str, Any]],
    *,
    requested_split: str = "test",
    max_questions: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    incorrect_by_key: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for record in base_records:
        if str(record.get("template_type", "") or "") != "incorrect_suggestion":
            continue
        key = (
            str(record.get("split", "") or ""),
            str(record.get("question_id", "") or ""),
            int(record.get("draw_idx", 0) or 0),
        )
        incorrect_by_key[key] = dict(record)

    congruent_by_source_record_id: Dict[int, Dict[str, Any]] = {}
    congruent_by_key: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for record in congruent_records:
        if str(record.get("template_type", "") or "") != "model_congruent_suggestion":
            continue
        key = (
            str(record.get("split", "") or ""),
            str(record.get("question_id", "") or ""),
            int(record.get("draw_idx", 0) or 0),
        )
        congruent_by_key[key] = dict(record)
        try:
            neutral_source_record_id = int(record.get("neutral_source_record_id"))
        except Exception:
            continue
        congruent_by_source_record_id[neutral_source_record_id] = dict(record)

    pair_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for record in base_records:
        if str(record.get("template_type", "") or "") != "neutral":
            continue
        split = str(record.get("split", "") or "")
        question_id = str(record.get("question_id", "") or "")
        draw_idx = int(record.get("draw_idx", 0) or 0)
        key = (split, question_id, draw_idx)

        coverage_row: Dict[str, Any] = {
            "split": split,
            "question_id": question_id,
            "draw_idx": draw_idx,
            "neutral_record_id": record.get("record_id"),
            "neutral_selected_choice": resolve_selected_choice(record),
            "correct_choice": normalize_choice(record.get("correct_letter")),
            "endorsed_wrong_choice": "",
            "incorrect_record_id": pd.NA,
            "congruent_record_id": pd.NA,
            "included": False,
            "exclusion_reason": "",
        }

        if split != str(requested_split):
            coverage_row["exclusion_reason"] = "split_mismatch"
            coverage_rows.append(coverage_row)
            continue
        if str(record.get("task_format", "") or "") != "multiple_choice":
            coverage_row["exclusion_reason"] = "not_multiple_choice"
            coverage_rows.append(coverage_row)
            continue
        if str(record.get("mc_mode", "") or "") != "strict_mc":
            coverage_row["exclusion_reason"] = "not_strict_mc"
            coverage_rows.append(coverage_row)
            continue
        if not record_is_usable_for_metrics(record):
            coverage_row["exclusion_reason"] = "neutral_not_usable"
            coverage_rows.append(coverage_row)
            continue

        correct_choice = normalize_choice(record.get("correct_letter"))
        neutral_wrong_choice = normalize_choice(record.get("incorrect_letter"))
        neutral_selected_choice = coverage_row["neutral_selected_choice"]
        if not correct_choice or not neutral_wrong_choice:
            coverage_row["exclusion_reason"] = "missing_neutral_choice_metadata"
            coverage_rows.append(coverage_row)
            continue
        if correct_choice == neutral_wrong_choice:
            coverage_row["endorsed_wrong_choice"] = neutral_wrong_choice
            coverage_row["exclusion_reason"] = "non_distinct_choice_metadata"
            coverage_rows.append(coverage_row)
            continue
        if neutral_selected_choice != correct_choice:
            coverage_row["endorsed_wrong_choice"] = neutral_wrong_choice
            coverage_row["exclusion_reason"] = "neutral_not_correct"
            coverage_rows.append(coverage_row)
            continue

        incorrect_record = incorrect_by_key.get(key)
        if incorrect_record is None:
            coverage_row["endorsed_wrong_choice"] = neutral_wrong_choice
            coverage_row["exclusion_reason"] = "missing_incorrect_pair"
            coverage_rows.append(coverage_row)
            continue
        coverage_row["incorrect_record_id"] = incorrect_record.get("record_id", pd.NA)

        incorrect_wrong_choice = normalize_choice(incorrect_record.get("incorrect_letter"))
        if incorrect_wrong_choice and neutral_wrong_choice and incorrect_wrong_choice != neutral_wrong_choice:
            coverage_row["endorsed_wrong_choice"] = incorrect_wrong_choice
            coverage_row["exclusion_reason"] = "inconsistent_endorsed_wrong_choice"
            coverage_rows.append(coverage_row)
            continue
        endorsed_wrong_choice = incorrect_wrong_choice or neutral_wrong_choice
        coverage_row["endorsed_wrong_choice"] = endorsed_wrong_choice
        if not endorsed_wrong_choice or endorsed_wrong_choice == correct_choice:
            coverage_row["exclusion_reason"] = "non_distinct_choice_metadata"
            coverage_rows.append(coverage_row)
            continue

        try:
            neutral_record_id = int(record.get("record_id"))
        except Exception:
            neutral_record_id = None
        congruent_record = (
            congruent_by_source_record_id.get(neutral_record_id)
            if neutral_record_id is not None
            else None
        )
        if congruent_record is None:
            congruent_record = congruent_by_key.get(key)
        if congruent_record is None:
            coverage_row["exclusion_reason"] = "missing_congruent_pair"
            coverage_rows.append(coverage_row)
            continue
        coverage_row["congruent_record_id"] = congruent_record.get("record_id", pd.NA)

        if not record_supports_candidate_letters(
            record,
            correct_choice=correct_choice,
            endorsed_wrong_choice=endorsed_wrong_choice,
        ):
            coverage_row["exclusion_reason"] = "neutral_missing_candidate_letters"
            coverage_rows.append(coverage_row)
            continue
        if not record_supports_candidate_letters(
            incorrect_record,
            correct_choice=correct_choice,
            endorsed_wrong_choice=endorsed_wrong_choice,
        ):
            coverage_row["exclusion_reason"] = "incorrect_missing_candidate_letters"
            coverage_rows.append(coverage_row)
            continue
        if not record_supports_candidate_letters(
            congruent_record,
            correct_choice=correct_choice,
            endorsed_wrong_choice=endorsed_wrong_choice,
        ):
            coverage_row["exclusion_reason"] = "congruent_missing_candidate_letters"
            coverage_rows.append(coverage_row)
            continue

        coverage_row["included"] = True
        coverage_rows.append(coverage_row)
        pair_rows.append(
            {
                "split": split,
                "question_id": question_id,
                "draw_idx": draw_idx,
                "neutral_record": dict(record),
                "incorrect_record": dict(incorrect_record),
                "congruent_record": dict(congruent_record),
                "correct_choice": correct_choice,
                "endorsed_wrong_choice": endorsed_wrong_choice,
            }
        )
        if max_questions is not None and len(pair_rows) >= int(max_questions):
            break

    coverage_df = pd.DataFrame(coverage_rows)
    return pair_rows, coverage_df


def model_device(model) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError("Could not infer model device.") from exc


def pad_token_id_for(tokenizer) -> int:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        token_id = 0
    return int(token_id)


def encode_with_last_token_index(
    tokenizer,
    prompt_messages: Sequence[Dict[str, Any]],
    completion: str,
) -> tuple[List[int], int]:
    messages = list(prompt_messages) + [{"type": "assistant", "content": completion}]
    encoded = encode_chat(tokenizer, messages, add_generation_prompt=False)
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids[0].detach().cpu().tolist()
    else:
        input_ids = encoded[0].detach().cpu().tolist()
    token_ids = [int(token_id) for token_id in input_ids]
    last_idx = _assistant_text_last_token_index(tokenizer, token_ids, completion)
    return token_ids, int(last_idx)


def extract_choice_feature_map(
    model,
    tokenizer,
    prompt_messages: Sequence[Dict[str, Any]],
    choices: Sequence[str],
    *,
    layer: int,
) -> Dict[str, np.ndarray]:
    import torch

    if not choices:
        return {}

    encoded_rows: List[List[int]] = []
    last_token_indices: List[int] = []
    for choice in choices:
        token_ids, last_idx = encode_with_last_token_index(tokenizer, prompt_messages, str(choice))
        encoded_rows.append(token_ids)
        last_token_indices.append(last_idx)

    target_device = model_device(model)
    pad_token_id = pad_token_id_for(tokenizer)
    batch_size = len(encoded_rows)
    max_len = max(len(token_ids) for token_ids in encoded_rows)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=target_device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=target_device)
    for row_idx, token_ids in enumerate(encoded_rows):
        row_tensor = torch.tensor(token_ids, dtype=torch.long, device=target_device)
        input_ids[row_idx, : len(token_ids)] = row_tensor
        attention_mask[row_idx, : len(token_ids)] = 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    feature_map: Dict[str, np.ndarray] = {}
    layer_hidden = outputs.hidden_states[int(layer)]
    for row_idx, choice in enumerate(choices):
        last_idx = last_token_indices[row_idx]
        feature_map[str(choice)] = layer_hidden[row_idx, last_idx].detach().float().cpu().numpy()

    del outputs
    del input_ids
    del attention_mask
    return feature_map


def decompose_probe_direction(
    delta_vec: np.ndarray,
    probe_weights: np.ndarray,
) -> Dict[str, float]:
    metrics = decompose_probe_delta(delta_vec, probe_weights)
    delta_l2 = float(np.sqrt(float(metrics["delta_l2_sq"])))
    parallel_l2 = float(np.sqrt(float(metrics["parallel_l2_sq"])))
    orthogonal_l2 = float(np.sqrt(float(metrics["orthogonal_l2_sq"])))
    probe_weight_norm = float(np.linalg.norm(np.asarray(probe_weights, dtype=float)))
    return {
        "score_shift_linear": float(metrics["delta_probe_logit"]),
        "delta_l2": delta_l2,
        "parallel_l2": parallel_l2,
        "orthogonal_l2": orthogonal_l2,
        "orthogonal_fraction": 0.0 if delta_l2 <= 0.0 else float(orthogonal_l2 / delta_l2),
        "parallel_fraction": 0.0 if delta_l2 <= 0.0 else float(parallel_l2 / delta_l2),
        "reconstruction_error": float(metrics["reconstruction_error"]),
        "probe_weight_norm": probe_weight_norm,
    }


def build_pairwise_rows_for_pair(
    *,
    pair: Dict[str, Any],
    probe_name: str,
    layer: int,
    probe_weights: np.ndarray,
    neutral_feature_map: Dict[str, np.ndarray],
    condition_feature_map: Dict[str, np.ndarray],
    condition: str,
    model_name: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    neutral_record = pair["neutral_record"]
    condition_record = pair["incorrect_record"] if condition == "incorrect_suggestion" else pair["congruent_record"]
    correct_choice = pair["correct_choice"]
    endorsed_wrong_choice = pair["endorsed_wrong_choice"]
    rows: List[Dict[str, Any]] = []
    for candidate_role, candidate_choice in [
        ("correct_choice", correct_choice),
        ("endorsed_wrong_choice", endorsed_wrong_choice),
    ]:
        neutral_vec = neutral_feature_map[candidate_choice]
        condition_vec = condition_feature_map[candidate_choice]
        delta_vec = condition_vec - neutral_vec
        metrics = decompose_probe_direction(delta_vec, probe_weights)
        rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "probe_name": probe_name,
                "probe_layer": int(layer),
                "split": pair["split"],
                "question_id": pair["question_id"],
                "draw_idx": int(pair["draw_idx"]),
                "condition": condition,
                "candidate_role": candidate_role,
                "candidate_choice": candidate_choice,
                "candidate_is_correct": bool(candidate_role == "correct_choice"),
                "correct_choice": correct_choice,
                "endorsed_wrong_choice": endorsed_wrong_choice,
                "neutral_record_id": neutral_record.get("record_id"),
                "neutral_prompt_id": str(neutral_record.get("prompt_id", "") or ""),
                "condition_record_id": condition_record.get("record_id"),
                "condition_prompt_id": str(condition_record.get("prompt_id", "") or ""),
                "neutral_selected_choice": resolve_selected_choice(neutral_record),
                "condition_selected_choice": resolve_selected_choice(condition_record),
                "score_shift_linear": metrics["score_shift_linear"],
                "delta_l2": metrics["delta_l2"],
                "parallel_l2": metrics["parallel_l2"],
                "orthogonal_l2": metrics["orthogonal_l2"],
                "orthogonal_fraction": metrics["orthogonal_fraction"],
                "parallel_fraction": metrics["parallel_fraction"],
                "reconstruction_error": metrics["reconstruction_error"],
                "probe_weight_norm": metrics["probe_weight_norm"],
            }
        )
    return rows


def summarize_probe_displacement(pair_df: pd.DataFrame) -> pd.DataFrame:
    summary_columns = [
        "condition",
        "candidate_role",
        "n_pairs",
        "n_questions",
        "mean_delta_l2",
        "median_delta_l2",
        "mean_orthogonal_l2",
        "median_orthogonal_l2",
        "mean_orthogonal_fraction",
        "mean_score_shift_linear",
        "mean_abs_score_shift_linear",
    ]
    if pair_df.empty:
        return pd.DataFrame(columns=summary_columns)

    rows: List[Dict[str, Any]] = []

    def append_summary(subset: pd.DataFrame, *, condition: str, candidate_role: str) -> None:
        if subset.empty:
            return
        rows.append(
            {
                "condition": condition,
                "candidate_role": candidate_role,
                "n_pairs": int(len(subset)),
                "n_questions": int(subset["question_id"].nunique()),
                "mean_delta_l2": float(pd.to_numeric(subset["delta_l2"], errors="coerce").mean()),
                "median_delta_l2": float(pd.to_numeric(subset["delta_l2"], errors="coerce").median()),
                "mean_orthogonal_l2": float(pd.to_numeric(subset["orthogonal_l2"], errors="coerce").mean()),
                "median_orthogonal_l2": float(pd.to_numeric(subset["orthogonal_l2"], errors="coerce").median()),
                "mean_orthogonal_fraction": float(
                    pd.to_numeric(subset["orthogonal_fraction"], errors="coerce").mean()
                ),
                "mean_score_shift_linear": float(
                    pd.to_numeric(subset["score_shift_linear"], errors="coerce").mean()
                ),
                "mean_abs_score_shift_linear": float(
                    pd.to_numeric(subset["score_shift_linear"], errors="coerce").abs().mean()
                ),
            }
        )

    for condition in DEFAULT_CONDITION_ORDER:
        condition_subset = pair_df.loc[pair_df["condition"].astype(str).eq(condition)].copy()
        if condition_subset.empty:
            continue
        for candidate_role in DEFAULT_CANDIDATE_ROLE_ORDER:
            append_summary(
                condition_subset.loc[condition_subset["candidate_role"].astype(str).eq(candidate_role)],
                condition=condition,
                candidate_role=candidate_role,
            )
        append_summary(condition_subset, condition=condition, candidate_role="all")

    return pd.DataFrame(rows, columns=summary_columns)


def maybe_clear_device_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def load_probe_metadata(run_dir: Path, probe_name: str) -> Dict[str, Any]:
    probe_dir = run_dir / "probes" / "chosen_probe" / probe_name
    metadata_path = probe_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing chosen probe metadata: {metadata_path}")
    return load_json(metadata_path)


def load_probe_weights(run_dir: Path, probe_name: str) -> tuple[np.ndarray, Dict[str, Any]]:
    probe_dir = run_dir / "probes" / "chosen_probe" / probe_name
    metadata = load_probe_metadata(run_dir, probe_name)
    model_path = probe_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing chosen probe model: {model_path}")
    with model_path.open("rb") as handle:
        clf = pickle.load(handle)
    coef = getattr(clf, "coef_", None)
    if coef is None:
        raise ValueError(f"Saved probe model is missing coef_: {model_path}")
    probe_weights = np.asarray(coef[0], dtype=float)
    if probe_weights.ndim != 1:
        raise ValueError(f"Expected a 1D probe weight vector, got shape={probe_weights.shape}")
    return probe_weights, metadata


def main() -> None:
    args = build_parser().parse_args()
    if str(args.split) != "test":
        raise ValueError(
            f"This analysis currently supports only --split test, got --split {args.split!r}."
        )
    run_dir = resolve_repo_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    run_config_path = run_dir / "run_config.json"
    sampling_records_path = run_dir / "logs" / "sampling_records.jsonl"
    congruent_records_path = run_dir / DEFAULT_CONGRUENT_SUBDIR / "sampling_records.jsonl"
    backfill_script_path = (REPO_ROOT / "scripts" / "backfill_model_congruent_prompts.py").resolve()

    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {run_dir}")
    if not sampling_records_path.exists():
        raise FileNotFoundError(f"Missing base sampling_records.jsonl in {sampling_records_path}")
    if not congruent_records_path.exists():
        raise FileNotFoundError(
            "Missing model_congruent_suggestion backfill records at "
            f"{congruent_records_path}. Create them first with {backfill_script_path}."
        )

    run_config = load_json(run_config_path)
    probe_metadata = load_probe_metadata(run_dir, args.probe_name)
    if str(probe_metadata.get("template_type", "") or "") != "neutral":
        raise ValueError(
            f"Expected a neutral chosen probe, but {args.probe_name!r} has template_type="
            f"{probe_metadata.get('template_type')!r}."
        )

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else (run_dir / DEFAULT_OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_records = load_jsonl_records(sampling_records_path)
    congruent_records = load_jsonl_records(congruent_records_path)
    paired_questions, coverage_df = build_displacement_pairs(
        base_records,
        congruent_records,
        requested_split=args.split,
        max_questions=args.max_questions,
    )
    if coverage_df.empty:
        raise RuntimeError("No neutral records were found while building displacement pairs.")
    if not paired_questions:
        raise RuntimeError(
            "No neutral-correct question pairs survived filtering. "
            "Inspect the manifest coverage summary for exclusion reasons."
        )

    coverage_counts = Counter(str(reason or "included") for reason in coverage_df["exclusion_reason"].fillna(""))
    included_count = int(coverage_df["included"].astype(bool).sum())
    if "" in coverage_counts:
        del coverage_counts[""]
    coverage_counts["included"] = included_count

    manifest: Dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "model_name": str(run_config.get("model", "") or ""),
        "dataset_name": str(run_config.get("dataset_name", "") or ""),
        "requested_split": str(args.split),
        "probe_name": str(args.probe_name),
        "probe_training_template_type": str(probe_metadata.get("template_type", "") or ""),
        "probe_layer": int(probe_metadata["layer"]),
        "conditions": list(DEFAULT_CONDITION_ORDER),
        "candidate_roles": list(DEFAULT_CANDIDATE_ROLE_ORDER),
        "comparison_definition": (
            "Teacher-forced same-candidate comparison at the final answer token. For each neutral-correct "
            "question, compare neutral vs incorrect_suggestion and neutral vs model_congruent_suggestion "
            "for the gold answer c and the incorrect-suggestion endorsed wrong answer b."
        ),
        "congruent_wrong_choice_convention": (
            "Reuse the incorrect_suggestion endorsed wrong answer b under the congruent prompt as well."
        ),
        "record_counts": {
            "base_sampling_records": int(len(base_records)),
            "congruent_sampling_records": int(len(congruent_records)),
            "paired_question_count": int(len(paired_questions)),
        },
        "coverage_summary": dict(sorted(coverage_counts.items())),
        "source_artifacts": {
            "base_sampling_records": str(sampling_records_path),
            "congruent_sampling_records": str(congruent_records_path),
            "probe_metadata": str((run_dir / "probes" / "chosen_probe" / args.probe_name / "metadata.json").resolve()),
            "probe_model": str((run_dir / "probes" / "chosen_probe" / args.probe_name / "model.pkl").resolve()),
        },
        "dry_run": bool(args.dry_run),
        "files": {},
    }

    manifest_path = output_dir / "probe_displacement_manifest.json"
    if args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[dry-run] paired_question_count={len(paired_questions)}")
        print(f"[dry-run] manifest={manifest_path}")
        return

    probe_weights, probe_metadata = load_probe_weights(run_dir, args.probe_name)
    probe_layer = int(probe_metadata["layer"])
    resolved_device = resolve_device(args.device)
    hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir or run_config.get("hf_cache_dir"))
    model_name = str(run_config.get("model", "") or "")
    if not model_name:
        raise ValueError("run_config.json is missing model.")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=resolved_device,
        device_map_auto=bool(args.device_map_auto),
        hf_cache_dir=hf_cache_dir,
    )

    pair_rows: List[Dict[str, Any]] = []
    progress = tqdm(paired_questions, desc="Extracting probe displacement rows", unit="question")
    for pair_idx, pair in enumerate(progress, start=1):
        correct_choice = pair["correct_choice"]
        endorsed_wrong_choice = pair["endorsed_wrong_choice"]
        target_choices = [correct_choice, endorsed_wrong_choice]

        neutral_feature_map = extract_choice_feature_map(
            model,
            tokenizer,
            pair["neutral_record"]["prompt_messages"],
            target_choices,
            layer=probe_layer,
        )
        incorrect_feature_map = extract_choice_feature_map(
            model,
            tokenizer,
            pair["incorrect_record"]["prompt_messages"],
            target_choices,
            layer=probe_layer,
        )
        congruent_feature_map = extract_choice_feature_map(
            model,
            tokenizer,
            pair["congruent_record"]["prompt_messages"],
            target_choices,
            layer=probe_layer,
        )

        pair_rows.extend(
            build_pairwise_rows_for_pair(
                pair=pair,
                probe_name=args.probe_name,
                layer=probe_layer,
                probe_weights=probe_weights,
                neutral_feature_map=neutral_feature_map,
                condition_feature_map=incorrect_feature_map,
                condition="incorrect_suggestion",
                model_name=model_name,
                dataset_name=str(run_config.get("dataset_name", "") or ""),
            )
        )
        pair_rows.extend(
            build_pairwise_rows_for_pair(
                pair=pair,
                probe_name=args.probe_name,
                layer=probe_layer,
                probe_weights=probe_weights,
                neutral_feature_map=neutral_feature_map,
                condition_feature_map=congruent_feature_map,
                condition="model_congruent_suggestion",
                model_name=model_name,
                dataset_name=str(run_config.get("dataset_name", "") or ""),
            )
        )

        if pair_idx % 32 == 0:
            maybe_clear_device_cache()

    if not pair_rows:
        raise RuntimeError("Hidden-state extraction finished, but no pairwise rows were produced.")

    pair_df = pd.DataFrame(pair_rows)
    summary_df = summarize_probe_displacement(pair_df)

    pairwise_path = output_dir / "pairwise_probe_displacement.csv"
    summary_path = output_dir / "probe_displacement_summary.csv"
    pair_df.to_csv(pairwise_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    manifest["resolved_device"] = resolved_device
    manifest["hf_cache_dir"] = hf_cache_dir
    manifest["probe_weight_norm"] = float(np.linalg.norm(probe_weights))
    manifest["record_counts"]["pairwise_rows"] = int(len(pair_df))
    manifest["record_counts"]["summary_rows"] = int(len(summary_df))
    manifest["files"]["pairwise_probe_displacement_csv"] = str(pairwise_path)
    manifest["files"]["probe_displacement_summary_csv"] = str(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] paired_question_count={len(paired_questions)}")
    print(f"[done] pairwise_rows={len(pair_df)}")
    print(f"[done] pairwise_csv={pairwise_path}")
    print(f"[done] summary_csv={summary_path}")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()

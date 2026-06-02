from __future__ import annotations

import argparse
import gc
import json
import sys
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
    / "meta_llama_Llama_3_1_8B_Instruct"
    / "aqua_mc"
    / "full_aqua_mc_llama31_8b_20260320_auto_allq_l32_seas"
)
DEFAULT_OUTPUT_SUBDIR = "analysis/activation_change_teacher_forced_choices"


from llmssycoph.cli import resolve_device, resolve_hf_cache_dir
from llmssycoph.grading import record_is_usable_for_metrics
from llmssycoph.llm.generation import encode_chat
from llmssycoph.llm.loading import load_model_and_tokenizer
from llmssycoph.probes.features import _assistant_text_last_token_index


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how strict-MC probe-input activations change between neutral prompts and "
            "biased prompts for one completed Hugging Face run. The comparison is defined on "
            "teacher-forced answer-choice letters, which matches the probe input semantics used "
            "for strict-MC choice-candidate probes."
        ),
    )
    parser.add_argument(
        "--run_dir",
        default=str(DEFAULT_RUN_DIR),
        help=(
            "Run directory to analyze. Defaults to the completed "
            "Llama-3.1-8B / aqua_mc strict-MC run bundled in this repo."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Optional output directory. Defaults to <run_dir>/"
            f"{DEFAULT_OUTPUT_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--bias_types",
        default=None,
        help=(
            "Comma-separated bias template types to compare against neutral. "
            "Defaults to the run's saved bias_types."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used to load the Hugging Face model for activation extraction.",
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
        "--layer_min",
        type=int,
        default=1,
        help="Minimum transformer layer index to analyze.",
    )
    parser.add_argument(
        "--layer_max",
        type=int,
        default=None,
        help="Maximum transformer layer index to analyze. Defaults to the model's last layer.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional limit on the number of paired question groups to analyze.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate the run layout and pairing coverage without loading the model or extracting "
            "activations."
        ),
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


def parse_csv_choices(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def choice_letters(record: Dict[str, Any]) -> List[str]:
    letters = str(record.get("letters", "") or "").strip().upper()
    return [letter for letter in letters if letter.strip()]


def usable_strict_mc_records(
    records: Sequence[Dict[str, Any]],
    *,
    bias_types: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_templates = {"neutral", *[str(bias_type) for bias_type in bias_types]}
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if not record_is_usable_for_metrics(record):
            continue
        if str(record.get("task_format", "") or "") != "multiple_choice":
            continue
        if str(record.get("mc_mode", "") or "") != "strict_mc":
            continue
        if str(record.get("template_type", "") or "") not in allowed_templates:
            continue
        if not choice_letters(record):
            continue
        if not isinstance(record.get("prompt_messages"), list):
            continue
        filtered.append(dict(record))
    return filtered


def build_paired_question_groups(
    records: Sequence[Dict[str, Any]],
    *,
    bias_types: Sequence[str],
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    grouped: Dict[tuple[str, str, int], Dict[str, Dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("split", "") or ""),
            str(record.get("question_id", "") or ""),
            int(record.get("draw_idx", 0) or 0),
        )
        grouped.setdefault(key, {})[str(record.get("template_type", "") or "")] = record

    pairable_groups: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for (split, question_id, draw_idx), by_template in sorted(grouped.items()):
        coverage_row = {
            "split": split,
            "question_id": question_id,
            "draw_idx": draw_idx,
            "has_neutral": "neutral" in by_template,
        }
        missing_biases = [bias_type for bias_type in bias_types if bias_type not in by_template]
        for bias_type in bias_types:
            coverage_row[f"has_{bias_type}"] = bias_type in by_template
        coverage_row["missing_bias_count"] = len(missing_biases)
        coverage_row["missing_biases"] = ",".join(missing_biases)
        coverage_row["fully_paired"] = coverage_row["has_neutral"] and not missing_biases
        coverage_rows.append(coverage_row)
        if not coverage_row["fully_paired"]:
            continue
        pairable_groups.append(
            {
                "split": split,
                "question_id": question_id,
                "draw_idx": draw_idx,
                "neutral_record": by_template["neutral"],
                "biased_records": {bias_type: by_template[bias_type] for bias_type in bias_types},
            }
        )
    return pairable_groups, pd.DataFrame(coverage_rows)


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


def extract_choice_feature_matrices(
    model,
    tokenizer,
    prompt_messages: Sequence[Dict[str, Any]],
    choices: Sequence[str],
    layer_grid: Sequence[int],
) -> Dict[str, np.ndarray]:
    import torch

    if not choices:
        return {}

    encoded_rows: List[List[int]] = []
    last_token_indices: List[int] = []
    for choice in choices:
        token_ids, last_idx = encode_with_last_token_index(tokenizer, prompt_messages, choice)
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
    for row_idx, choice in enumerate(choices):
        layer_vectors: List[np.ndarray] = []
        last_idx = last_token_indices[row_idx]
        for layer in layer_grid:
            layer_hidden = outputs.hidden_states[int(layer)]
            layer_vectors.append(layer_hidden[row_idx, last_idx].detach().float().cpu().numpy())
        feature_map[str(choice)] = np.stack(layer_vectors, axis=0)

    del outputs
    del input_ids
    del attention_mask
    return feature_map


def choice_probability(record: Dict[str, Any], choice: str) -> float:
    raw = record.get("choice_probabilities", {})
    if not isinstance(raw, dict):
        return float("nan")
    try:
        return float(raw.get(choice, float("nan")))
    except Exception:
        return float("nan")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(vec_a, vec_b) / denom)


def build_pair_rows_for_group(
    *,
    group: Dict[str, Any],
    bias_types: Sequence[str],
    layer_grid: Sequence[int],
    neutral_feature_map: Dict[str, np.ndarray],
    biased_feature_maps: Dict[str, Dict[str, np.ndarray]],
    model_name: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    neutral_record = group["neutral_record"]
    neutral_choices = choice_letters(neutral_record)
    neutral_choice_set = set(neutral_choices)
    neutral_selected_choice = str(neutral_record.get("response_raw", "") or "").strip().upper()
    neutral_correct_choice = str(neutral_record.get("correct_letter", "") or "").strip().upper()

    for bias_type in bias_types:
        biased_record = group["biased_records"][bias_type]
        biased_choices = choice_letters(biased_record)
        shared_choices = [choice for choice in neutral_choices if choice in set(biased_choices)]
        biased_selected_choice = str(biased_record.get("response_raw", "") or "").strip().upper()
        biased_correct_choice = str(biased_record.get("correct_letter", "") or "").strip().upper()
        biased_feature_map = biased_feature_maps[bias_type]

        for choice in shared_choices:
            if choice not in neutral_choice_set:
                continue
            if choice not in neutral_feature_map or choice not in biased_feature_map:
                continue
            neutral_layers = neutral_feature_map[choice]
            biased_layers = biased_feature_map[choice]
            candidate_is_correct = choice == neutral_correct_choice == biased_correct_choice
            neutral_is_selected = choice == neutral_selected_choice
            biased_is_selected = choice == biased_selected_choice
            selected_in_either = neutral_is_selected or biased_is_selected
            for layer_idx, layer in enumerate(layer_grid):
                neutral_vec = neutral_layers[layer_idx]
                biased_vec = biased_layers[layer_idx]
                delta_vec = biased_vec - neutral_vec
                neutral_norm = float(np.linalg.norm(neutral_vec))
                biased_norm = float(np.linalg.norm(biased_vec))
                delta_l2 = float(np.linalg.norm(delta_vec))
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "split": group["split"],
                        "question_id": group["question_id"],
                        "draw_idx": int(group["draw_idx"]),
                        "bias_type": str(bias_type),
                        "candidate_choice": str(choice),
                        "candidate_is_correct": bool(candidate_is_correct),
                        "neutral_selected_choice": neutral_selected_choice,
                        "biased_selected_choice": biased_selected_choice,
                        "neutral_candidate_is_selected": bool(neutral_is_selected),
                        "biased_candidate_is_selected": bool(biased_is_selected),
                        "candidate_selected_in_either": bool(selected_in_either),
                        "neutral_choice_probability": choice_probability(neutral_record, choice),
                        "biased_choice_probability": choice_probability(biased_record, choice),
                        "layer": int(layer),
                        "neutral_norm": neutral_norm,
                        "biased_norm": biased_norm,
                        "delta_l2": delta_l2,
                        "relative_delta_l2": float(delta_l2 / neutral_norm) if neutral_norm > 0.0 else float("nan"),
                        "cosine_similarity": cosine_similarity(neutral_vec, biased_vec),
                    }
                )
    return rows


def summarize_pair_df(pair_df: pd.DataFrame) -> pd.DataFrame:
    subset_masks = {
        "all": pd.Series(True, index=pair_df.index),
        "correct_choice": pair_df["candidate_is_correct"].astype(bool),
        "selected_under_neutral": pair_df["neutral_candidate_is_selected"].astype(bool),
        "selected_under_biased": pair_df["biased_candidate_is_selected"].astype(bool),
        "selected_under_either": pair_df["candidate_selected_in_either"].astype(bool),
    }

    summary_frames: List[pd.DataFrame] = []
    for subset_name, mask in subset_masks.items():
        subset_df = pair_df.loc[mask].copy()
        if subset_df.empty:
            continue
        grouped = (
            subset_df.groupby(["bias_type", "layer"], as_index=False)
            .agg(
                pair_count=("delta_l2", "size"),
                question_count=("question_id", "nunique"),
                mean_delta_l2=("delta_l2", "mean"),
                median_delta_l2=("delta_l2", "median"),
                mean_relative_delta_l2=("relative_delta_l2", "mean"),
                mean_cosine_similarity=("cosine_similarity", "mean"),
                mean_neutral_norm=("neutral_norm", "mean"),
                mean_biased_norm=("biased_norm", "mean"),
            )
        )
        grouped.insert(0, "candidate_subset", subset_name)
        summary_frames.append(grouped)
    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)


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


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve_repo_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    run_config_path = run_dir / "run_config.json"
    sampling_records_path = run_dir / "logs" / "sampling_records.jsonl"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {run_dir}")
    if not sampling_records_path.exists():
        raise FileNotFoundError(f"Missing sampling_records.jsonl in {sampling_records_path}")

    run_config = load_json(run_config_path)
    bias_types = parse_csv_choices(args.bias_types) or parse_csv_choices(str(run_config.get("bias_types", "") or ""))
    if not bias_types:
        raise ValueError("Could not determine bias_types from --bias_types or run_config.json.")

    records = load_jsonl_records(sampling_records_path)
    strict_mc_records = usable_strict_mc_records(records, bias_types=bias_types)
    if not strict_mc_records:
        raise RuntimeError(
            "No usable strict-MC records were found. This script only supports completed strict-MC Hugging Face runs."
        )

    paired_groups, coverage_df = build_paired_question_groups(strict_mc_records, bias_types=bias_types)
    if coverage_df.empty:
        raise RuntimeError("No candidate prompt groups were found after filtering.")
    if not paired_groups:
        raise RuntimeError("No fully paired neutral/bias question groups were found.")

    total_paired_groups = len(paired_groups)
    if args.max_questions is not None and args.max_questions > 0:
        paired_groups = paired_groups[: int(args.max_questions)]

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else (run_dir / DEFAULT_OUTPUT_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = output_dir / "question_pair_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)

    manifest: Dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "model_name": str(run_config.get("model", "") or ""),
        "dataset_name": str(run_config.get("dataset_name", "") or ""),
        "bias_types": list(bias_types),
        "comparison_definition": (
            "Teacher-forced same-choice comparison: for each strict-MC question, compare the "
            "last-token hidden state of the same answer-choice letter under the neutral prompt "
            "versus each biased prompt."
        ),
        "probe_input_alignment": (
            "Matches strict-MC choice-candidate probe inputs, where the probe completion text is "
            "the candidate answer letter."
        ),
        "record_counts": {
            "all_sampling_records": len(records),
            "usable_strict_mc_records": len(strict_mc_records),
            "total_pairable_question_groups": total_paired_groups,
            "analyzed_question_groups": len(paired_groups),
        },
        "files": {
            "question_pair_coverage_csv": str(coverage_path),
        },
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        manifest_path = output_dir / "activation_change_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[dry-run] paired_question_groups={len(paired_groups)}")
        print(f"[dry-run] coverage_csv={coverage_path}")
        print(f"[dry-run] manifest={manifest_path}")
        return

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

    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    if n_layers <= 0:
        raise RuntimeError("Could not determine the number of hidden layers from the loaded model.")
    layer_min = max(1, int(args.layer_min))
    layer_max = min(int(args.layer_max) if args.layer_max is not None else n_layers, n_layers)
    if layer_min > layer_max:
        raise ValueError(f"Invalid layer range: layer_min={layer_min}, layer_max={layer_max}")
    layer_grid = list(range(layer_min, layer_max + 1))

    pair_rows: List[Dict[str, Any]] = []
    progress = tqdm(paired_groups, desc="Extracting activation deltas", unit="question")
    for group_idx, group in enumerate(progress, start=1):
        neutral_record = group["neutral_record"]
        neutral_choices = choice_letters(neutral_record)
        neutral_feature_map = extract_choice_feature_matrices(
            model,
            tokenizer,
            neutral_record["prompt_messages"],
            neutral_choices,
            layer_grid,
        )

        biased_feature_maps: Dict[str, Dict[str, np.ndarray]] = {}
        for bias_type in bias_types:
            biased_record = group["biased_records"][bias_type]
            biased_feature_maps[bias_type] = extract_choice_feature_matrices(
                model,
                tokenizer,
                biased_record["prompt_messages"],
                choice_letters(biased_record),
                layer_grid,
            )

        pair_rows.extend(
            build_pair_rows_for_group(
                group=group,
                bias_types=bias_types,
                layer_grid=layer_grid,
                neutral_feature_map=neutral_feature_map,
                biased_feature_maps=biased_feature_maps,
                model_name=model_name,
                dataset_name=str(run_config.get("dataset_name", "") or ""),
            )
        )

        if group_idx % 32 == 0:
            maybe_clear_device_cache()

    if not pair_rows:
        raise RuntimeError("Activation extraction finished, but no paired delta rows were produced.")

    pair_df = pd.DataFrame(pair_rows)
    summary_df = summarize_pair_df(pair_df)

    pair_path = output_dir / "pairwise_activation_change_by_layer.csv"
    summary_path = output_dir / "layerwise_activation_change_summary.csv"
    pair_df.to_csv(pair_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    manifest["resolved_device"] = resolved_device
    manifest["hf_cache_dir"] = hf_cache_dir
    manifest["layer_grid"] = layer_grid
    manifest["record_counts"]["pairwise_rows"] = int(len(pair_df))
    manifest["files"]["pairwise_activation_change_csv"] = str(pair_path)
    manifest["files"]["layerwise_activation_change_summary_csv"] = str(summary_path)

    manifest_path = output_dir / "activation_change_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] paired_question_groups={len(paired_groups)}")
    print(f"[done] pairwise_rows={len(pair_df)}")
    print(f"[done] pairwise_csv={pair_path}")
    print(f"[done] summary_csv={summary_path}")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()

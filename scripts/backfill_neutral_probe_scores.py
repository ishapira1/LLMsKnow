from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def _bootstrap_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    return repo_root


REPO_ROOT = _bootstrap_src_path()

from llmssycoph.grading.probe_data import build_choice_candidate_records
from llmssycoph.llm.loading import load_model_and_tokenizer
from llmssycoph.probes.score import score_records_with_probe
from llmssycoph.saving_manager import build_mc_probe_scores_by_prompt_df, to_probe_candidate_scores_df


warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator LogisticRegression",
)


DEFAULT_RUN_DIRS = [
    "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/commonsense_qa/full_commonsense_qa_llama31_8b_20260321_allq_fulldepth_seas",
    "results/sycophancy_bias_probe/meta_llama_Llama_3_1_8B_Instruct/arc_challenge/full_arc_challenge_llama31_8b_20260321_allq_fulldepth_seas",
    "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/commonsense_qa/full_commonsense_qa_qwen25_7b_20260322_allq_fulldepth_seas",
    "results/sycophancy_bias_probe/Qwen_Qwen2_5_7B_Instruct/arc_challenge/full_arc_challenge_qwen25_7b_20260322_allq_fulldepth_seas_nanfix_rerun",
]
RESULTS_ROOT = REPO_ROOT / "results" / "sycophancy_bias_probe"


_MODEL_RUNTIME_CACHE: Dict[str, Any] = {
    "model_name": None,
    "model": None,
    "tokenizer": None,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill prompt-level probe scores by evaluating the saved neutral chosen probe "
            "on all prompt templates for completed strict-MC runs."
        ),
    )
    parser.add_argument(
        "--run_dir",
        action="append",
        default=None,
        help=(
            "Absolute or repo-relative run directory. Repeat this flag to process multiple runs. "
            "If omitted, the script uses the four default 20260321-20260322 Llama/Qwen runs."
        ),
    )
    parser.add_argument(
        "--template_type",
        action="append",
        default=None,
        help=(
            "Prompt template type to score, such as neutral or incorrect_suggestion. "
            "Repeat to include multiple template types. Defaults to all templates present in each run."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to score. Repeat to include multiple splits. Defaults to train, val, and test.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help=(
            "Optional Hugging Face cache directory override. "
            "If omitted, the script uses HUGGINGFACE_HUB_CACHE / HF_HUB_CACHE when set, "
            "otherwise the run's saved hf_cache_dir."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override. Defaults to the run's resolved_device / device.",
    )
    parser.add_argument(
        "--output_subdir",
        default="probes/backfills/probe_no_bias_all_templates",
        help=(
            "Repo-relative subdirectory inside each run where the backfilled artifacts should be written. "
            "Defaults to probes/backfills/probe_no_bias_all_templates."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing backfilled artifacts instead of skipping completed runs.",
    )
    return parser


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def unload_cached_model() -> None:
    model = _MODEL_RUNTIME_CACHE.get("model")
    tokenizer = _MODEL_RUNTIME_CACHE.get("tokenizer")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    _MODEL_RUNTIME_CACHE["model_name"] = None
    _MODEL_RUNTIME_CACHE["model"] = None
    _MODEL_RUNTIME_CACHE["tokenizer"] = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_run_dirs(cli_run_dirs: Optional[Sequence[str]]) -> List[Path]:
    run_dirs = list(cli_run_dirs) if cli_run_dirs else list(DEFAULT_RUN_DIRS)
    resolved: List[Path] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        else:
            path = path.resolve()
        if path.exists() and (path / "run_config.json").exists():
            resolved.append(path)
            continue

        run_name = path.name
        candidates = sorted(
            candidate.parent
            for candidate in RESULTS_ROOT.glob(f"**/{run_name}/run_config.json")
        )
        if len(candidates) == 1:
            resolved.append(candidates[0].resolve())
            continue
        if len(candidates) > 1:
            raise ValueError(
                f"Run name {run_name!r} matched multiple directories under {RESULTS_ROOT}: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        raise FileNotFoundError(
            f"Could not resolve run_dir={run_dir!r}. "
            f"Tried exact path {path} and no unique run named {run_name!r} was found under {RESULTS_ROOT}."
        )
    return resolved


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))


def load_neutral_probe_bundle(run_dir: Path) -> tuple[Any, int, Dict[str, Any]]:
    probe_dir = run_dir / "probes" / "chosen_probe" / "probe_no_bias"
    metadata_path = probe_dir / "metadata.json"
    model_path = probe_dir / "model.pkl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as handle:
        clf = pickle.load(handle)
    return clf, int(metadata["layer"]), metadata


def resolve_cache_dir(cli_hf_cache_dir: Optional[str], run_cfg: Dict[str, Any]) -> Optional[str]:
    if cli_hf_cache_dir:
        return str(Path(cli_hf_cache_dir).expanduser())
    env_cache = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    if env_cache:
        return str(Path(env_cache).expanduser())
    run_cache = run_cfg.get("hf_cache_dir")
    if run_cache:
        return str(Path(str(run_cache)).expanduser())
    return None


def resolve_device(cli_device: Optional[str], run_cfg: Dict[str, Any]) -> str:
    if cli_device:
        return str(cli_device).strip()
    device = str(run_cfg.get("resolved_device") or run_cfg.get("device") or "cpu").strip()
    return "cpu" if device == "auto" else device


def get_model_bundle_for_run(
    run_cfg: Dict[str, Any],
    *,
    hf_cache_dir: Optional[str],
    device_override: Optional[str],
) -> tuple[Any, Any]:
    backend = str(run_cfg.get("model_backend", "huggingface") or "huggingface")
    if backend != "huggingface":
        raise ValueError(f"Only huggingface runs are supported, got backend={backend!r}.")

    model_name = str(run_cfg.get("model", "") or "").strip()
    if not model_name:
        raise ValueError("Run config is missing a model name.")

    if _MODEL_RUNTIME_CACHE["model_name"] != model_name:
        unload_cached_model()
        device = resolve_device(device_override, run_cfg)
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            device_map_auto=bool(run_cfg.get("device_map_auto", False)),
            hf_cache_dir=hf_cache_dir,
        )
        _MODEL_RUNTIME_CACHE["model_name"] = model_name
        _MODEL_RUNTIME_CACHE["model"] = model
        _MODEL_RUNTIME_CACHE["tokenizer"] = tokenizer

    return _MODEL_RUNTIME_CACHE["model"], _MODEL_RUNTIME_CACHE["tokenizer"]


def normalize_requested_values(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def filter_records(
    records: Sequence[Dict[str, Any]],
    *,
    template_types: Optional[set[str]],
    splits: Optional[set[str]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for record in records:
        template_type = str(record.get("template_type", "") or "").strip()
        split = str(record.get("split", "") or "").strip()
        if template_types is not None and template_type not in template_types:
            continue
        if splits is not None and split not in splits:
            continue
        filtered.append(dict(record))
    return filtered


def build_output_paths(run_dir: Path, output_subdir: str) -> Dict[str, Path]:
    backfill_dir = run_dir / output_subdir
    return {
        "backfill_dir": backfill_dir,
        "candidate_scores_csv": backfill_dir / "probe_candidate_scores.csv",
        "prompt_scores_csv": backfill_dir / "probe_scores_by_prompt.csv",
        "metadata_json": backfill_dir / "metadata.json",
    }


def process_run(
    run_dir: Path,
    *,
    requested_template_types: Optional[set[str]],
    requested_splits: Optional[set[str]],
    output_subdir: str,
    hf_cache_dir: Optional[str],
    device_override: Optional[str],
    force: bool,
) -> Dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    run_cfg = load_run_config(run_dir)
    output_paths = build_output_paths(run_dir, output_subdir)
    if (
        not force
        and output_paths["candidate_scores_csv"].exists()
        and output_paths["prompt_scores_csv"].exists()
        and output_paths["metadata_json"].exists()
    ):
        metadata = json.loads(output_paths["metadata_json"].read_text(encoding="utf-8"))
        return {
            "run_dir": str(run_dir),
            "status": "skipped_existing",
            "model_name": run_cfg.get("model"),
            "dataset_name": run_cfg.get("dataset_name"),
            "n_source_records": metadata.get("n_source_records"),
            "n_candidate_rows": metadata.get("n_candidate_rows"),
            "n_prompt_rows": metadata.get("n_prompt_rows"),
            "output_dir": str(output_paths["backfill_dir"]),
        }

    records = load_jsonl_records(run_dir / "logs" / "sampling_records.jsonl")
    filtered_records = filter_records(
        records,
        template_types=requested_template_types,
        splits=requested_splits,
    )
    if not filtered_records:
        raise ValueError(f"No records matched the requested template/split filter in {run_dir}.")

    candidate_records = build_choice_candidate_records(
        filtered_records,
        probe_name="probe_no_bias",
        example_weighting=str(run_cfg.get("probe_example_weighting", "model_probability") or "model_probability"),
    )
    if not candidate_records:
        raise ValueError(
            "No strict-MC candidate records were created. "
            f"Run may not be compatible with choice-candidate rescoring: {run_dir}"
        )

    model, tokenizer = get_model_bundle_for_run(
        run_cfg,
        hf_cache_dir=hf_cache_dir,
        device_override=device_override,
    )
    clf, layer, probe_metadata = load_neutral_probe_bundle(run_dir)
    score_records_with_probe(
        model=model,
        tokenizer=tokenizer,
        records=candidate_records,
        clf=clf,
        layer=layer,
        score_key="probe_score",
        desc=f"{run_dir.name} neutral_probe_all_templates",
    )

    candidate_scores_df = to_probe_candidate_scores_df(
        candidate_records,
        model_name=str(run_cfg.get("model", "") or ""),
    )
    prompt_scores_df = build_mc_probe_scores_by_prompt_df(candidate_scores_df)

    output_paths["backfill_dir"].mkdir(parents=True, exist_ok=True)
    candidate_scores_df.to_csv(output_paths["candidate_scores_csv"], index=False)
    prompt_scores_df.to_csv(output_paths["prompt_scores_csv"], index=False)

    metadata = {
        "artifact_schema_version": 1,
        "artifact_kind": "neutral_probe_cross_template_backfill",
        "created_at_utc": utc_now_iso(),
        "source_run_dir": str(run_dir),
        "model_name": str(run_cfg.get("model", "") or ""),
        "dataset_name": str(run_cfg.get("dataset_name", "") or ""),
        "probe_name": "probe_no_bias",
        "probe_training_template_type": "neutral",
        "probe_model_path": str(run_dir / "probes" / "chosen_probe" / "probe_no_bias" / "model.pkl"),
        "probe_metadata_path": str(run_dir / "probes" / "chosen_probe" / "probe_no_bias" / "metadata.json"),
        "probe_layer": int(layer),
        "probe_template_type": str(probe_metadata.get("template_type", "") or "neutral"),
        "probe_construction": str(probe_metadata.get("training", {}).get("probe_construction", "") or ""),
        "probe_example_weighting": str(probe_metadata.get("training", {}).get("example_weighting", "") or ""),
        "requested_template_types": sorted(requested_template_types) if requested_template_types is not None else None,
        "requested_splits": sorted(requested_splits) if requested_splits is not None else None,
        "scored_template_types": sorted({str(record.get("template_type", "") or "") for record in filtered_records}),
        "scored_splits": sorted({str(record.get("split", "") or "") for record in filtered_records}),
        "n_source_records": int(len(filtered_records)),
        "n_candidate_rows": int(len(candidate_scores_df)),
        "n_prompt_rows": int(len(prompt_scores_df)),
        "files": {
            "probe_candidate_scores": str(output_paths["candidate_scores_csv"]),
            "probe_scores_by_prompt": str(output_paths["prompt_scores_csv"]),
        },
    }
    output_paths["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "status": "rescored",
        "model_name": run_cfg.get("model"),
        "dataset_name": run_cfg.get("dataset_name"),
        "n_source_records": len(filtered_records),
        "n_candidate_rows": len(candidate_scores_df),
        "n_prompt_rows": len(prompt_scores_df),
        "output_dir": str(output_paths["backfill_dir"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    requested_template_types = normalize_requested_values(args.template_type)
    requested_splits = normalize_requested_values(args.split) or {"train", "val", "test"}
    run_dirs = resolve_run_dirs(args.run_dir)

    summary_rows: List[Dict[str, Any]] = []
    run_cfgs = {run_dir: load_run_config(run_dir) for run_dir in run_dirs}
    ordered_run_dirs = sorted(
        run_dirs,
        key=lambda run_dir: (
            str(run_cfgs[run_dir].get("model", "") or ""),
            str(run_cfgs[run_dir].get("dataset_name", "") or ""),
            str(run_dir),
        ),
    )

    hf_cache_dir = args.hf_cache_dir
    if hf_cache_dir:
        hf_cache_dir = str(Path(hf_cache_dir).expanduser())

    try:
        for run_dir in ordered_run_dirs:
            print(f"[neutral-probe-backfill] processing run_dir={run_dir}")
            row = process_run(
                run_dir,
                requested_template_types=requested_template_types,
                requested_splits=requested_splits,
                output_subdir=args.output_subdir,
                hf_cache_dir=hf_cache_dir or resolve_cache_dir(None, run_cfgs[run_dir]),
                device_override=args.device,
                force=bool(args.force),
            )
            summary_rows.append(row)
            print(
                "[neutral-probe-backfill] completed "
                f"status={row['status']} model={row['model_name']} dataset={row['dataset_name']} "
                f"source_records={row['n_source_records']} candidate_rows={row['n_candidate_rows']} "
                f"prompt_rows={row['n_prompt_rows']}"
            )
    finally:
        unload_cached_model()

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        print()
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

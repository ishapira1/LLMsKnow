from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..cli import resolve_hf_cache_dir
from ..llm.registry import load_llm
from ..logging_utils import clear_run_logging, configure_run_logging, log_status, warn_status
from ..runtime import build_default_run_name, model_slug, utc_now_iso, write_csv_atomic, write_json_atomic, write_jsonl_atomic, write_text_atomic
from .cli import parse_args
from .data import CalibrationExample, PruningDatasets, build_pruning_datasets
from .losses import choice_token_probabilities, loss_for_example
from .masks import (
    apply_mask,
    build_magnitude_mask,
    build_random_mask,
    count_masked_weights,
    restore_masked_values,
    select_pruning_mask,
)
from .metrics import choose_selected_sparsity, compute_item_metrics, summarize_item_metrics
from .scores import collect_prunable_linear_weights, score_weight_importance


def _import_torch():
    import torch

    return torch


def _run_dir(args: Any) -> Path:
    name = str(args.run_name or "").strip() or build_default_run_name()
    if "/" in name or name in {".", ".."}:
        raise ValueError(f"Invalid run_name={name!r}. Use a single directory-safe token.")
    path = Path(args.out_dir) / model_slug(args.model) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_payload(args: Any, status: str, error: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "status": status,
        "updated_at_utc": utc_now_iso(),
        "model": args.model,
        "datasets": list(args.datasets),
        "run_name": args.run_name,
    }
    if error:
        payload["error"] = str(error)
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _example_rows(examples: Sequence[CalibrationExample]) -> List[Dict[str, Any]]:
    rows = []
    for example in examples:
        row = example.to_loss_dict()
        row["messages"] = list(example.messages)
        rows.append(row)
    return rows


def _mean_loss(model: Any, tokenizer: Any, examples: Sequence[Mapping[str, Any]]) -> float:
    torch = _import_torch()
    if not examples:
        return float("nan")
    losses = []
    with torch.no_grad():
        for example in examples:
            losses.append(float(loss_for_example(model, tokenizer, example).item()))
    return float(sum(losses) / len(losses))


def _evaluate_pairs(
    model: Any,
    tokenizer: Any,
    datasets: PruningDatasets,
    *,
    sparsity: float,
    mask_name: str,
) -> pd.DataFrame:
    rows = []
    for pair in datasets.eval_pairs:
        neutral_probs = choice_token_probabilities(
            model,
            tokenizer,
            pair.neutral_messages,
            choices=pair.choices,
        )
        biased_probs = choice_token_probabilities(
            model,
            tokenizer,
            pair.biased_messages,
            choices=pair.choices,
        )
        rows.append(
            compute_item_metrics(
                pair,
                neutral_probabilities=neutral_probs,
                biased_probabilities=biased_probs,
                sparsity=sparsity,
                mask_name=mask_name,
            )
        )
    return pd.DataFrame(rows)


def _evaluate_with_mask(
    model: Any,
    tokenizer: Any,
    datasets: PruningDatasets,
    masks: Mapping[str, Any],
    *,
    sparsity: float,
    mask_name: str,
    baseline_preservation_loss: float,
) -> tuple[pd.DataFrame, float, float]:
    if count_masked_weights(masks) <= 0:
        item_df = _evaluate_pairs(model, tokenizer, datasets, sparsity=sparsity, mask_name=mask_name)
        pres_loss = _mean_loss(model, tokenizer, [example.to_loss_dict() for example in datasets.preservation])
    else:
        originals = apply_mask(model, masks)
        try:
            item_df = _evaluate_pairs(model, tokenizer, datasets, sparsity=sparsity, mask_name=mask_name)
            pres_loss = _mean_loss(model, tokenizer, [example.to_loss_dict() for example in datasets.preservation])
        finally:
            restore_masked_values(model, masks, originals)
    increase = 0.0
    if baseline_preservation_loss and baseline_preservation_loss == baseline_preservation_loss:
        increase = (pres_loss - baseline_preservation_loss) / abs(baseline_preservation_loss)
    return item_df, float(pres_loss), float(increase)


def _save_mask(path: Path, masks: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    torch = _import_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "masks": {name: mask.cpu() for name, mask in masks.items()},
        },
        path,
    )


def _plot_delta_bucket(item_df: pd.DataFrame, selected_sparsity: float, output_dir: Path) -> None:
    if item_df.empty:
        return
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_style("white")
    subset = item_df[
        item_df["mask_name"].astype(str).eq("sycophancy")
        & item_df["split"].astype(str).eq("test")
        & item_df["condition"].astype(str).eq("incorrect_suggestion")
        & item_df["sparsity"].astype(float).isin([0.0, float(selected_sparsity)])
    ].copy()
    if subset.empty:
        return
    baseline = subset[subset["sparsity"].astype(float).eq(0.0)][["pair_id", "p_neutral_c"]].copy()
    if baseline["p_neutral_c"].nunique() < 2:
        return
    labels = ["Q1", "Q2", "Q3", "Q4"]
    baseline["neutral_confidence_bucket"] = pd.qcut(
        baseline["p_neutral_c"],
        q=min(4, baseline["p_neutral_c"].nunique()),
        labels=labels[: min(4, baseline["p_neutral_c"].nunique())],
        duplicates="drop",
    )
    plot_df = subset.merge(baseline[["pair_id", "neutral_confidence_bucket"]], on="pair_id", how="inner")
    plot_df["state"] = np.where(plot_df["sparsity"].astype(float).eq(0.0), "Before pruning", "After pruning")
    summary = (
        plot_df.groupby(["neutral_confidence_bucket", "state"], as_index=False, observed=False)
        .agg(mean_delta_p_b=("delta_p_b", "mean"), n_pairs=("pair_id", "nunique"))
    )
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.lineplot(
        data=summary,
        x="neutral_confidence_bucket",
        y="mean_delta_p_b",
        hue="state",
        marker="o",
        palette={"Before pruning": "#d4651a", "After pruning": "#73b3ab"},
        ax=ax,
    )
    ax.set_title("Sycophantic Movement by Neutral Confidence", fontsize=22)
    ax.set_xlabel("Neutral confidence bucket", fontsize=15)
    ax.set_ylabel("Mean delta P(b)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    if legend is not None:
        legend.set_title("")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "delta_p_by_neutral_confidence_bucket.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "delta_p_by_neutral_confidence_bucket.pdf", bbox_inches="tight")
    plt.close(fig)


def _control_status_row(control_name: str, status: str, reason: str = "") -> Dict[str, Any]:
    return {
        "control_name": control_name,
        "status": status,
        "reason": reason,
        "n_pairs": 0,
        "mean_delta_p_b": None,
        "mean_gap_closure": None,
        "flip_rate_to_b": None,
        "neutral_accuracy": None,
        "biased_accuracy": None,
    }


def _summarize_control(control_name: str, item_df: pd.DataFrame) -> Dict[str, Any]:
    if item_df.empty:
        return _control_status_row(control_name, "empty")
    subset = item_df[
        item_df["split"].astype(str).eq("test")
        & item_df["condition"].astype(str).eq("incorrect_suggestion")
    ].copy()
    if subset.empty:
        subset = item_df.copy()
    return {
        "control_name": control_name,
        "status": "completed",
        "reason": "",
        "n_pairs": int(subset["pair_id"].nunique()),
        "mean_delta_p_b": float(subset["delta_p_b"].mean()),
        "mean_gap_closure": float(subset["gap_closure"].mean()),
        "flip_rate_to_b": float(subset["flip_rate_to_b"].mean()),
        "neutral_accuracy": float(subset["neutral_accuracy"].mean()),
        "biased_accuracy": float(subset["biased_accuracy"].mean()),
    }


def _write_summary(path: Path, selected_sparsity: float, sweep_summary: pd.DataFrame, control_df: pd.DataFrame) -> None:
    lines = [
        "# Sycophancy Pruning Summary",
        "",
        f"- Selected sparsity: `{selected_sparsity:g}`",
        f"- Sweep rows: `{len(sweep_summary)}`",
        f"- Controls: `{len(control_df)}`",
    ]
    if not control_df.empty:
        lines.append("")
        lines.append("## Controls")
        for _, row in control_df.iterrows():
            lines.append(
                f"- `{row['control_name']}`: {row['status']}"
                + (f" ({row['reason']})" if str(row.get("reason", "") or "") else "")
            )
    write_text_atomic(path, "\n".join(lines) + "\n")


def run(args: Any) -> Path:
    run_dir = _run_dir(args)
    log_path = run_dir / "logs" / "run.log"
    configure_run_logging(log_path, run_dir / "logs" / "warnings.log")
    write_json_atomic(run_dir / "status.json", _status_payload(args, "running"))
    try:
        hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir)
        if hf_cache_dir:
            os.environ["HF_HUB_CACHE"] = hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        log_status("pruning/runner.py", f"loading model={args.model} device={args.resolved_device}")
        llm = load_llm(
            args.model,
            device=args.resolved_device,
            device_map_auto=bool(args.device_map_auto),
            hf_cache_dir=hf_cache_dir,
            torch_dtype=args.torch_dtype,
        )
        model, tokenizer = llm.get_model_and_tokenizer()

        def choice_scorer(messages: List[Dict[str, Any]], choices: Sequence[str]) -> Dict[str, float]:
            return choice_token_probabilities(model, tokenizer, messages, choices=choices)

        datasets = build_pruning_datasets(args, choice_scorer=choice_scorer)
        calibration_dir = run_dir / "calibration"
        write_jsonl_atomic(calibration_dir / "sycophancy.jsonl", _example_rows(datasets.sycophancy))
        write_jsonl_atomic(calibration_dir / "preservation.jsonl", _example_rows(datasets.preservation))
        write_jsonl_atomic(calibration_dir / "truthful_correction.jsonl", _example_rows(datasets.truthful_correction))
        write_jsonl_atomic(calibration_dir / "neutral_wrong.jsonl", _example_rows(datasets.neutral_wrong))
        write_jsonl_atomic(calibration_dir / "eval_pairs.jsonl", [pair.as_dict() for pair in datasets.eval_pairs])
        write_json_atomic(
            run_dir / "run_config.json",
            {
                **_json_ready(vars(args)),
                "run_dir": str(run_dir),
                "hf_cache_dir": hf_cache_dir,
                "n_sycophancy_examples": len(datasets.sycophancy),
                "n_preservation_examples": len(datasets.preservation),
                "n_eval_pairs": len(datasets.eval_pairs),
            },
        )

        syc_scores = score_weight_importance(
            model,
            tokenizer,
            [example.to_loss_dict() for example in datasets.sycophancy],
            desc="sycophancy SNIP scores",
        )
        pres_scores = score_weight_importance(
            model,
            tokenizer,
            [example.to_loss_dict() for example in datasets.preservation],
            desc="preservation SNIP scores",
        )
        prunable = collect_prunable_linear_weights(model)
        baseline_pres_loss = _mean_loss(model, tokenizer, [example.to_loss_dict() for example in datasets.preservation])
        all_items = []
        sweep_metadata = []

        for sparsity in args.sparsities:
            selection = select_pruning_mask(
                syc_scores,
                pres_scores,
                sparsity=float(sparsity),
                preserve_exclude_fraction=float(args.preserve_exclude_fraction),
            )
            if float(sparsity) > 0.0 and (args.save_all_sweep_masks or float(sparsity) == max(args.sparsities)):
                _save_mask(
                    run_dir / "masks" / f"sycophancy_sparsity_{sparsity:g}.pt",
                    selection.masks,
                    {
                        "mask_name": "sycophancy",
                        "sparsity": float(sparsity),
                        "selected_count": selection.selected_count,
                    },
                )
            item_df, pres_loss, pres_increase = _evaluate_with_mask(
                model,
                tokenizer,
                datasets,
                selection.masks,
                sparsity=float(sparsity),
                mask_name="sycophancy",
                baseline_preservation_loss=baseline_pres_loss,
            )
            item_df["preservation_loss"] = pres_loss
            item_df["preservation_loss_increase"] = pres_increase
            all_items.append(item_df)
            sweep_metadata.append(
                {
                    "mask_name": "sycophancy",
                    "sparsity": float(sparsity),
                    "requested_count": selection.requested_count,
                    "selected_count": selection.selected_count,
                    "preservation_loss": pres_loss,
                    "preservation_loss_increase": pres_increase,
                }
            )

        item_df = pd.concat(all_items, ignore_index=True) if all_items else pd.DataFrame()
        sweep_summary = summarize_item_metrics(item_df)
        sweep_meta_df = pd.DataFrame(sweep_metadata)
        if not sweep_summary.empty and not sweep_meta_df.empty:
            sweep_summary = sweep_summary.merge(
                sweep_meta_df,
                on=["mask_name", "sparsity"],
                how="left",
            )
        selected_sparsity = choose_selected_sparsity(
            sweep_summary,
            syc_reduction_target=args.syc_reduction_target,
            preservation_loss_budget=args.preservation_loss_budget,
            neutral_accuracy_drop_budget=args.neutral_accuracy_drop_budget,
        )
        selected_selection = select_pruning_mask(
            syc_scores,
            pres_scores,
            sparsity=float(selected_sparsity),
            preserve_exclude_fraction=float(args.preserve_exclude_fraction),
        )
        _save_mask(
            run_dir / "masks" / "selected_sycophancy.pt",
            selected_selection.masks,
            {
                "mask_name": "sycophancy",
                "sparsity": float(selected_sparsity),
                "selected_count": selected_selection.selected_count,
            },
        )

        controls = []
        control_items = []
        matched_count = selected_selection.selected_count
        del selected_selection
        control_builders = [
            ("random", lambda: build_random_mask(prunable, count=matched_count, seed=args.seed)),
            ("magnitude", lambda: build_magnitude_mask(prunable, count=matched_count)),
        ]
        for control_name, build_masks in control_builders:
            masks = build_masks()
            _save_mask(run_dir / "masks" / f"{control_name}.pt", masks, {"mask_name": control_name, "matched_count": matched_count})
            c_item_df, _pres_loss, _pres_increase = _evaluate_with_mask(
                model,
                tokenizer,
                datasets,
                masks,
                sparsity=float(selected_sparsity),
                mask_name=control_name,
                baseline_preservation_loss=baseline_pres_loss,
            )
            control_items.append(c_item_df)
            controls.append(_summarize_control(control_name, c_item_df))
            del masks

        control_score_specs = [
            ("neutral_wrong", datasets.neutral_wrong, args.wrong_control_min_examples),
            ("truthful_correction", datasets.truthful_correction, 1),
        ]
        for control_name, examples, minimum in control_score_specs:
            if len(examples) < int(minimum):
                controls.append(
                    _control_status_row(
                        control_name,
                        "insufficient_examples",
                        f"found {len(examples)} examples; minimum is {minimum}",
                    )
                )
                continue
            control_scores = score_weight_importance(
                model,
                tokenizer,
                [example.to_loss_dict() for example in examples],
                desc=f"{control_name} SNIP scores",
            )
            selection = select_pruning_mask(
                control_scores,
                pres_scores,
                sparsity=float(selected_sparsity),
                preserve_exclude_fraction=float(args.preserve_exclude_fraction),
            )
            _save_mask(
                run_dir / "masks" / f"{control_name}.pt",
                selection.masks,
                {"mask_name": control_name, "selected_count": selection.selected_count},
            )
            c_item_df, _pres_loss, _pres_increase = _evaluate_with_mask(
                model,
                tokenizer,
                datasets,
                selection.masks,
                sparsity=float(selected_sparsity),
                mask_name=control_name,
                baseline_preservation_loss=baseline_pres_loss,
            )
            control_items.append(c_item_df)
            controls.append(_summarize_control(control_name, c_item_df))
            del selection

        control_df = pd.DataFrame(controls)
        if control_items:
            item_df = pd.concat([item_df, *control_items], ignore_index=True)

        metrics_dir = run_dir / "metrics"
        write_csv_atomic(metrics_dir / "item_metrics.csv", item_df)
        write_csv_atomic(metrics_dir / "sweep_metrics.csv", sweep_summary)
        write_csv_atomic(metrics_dir / "control_metrics.csv", control_df)
        _plot_delta_bucket(item_df, float(selected_sparsity), run_dir / "plots")
        _write_summary(run_dir / "reports" / "summary.md", float(selected_sparsity), sweep_summary, control_df)
        write_json_atomic(
            run_dir / "status.json",
            {
                **_status_payload(args, "completed"),
                "selected_sparsity": float(selected_sparsity),
                "completed_at_utc": utc_now_iso(),
            },
        )
        log_status("pruning/runner.py", f"completed pruning run: {run_dir}")
        return run_dir
    except Exception as exc:
        warn_status("pruning/runner.py", "run_failed", str(exc))
        write_json_atomic(run_dir / "status.json", _status_payload(args, "failed", error=str(exc)))
        raise
    finally:
        clear_run_logging()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


__all__ = ["main", "run"]

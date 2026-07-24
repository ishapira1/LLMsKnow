from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from llmssycoph.interventions.experiment import _read_result_tree, _steering_condition_did


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def prepare_selection(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    directions_path = args.directions_path.expanduser().resolve()
    directions_manifest = directions_path.with_name("manifest.json")
    source_run_dir = args.source_run_dir.expanduser().resolve()
    if not directions_path.is_file() or not directions_manifest.is_file():
        raise FileNotFoundError(f"Missing direction artifact beside {directions_path}")
    metadata = json.loads(directions_manifest.read_text(encoding="utf-8"))
    if Path(metadata["source_run_dir"]).resolve() != source_run_dir:
        raise ValueError("Direction artifact and requested source run do not match.")
    if metadata.get("model_name") != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(f"Unexpected direction model: {metadata.get('model_name')!r}")
    if metadata.get("dataset_name") != "commonsense_qa":
        raise ValueError(f"Unexpected direction dataset: {metadata.get('dataset_name')!r}")
    if metadata.get("max_questions") is not None:
        raise ValueError("Pilot must reuse the full train-fitted direction artifact.")
    layers = _csv_ints(args.layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError(f"Layers must be nonempty and unique: {layers}")
    if min(layers) < 1 or max(layers) > 27:
        raise ValueError(f"Qwen nonterminal pilot layers must be within 1..27: {layers}")
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "prespecified_pilot_layer_grid",
        "selection_split": "val",
        "selection_subset": "prespecified evenly spaced nonterminal layers",
        "candidate_layers": layers,
        "top_k": len(layers),
        "source_run_dir": str(source_run_dir),
        "directions_manifest_sha256": _sha256(directions_manifest),
        "directions_npz_sha256": _sha256(directions_path),
        "model_name": metadata["model_name"],
        "dataset_name": metadata["dataset_name"],
        "pilot": True,
        "max_questions": int(args.max_questions),
        "frozen_before_dose_tuning": True,
        "test_confirmation_allowed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "selected_patch_layers.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite pilot layer plan: {target}")
    target.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


def summarize(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    frame = _read_result_tree(
        output_root,
        split="val",
        protocol="dose-tune",
        interventions=(
            "steer_restoration_meandiff",
            "steer_rademacher_null",
            "steer_random_direction",
        ),
    )
    treatment = _steering_condition_did(frame, "steer_restoration_meandiff")
    null = _steering_condition_did(frame, "steer_rademacher_null")
    random = _steering_condition_did(frame, "steer_random_direction")
    treatment = treatment[~treatment["hidden_truth_flip"].astype(bool)].copy()
    null = null[~null["hidden_truth_flip"].astype(bool)].copy()
    random = random[~random["hidden_truth_flip"].astype(bool)].copy()

    keys = ["layer", "alpha"]
    treatment_summary = treatment.groupby(keys, as_index=False).agg(
        n_items=("question_id", "nunique"),
        meandiff_mitigation_did=("mitigation_did", "mean"),
        biased_delta_margin=("biased_delta_margin", "mean"),
        neutral_delta_margin=("neutral_delta_margin", "mean"),
        neutral_accuracy_change=("neutral_accuracy_change", "mean"),
        biased_accuracy_change=("biased_accuracy_change", "mean"),
    )
    null_summary = (
        null.groupby(keys, as_index=False)["mitigation_did"]
        .mean()
        .rename(columns={"mitigation_did": "label_sign_null_mitigation_did"})
    )
    random_summary = (
        random.groupby(keys, as_index=False)["mitigation_did"]
        .mean()
        .rename(columns={"mitigation_did": "random_direction_mitigation_did"})
    )
    summary = treatment_summary.merge(null_summary, on=keys).merge(random_summary, on=keys)
    summary["meandiff_minus_label_sign_null"] = (
        summary["meandiff_mitigation_did"] - summary["label_sign_null_mitigation_did"]
    )
    summary["meandiff_minus_random_direction"] = (
        summary["meandiff_mitigation_did"] - summary["random_direction_mitigation_did"]
    )
    summary = summary.sort_values(keys).reset_index(drop=True)

    aggregate_dir = output_root / "aggregate"
    plot_dir = aggregate_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "pilot_dose_response.csv"
    summary.to_csv(summary_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    figure, axis = plt.subplots(figsize=(11.5, 7))
    layers = sorted(summary["layer"].astype(int).unique())
    palette = dict(zip(layers, sns.color_palette("viridis", n_colors=len(layers))))
    sns.lineplot(
        data=summary,
        x="alpha",
        y="meandiff_mitigation_did",
        hue="layer",
        marker="o",
        linewidth=2.3,
        palette=palette,
        ax=axis,
    )
    axis.axhline(0.0, color="#222222", linewidth=1.0, alpha=0.75)
    axis.set_title("Qwen MeanDiff pilot: mitigation dose response by layer", fontsize=20, pad=14)
    axis.set_xlabel("Dose α (training-pair projection standard deviations)", fontsize=16)
    axis.set_ylabel(
        "Mean mitigation DiD\nΔmargin(strong wrong suggestion) − Δmargin(neutral)",
        fontsize=15,
    )
    axis.tick_params(axis="both", labelsize=12)
    axis.legend(
        title="Residual layer",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
        frameon=True,
        fontsize=12,
        title_fontsize=12,
    )
    sns.despine(axis=axis)
    figure.tight_layout()
    plot_path = plot_dir / "pilot_meandiff_dose_response_by_layer.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Preliminary validation-only pilot. It is not held-out confirmation and must not "
            "be combined with the full DAG as if independently selected."
        ),
        "n_layers": len(layers),
        "layers": layers,
        "dose_response_csv": str(summary_path),
        "dose_response_plot": str(plot_path),
    }
    manifest_path = aggregate_dir / "pilot_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and summarize the independent Qwen MeanDiff pilot.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--source-run-dir", type=Path, required=True)
    prepare.add_argument("--directions-path", type=Path, required=True)
    prepare.add_argument("--layers", required=True)
    prepare.add_argument("--max-questions", type=int, required=True)
    prepare.set_defaults(function=prepare_selection)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    summary.set_defaults(function=summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


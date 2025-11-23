#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLMsKnow pipeline end-to-end for a given model and dataset."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id, e.g. mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--train_dataset",
        default="triviaqa",
        help="Train split name, e.g. triviaqa",
    )
    parser.add_argument(
        "--test_dataset",
        default="triviaqa_test",
        help="Test split name, e.g. triviaqa_test",
    )
    parser.add_argument(
        "--probe_layer",
        type=int,
        default=15,
        help="Layer index to probe.",
    )
    parser.add_argument(
        "--probe_token",
        default="exact_answer_last_token",
        help="Token position spec.",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=30,
        help="Number of resamples per question.",
    )
    parser.add_argument(
        "--skip_probe_scan",
        action="store_true",
        help="Skip probe_all_layers_and_tokens sweep.",
    )

    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent
    output_dir = repo_root / "output"

    # 0. Ensure output dir exists
    output_dir.mkdir(exist_ok=True)
    print(f"Ensured output dir exists at {output_dir}")

    # 1. Generate answers for train and test
    run(
        [
            sys.executable,
            "generate_model_answers.py",
            "--model",
            args.model,
            "--dataset",
            args.train_dataset,
        ]
    )
    run(
        [
            sys.executable,
            "generate_model_answers.py",
            "--model",
            args.model,
            "--dataset",
            args.test_dataset,
        ]
    )

    # 2. Extract exact answers (train + test)
    run(
        [
            sys.executable,
            "extract_exact_answer.py",
            "--model",
            args.model,
            "--dataset",
            args.train_dataset,
        ]
    )
    run(
        [
            sys.executable,
            "extract_exact_answer.py",
            "--model",
            args.model,
            "--dataset",
            args.test_dataset,
        ]
    )

    # 3. Optional: layer/token heatmap on train
    if not args.skip_probe_scan:
        run(
            [
                sys.executable,
                "probe_all_layers_and_tokens.py",
                "--model",
                args.model,
                "--probe_at",
                "mlp_last_layer_only_input",
                "--seed",
                "0",
                "--n_samples",
                "1000",
                "--dataset",
                args.train_dataset,
            ]
        )

    # 4. Train a single correctness probe on train
    run(
        [
            sys.executable,
            "probe.py",
            "--model",
            args.model,
            "--extraction_model",
            args.model,
            "--probe_at",
            "mlp",
            "--seeds",
            "0",
            "5",
            "26",
            "42",
            "63",
            "--n_samples",
            "all",
            "--save_clf",
            "--dataset",
            args.train_dataset,
            "--layer",
            str(args.probe_layer),
            "--token",
            args.probe_token,
        ]
    )

    # 5. Resampling on the test set
    run(
        [
            sys.executable,
            "resampling.py",
            "--model",
            args.model,
            "--seed",
            "0",
            "--n_resamples",
            str(args.n_resamples),
            "--dataset",
            args.test_dataset,
        ]
    )

    # 6. Extract exact answers for resamples
    run(
        [
            sys.executable,
            "extract_exact_answer.py",
            "--model",
            args.model,
            "--dataset",
            args.test_dataset,
            "--do_resampling",
            str(args.n_resamples),
        ]
    )

    # 7. Probe-based answer choice on test
    run(
        [
            sys.executable,
            "probe_choose_answer.py",
            "--model",
            args.model,
            "--probe_at",
            "mlp",
            "--layer",
            str(args.probe_layer),
            "--token",
            args.probe_token,
            "--dataset",
            args.test_dataset,
            "--n_resamples",
            str(args.n_resamples),
            "--seeds",
            "0",
            "5",
            "26",
            "42",
            "63",
        ]
    )

    print("\nDone. Check ../output and wandb for results.")


if __name__ == "__main__":
    main()

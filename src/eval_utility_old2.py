import argparse
import sys
import traceback

import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
# from eval_utils import evaluate_triviaqa

from lm_eval import evaluator, utils

from eval_utils import eval_triviaqa
from prune import load_model_and_tokenizer

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


# Per-category eval configs.
#
# Covers the *objective* subset of the Qwen2.5-14B-Instruct evaluation suite that
# lm-evaluation-harness can run directly. Not covered here (need separate harnesses):
#   - MT-Bench / Arena-Hard / AlignBench  -> LLM-as-judge (FastChat, arena-hard-auto)
#   - MultiPL-E                           -> bigcode-evaluation-harness
#   - LiveCodeBench                       -> LiveCodeBench repo
#
# NOTE: task strings drift across harness versions. Verify against your install:
#   python -m lm_eval --tasks list | grep -E "mmlu_pro|gpqa|ifeval|math"
#
# NOTE: generative tasks (math, ifeval, code, and CoT knowledge tasks) are run with
# apply_chat_template=True to match instruct-model evaluation. The plain multiple-choice
# loglikelihood tasks (zero_shot) keep it False for base-model-style scoring; flip it to
# True if you want to match the official instruct numbers.
EVAL_CONFIGS = {
    "zero_shot": {
        "tasks": [
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "openbookqa",
        ],
        "apply_chat_template": False,
        "confirm_run_unsafe_code": False,
    },
    "knowledge": {
        # MMLU-redux is not a standard harness task; `mmlu` is the closest stand-in.
        "tasks": [
            "mmlu",
            "mmlu_pro",
            "gpqa_main_zeroshot",
            "truthfulqa_mc2",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
    },
    "math": {
        # minerva_math == the harness's implementation of the Hendrycks MATH dataset.
        "tasks": [
            "gsm8k",
            "minerva_math",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
    },
    "instruction_following": {
        "tasks": [
            "ifeval",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": False,
    },
    "code": {
        "tasks": [
            "humaneval",
            "mbpp",
        ],
        "apply_chat_template": True,
        "confirm_run_unsafe_code": True,
    },
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="harm_compression_eval_utility")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional path to model checkpoint to load, should have the same architecture as --model")
    parser.add_argument("--triviaqa_nsamples", type=int, default=1000, help="Number of samples to evaluate on triviaqa (only)")
    parser.add_argument("--evaluation", type=str,
                        choices=["zero_shot", "triviaqa", "code", "knowledge", "math", "instruction_following"])
    return parser


def main(args):
    trigger_sync = TriggerWandbSyncHook()  # this is a hook that triggers wandb to sync the results

    model_name = args.model
    print(f"running evaluation on model: {model_name}")
    wandb.init(project=args.run_name, config=vars(args))

    ft_dataset_name = "none"
    prune_dataset_name = "none"
    lr = None
    scheduler = None
    lora_rank = None
    lora_alpha = None
    batch_size = None
    grad_accum_steps = None
    warmup_steps = None
    seed = None

    wandb.log({"model_name": model_name})
    wandb.log({"prune_dataset_name": prune_dataset_name})
    wandb.log({"ft_dataset_name": ft_dataset_name})
    wandb.log({"lr": lr})
    wandb.log({"scheduler": scheduler})
    wandb.log({"lora_rank": lora_rank})
    wandb.log({"lora_alpha": lora_alpha})
    wandb.log({"batch_size": batch_size})
    wandb.log({"grad_accum_steps": grad_accum_steps})
    wandb.log({"seed": seed})

    trigger_sync()

    if args.evaluation == "triviaqa":
        model, tokenizer = load_model_and_tokenizer(args)
        results = eval_triviaqa(args, model, tokenizer)
        wandb.summary["triviaqa_acc"] = results['correctness'].mean().item()
        exit()

    if args.evaluation not in EVAL_CONFIGS:
        raise ValueError(f"Unknown evaluation: {args.evaluation}")

    cfg = EVAL_CONFIGS[args.evaluation]
    task_list = cfg["tasks"]
    apply_chat_template = cfg["apply_chat_template"]
    confirm_run_unsafe_code = cfg["confirm_run_unsafe_code"]

    if args.model_path is None:
        args.model_path = args.model

    if args.tokenizer is None:
        args.tokenizer = args.model

    results = evaluator.simple_evaluate(
        model='vllm',
        model_args={'pretrained': args.model_path, 'tokenizer': args.tokenizer,
                    'tensor_parallel_size': args.tensor_parallel_size, 'enforce_eager': True},
        # for qwen2.5, we need to add enforce_eager=True
        tasks=task_list,
        batch_size=100000,
        apply_chat_template=apply_chat_template,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        confirm_run_unsafe_code=confirm_run_unsafe_code
    )

    trigger_sync()

    for task in results['results']:
        for metric in results['results'][task]:
            if ',' in metric:
                metric_name = metric.split(',')[0]
            else:
                continue

            wandb.log({
                f"{task}/{metric_name}": results['results'][task][metric],
            })
            trigger_sync()

    wandb.finish()

if __name__ == '__main__':
    args = build_parser().parse_args()

    try:
        main(args)
    except Exception:
        # Report the failure to wandb, then hard-exit. os._exit bypasses the
        # multiprocessing/NCCL teardown that otherwise deadlocks on a dead
        # vLLM tensor-parallel worker and leaves the slurm job hanging.
        traceback.print_exc()
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        try:
            wandb.finish()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # vLLM TP also sometimes hangs on a *clean* exit; hard-exit avoids that too.
        os._exit(0)
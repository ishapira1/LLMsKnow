import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from eval_utils import ALPACA_JUDGE_MODEL, generate_responses_for_harmful_requests, \
    strongreject_classifier, \
    eval_zero_shot, eval_triviaqa, finetuning_attack, generate_responses_financial_advice, \
    generate_responses_for_alpaca, \
    judge_for_alpaca, judge_for_safety, coherency_judge, cohere_explanation_judge, eval_ppl_wikitext, \
    load_model_and_tokenizer, eval_output_harm, eval_explanations
from prune_utils import prune_attribution_score, prune_attribution_score_set_difference, check_sparsity, \
    prune_from_indices, find_intersection_indices, log_pruning_distribution, \
    prune_attribution_score_set_difference_global, prune_attribution_score_set_difference_with_refusal
from paper_pruning import (
    copy_mask_artifacts,
    evaluate_preservation_manifest,
    run_manifest_global_pruning,
    sha256_file,
    summarize_alpaca_scores,
    summarize_zero_shot_results,
    update_evaluation_metadata,
    write_evaluation_artifact,
    write_evaluation_items,
    write_evaluation_metadata,
)


def float_or_str(value):
    try:
        return float(value)
    except ValueError:
        return value  # Return as string if conversion fails


def parse_args():
    parser = argparse.ArgumentParser()
    # General run configuration
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to prune, e.g. 'Qwen/Qwen3-32B' or 'allenai/Olmo-3-7B-Instruct-SFT'")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional path to model checkpoint to load, should have the same architecture as --model")
    parser.add_argument("--revision", type=str, default=None,
                        help="Pinned Hugging Face revision for both model and (by default) tokenizer.")
    parser.add_argument("--after_attack_model_path", type=str, default=None,
                        help="Optional path to model checkpoint to load for evaluation after a jailbreak. If specified, no jailbreak will run. Should have the same architecture as --model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path, defaults to same as model")
    parser.add_argument("--tokenizer_revision", type=str, default=None,
                        help="Optional tokenizer revision. Defaults to --revision.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--wandb_project", type=str, default="pruning_safety")
    parser.add_argument("--log_sparsity_distribution", action="store_true", help="Whether to log sparsity distribution plots to wandb.")



    # Pruning configuration
    parser.add_argument("--nsamples", type=int, default=412, help="Number of samples of pruning data.")
    parser.add_argument("--nsamples_preserve", type=int, default=None, help="Number of samples of preservation data. Defaults to --nsamples.")
    parser.add_argument("--prune_method", type=str,
                        choices=[
                            "none",
                            "attribution_score",
                            "attribution_score_set_difference",
                            "attribution_score_set_difference_global",
                            "given_indices",
                            "intersect_indices",
                            "random"
                        ], default="attribution_score_set_difference", help="Pruning method to use"
                        )
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to prune, 0-indexed. If not specified, prunes all layers.")
    parser.add_argument("--prune_data", type=str, default="alpaca_cleaned_no_safety", help="Dataset to use for pruning.")
    parser.add_argument("--prune_data_list", type=str, nargs="+", default=None)
    parser.add_argument("--preserve_data", type=str, default="alpaca_cleaned_no_safety_train_raw", help="Dataset to use for preservation (only used for set difference pruning).")
    parser.add_argument("--prune_manifest", type=str, default=None,
                        help="Paper-faithful JSONL pruning manifest. Requires --preserve_manifest.")
    parser.add_argument("--preserve_manifest", type=str, default=None,
                        help="Paper-faithful JSONL preservation manifest. Requires --prune_manifest.")
    parser.add_argument("--score_format", choices=["raw", "chat"], default="raw",
                        help="Render manifest prompts as raw pretraining text or with the pinned chat template.")
    parser.add_argument("--loss_mode", choices=["completion_nll", "choice_token"],
                        default="completion_nll",
                        help="Full response-token NLL, or candidate-renormalized MC choice loss.")
    parser.add_argument("--attribution_variant", choices=["paper", "released_abs"], default="paper",
                        help="Paper dataset-mean score or labeled released-code |w|*sum(|example grad|) sensitivity.")
    parser.add_argument("--max_score_length", type=int, default=4096,
                        help="Fail if a manifest prompt+target exceeds this length; scoring never truncates.")
    parser.add_argument("--artifact_root", type=str, default="artifacts/paper_pruning",
                        help="Root for identity-keyed scores, sparse masks, metadata, and saved models.")
    parser.add_argument("--score_cache", type=str, default=None,
                        help="Optional explicit score-cache directory; its identity must exactly match this run.")
    parser.add_argument("--score_role", choices=["both", "prune", "preserve"], default="both",
                        help="With --dump_score, compute both score sets or one independently sharded role.")
    parser.add_argument("--control", choices=["none", "structure_matched", "alpaca_only", "random_magnitude"],
                        default="none",
                        help="Label/control mode. Structure-matched and Alpaca-only use matching supplied manifests; random_magnitude replaces the selected mask.")
    parser.add_argument("--neg_prune", action="store_true", help="Whether to do negative pruning, i.e. prune the lowest scored (most negative, if not absolute value) weights instead of the highest scored weights.")
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Use combined with attribution_score_set_difference, the top p scored elements in the first set (alpaca_no_safety)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.5,
        help="Use combined with attribution_score_set_difference, the top q scored elements in the second set (align anti))",
    )
    parser.add_argument("--preserve_pretrained_format", action="store_true", help='Whether to use chat template for the preservation data SNIP score computation.')
    parser.add_argument("--prune_pretrained_format", action="store_true", help='Whether to use chat template for the prune data SNIP score computation.')
    parser.add_argument("--use_saved_scores", action="store_true", help='Use the SNIP scores saved from a previous run with --dump_score')
    parser.add_argument("--freeze_first_top_q", action="store_true", help='Avoid pruning the first top q weights, and use the 2nd top q weights instead (still taking the difference from the top p preservation data).')
    parser.add_argument("--indices_path", type=str, default=None, help="To be used with given_indices prune_method, path to json file containing list of indices to prune for each layer.")
    parser.add_argument("--match_bins", type=int, default=20,
                        help="Number of magnitude bins for the random matched control.")

    # Saving configuration
    parser.add_argument("--save_model", action='store_true', help="Save pruned model.")
    parser.add_argument("--save_model_after_attack", action='store_true', help="Save model after jailbreak.")
    parser.add_argument("--no_abs", action="store_true", help='Do not use absolute values for computing the SNIP score.')
    parser.add_argument("--abs_preserve", action="store_true", help='Combined only with --no_abs. Use absolute values for the preservation data.')
    parser.add_argument("--abs_prune", action="store_true", help='Combined only with --no_abs. Use absolute values for the prune data.')
    parser.add_argument("--alpha", type=float_or_str, default=0, help='Whether to completely prune the weight (alpha=0), or multiply it with a number alpha.')
    parser.add_argument("--hp_search", action="store_true", help="Will create a different wandb run if true (signals that this run is part of a hyperparameter search, so that results can be easily compared in wandb UI).")

    # Jailbreak configuration
    ## For pruning attack (ablating refusal)
    parser.add_argument("--attack_q", default=0.01, type=float, help='q for refusal_ablation jailbreak')
    parser.add_argument("--attack_p", default=0.01, type=float, help='p for refusal_ablation jailbreak')


    # Saving intermediate pruning results configuration
    parser.add_argument(
        "--dump_score", action="store_true",
        help="Whether to dump the weight scores. If true, not actual pruning will be performed in this run."
    )
    parser.add_argument("--do_not_dump_prune_scores", action="store_true", help="Whether to not dump the prune scores when dumping scores. By default, both preserve and prune scores will be dumped when --dump_score is used. This flag can be used to only dump the preserve scores.")
    parser.add_argument("--do_not_dump_preserve_score", action="store_true", help="Whether to not dump the preserve scores when dumping scores. By default, both preserve and prune scores will be dumped when --dump_score is used. This flag can be used to only dump the prune scores.")
    parser.add_argument("--dump_gradients_only", action="store_true", help="Dump gradients for analysis. To be used jointly with --dump_score. If true, only gradients will be dumped and no actual pruning will be performed in this run.")

    parser.add_argument(
        "--dump_mask", action="store_true", help="Whether to dump the final pruning mask."
    )
    parser.add_argument(
        "--dump_indices", action="store_true", help="Whether to dump the final pruning indices."
    )
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help=(
            "For manifest pruning, exit immediately after selecting, saving, and "
            "applying the sparse mask. This intentionally skips preservation-loss, "
            "WikiText, and other evaluations; use only when evaluation is run by a "
            "separate strict mask-replay stage."
        ),
    )

    # Finetuning jailbreak configuration
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for a finetuning jailbreak")
    parser.add_argument("--finetuning_before_epochs", type=int, default=5, help="Number of epochs for a finetuning jailbreak before pruning (for flag --finetuning_before)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning for a finetuning jailbreak")
    parser.add_argument("--finetuning_before_lr", type=float, default=2e-4, help="Learning for a finetuning jailbreak before pruning (for flag --finetuning_before)")

    # Evaluation configuration
    parser.add_argument("--eval_zero_shot", action="store_true", help='Utility evaluation (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Challenge, and OpenBookQA)')
    parser.add_argument("--eval_triviaqa", action="store_true", help='Evaluate TriviaQA accuracy (freeform generation)')
    parser.add_argument("--triviaqa_nsamples", type=int, default=1000, help='Number of samples to evaluate on TriviaQA')
    parser.add_argument("--eval_alpaca", action="store_true", help="Evaluate on Alpaca dataset instructions, using an LLM judge.")
    parser.add_argument("--alpaca_nsamples", default=1000, type=int, help='Number of samples to evaluate on Alpaca instruct.')
    parser.add_argument(
        "--alpaca_eval_seed",
        default=5,
        type=int,
        help="Fixed seed used only to select the Alpaca utility-evaluation subset.",
    )
    parser.add_argument(
        "--alpaca_eval_data",
        type=str,
        default=None,
        help=(
            "Local .json, .jsonl/.ndjson, or .csv Alpaca utility dataset. "
            "Required for manifest pruning with --eval_alpaca."
        ),
    )
    parser.add_argument("--eval_harmfulness_detection", action="store_true", help="Evaluate detection capabilities (is this request harmful? yes/no)")
    parser.add_argument("--eval_harmfulness_explanation", action="store_true", help="Evaluate explanation capabilities (explain why this request is harmful)")
    parser.add_argument("--eval_financial_advice", action="store_true", help="Evaluate on a related-to-harmful domain but non-harmful requests of financial advice. Paper shows this is a collateral damage of pruning.")
    parser.add_argument("--eval_refusal", action="store_true", help="Evaluate refusal rate (keyword-based)")
    parser.add_argument("--eval_coherency", action="store_true",
                        help="Run Cohere coherency judge on generation, explanation or TriviaQA outputs")
    parser.add_argument("--eval_safety", action="store_true", help="Generate responses to harmful requests (possibly with a jailbreak specific by --attack) and evaluate with a StrongReject classifier")
    parser.add_argument("--safety_llm_judge", action="store_true", help='Run an LLM judge in addition to the StrongReject evaluation')

    # only relevant if eval_safety is True
    parser.add_argument("--attack", type=str, default="none",
                        choices=["none", "prefilling", "finetuning", "finetuning_prefilling", "refusal_ablation",
                                 "refusal_ablation_prefilling"], help="Jailbreak to use. Only relevan with --eval_safety.")
    parser.add_argument("--dataset", choices=["hex_phi", "advbench"], help='Prompts to use to evaluate safety.')
    parser.add_argument("--category", choices=['adult_content', 'economic_harm', 'financial_advice',
                                               'fraudulent_deceptive', 'harm_children', 'hate_harass_violence',
                                               'illegal_activity', 'malware', 'physical_harm', 'political',
                                               'privacy_violation', 'chosen_five', None], default=None, help='Only relevant for --dataset hex_phi. Specifies the category to test.')
    parser.add_argument("--minus_category", choices=['adult_content', 'economic_harm', 'financial_advice',
                                                     'fraudulent_deceptive', 'harm_children', 'hate_harass_violence',
                                                     'illegal_activity', 'malware', 'physical_harm', 'political',
                                                     'privacy_violation', None], default=None, help='Only relevant for --dataset hex_phi. Specifies the category to *not* test.')
    parser.add_argument("--finetuning_before", action="store_true", help='fine-tune on harmful requests+responses before pruning')
    parser.add_argument("--finetuning_after", action="store_true", help='fine-tune on harmful requests+responses after pruning')
    parser.add_argument("--no_chat_template", action="store_true", help='Do not use chat template. Models are often more susceptible to jailbreaks without it. Only relevant to instruct models.')

    parser.add_argument(
        "--p2",
        type=float,
        default=0.5,
        help="Use combined with attribution_score_set_difference, the top p2 scored elements in the third set (align))",
    )
    parser.add_argument(
        "--abs_refusal", action="store_true", help="Whether to use absolute values for the refusal data when computing the SNIP scores for the refusal_ablation attack"
    )
    parser.add_argument("--refusal_pretrained_format", action="store_true", help='Whether to use chat template for the refusal data SNIP score computation for the refusal_ablation attack.')
    parser.add_argument("--refusal_data", type=str, default=None, help="Dataset to use for the refusal data in the refusal_ablation attack. If not specified, will use the same dataset as --preserve_data")

    args = parser.parse_args()
    if (args.prune_manifest is None) != (args.preserve_manifest is None):
        parser.error("--prune_manifest and --preserve_manifest must be provided together")
    if args.prune_manifest is not None:
        if args.model_path is not None or args.after_attack_model_path is not None:
            parser.error(
                "manifest pruning scores the pinned --model revision directly; "
                "--model_path and --after_attack_model_path are not supported"
            )
        if args.prune_method != "attribution_score_set_difference_global":
            parser.error(
                "manifest scoring requires --prune_method "
                "attribution_score_set_difference_global"
            )
        if not args.revision:
            parser.error("manifest scoring requires a pinned --revision")
        if getattr(args, "tokenizer_revision", None) not in (None, args.revision):
            parser.error(
                "manifest scoring pins model and tokenizer to the same --revision; "
                "--tokenizer_revision may only repeat that value"
            )
        if args.prune_pretrained_format or args.preserve_pretrained_format:
            parser.error(
                "manifest scoring replaces the ambiguous *_pretrained_format flags; "
                "use --score_format raw or --score_format chat"
            )
        if args.dump_score and args.use_saved_scores:
            parser.error("--dump_score and --use_saved_scores are mutually exclusive")
        if args.score_role != "both" and not args.dump_score:
            parser.error("--score_role prune/preserve is only valid with --dump_score")
        if args.mask_only and args.dump_score:
            parser.error("--mask_only and --dump_score are mutually exclusive")
        if args.mask_only and not (args.dump_mask or args.dump_indices):
            parser.error("--mask_only requires --dump_mask or --dump_indices")
        if args.eval_alpaca and not args.alpaca_eval_data:
            parser.error(
                "manifest pruning with --eval_alpaca requires "
                "--alpaca_eval_data <json|jsonl|csv>"
            )
    print(args)
    return args

def save_model(args, model, tokenizer):
    if args.prune_manifest is not None:
        result = getattr(args, "_paper_pruning_result", {})
        mask_dir = Path(result.get("mask_dir", "unlabeled"))
        try:
            relative_mask = mask_dir.relative_to(Path(args.artifact_root) / "masks")
        except ValueError:
            relative_mask = Path(args.model.replace("/", "_")) / mask_dir.name
        save_path = Path(args.artifact_root) / "models" / relative_mask
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        with open(save_path / "pruning_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        copied_mask_artifacts = copy_mask_artifacts(mask_dir, save_path)
        if copied_mask_artifacts:
            print(
                "copied sparse mask artifacts into checkpoint: "
                + ", ".join(copied_mask_artifacts)
            )
        print(f"successfully saved model and tokenizer to {save_path}")
        return
    abs_indicator = "abs" if not args.no_abs else "no_abs"
    if args.prune_method == "attribution_score_set_difference":
        p_q = f"p_{args.p}_q_{args.q}"
    else:
        p_q = ""
    save_path = f"../pruned_models/{args.model.split('/')[-1]}/seed_{args.seed}/prune_{args.prune_data}/nsamples_{args.nsamples}/preserve_{args.preserve_data}/{args.prune_method}/{abs_indicator}/neg_{args.neg_prune}/{p_q}/alpha_{args.alpha}/layers_{args.layers}/model.pt"
    if args.freeze_first_top_q:
        save_path = f"../pruned_models/{args.model.split('/')[-1]}/seed_{args.seed}/prune_{args.prune_data}/nsamples_{args.nsamples}/preserve_{args.preserve_data}/{args.prune_method}/{abs_indicator}/neg_{args.neg_prune}/{p_q}/alpha_{args.alpha}/layers_{args.layers}/freeze_top_q/model.pt"
    if args.model in ("allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-DPO"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"successfully saved model to {save_path}")


def prune(args, device, model, tokenizer):
    if args.prune_manifest is not None and (
        args.model_path is not None or args.after_attack_model_path is not None
    ):
        raise ValueError(
            "manifest pruning cannot be combined with --model_path or "
            "--after_attack_model_path"
        )
    if (args.model_path is not None) or (args.after_attack_model_path is not None):
        print("model loaded from checkpoint, skipping pruning")
        return

    print("pruning starts")
    if (args.prune_manifest is None) != (args.preserve_manifest is None):
        raise ValueError("--prune_manifest and --preserve_manifest must be provided together")
    if args.prune_manifest is not None:
        if args.prune_method != "attribution_score_set_difference_global":
            raise ValueError(
                "manifest scoring is implemented for "
                "--prune_method attribution_score_set_difference_global"
            )
        if not args.revision:
            raise ValueError(
                "manifest-driven pruning requires a pinned --revision for reproducibility"
            )
        if getattr(args, "tokenizer_revision", None) not in (None, args.revision):
            raise ValueError(
                "manifest-driven pruning requires model and tokenizer to use the same revision"
            )
        result = run_manifest_global_pruning(args, model, tokenizer)
        args._paper_pruning_result = result
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    if args.prune_method in ["attribution_score_set_difference", "attribution_score_set_difference_global", "random"]:
        if args.use_saved_scores:
            if args.refusal_data is not None:
                prune_attribution_score_set_difference_with_refusal(
                    args,
                    model,
                    p=args.p,
                    q=args.q,
                )
            elif args.prune_method in ["attribution_score_set_difference", "random"]:
                prune_attribution_score_set_difference(
                    args,
                    model,
                    p=args.p,
                    q=args.q,
                )
            else:
                prune_attribution_score_set_difference_global(
                    args,
                    model,
                    p=args.p,
                    q=args.q,
                )

        else:
            no_abs_prune = args.no_abs
            if args.no_abs and args.abs_prune:
                no_abs_prune = False
            scores_dict_prune = prune_attribution_score(
                args,
                model,
                tokenizer,
                device,
                prune_data=args.prune_data,
                return_score=True,
                no_abs=no_abs_prune,
                pretrained_format=args.prune_pretrained_format,
                dump_score=False if args.do_not_dump_prune_scores else args.dump_score
            )
            no_abs_preserve = args.no_abs
            if args.no_abs and args.abs_preserve:
                no_abs_preserve = False

            old_nsamples = None
            if args.nsamples_preserve is not None:
                old_nsamples = args.nsamples
                args.nsamples = args.nsamples_preserve
            scores_dict_preserve = prune_attribution_score(
                args,
                model,
                tokenizer,
                device,
                prune_data=args.preserve_data,
                return_score=True,
                no_abs=no_abs_preserve,
                pretrained_format=args.preserve_pretrained_format,
                dump_score=False if args.do_not_dump_preserve_score else args.dump_score
            )
            if old_nsamples is not None:
                args.nsamples = old_nsamples

            if args.dump_score:
                print("legacy score dump complete; skipping mask construction")
                return {"dump_only": True}

            if args.prune_method in ["attribution_score_set_difference", "random"]:
                prune_attribution_score_set_difference(
                        args,
                        model,
                        p=args.p,
                        q=args.q,
                        scores_dict_preserve=scores_dict_preserve,
                        scores_dict_prune=scores_dict_prune,
                    )
            else:
                prune_attribution_score_set_difference_global(
                    args,
                    model,
                    p=args.p,
                    q=args.q,
                    scores_dict_preserve=scores_dict_preserve,
                    scores_dict_prune=scores_dict_prune,
                )

    elif args.prune_method == "attribution_score":
        prune_attribution_score(
            args,
            model,
            tokenizer,
            device,
            prune_data=args.prune_data,
            pretrained_format=args.preserve_pretrained_format
        )

    elif args.prune_method == "given_indices":
        prune_from_indices(args, model)
    elif args.prune_method == "intersect_indices":
        if args.indices_path is not None:
            prune_from_indices(args, model)
        else:
            assert(args.prune_data_list is not None), "For intersect_indices prune_method, please provide datasets in prune_data_list"
            indices_to_prune = find_intersection_indices(args, device, model, tokenizer)
            prune_from_indices(args, model, indices_to_prune)

def eval_alpaca(output):
    judge_score, full_judge_responses = judge_for_alpaca(output)
    output['judge_score'] = judge_score
    output['full_judge_responses'] = full_judge_responses
    wandb.log({"output_alpaca": wandb.Table(dataframe=output)})
    metrics = summarize_alpaca_scores(judge_score)
    wandb.summary["alpaca_judge_score"] = metrics["mean_score"]
    wandb.summary["alpaca_judge_count"] = metrics["count"]
    return metrics


def save_model_after_attack(args, model):
    # there's no point of doing this for any attack other than finetuning or pruning
    if args.attack not in ["finetuning", "refusal_ablation", "refusal_ablation_prefilling"]:
        raise ValueError(f"save_model_after_attack not implemented for attack {args.attack}")
    abs_indicator = "abs" if not args.no_abs else "no_abs"
    if args.prune_method == "attribution_score_set_difference":
        p_q = f"p_{args.p}_q_{args.q}"
    else:
        p_q = ""
    base_path = "../pruned_models"
    if args.prune_method == "none":
        save_path = f"{base_path}/{args.model.split('/')[-1]}/seed_{args.seed}/{args.prune_method}/model_after_{args.attack}.pt"
    else:
        save_path = f"{base_path}/{args.model.split('/')[-1]}/seed_{args.seed}/prune_{args.prune_data}/nsamples_{args.nsamples}/preserve_{args.preserve_data}/{args.prune_method}/{abs_indicator}/neg_{args.neg_prune}/{p_q}/alpha_{args.alpha}/layers_{args.layers}/model_after_{args.attack}.pt"
    model.save_pretrained(save_path)
    print("Saved model after attack:", save_path)

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.hp_search:
        wandb.init(project="pruning_safety_hyperparams_search", config=vars(args))
    else:
        wandb.init(project=args.wandb_project, config=vars(args))

    # if args.dump_score:
    #     assert args.prune_method in [
    #         "attribution_score",
    #     ], "dump_score only works with attribut ion_score"

    model, tokenizer = load_model_and_tokenizer(args)
    device = torch.device("cuda:0")

    if args.finetuning_before:
        model = finetuning_attack(args, model, tokenizer, args.seed, lr=args.finetuning_before_lr,
                                  epochs=args.finetuning_before_epochs)

    prune_result = prune(args, device, model, tokenizer)

    # Score jobs are intentionally side-effect-free with respect to weights and
    # must not fall through to model saving, sparsity checks, WikiText, or evals.
    if args.dump_score:
        print("score dump complete; exiting before masking and evaluation")
        wandb.finish()
        return
    if args.mask_only:
        print(
            "mask-only run complete; exiting before preservation loss, WikiText, "
            "and downstream evaluations"
        )
        wandb.finish()
        return

    if args.finetuning_after:
        # finetuning attack after pruning, and then prune after to measure mitigation
        # to finetune without pruning afterwards, simply use --attack finetuning
        model = finetuning_attack(args, model, tokenizer, args.seed, lr=args.lr, epochs=args.epochs)
        prune_result = prune(args, device, model, tokenizer)

    is_manifest_mask_run = (
        args.prune_manifest is not None
        and isinstance(prune_result, dict)
        and not prune_result.get("dump_only", False)
        and prune_result.get("mask_dir") is not None
    )
    preservation_loss = None
    if is_manifest_mask_run:
        preservation_loss = evaluate_preservation_manifest(args, model, tokenizer)
        print(f"post-mask preservation manifest loss {preservation_loss}")
        wandb.summary["preservation_manifest_loss"] = preservation_loss

    if args.save_model:
        save_model(args, model, tokenizer)

    print("*" * 30)

    sparsity_data = check_sparsity(model)
    sparsity_ratio = sparsity_data['overall']
    print(f"sparsity sanity check {sparsity_ratio:.6f}")

    if not args.dump_score:
        wandb.summary["sparsity_ratio"] = sparsity_ratio

    if args.log_sparsity_distribution:
        log_pruning_distribution(sparsity_data, args.prune_data)

    print("*" * 30)

    ppl_test = eval_ppl_wikitext(args, model, tokenizer, args.model, device)
    print(f"wikitext perplexity {ppl_test}")
    wandb.summary["wikitest_ppl"] = ppl_test

    if is_manifest_mask_run:
        evaluation_path = write_evaluation_metadata(
            args,
            prune_result,
            preservation_loss=preservation_loss,
            wikitext_perplexity=ppl_test,
            sparsity=sparsity_ratio,
        )
        print(f"wrote manifest pruning evaluation metadata to {evaluation_path}")

    print("*" * 30)

    if args.eval_alpaca:
        output = generate_responses_for_alpaca(model, tokenizer, args, nsamples=args.alpaca_nsamples)
        alpaca_metrics = eval_alpaca(output)
        if is_manifest_mask_run:
            alpaca_data_path = Path(args.alpaca_eval_data).expanduser().resolve()
            alpaca_data_sha256 = sha256_file(alpaca_data_path)
            alpaca_eval_seed = int(args.alpaca_eval_seed)
            alpaca_identity = f"{alpaca_data_sha256[:12]}_seed{alpaca_eval_seed}"
            items_path = write_evaluation_items(
                prune_result,
                f"alpaca_items_{alpaca_identity}.jsonl",
                output.to_dict(orient="records"),
            )
            alpaca_artifact = {
                **alpaca_metrics,
                "data_path": str(alpaca_data_path),
                "data_sha256": alpaca_data_sha256,
                "evaluation_seed": alpaca_eval_seed,
                "requested_nsamples": int(args.alpaca_nsamples),
                "generation": {"do_sample": False, "max_new_tokens": 256},
                "judge": {"model": ALPACA_JUDGE_MODEL, "temperature": 0},
            }
            metrics_path = write_evaluation_artifact(
                prune_result,
                f"alpaca_metrics_{alpaca_identity}.json",
                alpaca_artifact,
            )
            update_evaluation_metadata(
                prune_result,
                {
                    "alpaca": {
                        **alpaca_artifact,
                        "metrics_path": metrics_path.name,
                        "items_path": items_path.name,
                    }
                },
            )

    if args.eval_zero_shot:
        print("*" * 30)
        results = eval_zero_shot(args, model, tokenizer)
        zero_shot_metrics = summarize_zero_shot_results(results)
        all_acc = []
        all_stderr = []

        print("#" * 10, "Zero shot results:")
        for k, v in results.items():
            print(k, v)
            print(f"{k}: {v['acc,none']}")
            wandb.summary[k] = v['acc,none']
            all_acc.append(v['acc,none'])
            wandb.summary[k + "_stderr"] = v['acc_stderr,none']
            all_stderr.append(v['acc_stderr,none'])

        print("Mean accuracy zero-shot:", np.mean(all_acc))
        wandb.summary["mean_accuracy_zero_shot"] = np.mean(all_acc)
        all_vars = np.array(all_stderr) ** 2
        stderr_overall = np.sqrt(np.sum(all_vars) / (len(all_stderr) ** 2))
        print("Overall stderr zero-shot:", stderr_overall)
        wandb.summary["mean_accuracy_zero_shot_stderr"] = stderr_overall
        if is_manifest_mask_run:
            metrics_path = write_evaluation_artifact(
                prune_result,
                "zero_shot_metrics.json",
                zero_shot_metrics,
            )
            update_evaluation_metadata(
                prune_result,
                {
                    "zero_shot": {
                        **zero_shot_metrics,
                        "metrics_path": metrics_path.name,
                    }
                },
            )

    if args.eval_triviaqa:
        print("*" * 30)
        results = eval_triviaqa(args, model, tokenizer)
        wandb.summary["triviaqa_acc"] = results['correctness'].mean().item()
        if args.eval_coherency:
            print("Running coherency evaluation on TriviaQA outputs...")
            coherency_scores_all, coherency_reasoning_all = coherency_judge(results)
            results['coherency_score'] = coherency_scores_all
            results['coherency_reasoning'] = coherency_reasoning_all
            valid_coherency_all = [s for s in coherency_scores_all if not np.isnan(s)]
            if valid_coherency_all:
                wandb.summary["triviaqa_mean_coherency"] = np.mean(valid_coherency_all)
                print(f"TriviaQA mean coherency (all): {np.mean(valid_coherency_all):.3f}")

            coherency_scores_correct, coherency_reasoning_correct = coherency_judge(results,
                                                                                    filter_by='correctness')
            results['coherency_score_correct_only'] = coherency_scores_correct
            valid_coherency_correct = [s for s in coherency_scores_correct if not np.isnan(s)]

            if len(valid_coherency_correct) != len(coherency_reasoning_correct):
                print("WARNING valid_coherency_correct length does not match coherency_reasoning_correct length, some judge answers were invalid")
            if valid_coherency_correct:
                wandb.summary["triviaqa_mean_coherency_correct_only"] = np.mean(valid_coherency_correct)
                print(f"TriviaQA mean coherency (correct only): {np.mean(valid_coherency_correct):.3f}")

            wandb.log({"output_triviaqa_with_coherency": wandb.Table(dataframe=results)})

    if args.eval_safety:
        output = generate_responses_for_harmful_requests(model, tokenizer, args)
        output = eval_output_harm(args, output)

        if ("refusal_ablation" in args.attack) or ("finetuning" in args.attack):
            ppl_test = eval_ppl_wikitext(args, model, tokenizer, args.model, device)
            print(f"wikitext perplexity {ppl_test}")
            wandb.summary["wikitest_ppl_after_attack"] = ppl_test

        if args.eval_harmfulness_explanation:
            output = eval_explanations(args, output)

        wandb.log({"output": wandb.Table(dataframe=output)})

    if args.save_model_after_attack:
        save_model_after_attack(args, model)

if __name__ == "__main__":
    main()

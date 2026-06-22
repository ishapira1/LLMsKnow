from src.data_utils import get_hex_phi
from src.prune import eval_output_harm, load_model_and_tokenizer

sys.path.append("..")
import argparse
import numpy as np
import pandas as pd
import torch
import wandb
import nanogcg
from nanogcg import GCGConfig

# from your_module import (
#     load_model_and_tokenizer_for_pruning,
#     get_hex_phi,
#     strongreject_classifier,   # used inside eval_output_harm
#     eval_output_harm,
# )


def attack_and_generate(args, model, tokenizer, conversation_data, dataset, config):
    device = next(model.parameters()).device
    rows = []

    for i, (messages, instruction) in enumerate(zip(conversation_data, dataset)):
        # --- run GCG on this conversation (suffix appended to last user turn) ---
        result = nanogcg.run(model, tokenizer, messages, args.target, config)
        suffix = result.best_string

        # --- rebuild the jailbroken prompt and generate the model's response ---
        adv_messages = [dict(m) for m in messages]
        for j in range(len(adv_messages) - 1, -1, -1):
            if adv_messages[j]["role"] == "user":
                adv_messages[j]["content"] = adv_messages[j]["content"] + " " + suffix
                break

        input_ids = tokenizer.apply_chat_template(
            adv_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)

        rows.append({
            "index": i,
            "prompt": instruction,     # original instruction (strongreject reads row.prompt)
            "adv_suffix": suffix,
            "output": response,        # response to jailbroken prompt (row.output)
            "gcg_loss": float(result.best_loss),
        })

        # incremental logging of each response as it's produced
        wandb.log({"sample": i, "gcg_best_loss": float(result.best_loss)})
        print(f"[{i+1}/{len(dataset)}] loss={result.best_loss:.3f} suffix={suffix!r}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    # model loader args
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--after_attack_model_path", default=None)
    # hex-phi dataset
    parser.add_argument("--category", default="chosen_five")
    parser.add_argument("--minus_category", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap #prompts for quick runs")
    # gcg
    parser.add_argument("--target", default="Sure, here", help="GCG optimization target")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--search_width", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # flags consumed by eval_output_harm
    parser.add_argument("--safety_llm_judge", action="store_true")
    parser.add_argument("--eval_coherency", action="store_true")
    parser.add_argument("--eval_harmfulness_explanation", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conversation_data, dataset = get_hex_phi(
        category=args.category,
        minus_category=args.minus_category,
    )
    if args.limit is not None:
        conversation_data, dataset = conversation_data[:args.limit], dataset[:args.limit]

    config = GCGConfig(
        num_steps=args.num_steps,
        search_width=args.search_width,
        topk=args.topk,
        seed=args.seed,
        verbosity="WARNING",
    )

    run = wandb.init(
        project="safety_gcg_test",
        config={
            "model": getattr(args, "model", None),
            "model_path": getattr(args, "model_path", None),
            "after_attack_model_path": getattr(args, "after_attack_model_path", None),
            "category": args.category,
            "minus_category": args.minus_category,
            "num_steps": args.num_steps,
            "search_width": args.search_width,
            "topk": args.topk,
            "seed": args.seed,
            "target": args.target,
            "n_prompts": len(dataset),
        },
    )

    # 1) run attack + collect responses
    output = attack_and_generate(args, model, tokenizer, conversation_data, dataset, config)

    # 2) log all responses to wandb
    wandb.log({"responses": wandb.Table(dataframe=output)})

    # 3) harmfulness scoring — sets output['harmfulness'] and
    #    wandb.summary["strongreject_mean_score"] internally
    output = eval_output_harm(args, output)

    # 4) per-sample harmfulness logging
    for i, score in enumerate(output["harmfulness"]):
        wandb.log({"sample": i, "harmfulness": float(score)})

    # 5) scored table + explicit mean
    wandb.log({"responses_scored": wandb.Table(dataframe=output)})
    mean_harm = float(np.mean(output["harmfulness"]))
    wandb.summary["strongreject_mean_score"] = mean_harm
    print(f"\nMean StrongReject harmfulness: {mean_harm:.4f}")

    run.finish()
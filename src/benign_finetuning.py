import sys

from eval_utils import load_model_and_tokenizer, eval_output_harm

sys.path.append("..")
from data_utils import get_hex_phi

import argparse
import datasets
import numpy as np
import pandas as pd
import torch
import wandb
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# from your_module import (
#     load_model_and_tokenizer,
#     get_hex_phi,
#     strongreject_classifier,   # used inside eval_output_harm
#     eval_output_harm,
# )


# --------------------------------------------------------------------------- #
# Refusal filtering (ported from QuixiAI/open-instruct-uncensored:
# remove_refusals.py). Case-insensitive substring match against the response;
# an example is dropped if it contains ANY phrase. VERIFY this list against the
# upstream file -- this is a reconstruction, not a byte-for-byte copy.
# --------------------------------------------------------------------------- #
REFUSAL_PHRASES = [
    "text-based AI language model", "domestic violence", "please refrain", "derogatory",
    "inappropriate", "offensive", "racism", "racist", "racial", "discriminate",
    "discriminatory", "discrimination", "sexist", "sexism", "unacceptable",
    "inclusive workplace", "lgbt", "morals", "ethics", "ethical", "legality",
    "illegal", "illegality", "hateful", "harmful", "it is never okay", "It is important to",
    "It's important to", "real-world consequences", "hate speech", "glorify",
    "not be appropriate", "supremacist", "extremist", "responsible AI", "AI principles",
    "AI assistant", "an AI language", "ableist", "hurtful", "gender stereotype",
    "gender inequality", "underrepresentation", "safe spaces", "gender-based",
    "inclusivity", "feminist", "feminism", "transgender", "empowerment", "communist",
    "capitalism", "stereotypes", "biases", "bias", "Microaggression",
    "prioritize human safety", "as a language model", "as an AI language model",
    "As a large language model", "As an AI", "ethical principles", "consensual",
    "it is not appropriate", "it's not appropriate", "I cannot fulfill your request",
    "harmful to human beings", "ethical guidelines", "my guidelines",
    "prioritize user safety", "adhere to ethical guidelines", "harmful consequences",
    "potentially harmful", "dangerous activities", "promote safety",
    "well-being of all users", "responsible information sharing", "jeopardize the safety",
    "illegal actions or intentions", "undermine the stability", "promote the well-being",
    "illegal activities or actions", "adherence to the law", "potentially be harmful",
    "illegal substances or activities", "committed to promoting", "safe information",
    "lawful information", "cannot provide guidance", "cannot provide information",
    "unable to offer assistance", "cannot engage in discussions", "programming prohibits",
    "follow ethical guidelines", "ensure the safety", "involves an illegal subject",
    "prioritize safety", "illegal subject", "prioritize user well-being",
    "cannot support or promote", "activities that could harm", "pose a risk to others",
    "against my programming", "activities that could undermine", "potentially dangerous",
    "not within the scope", "designed to prioritize safety", "not able to provide",
    "maintain user safety", "adhere to safety guidelines", "dangerous or harmful",
    "cannot provide any information", "focus on promoting safety",
]


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in REFUSAL_PHRASES)


# --------------------------------------------------------------------------- #
# Fine-tuning datasets
# --------------------------------------------------------------------------- #
def get_harmful_finetuning_data():
    """Risk Level-1 explicit harmful examples (your original loader)."""
    finetuning_data = pd.read_csv("../data/red_teaming_prompts.csv")
    finetuning_data = finetuning_data[finetuning_data["harmful"] == 1]
    all_msgs = []
    for row in finetuning_data.iterrows():
        all_msgs.append({
            "prompt": [{"role": "user", "content": row[1]["text"]}],
            "completion": [{"role": "assistant", "content": row[1]["rejected_response"]}],
        })
    return datasets.Dataset.from_pandas(pd.DataFrame(all_msgs))


def get_benign_finetuning_data(dataset_name="alpaca", remove_safety_samples=True):
    """Risk Level-3 benign fine-tuning (Section 4.4 / Appendix G)."""
    if dataset_name == "alpaca":
        raw = datasets.load_dataset("tatsu-lab/alpaca", split="train")
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.")
        instr_key, input_key, output_key = "instruction", "input", "output"
    elif dataset_name == "dolly":
        raw = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
        system_prompt = "You are a helpful assistant."
        instr_key, input_key, output_key = "instruction", "context", "response"
    else:
        raise ValueError(f"unknown benign dataset: {dataset_name}")

    all_msgs = []
    for ex in raw:
        completion = (ex[output_key] or "").strip()
        if remove_safety_samples and _is_refusal(completion):
            continue
        user_content = (ex[instr_key] or "").strip()
        ctx = (ex.get(input_key, "") or "").strip()
        if ctx:
            user_content = f"{user_content}\n\n{ctx}"
        all_msgs.append({
            "prompt": [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_content}],
            "completion": [{"role": "assistant", "content": completion}],
        })
    return datasets.Dataset.from_pandas(pd.DataFrame(all_msgs))


def get_finetuning_dataset(args):
    if args.finetune_data == "harmful":
        return get_harmful_finetuning_data()
    return get_benign_finetuning_data(
        dataset_name=args.finetune_data,
        remove_safety_samples=not args.keep_safety_samples,
    )


# --------------------------------------------------------------------------- #
# The "attack": LoRA fine-tuning
# --------------------------------------------------------------------------- #
def finetuning_attack(args, model, tokenizer, seed, dataset=None,
                      lora_rank=128, lora_alpha=16, lora_dropout=0,
                      batch_size=16, grad_accum_steps=1, epochs=1, lr=1e-4,
                      warmup_ratio=0.05, weight_decay=0.01):
    if args.after_attack_model_path is not None:
        return model

    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if dataset is None:
        dataset = get_finetuning_dataset(args)
    print("N samples finetuning dataset:", len(dataset))

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_rank,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    config = SFTConfig(
        completion_only_loss=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="none",
        save_strategy="no",
        gradient_checkpointing=False,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=config,
        processing_class=tokenizer,
    )
    trainer.can_return_loss = True
    trainer.train()

    return trainer.model.merge_and_unload()


# --------------------------------------------------------------------------- #
# Post-attack generation on the safety eval set
# --------------------------------------------------------------------------- #
def generate_responses(args, model, tokenizer, conversation_data, dataset):
    device = next(model.parameters()).device
    rows = []

    for i, (messages, instruction) in enumerate(zip(conversation_data, dataset)):
        eval_messages = [dict(m) for m in messages]

        # Honor the paper's requirement (Section 4.1): use the SAME system prompt
        # at eval time as during fine-tuning so the safety delta is attributable
        # to fine-tuning, not the prompt.
        if args.eval_system_prompt is not None:
            if eval_messages and eval_messages[0]["role"] == "system":
                eval_messages[0]["content"] = args.eval_system_prompt
            else:
                eval_messages.insert(
                    0, {"role": "system", "content": args.eval_system_prompt})

        input_ids = tokenizer.apply_chat_template(
            eval_messages, add_generation_prompt=True, return_tensors="pt"
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
            "prompt": instruction,   # original instruction (strongreject reads row.prompt)
            "output": response,      # post-finetuning response (row.output)
        })

        wandb.log({"sample": i})
        print(f"[{i+1}/{len(dataset)}] {instruction[:60]!r} -> {response[:60]!r}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    # model loader args
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--after_attack_model_path", default=None)
    # hex-phi eval dataset
    parser.add_argument("--category", default="chosen_five")
    parser.add_argument("--minus_category", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap #prompts for quick runs")
    parser.add_argument("--eval_system_prompt", default=None,
                        help="override eval system prompt to match fine-tuning")
    # fine-tuning ("attack") args
    parser.add_argument("--finetune_data", default="dolly",
                        choices=["alpaca", "dolly", "harmful"])
    parser.add_argument("--keep_safety_samples", action="store_true",
                        help="skip refusal filtering of the benign dataset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # flags consumed by eval_output_harm
    parser.add_argument("--safety_llm_judge", action="store_true")
    parser.add_argument("--eval_coherency", action="store_true")
    parser.add_argument("--eval_harmfulness_explanation", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conversation_data, dataset = get_hex_phi(
        category=args.category,
        minus_category=args.minus_category,
    )
    if args.limit is not None:
        conversation_data, dataset = conversation_data[:args.limit], dataset[:args.limit]

    run = wandb.init(
        project="safety_finetuning_test",
        config={
            "model": getattr(args, "model", None),
            "model_path": getattr(args, "model_path", None),
            "after_attack_model_path": getattr(args, "after_attack_model_path", None),
            "category": args.category,
            "minus_category": args.minus_category,
            "finetune_data": args.finetune_data,
            "keep_safety_samples": args.keep_safety_samples,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
            "n_prompts": len(dataset),
        },
    )

    # 1) run the fine-tuning attack
    model = finetuning_attack(
        args, model, tokenizer, args.seed,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, epochs=args.epochs, lr=args.lr,
    )
    model.eval()

    # 2) generate responses on the safety eval set
    output = generate_responses(args, model, tokenizer, conversation_data, dataset)

    # 3) log all responses to wandb
    wandb.log({"responses": wandb.Table(dataframe=output)})

    # 4) harmfulness scoring -- sets output['harmfulness'] and
    #    wandb.summary["strongreject_mean_score"] internally
    output = eval_output_harm(args, output)

    # 5) scored table + explicit mean
    wandb.log({"responses_scored": wandb.Table(dataframe=output)})
    mean_harm = float(np.mean(output["harmfulness"]))
    wandb.summary["strongreject_mean_score"] = mean_harm
    print(f"\nMean StrongReject harmfulness: {mean_harm:.4f}")

    run.finish()


if __name__ == "__main__":
    main()
import sys

from tqdm.auto import tqdm

from eval_utils import load_model_and_tokenizer, eval_output_harm

sys.path.append("..")
from data_utils import get_hex_phi

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import nanogcg
from nanogcg import GCGConfig


# ---------------------------------------------------------------------------
# Universal (multi-prompt) GCG  --  Algorithm 2 of Zou et al. (2023).
#
# nanogcg.run only optimizes a suffix for a SINGLE conversation, so the
# universal attack is implemented directly here: one shared suffix p_{1:l} is
# optimized while aggregating both gradients and losses across many prompts,
# with prompts added incrementally (mc) once the current suffix succeeds on the
# active set. Mechanics (one-hot token gradients -> top-k -> sampled candidate
# eval -> greedy best) mirror nanogcg / AutoPrompt.
#
# Per the paper, before aggregating, each prompt's gradient is clipped to unit
# norm. Candidate token-length filtering (nanogcg's filter_ids) is omitted for
# simplicity; verify suffix re-tokenization for your tokenizer if it matters.
# ---------------------------------------------------------------------------

class _UniversalResult:
    """Mimics the fields of a nanogcg result that this script consumes."""
    def __init__(self, best_string, best_loss):
        self.best_string = best_string
        self.best_loss = best_loss


def _universal_segments(tokenizer, messages, target, device):
    """Split one conversation into (before, after, target) token ids, where the
    shared suffix is inserted at the end of the last user turn."""
    SENTINEL = "<|__GCG_UNIVERSAL_SUFFIX__|>"
    msgs = [dict(m) for m in messages]
    for j in range(len(msgs) - 1, -1, -1):
        if msgs[j]["role"] == "user":
            msgs[j]["content"] = msgs[j]["content"] + " " + SENTINEL
            break
    templated = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    if SENTINEL not in templated:
        raise ValueError("Sentinel was lost during templating; cannot locate suffix slot.")
    before_str, after_str = templated.split(SENTINEL)
    # Chat templates usually already include BOS textually; only add it if absent.
    add_bos = not (tokenizer.bos_token and before_str.startswith(tokenizer.bos_token))

    before_ids = tokenizer(before_str, add_special_tokens=add_bos,
                           return_tensors="pt").input_ids.to(device)
    after_ids = tokenizer(after_str, add_special_tokens=False,
                          return_tensors="pt").input_ids.to(device)
    target_ids = tokenizer(target, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(device)
    return before_ids, after_ids, target_ids


def _build_pack(tokenizer, embed, messages, target, device):
    """Precompute the fixed (non-suffix) embeddings + target for one prompt."""
    before_ids, after_ids, target_ids = _universal_segments(tokenizer, messages, target, device)
    with torch.no_grad():
        before_embeds = embed(before_ids)
        after_embeds = embed(after_ids)
        target_embeds = embed(target_ids)
    return {
        "before_embeds": before_embeds,
        "after_embeds": after_embeds,
        "target_embeds": target_embeds,
        "target_ids": target_ids,
        "nb": before_ids.shape[1],
        "na": after_ids.shape[1],
        "nt": target_ids.shape[1],
    }


def _grad_loss(model, pack, optim_embeds):
    """Differentiable target CE loss for a single prompt given suffix embeds (1,L,d)."""
    full = torch.cat(
        [pack["before_embeds"], optim_embeds, pack["after_embeds"], pack["target_embeds"]],
        dim=1,
    )
    logits = model(inputs_embeds=full, use_cache=False).logits
    L = optim_embeds.shape[1]
    start = pack["nb"] + L + pack["na"]
    shift = logits[:, start - 1: start - 1 + pack["nt"], :].float()
    return F.cross_entropy(shift.reshape(-1, shift.size(-1)),
                           pack["target_ids"].reshape(-1))


def _batch_loss(model, embed, pack, cand_ids):
    """Target CE loss for a batch of candidate suffixes (B,L) -> (B,)."""
    B, L = cand_ids.shape
    optim_embeds = embed(cand_ids)
    be = pack["before_embeds"].expand(B, -1, -1)
    ae = pack["after_embeds"].expand(B, -1, -1)
    te = pack["target_embeds"].expand(B, -1, -1)
    full = torch.cat([be, optim_embeds, ae, te], dim=1)
    logits = model(inputs_embeds=full, use_cache=False).logits
    start = pack["nb"] + L + pack["na"]
    shift = logits[:, start - 1: start - 1 + pack["nt"], :].float()      # (B,nt,V)
    tgt = pack["target_ids"].expand(B, -1)                                # (B,nt)
    return F.cross_entropy(shift.transpose(1, 2), tgt, reduction="none").mean(dim=1)


def _succeeds(model, embed, pack, optim_ids):
    """Teacher-forced check: does greedy decoding reproduce the full target?"""
    with torch.no_grad():
        L = optim_ids.shape[1]
        optim_embeds = embed(optim_ids)
        full = torch.cat(
            [pack["before_embeds"], optim_embeds, pack["after_embeds"], pack["target_embeds"]],
            dim=1,
        )
        logits = model(inputs_embeds=full, use_cache=False).logits
        start = pack["nb"] + L + pack["na"]
        pred = logits[:, start - 1: start - 1 + pack["nt"], :].argmax(-1)
        return bool((pred == pack["target_ids"]).all())


def _get_nonascii_toks(tokenizer, device):
    """Token ids whose decoded form is not printable ASCII (llm-attacks bans
    these from candidate sampling when allow_non_ascii=False)."""
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    bad = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            bad.append(i)
    for t in (tokenizer.bos_token_id, tokenizer.eos_token_id,
              tokenizer.pad_token_id, tokenizer.unk_token_id):
        if t is not None:
            bad.append(t)
    return torch.tensor(sorted(set(bad)), device=device)


def _filter_and_reencode(tokenizer, cand_ids, curr_str, L, device):
    """llm-attacks get_filtered_cands: keep candidates whose decoded string
    re-encodes to exactly L tokens and differs from the current control, then
    re-encode the survivors back to ids. This guarantees the suffix we optimize
    is identical to the suffix that gets deployed as a string at generation
    time. Survivors are padded back up to the batch size by repeating the last
    valid one (matching the reference)."""
    B = cand_ids.shape[0]
    strings, ids = [], []
    for i in range(B):
        s = tokenizer.decode(cand_ids[i], skip_special_tokens=True)
        enc = tokenizer(s, add_special_tokens=False).input_ids
        if s != curr_str and len(enc) == L:
            strings.append(s)
            ids.append(enc)
    if not strings:
        # degenerate step: nothing valid -> keep current control unchanged
        return None, None
    # pad back to B by repeating the last valid candidate (reference behavior)
    while len(strings) < B:
        strings.append(strings[-1])
        ids.append(ids[-1])
    ids_t = torch.tensor(ids, device=device)        # (B, L)
    return ids_t, strings


def run_universal(model, tokenizer, conversation_list, target, config,
                  init="x x x x x x x x x x x x x x x x x x x x", eval_batch=32,
                  allow_non_ascii=False, anneal=True):
    """Optimize a single shared adversarial suffix across many conversations.

    Faithful to llm-attacks' Progressive + GCGMultiPromptAttack:
      * gradients: sum RAW per-prompt one-hot grads, then normalize PER POSITION
        (dim=-1) once on the summed gradient;
      * candidates: even-spread replacement positions, optional non-ascii ban,
        then get_filtered_cands (decode->reencode length match, differ from
        current) so the optimized ids == the deployed string;
      * selection: candidate minimizing the SUMMED target loss over the active
        prompts; control is carried forward (with optional annealed acceptance);
      * progression: start with 1 prompt, add one once all active prompts
        succeed, resetting prev_loss to inf (so the next step is accepted);
      * returned suffix: best control seen while the FULL prompt set is active
        (NOT a global-min snapshot, which would overfit to prompt 0).
    """
    device = next(model.parameters()).device
    embed = model.get_input_embeddings()
    emb_weight = embed.weight                       # (V, d)
    vocab_size = emb_weight.shape[0]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.seed)

    packs = [_build_pack(tokenizer, embed, msgs, target, device) for msgs in conversation_list]
    if not packs:
        raise ValueError("Universal attack needs at least one training prompt.")

    optim_ids = tokenizer(init, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    L = optim_ids.shape[1]

    k = config.topk
    B = config.search_width
    n_steps = config.num_steps
    not_allowed = None if allow_non_ascii else _get_nonascii_toks(tokenizer, device)

    mc = 1                                   # number of active training prompts
    control_str = tokenizer.decode(optim_ids[0], skip_special_tokens=True)
    prev_loss = float("inf")                 # for annealed acceptance (reset on add)

    # Best suffix is tracked ONLY while the full prompt set is active, so we never
    # return something overfit to the first prompt (the original bug).
    best_full_ids = optim_ids.clone()
    best_full_loss = float("inf")
    seen_full = False

    pbar = tqdm(range(n_steps), desc="[universal]", dynamic_ncols=True,
                file=sys.stderr)
    for step in pbar:
        active = packs[:mc]

        # --- (1) gradient: SUM raw per-prompt one-hot grads, then normalize ---
        #     per position once on the summed gradient (llm-attacks behavior).
        grad_sum = torch.zeros(L, vocab_size, device=device, dtype=torch.float32)
        for pack in active:
            one_hot = F.one_hot(optim_ids, vocab_size).to(emb_weight.dtype)
            one_hot.requires_grad_(True)
            optim_embeds = one_hot @ emb_weight
            loss = _grad_loss(model, pack, optim_embeds)
            (g,) = torch.autograd.grad(loss, one_hot)
            grad_sum += g[0].float()                       # raw, summed
        grad = grad_sum / (grad_sum.norm(dim=-1, keepdim=True) + 1e-8)  # per-pos norm

        if not_allowed is not None:
            grad[:, not_allowed] = float("inf")            # ban non-ascii tokens
        topk_ids = (-grad).topk(k, dim=1).indices          # (L, k)

        # --- (2) sample candidates: evenly-spread replacement positions ---
        cand_ids = optim_ids.repeat(B, 1)                   # (B, L)
        new_token_pos = torch.arange(0, L, L / B, device=device).long()[:B]  # (B,)
        col = torch.randint(0, k, (B, 1), generator=gen).to(device)
        new_token_val = torch.gather(topk_ids[new_token_pos], 1, col)        # (B,1)
        cand_ids.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

        # --- (3) filter to tokenization-stable candidates, then re-encode ---
        #     so the optimized ids are exactly what gets deployed as a string.
        filt_ids, filt_strs = _filter_and_reencode(tokenizer, cand_ids, control_str, L, device)
        if filt_ids is None:
            pbar.set_postfix(active=f"{mc}/{len(packs)}", loss="skip", best=f"{best_full_loss:.3f}")
            continue

        # --- (4) score candidates by SUMMED target loss over active prompts ---
        total = torch.zeros(filt_ids.shape[0], device=device)
        for pack in active:
            for s in range(0, filt_ids.shape[0], eval_batch):
                chunk = filt_ids[s:s + eval_batch]
                with torch.no_grad():
                    total[s:s + eval_batch] += _batch_loss(model, embed, pack, chunk)

        best_idx = int(total.argmin())
        cand_loss = float(total[best_idx]) / mc            # mean per active prompt

        # --- (5) accept (annealed) and carry control forward ---
        if anneal:
            T = max(1.0 - (step + 1) / n_steps, 1e-7)
            accept = (cand_loss < prev_loss) or (np.exp(-(cand_loss - prev_loss) / T) >= np.random.random())
        else:
            accept = cand_loss < prev_loss
        if accept:
            optim_ids = filt_ids[best_idx:best_idx + 1].clone()
            control_str = filt_strs[best_idx]
        prev_loss = cand_loss

        # track best only over the full active set
        if mc == len(packs):
            seen_full = True
            if cand_loss < best_full_loss:
                best_full_loss = cand_loss
                best_full_ids = optim_ids.clone()

        # --- (6) progression: add a prompt once all active ones succeed ---
        if mc < len(packs) and all(_succeeds(model, embed, p, optim_ids) for p in active):
            mc += 1
            prev_loss = float("inf")                        # reset so next step is accepted

        pbar.set_postfix(active=f"{mc}/{len(packs)}",
                         loss=f"{cand_loss:.3f}",
                         best=f"{best_full_loss if seen_full else float('nan'):.3f}")

    # If we never reached the full set, fall back to the last accepted control.
    final_ids = best_full_ids if seen_full else optim_ids
    with torch.no_grad():
        final = sum(float(_batch_loss(model, embed, p, final_ids)) for p in packs) / len(packs)

    best_string = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    print(f"[universal] done. suffix={best_string!r} mean_train_loss={final:.3f} "
          f"(reached_full_set={seen_full})")
    return _UniversalResult(best_string, final)


def attack_and_generate(args, model, tokenizer, conversation_data, dataset, config,
                        universal_suffix=None, universal_loss=None, train_size=None):
    device = next(model.parameters()).device
    rows = []

    for i, (messages, instruction) in enumerate(zip(conversation_data, dataset)):
        if universal_suffix is None:
            # --- run GCG on this conversation (suffix appended to last user turn) ---
            result = nanogcg.run(model, tokenizer, messages, args.target, config)
            suffix = result.best_string
            best_loss = float(result.best_loss)
            print("Finished running nanogcg, result string:", suffix)
        else:
            # --- universal mode: reuse the single shared suffix for every prompt ---
            suffix = universal_suffix
            best_loss = float(universal_loss)

        # --- rebuild the jailbroken prompt ---
        adv_messages = [dict(m) for m in messages]
        for j in range(len(adv_messages) - 1, -1, -1):
            if adv_messages[j]["role"] == "user":
                adv_messages[j]["content"] = adv_messages[j]["content"] + " " + suffix
                break

        input_ids = tokenizer.apply_chat_template(
            adv_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        # --- generate args.num_generations sampled responses for this prompt ---
        for g in range(args.num_generations):
            # seed per generation so runs are reproducible but the samples differ.
            # gen_idx g uses seed (args.seed + g) for EVERY prompt, so g indexes a
            # full-dataset run -> treat each g as one "seed".
            torch.manual_seed(args.seed + g)

            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                gen[0, input_ids.shape[1]:], skip_special_tokens=True
            )

            row = {
                "index": i,                # prompt index
                "gen_idx": g,              # which of the N samples / "seeds" this is
                "prompt": instruction,     # original instruction (strongreject reads row.prompt)
                "adv_suffix": suffix,
                "output": response,        # response to jailbroken prompt (row.output)
                "gcg_loss": best_loss,
            }
            if train_size is not None:
                # universal mode: mark whether this prompt was in the optimization set
                row["is_train"] = bool(i < train_size)
            rows.append(row)
            print(f"  gen {g+1}/{args.num_generations} response: {response}")

        # incremental logging of GCG progress as each prompt is finished
        # wandb.log({"sample": i, "gcg_best_loss": best_loss})
        print(f"[{i+1}/{len(dataset)}] loss={best_loss:.3f} suffix={suffix!r}")

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
    # universal (multi-prompt) attack -- Algorithm 2
    parser.add_argument("--universal", action="store_true",
                        help="optimize ONE shared suffix across prompts (Alg. 2) "
                             "instead of a separate suffix per prompt")
    parser.add_argument("--universal_train_size", type=int, default=25,
                        help="number of (leading) prompts to optimize the shared suffix on")
    parser.add_argument("--universal_init",
                        default="x x x x x x x x x x x x x x x x x x x x",
                        help="initial suffix string (its token length sets the suffix length)")
    parser.add_argument("--universal_eval_batch", type=int, default=32,
                        help="candidate eval chunk size during universal optimization (OOM guard)")
    parser.add_argument("--universal_allow_non_ascii", action="store_true",
                        help="allow non-ascii tokens in the suffix (llm-attacks bans them by default)")
    parser.add_argument("--universal_no_anneal", action="store_true",
                        help="disable annealed acceptance (greedy accept-if-better instead)")
    # sampling / multi-generation
    parser.add_argument("--num_generations", type=int, default=3,
                        help="number of sampled responses per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
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

    # Shuffle prompts before selecting the first universal_train_size examples.
    # Important: shuffle both lists with the same permutation to preserve alignment.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(dataset))

    conversation_data = [conversation_data[i] for i in perm]
    dataset = [dataset[i] for i in perm]

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
            "universal": args.universal,
            "universal_train_size": args.universal_train_size,
            "universal_init": args.universal_init,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n_prompts": len(dataset),
        },
    )

    # 1) run attack + collect responses (N generations per prompt)
    if args.universal:
        train_size = min(args.universal_train_size, len(conversation_data))
        print(f"[universal] optimizing one shared suffix on {train_size} prompts...")
        uni = run_universal(
            model, tokenizer,
            conversation_data[:train_size],
            args.target, config,
            init=args.universal_init,
            eval_batch=args.universal_eval_batch,
            allow_non_ascii=args.universal_allow_non_ascii,
            anneal=not args.universal_no_anneal,
        )
        output = attack_and_generate(
            args, model, tokenizer, conversation_data, dataset, config,
            universal_suffix=uni.best_string,
            universal_loss=uni.best_loss,
            train_size=train_size,
        )
    else:
        output = attack_and_generate(args, model, tokenizer, conversation_data, dataset, config)

    # 2) log all responses to wandb
    # wandb.log({"responses": wandb.Table(dataframe=output)})

    # 3) harmfulness scoring — scores every row (i.e. every generation),
    #    setting output['harmfulness']
    output = eval_output_harm(args, output)

    # 4) scored table (one row per generation)
    wandb.log({"responses_scored": wandb.Table(dataframe=output)})

    # 5) one dataset-level mean per seed (= per gen_idx).
    #    Each gen_idx is a full pass over the dataset under a fixed seed, so this
    #    gives exactly num_generations dataset-level means -- the analog of running
    #    the other methods 3 times with different seeds.
    per_seed = (
        output.groupby("gen_idx")["harmfulness"]
        .mean()
        .reset_index(name="dataset_mean")
    )

    # for _, r in per_seed.iterrows():
    #     wandb.log({
    #         "gen_idx": int(r["gen_idx"]),
    #         "dataset_mean_harm": float(r["dataset_mean"]),
    #     })
    wandb.log({"per_seed_harm": wandb.Table(dataframe=per_seed)})

    # 6) report: mean and std OVER the per-seed dataset means
    mean_over_seeds = float(per_seed["dataset_mean"].mean())
    std_over_seeds = (
        float(per_seed["dataset_mean"].std(ddof=1)) if len(per_seed) > 1 else 0.0
    )

    wandb.summary["harm_mean"] = mean_over_seeds
    wandb.summary["harm_std"] = std_over_seeds

    seed_means = [round(float(x), 4) for x in per_seed["dataset_mean"]]
    print(f"\nPer-seed dataset means ({len(per_seed)} seeds): {seed_means}")
    print(f"Mean over seeds: {mean_over_seeds:.4f}")
    print(f"Std  over seeds: {std_over_seeds:.4f}")

    # 6b) universal mode only: additionally break the same per-seed means down
    #     into train vs held-out test prompts (paper-style ASR reporting).
    if args.universal and "is_train" in output.columns:
        for flag, name in [(True, "train"), (False, "test")]:
            sub = output[output["is_train"] == flag]
            if len(sub) == 0:
                continue
            per_seed_split = sub.groupby("gen_idx")["harmfulness"].mean()
            m = float(per_seed_split.mean())
            s = float(per_seed_split.std(ddof=1)) if len(per_seed_split) > 1 else 0.0
            wandb.summary[f"harm_mean_{name}"] = m
            wandb.summary[f"harm_std_{name}"] = s
            print(f"[universal/{name}] mean={m:.4f} std={s:.4f} (n_prompts={sub['index'].nunique()})")

    run.finish()


if __name__ == "__main__":
    main()
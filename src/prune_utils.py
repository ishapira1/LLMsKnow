import gc
import io
import os
import pickle
import re
from collections import defaultdict
from functools import reduce

import torch
import torch.nn as nn

from data_utils import get_loaders
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import wandb

MODEL_TO_N_LAYERS = {
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "meta-llama/Llama-3.1-8B": 32,
    "Qwen/Qwen2.5-14B-Instruct": 48,
}
MODEL_TO_COMPONENTS = {
    "meta-llama/Llama-3.1-8B-Instruct": ['mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj', 'self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
    "meta-llama/Llama-3.1-8B": ['mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj', 'self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
    "Qwen/Qwen2.5-14B-Instruct": ['mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj', 'self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.o_proj']
}

class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        # self.register_buffer('activation_norms', torch.zeros([base.in_features], device=self.base.weight.device, requires_grad=False))
        self.activation_norms = torch.zeros(
            [base.in_features], device=self.base.weight.device, requires_grad=False
        )
        self.n_samples = 0
        self.record_activation = True

    def clear_act_buffer(self):
        self.activation_norms.fill_(0.0)
        self.n_samples = 0

    def forward(self, x):

        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask]
            else:
                x_ = x

            bs = x_.nelement() // x_.shape[-1]
            self.activation_norms = self.activation_norms * (
                    self.n_samples / (self.n_samples + bs)
            ) + (x_ * x_).view(-1, x_.shape[-1]).sum(dim=0) * (
                                            1.0 / (self.n_samples + bs)
                                    )
            self.n_samples += bs

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


def make_Act(model, verbose=False):
    replace_map = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
                    # setattr(model, name_suffix, v)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            # Extract the base nn.Linear module from ActLinear
            linear_module = module.base
            # Navigate to the parent module of the ActLinear module
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            print(f"Reverting {name}, parent: {parent_name}")
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            # Replace the ActLinear module with the extracted nn.Linear module
            setattr(parent_module, name.split(".")[-1], linear_module)

    return model


def _prune_core(
        args,
        model,
        prune_mode="activation",
        name_filter_fn=None,
        return_score=False,
        no_abs=None,
        prune_data="wikitext",
        pretrained_format=False,
        dump_score=False
):

    if no_abs is None:
        no_abs = args.no_abs

    if no_abs:
        abs_indicator = "no_abs"
    else:
        abs_indicator = "abs"
    if pretrained_format:
        prune_data = prune_data + "_pretrained_format"


    if args.dump_gradients_only:
        save_path = f"gradients/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}"
    else:
        save_path = f"scores/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}"

    if return_score:
        scores_dict = {}

    for name, module in model.named_modules():
        if name_filter_fn is not None and not name_filter_fn(name):
            continue

        if isinstance(module, ActLinear):
            print("pruning:", name)

            i = re.search(r"\d+", name)
            if i:
                i = int(i.group())
            else:
                i = 0

            print("layer", i)

            if no_abs:
                magnitude = module.base.weight.data
            else:
                magnitude = torch.abs(module.base.weight.data)

            if prune_mode == "activation":
                act = (module.activation_norms ** 0.5).unsqueeze(0)
            elif prune_mode == "gradient":
                if no_abs:
                    act = module.base.weight.grad
                else:
                    act = module.base.weight.grad.abs()
            else:
                raise NotImplemented

            W_metric = magnitude * act
            if dump_score and args.dump_gradients_only:
                W_metric = 1 * act
            if (args.neg_prune) and (not dump_score) and (not return_score):
                W_metric = -W_metric

            if dump_score:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                target_file = os.path.join(
                    save_path, f"W_metric_layer_{i}_name_{name}_weight.pkl"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing W_metric in layer {} and name {} with {} to the file {}".format(
                            i, name, prune_data, save_path
                        )
                    )
                    pickle.dump(W_metric, f)
            if return_score:
                scores_dict[name] = W_metric.clone()

            if dump_score or return_score:
                continue

            W_mask = (
                    torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # unstructured pruning
            indices = sort_res[1][
                      :, : int(W_metric.shape[1] * args.sparsity_ratio)
                      ]
            W_mask.scatter_(1, indices, True)

            print("args.alpha", args.alpha)
            if args.alpha:
                if args.alpha == 'mean':
                    module.base.weight.data[W_mask] = module.base.weight.data.mean()
                else:
                    module.base.weight.data[W_mask] = module.base.weight.data[W_mask] * args.alpha
            else:
                module.base.weight.data[W_mask] = 0  ## set weights to zero

            if args.dump_mask:
                mask_save_path = f"masks/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}"
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                target_file = os.path.join(
                    mask_save_path, f"Mask_layer_{i}_name_{name}_weight.pkl"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing Mask in layer {} and name {} with {} to the file".format(
                            i, name, prune_data
                        )
                    )
                    pickle.dump(W_mask, f)

            if args.dump_indices:
                indices = torch.nonzero(W_mask, as_tuple=False)
                scores_save_path = f"../indices/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}"
                if not os.path.exists(scores_save_path):
                    os.makedirs(scores_save_path)
                target_file = os.path.join(
                    scores_save_path, f"indices_layer_{i}_name_{name}_weight.pt"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing Indices in layer {} and name {} with {} to the file".format(
                            i, name, prune_data
                        )
                    )
                torch.save(indices, target_file)

    if return_score:
        return scores_dict


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def prune_attribution_score(
        args,
        model,
        tokenizer,
        device=torch.device("cuda:0"),
        prune_data="wikitext",
        return_score=False,
        no_abs=None,
        pretrained_format=False,
        dump_score=None
):
    if no_abs is None:
        no_abs = args.no_abs
    model = make_Act(model, verbose=False)

    if dump_score is None:
        dump_score = args.dump_score

    print(f"loading calibration data {prune_data}")

    dataloader, _ = get_loaders(
        args,
        prune_data,
        args.model,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
    )
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers
    saved_grad = {}
    scores_dict = {}
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )

        model.zero_grad()
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("enabling grad for", name)
                module.base.requires_grad_(True)
                saved_grad[name] = torch.zeros_like(
                    module.base.weight, device=module.base.weight.device
                )
                module.base.zero_grad()

        for batch in dataloader:
            inp, tar = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            with no_act_recording(model):
                loss = model(input_ids=inp, labels=tar)[0]
            loss.backward()
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    if no_abs:
                        saved_grad[name] += module.base.weight.grad
                    else:
                        saved_grad[name] += module.base.weight.grad.abs()

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.base.weight.grad.copy_(saved_grad[name])
                saved_grad.pop(name)

        if return_score:
            scores_dict_for_layer = _prune_core(
                args,
                model,
                prune_mode="gradient",
                name_filter_fn=layer_filter_fn,
                return_score=True,
                no_abs=no_abs,
                prune_data=prune_data,
                pretrained_format=pretrained_format,
                dump_score=dump_score
            )
            scores_dict.update(scores_dict_for_layer)
        else:
            _prune_core(
                args,
                model,
                prune_mode="gradient",
                name_filter_fn=layer_filter_fn,
                no_abs=no_abs,
                prune_data=prune_data,
                pretrained_format=pretrained_format,
                dump_score=dump_score
            )

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory

    if return_score:
        return scores_dict

def prune_attribution_score_set_difference_global(
        args,
        model,
        p=0.5,
        q=0.5,
        scores_dict_preserve=None,
        scores_dict_prune=None
):
    """
    GLOBAL version of the set-difference pruner.

    Instead of computing the top-p (utility) and top-q (safety) thresholds
    independently inside every weight matrix, this pools the attribution
    scores of *all* selected layers/modules into one flat tensor, picks the
    global top-p and top-q across that pool, takes the set difference
    (top-q minus top-p) globally, and then maps the surviving indices back to
    each individual weight matrix to apply the mask.

    Net effect: `top_utility = int(p * N_total)` and
    `top_safety = int(q * N_total)` where N_total is the total number of
    weights across every gathered module, rather than per-matrix counts.
    """
    model.config.use_cache = False
    layers = model.model.layers

    print(
        "GLOBAL prune p = {}, q = {}, with preserve_data = {}, prune_data = {}".format(
            p, q, args.preserve_data, args.prune_data
        )
    )

    # assert (
    #         args.dump_score == False
    # )  # Only pruning from the saved score, won't save score again

    if args.no_abs:
        abs_indicator = "no_abs"
    else:
        abs_indicator = "abs"

    if args.abs_preserve:
        abs_indicator_preserve = "abs"
    else:
        abs_indicator_preserve = abs_indicator

    # ------------------------------------------------------------------ #
    # Pass 1: gather every module's preserve/prune scores into a single  #
    #         global pool, recording where each module lives in it.      #
    # ------------------------------------------------------------------ #
    preserve_flat_list = []
    prune_flat_list = []
    block_meta = []  # one entry per (layer, name) block
    offset = 0

    for i in range(len(layers)):

        if args.layers is not None:
            if i not in args.layers:
                print(f"Skipping layer {i}")
                continue

        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"gathering scores: layer {i} name {name}")

            # ---- load preserve (utility) metric ----
            if scores_dict_preserve is None:
                if args.preserve_pretrained_format:
                    preserve_data = args.preserve_data + "_pretrained_format"
                else:
                    preserve_data = args.preserve_data
                path = f"scores/{args.model.split('/')[-1]}/{args.prune_method}/seed_0/{preserve_data}/{abs_indicator_preserve}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded preserve weight from ", path)
                W_metric_preserve = smart_load(path)
            else:
                W_metric_preserve = scores_dict_preserve[f"model.layers.{i}.{name}"]

            # ---- load prune (safety) metric ----
            if scores_dict_prune is None:
                if args.prune_pretrained_format:
                    prune_data = args.prune_data + "_pretrained_format"
                else:
                    prune_data = args.prune_data
                path = f"scores/{args.model.split('/')[-1]}/{args.prune_method}/seed_0/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded prune weight from ", path)
                W_metric_prune = smart_load(path)
            else:
                W_metric_prune = scores_dict_prune[f"model.layers.{i}.{name}"]

            if args.neg_prune:
                if not args.abs_preserve:
                    W_metric_preserve = -W_metric_preserve
                if not args.abs_prune:
                    W_metric_prune = -W_metric_prune

            rows, cols = W_metric_prune.shape
            numel = rows * cols

            # Move to CPU so the concatenated global pool doesn't blow up GPU
            # memory; topk/membership are computed on CPU.
            preserve_flat_list.append(W_metric_preserve.reshape(-1).cpu())
            prune_flat_list.append(W_metric_prune.reshape(-1).cpu())

            block_meta.append(
                {
                    "i": i,
                    "name": name,
                    "module": subset[name],
                    "offset": offset,
                    "rows": rows,
                    "cols": cols,
                    "numel": numel,
                }
            )
            offset += numel

            del W_metric_preserve
            del W_metric_prune
            gc.collect()

    total_numel = offset
    print(f"Total weights gathered across all layers: {total_numel}")

    # ------------------------------------------------------------------ #
    # Global top-p / top-q over the concatenated pool, then the          #
    # set difference (top-q that is NOT in top-p).                       #
    # ------------------------------------------------------------------ #
    preserve_flat = torch.cat(preserve_flat_list)
    prune_flat = torch.cat(prune_flat_list)
    del preserve_flat_list
    del prune_flat_list
    gc.collect()

    top_utility = int(p * total_numel)
    top_safety = int(q * total_numel)

    top_p_indices = torch.topk(preserve_flat, top_utility, largest=True)[1]
    top_q_indices = torch.topk(prune_flat, top_safety, largest=True)[1]

    # Boolean membership over the whole pool is far cheaper than torch.isin
    # on billion-element index tensors, and is exactly the same set logic.
    in_top_p = torch.zeros(total_numel, dtype=torch.bool)
    in_top_p[top_p_indices] = True

    if args.freeze_first_top_q:
        top_safety2 = int((2 * q) * total_numel)
        top_2q_indices = torch.topk(prune_flat, top_safety2, largest=True)[1]

        in_top_q = torch.zeros(total_numel, dtype=torch.bool)
        in_top_q[top_q_indices] = True

        keep = (~in_top_p[top_2q_indices]) & (~in_top_q[top_2q_indices])
        filtered_indices = top_2q_indices[keep]
        del in_top_q
    else:
        keep = ~in_top_p[top_q_indices]
        filtered_indices = top_q_indices[keep]

    del preserve_flat, prune_flat, top_p_indices, top_q_indices, in_top_p
    gc.collect()

    print("total n_pruned_weights (global):", len(filtered_indices))

    # Sort once so each block can be sliced out of the global index list.
    filtered_indices, _ = torch.sort(filtered_indices)

    # ------------------------------------------------------------------ #
    # Pass 2: map the surviving global indices back into each module and #
    #         apply the mask exactly as before.                          #
    # ------------------------------------------------------------------ #
    for meta in block_meta:
        i = meta["i"]
        name = meta["name"]
        module = meta["module"]
        off = meta["offset"]
        numel = meta["numel"]
        cols = meta["cols"]

        # Global indices that fall inside this block's contiguous range.
        block_sel = (filtered_indices >= off) & (filtered_indices < off + numel)
        local_indices = filtered_indices[block_sel] - off

        device = module.weight.data.device
        filtered_indices_rows = (local_indices // cols).to(device)
        filtered_indices_cols = (local_indices % cols).to(device)

        print(
            f"pruning layer {i} name {name}, n_pruned_weights: {len(filtered_indices_rows)}"
        )

        W_mask = torch.zeros_like(module.weight.data) == 1
        W_mask[filtered_indices_rows, filtered_indices_cols] = (
            True  # prune weights with relatively high (global) safety while not in top (global) utility scores
        )

        if args.alpha:
            if args.alpha == 'mean':
                module.weight.data[W_mask] = module.weight.data[W_mask].mean()
            else:
                module.weight.data[W_mask] = module.weight.data[W_mask] * args.alpha
        else:
            module.weight.data[W_mask] = 0  ## set weights to zero

        if args.dump_mask:
            mask_save_path = f"masks/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{args.prune_data}_not_{args.preserve_data}/{abs_indicator}/nsamples_{args.nsamples}"
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
            target_file = os.path.join(
                mask_save_path, f"Mask_layer_{i}_name_{name}_weight.pkl"
            )
            with open(target_file, "wb") as f:
                print(
                    "Writing Mask in layer {} and name {} with {} to the file".format(
                        i, name, args.prune_data
                    )
                )
                pickle.dump(W_mask, f)

        if args.dump_indices:
            indices = torch.nonzero(W_mask, as_tuple=False)
            scores_save_path = f"../indices/{args.model.split('/')[-1]}/{args.prune_method}_not_{args.preserve_data}/seed_{args.seed}/{args.prune_data}/{abs_indicator}/nsamples_{args.nsamples}"
            if not os.path.exists(scores_save_path):
                os.makedirs(scores_save_path)
            target_file = os.path.join(
                scores_save_path, f"indices_layer_{i}_name_{name}_weight.pt"
            )
            with open(target_file, "wb") as f:
                print(
                    "Writing Indices in layer {} and name {} with {} to the file".format(
                        i, name, args.prune_data
                    )
                )
            torch.save(indices, target_file)

        gc.collect()

def prune_attribution_score_set_difference_with_refusal(
        args,
        model,
        p=0.5,
        q=0.5,
        scores_dict_preserve=None,
        scores_dict_prune=None,
        scores_dict_refusal=None,
):
    model.config.use_cache = False
    layers = model.model.layers

    print(
        "prune p = {}, q = {}, p2 = {}, with preserve_data = {}, prune_data = {}, refusal_data = {}".format(
            p, q, getattr(args, "p2", None), args.preserve_data, args.prune_data, args.refusal_data
        )
    )

    for i in range(len(layers)):

        if args.layers is not None:
            if i not in args.layers:
                print(f"Skipping layer {i}")
                continue

        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")

            if args.no_abs:
                abs_indicator = "no_abs"
            else:
                abs_indicator = "abs"

            if args.abs_preserve:
                abs_indicator_preserve = "abs"
            else:
                abs_indicator_preserve = abs_indicator

            if scores_dict_preserve is None:
                if args.preserve_pretrained_format:
                    preserve_data = args.preserve_data + "_pretrained_format"
                else:
                    preserve_data = args.preserve_data
                path = f"scores/{args.model.split('/')[-1]}/{args.prune_method}/seed_0/{preserve_data}/{abs_indicator_preserve}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded weight from ", path)
                W_metric_preserve = smart_load(path)
            else:
                W_metric_preserve = scores_dict_preserve[f"model.layers.{i}.{name}"]

            if scores_dict_prune is None:
                if args.prune_pretrained_format:
                    prune_data = args.prune_data + "_pretrained_format"
                else:
                    prune_data = args.prune_data
                path = f"scores/{args.model.split('/')[-1]}/{args.prune_method}/seed_0/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded weight from ", path)
                W_metric_prune = smart_load(path)
            else:
                W_metric_prune = scores_dict_prune[f"model.layers.{i}.{name}"]

            # --- refusal preservation set ---
            if getattr(args, "abs_refusal", False):
                abs_indicator_refusal = "abs"
            else:
                abs_indicator_refusal = abs_indicator

            if scores_dict_refusal is None:
                if getattr(args, "refusal_pretrained_format", False):
                    refusal_data = args.refusal_data + "_pretrained_format"
                else:
                    refusal_data = args.refusal_data
                path = f"scores/{args.model.split('/')[-1]}/{args.prune_method}/seed_0/{refusal_data}/{abs_indicator_refusal}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded weight from ", path)
                W_metric_refusal = smart_load(path)
            else:
                W_metric_refusal = scores_dict_refusal[f"model.layers.{i}.{name}"]

            if args.neg_prune:
                if not args.abs_preserve:
                    W_metric_preserve = -W_metric_preserve
                if not args.abs_prune:
                    W_metric_prune = -W_metric_prune
                if args.abs_refusal:
                    W_metric_refusal = -W_metric_refusal
                # if not getattr(args, "abs_refusal", False):
                #     W_metric_refusal = -W_metric_refusal

            top_utility = int(p * W_metric_preserve.shape[1] * W_metric_preserve.shape[0])
            top_safety = int(q * W_metric_prune.shape[1] * W_metric_prune.shape[0])
            top_refusal = int(args.p2 * W_metric_refusal.shape[1] * W_metric_refusal.shape[0])

            top_p_indices = torch.topk(W_metric_preserve.flatten(), top_utility, largest=True)[1]
            top_q_indices = torch.topk(W_metric_prune.flatten(), top_safety, largest=True)[1]
            top_p2_indices = torch.topk(W_metric_refusal.flatten(), top_refusal, largest=True)[1]

            unique_p = torch.unique(top_p_indices)
            unique_q = torch.unique(top_q_indices)
            unique_refusal = torch.unique(top_p2_indices)

            if args.freeze_first_top_q:
                top_safety2 = int((2 * q) * W_metric_prune.shape[1] * W_metric_prune.shape[0])
                top_2q_indices = torch.topk(W_metric_prune.flatten(), top_safety2, largest=True)[1]
                unique_2q = torch.unique(top_2q_indices)
                mask = (
                    ~torch.isin(unique_2q, unique_p)
                    & ~torch.isin(unique_2q, unique_q)
                    & ~torch.isin(unique_2q, unique_refusal)
                )
                filtered_indices = unique_2q[mask]
            else:
                # Create a boolean mask for elements in unique_q that are not in
                # unique_p (utility) and not in unique_refusal (refusal)
                mask = ~torch.isin(unique_q, unique_p) & ~torch.isin(unique_q, unique_refusal)
                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]

            weight_dim = subset[name].weight.data.shape[1]
            filtered_indices_rows = filtered_indices // weight_dim
            filtered_indices_cols = filtered_indices % weight_dim

            print("n_pruned_weights:", len(filtered_indices_rows))
            #
            # assert (
            #         args.dump_score == False
            # )  # Only pruning from the saved score, won't save score again

            W_mask = torch.zeros_like(subset[name].weight.data) == 1
            W_mask[filtered_indices_rows, filtered_indices_cols] = (
                True  # prune weights that has relatively high safety while not in top utility scores
            )

            if args.alpha:
                if args.alpha == 'mean':
                    subset[name].weight.data[W_mask] = subset[name].weight.data[W_mask].mean()
                else:
                    subset[name].weight.data[W_mask] = subset[name].weight.data[W_mask] * args.alpha
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero

            if args.dump_mask:
                mask_save_path = f"masks/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{args.prune_data}_not_{args.preserve_data}/{abs_indicator}/nsamples_{args.nsamples}"
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                target_file = os.path.join(
                    mask_save_path, f"Mask_layer_{i}_name_{name}_weight.pkl"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing Mask in layer {} and name {} with {} to the file".format(
                            i, name, args.prune_data
                        )
                    )
                    pickle.dump(W_mask, f)

            if args.dump_indices:
                indices = torch.nonzero(W_mask, as_tuple=False)
                scores_save_path = f"../indices/{args.model.split('/')[-1]}/{args.prune_method}_not_{args.preserve_data}/seed_{args.seed}/{args.prune_data}/{abs_indicator}/nsamples_{args.nsamples}"
                if not os.path.exists(scores_save_path):
                    os.makedirs(scores_save_path)
                target_file = os.path.join(
                    scores_save_path, f"indices_layer_{i}_name_{name}_weight.pt"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing Indices in layer {} and name {} with {} to the file".format(
                            i, name, args.prune_data
                        )
                    )
                torch.save(indices, target_file)

            del W_metric_preserve
            del W_metric_prune
            del W_metric_refusal
            gc.collect()

def prune_attribution_score_set_difference(
        args,
        model,
        p=0.5,
        q=0.5,
        scores_dict_preserve=None,
        scores_dict_prune=None
):
    model.config.use_cache = False
    layers = model.model.layers

    print(
        "prune p = {}, q = {}, with preserve_data = {}, prune_data = {}".format(
            p, q, args.preserve_data, args.prune_data
        )
    )

    for i in range(len(layers)):

        if args.layers is not None:
            if i not in args.layers:
                print(f"Skipping layer {i}")
                continue

        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")

            if args.no_abs:
                abs_indicator = "no_abs"
            else:
                abs_indicator = "abs"

            if args.abs_preserve:
                abs_indicator_preserve = "abs"
            else:
                abs_indicator_preserve = abs_indicator

            if scores_dict_preserve is None:
                if args.preserve_pretrained_format:
                    preserve_data = args.preserve_data + "_pretrained_format"
                else:
                    preserve_data = args.preserve_data

                path = f"scores/{args.model.split('/')[-1]}/attribution_score_set_difference/seed_0/{preserve_data}/{abs_indicator_preserve}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded weight from ", path)
                W_metric_preserve = smart_load(path)
            else:
                W_metric_preserve = scores_dict_preserve[f"model.layers.{i}.{name}"]

            if scores_dict_prune is None:
                if args.prune_pretrained_format:
                    prune_data = args.prune_data + "_pretrained_format"
                else:
                    prune_data = args.prune_data
                path = f"scores/{args.model.split('/')[-1]}/attribution_score_set_difference/seed_0/{prune_data}/{abs_indicator}/nsamples_{args.nsamples}/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl"
                print("Loaded weight from ", path)
                W_metric_prune = smart_load(path)
            else:
                W_metric_prune = scores_dict_prune[f"model.layers.{i}.{name}"]

            if args.neg_prune:
                if not args.abs_preserve:
                    W_metric_preserve = -W_metric_preserve
                if not args.abs_prune:
                    W_metric_prune = -W_metric_prune

            top_utility = int(p * W_metric_preserve.shape[1] * W_metric_preserve.shape[0])
            top_safety = int(q * W_metric_prune.shape[1] * W_metric_prune.shape[0])

            top_p_indices = torch.topk(W_metric_preserve.flatten(), top_utility, largest=True)[1]
            top_q_indices = torch.topk(W_metric_prune.flatten(), top_safety, largest=True)[1]


            unique_p = torch.unique(top_p_indices)
            unique_q = torch.unique(top_q_indices)

            if args.freeze_first_top_q:
                top_safety2 = int((2 * q) * W_metric_prune.shape[1] * W_metric_prune.shape[0])
                top_2q_indices = torch.topk(W_metric_prune.flatten(), top_safety2, largest=True)[1]
                unique_2q = torch.unique(top_2q_indices)
                mask = ~torch.isin(unique_2q, unique_p) & ~torch.isin(unique_2q, unique_q)
                filtered_indices = unique_2q[mask]
            else:
                # Create a boolean mask for elements in unique_q that are not in unique_p
                mask = ~torch.isin(unique_q, unique_p)
                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]

            if args.prune_method == "random":
                filtered_indices = match_random_indices_by_magnitude(
                    subset[name].weight.data,
                    filtered_indices,
                    n_bins=getattr(args, "match_bins", 20),
                )


            weight_dim = subset[name].weight.data.shape[1]
            filtered_indices_rows = filtered_indices // weight_dim
            filtered_indices_cols = filtered_indices % weight_dim

            print("n_pruned_weights:", len(filtered_indices_rows))

            # assert (
            #         args.dump_score == False
            # )  # Only pruning from the saved score, won't save score again

            W_mask = torch.zeros_like(subset[name].weight.data) == 1
            W_mask[filtered_indices_rows, filtered_indices_cols] = (
                True  # prune weights that has relatively high safety while not in top utility scores
            )

            if args.alpha:
                if args.alpha == 'mean':
                    subset[name].weight.data[W_mask] = subset[name].weight.data[W_mask].mean()
                else:
                    subset[name].weight.data[W_mask] = subset[name].weight.data[W_mask] * args.alpha
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero

            if args.dump_mask:
                mask_save_path = f"masks/{args.model.split('/')[-1]}/{args.prune_method}/seed_{args.seed}/{args.prune_data}_not_{args.preserve_data}/{abs_indicator}/nsamples_{args.nsamples}"
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                target_file = os.path.join(
                    mask_save_path, f"Mask_layer_{i}_name_{name}_weight.pkl"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing Mask in layer {} and name {} with {} to the file".format(
                            i, name, args.prune_data
                        )
                    )
                    pickle.dump(W_mask, f)

            if args.dump_indices:
                indices = torch.nonzero(W_mask, as_tuple=False)
                scores_save_path = f"../indices/{args.model.split('/')[-1]}/{args.prune_method}_not_{args.preserve_data}/seed_{args.seed}/{args.prune_data}/{abs_indicator}/nsamples_{args.nsamples}"
                if not os.path.exists(scores_save_path):
                    os.makedirs(scores_save_path)
                target_file = os.path.join(
                    scores_save_path, f"indices_layer_{i}_name_{name}_weight.pt"
                )
                with open(target_file, "wb") as f:
                    print(
                        "Writing Indices in layer {} and name {} with {} to the file".format(
                            i, name, args.prune_data
                        )
                    )
                torch.save(indices, target_file)

            del W_metric_preserve
            del W_metric_prune
            gc.collect()

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    layer_sparsity = []
    component_data = defaultdict(dict)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            zeros = (W == 0).sum().item()
            numel = W.numel()

            count += zeros
            total_params += numel
            sub_count += zeros
            sub_params += numel

            component_sparsity = zeros / numel if numel > 0 else 0
            component_data[i][name] = component_sparsity

        layer_sp = float(sub_count) / sub_params if sub_params > 0 else 0
        layer_sparsity.append(layer_sp)
        print(f"layer {i} sparsity {layer_sp:.6f}")

    model.config.use_cache = use_cache
    overall = float(count) / total_params if total_params > 0 else 0

    return {
        'overall': overall,
        'layer_wise': layer_sparsity,
        'component_wise': dict(component_data)
    }

def log_pruning_distribution(sparsity_data, prune_data_name):
    """Create and log visualizations of pruning distribution to WandB.

    Args:
        sparsity_data: Dict from check_sparsity() with overall, layer_wise, component_wise
        prune_data_name: Name of pruning dataset (e.g., 'advbench_align_anti')
    """

    try:
        # Visualization 1: Layer-wise sparsity bar chart
        fig, ax = plt.subplots(figsize=(14, 6))
        layers = list(range(len(sparsity_data['layer_wise'])))
        ax.bar(layers, sparsity_data['layer_wise'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Sparsity', fontsize=12)
        ax.set_title(f'Layer-wise Sparsity Distribution - {prune_data_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(sparsity_data['layer_wise']) * 1.1 if sparsity_data['layer_wise'] else 1)
        wandb.log({f"pruning_distribution/layer_wise_bar": wandb.Image(fig)})
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to create layer-wise bar chart: {e}")

    try:
        # Visualization 2: Component heatmap (layers x components)
        components = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                      'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        heatmap_data = []
        for layer_idx in sorted(sparsity_data['component_wise'].keys()):
            row = [sparsity_data['component_wise'][layer_idx].get(comp, 0) for comp in components]
            heatmap_data.append(row)

        fig, ax = plt.subplots(figsize=(10, 16))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_yticks(range(len(heatmap_data)))
        ax.set_yticklabels([f'L{i}' for i in range(len(heatmap_data))], fontsize=8)
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels([c.split('.')[-1] for c in components], rotation=45, ha='right')
        plt.colorbar(im, ax=ax, label='Sparsity')
        ax.set_title(f'Component-wise Sparsity Heatmap - {prune_data_name}', fontsize=12)
        wandb.log({f"pruning_distribution/component_heatmap": wandb.Image(fig)})
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to create component heatmap: {e}")

    try:
        # Visualization 3: Cumulative distribution
        sorted_sparsity = sorted(sparsity_data['layer_wise'])
        cumulative = np.cumsum(sorted_sparsity) / np.sum(sorted_sparsity) if np.sum(sorted_sparsity) > 0 else np.zeros_like(sorted_sparsity)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(cumulative)), cumulative, linewidth=2, color='darkblue')
        ax.set_xlabel('Layers (sorted by sparsity)', fontsize=12)
        ax.set_ylabel('Cumulative proportion of pruned weights', fontsize=12)
        ax.set_title(f'Cumulative Pruning Distribution - {prune_data_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(cumulative)-1)
        ax.set_ylim(0, 1)
        wandb.log({f"pruning_distribution/cumulative": wandb.Image(fig)})
        plt.close()

    except Exception as e:
        print(f"Warning: Failed to create cumulative distribution: {e}")

    try:
        # Visualization 4: Layer data table
        layer_table = wandb.Table(
            columns=['Layer', 'Sparsity'],
            data=[[i, s] for i, s in enumerate(sparsity_data['layer_wise'])]
        )
        wandb.log({f"pruning_distribution/layer_table": layer_table})

    except Exception as e:
        print(f"Warning: Failed to create layer table: {e}")

def prune_from_indices(args, model, indices_to_prune=None):

    if indices_to_prune is None:
        with open(args.indices_path, "rb") as f:
            indices_to_prune = pickle.load(f)

    model.config.use_cache = False
    layers = model.model.layers

    print("Starting pruning from indices")

    for i in range(len(layers)):

        if args.layers is not None:
            if i not in args.layers:
                print(f"Skipping layer {i}")
                continue

        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"pruning layer {i} name {name}")

            W_mask = torch.zeros_like(subset[name].weight.data) == 1
            if len(indices_to_prune[i][name]) > 0:
                filtered_indices_rows, filtered_indices_cols = zip(*list(indices_to_prune[i][name]))
                W_mask[filtered_indices_rows, filtered_indices_cols] = (
                    True  # prune weights that has relatively high safety while not in top utility scores
                )
                print("n_pruned_weights:", len(filtered_indices_rows))
            else:
                print("n_pruned_weights:", 0)

            if args.alpha:
                if args.alpha == 'mean':
                    subset[name].weight.data[W_mask] = subset[name].weight.data[W_mask].mean()
                else:
                    subset[name].weight.data[W_mask] = subset[name].weight.data[W_mask] * args.alpha
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero

def find_intersection_indices(args):

    indices_dict = defaultdict(lambda: defaultdict(dict))
    layers = range(0, MODEL_TO_N_LAYERS[args.model])
    components = MODEL_TO_COMPONENTS[args.model]

    nsamples_preserve = args.nsamples_preserve if args.nsamples_preserve is not None else args.nsamples

    for prune_data in args.prune_data_list:
        for l in layers:
            for c in components:
                weight_path_safety = f"../src/scores/{args.model.split('/')[-1]}/attribution_score/seed_0/{prune_data}/no_abs/nsamples_{nsamples_preserve}/W_metric_layer_{l}_name_model.layers.{l}.{c}_weight.pkl"
                weight_path_utility = f"../src/scores/{args.model.split('/')[-1]}/attribution_score/seed_0/{args.preserve_data}/no_abs/nsamples_{args.nsamples}/W_metric_layer_{l}_name_model.layers.{l}.{c}_weight.pkl"
                with open(weight_path_safety, "rb") as f:
                    W_metric_prune = pickle.load(f)
                with open(weight_path_utility, "rb") as f:
                    W_metric_utility = pickle.load(f)

                if args.neg_prune:
                    W_metric_prune = -W_metric_prune
                    W_metric_utility = -W_metric_utility

                top_utility = int(args.p * W_metric_utility.shape[1] * W_metric_utility.shape[0])
                top_safety = int(args.q * W_metric_prune.shape[1] * W_metric_prune.shape[0])

                top_p_indices = torch.topk(W_metric_utility.flatten(), top_utility, largest=True)[1]
                top_q_indices = torch.topk(W_metric_prune.flatten(), top_safety, largest=True)[1]

                unique_p = torch.unique(top_p_indices)
                unique_q = torch.unique(top_q_indices)

                # Create a boolean mask for elements in unique_q that are not in unique_p
                mask = ~torch.isin(unique_q, unique_p)

                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]
                weight_dim = W_metric_utility.shape[1]  # subset[name].weight.data.shape[1]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim

                print(f"n_pruned_weights ({prune_data}, {l}, {c}):", len(filtered_indices_rows))

                indices_to_prune = list(zip(filtered_indices_rows, filtered_indices_cols))
                indices_dict[prune_data][l][c] = indices_to_prune

    indices_dict_clean = defaultdict(lambda: defaultdict(dict))
    for prune_data in indices_dict:
        for l in indices_dict[prune_data]:
            for c in indices_dict[prune_data][l]:
                indices_dict_clean[prune_data][l][c] = [(a.item(), b.item()) for a, b in indices_dict[prune_data][l][c]]
    indices_dict = indices_dict_clean

    indices_to_prune = defaultdict(dict)
    for l in layers:
        for c in components:
            sets_per_cat = [
                set(indices_dict[prune_data][l][c])
                for prune_data in args.prune_data_list
                if l in indices_dict[prune_data] and c in indices_dict[prune_data][l]
            ]
            if not sets_per_cat:  # skip if nothing
                continue

            intersection = set.intersection(*sets_per_cat)
            indices_to_prune[l][c] = intersection

    if args.dump_indices is not None:
        with open(args.dump_indices, "wb") as f:
            pickle.dump(indices_to_prune, f)
        print(f"Dumped indices to {args.dump_indices}")
    return indices_to_prune

def smart_load(path, device: torch.device | None = None):
    """
    Load a pickle that contains PyTorch tensors, mapping all storages
    to the given device.

    - If device is None: use cuda:0 if available, else cpu.
    - Works for both CPU-only and GPU machines.
    """

    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Intercept low-level storage loading used by pickled tensors
            if module == 'torch.storage' and name == '_load_from_bytes':
                # b is a bytes object; wrap it, then torch.load with map_location
                return lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=True)
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return DeviceUnpickler(f).load()


def match_random_indices_by_magnitude(
        weight,
        targeted_flat_indices,
        n_bins=20,
        subsample_for_edges=1_000_000,
        verbose=True,
):
    """
    Given a weight matrix and the flat indices a method selected to prune,
    return a DISJOINT set of random flat indices of identical size whose
    |W| histogram matches the targeted set's |W| histogram (per magnitude bin).

    Relies on the global RNG, so seed once (torch.manual_seed) before calling.
    """
    device = weight.device
    Wabs = weight.detach().abs().reshape(-1)
    numel = Wabs.numel()
    targeted_flat_indices = targeted_flat_indices.to(device)
    n_target = targeted_flat_indices.numel()
    if n_target == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # --- bin edges from a (sub)sample of the full magnitude distribution ---
    # subsample because torch.quantile chokes on very large tensors.
    if numel > subsample_for_edges:
        sample_idx = torch.randint(0, numel, (subsample_for_edges,), device=device)
        sample = Wabs[sample_idx]
    else:
        sample = Wabs
    qs = torch.linspace(0, 1, n_bins + 1, device=device)
    edges = torch.quantile(sample.float(), qs)
    inner_edges = edges[1:-1].contiguous()            # interior cut points
    bins = torch.bucketize(Wabs, inner_edges)         # each elem -> bin in [0, n_bins-1]

    # how many targeted weights fall in each bin
    need = torch.bincount(bins[targeted_flat_indices], minlength=n_bins)

    # candidate pool = everything NOT targeted (guarantees disjointness)
    is_targeted = torch.zeros(numel, dtype=torch.bool, device=device)
    is_targeted[targeted_flat_indices] = True
    cand_idx = (~is_targeted).nonzero(as_tuple=False).squeeze(1)
    cand_bins = bins[cand_idx]

    chosen = []
    leftover = []
    deficit = 0
    for b in range(n_bins):
        pool = cand_idx[cand_bins == b]
        k = int(need[b].item())
        if k <= 0:
            leftover.append(pool)
            continue
        if pool.numel() <= k:                          # not enough in this bin
            chosen.append(pool)
            deficit += k - pool.numel()
        else:
            perm = torch.randperm(pool.numel(), device=device)
            chosen.append(pool[perm[:k]])
            leftover.append(pool[perm[k:]])

    # fill any deficit from leftover candidates (nearest-bin spillover, simplified to global)
    if deficit > 0:
        leftover = torch.cat([p for p in leftover if p.numel() > 0]) \
            if leftover else torch.empty(0, dtype=torch.long, device=device)
        if leftover.numel() > 0:
            perm = torch.randperm(leftover.numel(), device=device)
            chosen.append(leftover[perm[:deficit]])

    random_indices = torch.cat(chosen) if chosen else \
        torch.empty(0, dtype=torch.long, device=device)

    if verbose:
        tgt_mean = Wabs[targeted_flat_indices].mean().item()
        rnd_mean = Wabs[random_indices].mean().item() if random_indices.numel() else 0.0
        print(f"    match: n_target={n_target} n_random={random_indices.numel()} "
              f"deficit={deficit} | mean|W| target={tgt_mean:.4e} random={rnd_mean:.4e}")

    return random_indices

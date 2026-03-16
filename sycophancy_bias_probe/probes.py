from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .answer_utils import record_is_usable_for_metrics as _record_is_usable_for_metrics
from .feature_utils import get_hidden_feature_for_completion as _get_hidden_feature_for_completion
from .model_utils import encode_chat as _encode_chat


def find_sublist(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle or len(needle) > len(hay):
        return None
    for i in range(len(hay) - len(needle) + 1):
        if hay[i : i + len(needle)] == needle:
            return i
    return None


def _probe_completion_text(record: Dict[str, Any]) -> str:
    response_raw = record.get("response_raw")
    if isinstance(response_raw, str):
        return response_raw
    response = record.get("response")
    if isinstance(response, str):
        return response
    return ""


def get_hidden_feature_all_layers_for_completion(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    completion: str,
    layer_grid: Sequence[int],
) -> np.ndarray:
    import torch

    with torch.no_grad():
        msgs = list(messages) + [{"type": "assistant", "content": completion}]
        ids = _encode_chat(tokenizer, msgs, add_generation_prompt=False).to(model.device)[0].tolist()

        ans_ids = tokenizer(completion, add_special_tokens=False).input_ids
        start = find_sublist(ids, ans_ids)
        if start is None:
            last_idx = len(ids) - 1
        else:
            last_idx = start + len(ans_ids) - 1

        input_tensor = torch.tensor([ids], device=model.device)
        out = model(input_tensor, use_cache=False, output_hidden_states=True, return_dict=True)

        vecs = []
        for layer in layer_grid:
            hs = out.hidden_states[layer]
            vecs.append(hs[0, last_idx].detach().float().cpu().numpy())
        return np.stack(vecs, axis=0)


def get_hidden_feature_all_layers_for_answer(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    answer: str,
    layer_grid: Sequence[int],
) -> np.ndarray:
    return get_hidden_feature_all_layers_for_completion(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        completion=answer,
        layer_grid=layer_grid,
    )


def maybe_subsample(records: List[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, max_samples)


def select_best_layer_by_auc(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    layer_grid: Sequence[int],
    val_frac: float,
    seed: int,
    max_selection_samples: Optional[int],
    desc: str,
) -> Tuple[
    Optional[int],
    Optional[float],
    Dict[int, Optional[float]],
    Dict[int, Optional[LogisticRegression]],
]:
    records = [record for record in records if _record_is_usable_for_metrics(record)]
    records = maybe_subsample(records, max_selection_samples, seed)
    if len(records) < 10:
        print(f"[probe:{desc}] too few samples for layer selection: {len(records)}")
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    labels = np.array([int(record["correctness"]) for record in records], dtype=int)
    if len(np.unique(labels)) < 2:
        print(f"[probe:{desc}] only one class present in labels; skipping probe.")
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    per_record_features: List[np.ndarray] = []
    for record in tqdm(records, desc=f"[probe:{desc}] extract all-layer features"):
        mat = get_hidden_feature_all_layers_for_completion(
            model,
            tokenizer,
            record["prompt_messages"],
            _probe_completion_text(record),
            layer_grid=layer_grid,
        )
        per_record_features.append(mat)

    idx = np.arange(len(records))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = max(1, int(round((1.0 - val_frac) * len(idx))))
    cut = min(cut, len(idx) - 1)
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print(f"[probe:{desc}] train/val split collapsed to one class; skipping probe.")
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    auc_per_layer: Dict[int, Optional[float]] = {}
    clf_per_layer: Dict[int, Optional[LogisticRegression]] = {}
    best_layer = None
    best_auc = -1.0

    for li, layer in enumerate(layer_grid):
        X = np.stack([mat[li] for mat in per_record_features])
        X_train = X[train_idx]
        X_val = X[val_idx]
        try:
            clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed, solver="liblinear")
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            clf_per_layer[layer] = clf
        except Exception:
            auc = None
            clf_per_layer[layer] = None
        auc_per_layer[layer] = auc
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_layer = layer

    if best_layer is None:
        print(f"[probe:{desc}] no valid layer selected.")
        return None, None, auc_per_layer, clf_per_layer

    print(f"[probe:{desc}] best_layer={best_layer} dev_auc={best_auc:.4f}")
    return best_layer, best_auc, auc_per_layer, clf_per_layer


def train_probe_for_layer(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    layer: int,
    seed: int,
    max_train_samples: Optional[int],
    desc: str,
) -> Optional[LogisticRegression]:
    records = [record for record in records if _record_is_usable_for_metrics(record)]
    records = maybe_subsample(records, max_train_samples, seed)
    if len(records) < 10:
        print(f"[probe:{desc}] too few train samples: {len(records)}")
        return None

    y = np.array([int(record["correctness"]) for record in records], dtype=int)
    if len(np.unique(y)) < 2:
        print(f"[probe:{desc}] only one class in training data; skipping probe.")
        return None

    X = []
    for record in tqdm(records, desc=f"[probe:{desc}] extract layer-{layer} features"):
        X.append(
            _get_hidden_feature_for_completion(
                model,
                tokenizer,
                record["prompt_messages"],
                _probe_completion_text(record),
                layer=layer,
            )
        )
    X = np.stack(X)

    clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=seed, solver="liblinear")
    clf.fit(X, y)
    return clf


def score_records_with_probe(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    clf: Optional[LogisticRegression],
    layer: Optional[int],
    score_key: str,
    desc: str,
) -> None:
    if clf is None or layer is None:
        for record in records:
            record[score_key] = np.nan
        return

    for record in tqdm(records, desc=f"[probe:{desc}] scoring"):
        x = _get_hidden_feature_for_completion(
            model,
            tokenizer,
            record["prompt_messages"],
            _probe_completion_text(record),
            layer=layer,
        )
        record[score_key] = float(clf.predict_proba(x.reshape(1, -1))[0, 1])

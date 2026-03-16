from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .correctness import record_is_usable_for_metrics as _record_is_usable_for_metrics
from .feature_utils import _assistant_text_last_token_index
from .feature_utils import get_hidden_feature_for_completion as _get_hidden_feature_for_completion
from .logging_utils import log_status, tqdm_desc
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
        last_idx = _assistant_text_last_token_index(tokenizer, ids, completion)

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
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    layer_grid: Sequence[int],
    seed: int,
    max_selection_samples: Optional[int],
    desc: str,
) -> Tuple[
    Optional[int],
    Optional[float],
    Dict[int, Optional[float]],
    Dict[int, Optional[LogisticRegression]],
]:
    train_records = [record for record in train_records if _record_is_usable_for_metrics(record)]
    val_records = [record for record in val_records if _record_is_usable_for_metrics(record)]
    train_records = maybe_subsample(train_records, max_selection_samples, seed)
    val_records = maybe_subsample(val_records, max_selection_samples, seed + 1)
    log_status(
        "probes.py",
        f"layer selection for {desc}: train_records={len(train_records)} "
        f"val_records={len(val_records)} layers={len(layer_grid)}",
    )

    if len(train_records) < 2 or len(val_records) < 2:
        log_status(
            "probes.py",
            f"skipping layer selection for {desc}: too few samples "
            f"train={len(train_records)} val={len(val_records)}",
        )
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    y_train = np.array([int(record["correctness"]) for record in train_records], dtype=int)
    y_val = np.array([int(record["correctness"]) for record in val_records], dtype=int)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        log_status(
            "probes.py",
            f"skipping layer selection for {desc}: train or val split has only one class",
        )
        return (
            None,
            None,
            {layer: None for layer in layer_grid},
            {layer: None for layer in layer_grid},
        )

    train_features: List[np.ndarray] = []
    for record in tqdm(
        train_records,
        desc=tqdm_desc("probes.py", f"{desc} train all-layer features"),
        unit="record",
    ):
        train_features.append(
            get_hidden_feature_all_layers_for_completion(
                model,
                tokenizer,
                record["prompt_messages"],
                _probe_completion_text(record),
                layer_grid=layer_grid,
            )
        )

    val_features: List[np.ndarray] = []
    for record in tqdm(
        val_records,
        desc=tqdm_desc("probes.py", f"{desc} val all-layer features"),
        unit="record",
    ):
        val_features.append(
            get_hidden_feature_all_layers_for_completion(
                model,
                tokenizer,
                record["prompt_messages"],
                _probe_completion_text(record),
                layer_grid=layer_grid,
            )
        )

    auc_per_layer: Dict[int, Optional[float]] = {}
    clf_per_layer: Dict[int, Optional[LogisticRegression]] = {}
    best_layer = None
    best_auc = -1.0

    for li, layer in enumerate(
        tqdm(layer_grid, desc=tqdm_desc("probes.py", f"{desc} layer selection"), unit="layer")
    ):
        X_train = np.stack([mat[li] for mat in train_features])
        X_val = np.stack([mat[li] for mat in val_features])
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
        log_status("probes.py", f"no valid layer selected for {desc}")
        return None, None, auc_per_layer, clf_per_layer

    log_status("probes.py", f"selected best layer for {desc}: layer={best_layer} dev_auc={best_auc:.4f}")
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
    log_status(
        "probes.py",
        f"training probe for {desc}: records={len(records)} layer={layer}",
    )
    if len(records) < 2:
        log_status("probes.py", f"skipping training for {desc}: too few train samples={len(records)}")
        return None

    y = np.array([int(record["correctness"]) for record in records], dtype=int)
    if len(np.unique(y)) < 2:
        log_status("probes.py", f"skipping training for {desc}: only one class in training data")
        return None

    X = []
    for record in tqdm(
        records,
        desc=tqdm_desc("probes.py", f"{desc} layer-{layer} features"),
        unit="record",
    ):
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

    log_status(
        "probes.py",
        f"scoring records for {desc}: records={len(records)} layer={layer} score_key={score_key}",
    )
    for record in tqdm(
        records,
        desc=tqdm_desc("probes.py", f"{desc} scoring"),
        unit="record",
    ):
        x = _get_hidden_feature_for_completion(
            model,
            tokenizer,
            record["prompt_messages"],
            _probe_completion_text(record),
            layer=layer,
        )
        record[score_key] = float(clf.predict_proba(x.reshape(1, -1))[0, 1])

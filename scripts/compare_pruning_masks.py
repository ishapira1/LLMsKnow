#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Mask must be a module-to-index mapping: {path}")
    result = {}
    for name, indices in value.items():
        tensor = torch.as_tensor(indices, dtype=torch.int64).reshape(-1).unique(sorted=True)
        result[str(name)] = tensor
    return result


def _parse_masks(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --mask {value!r}; expected SEED=PATH")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate seed {seed}")
        result[seed] = Path(path_text).expanduser().resolve()
    if len(result) < 2:
        raise ValueError("At least two --mask SEED=PATH values are required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute exact pairwise sparse-mask overlap.")
    parser.add_argument("--mask", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = _parse_masks(args.mask)
    masks = {seed: _load(path) for seed, path in paths.items()}
    pairs = []
    seeds = sorted(masks)
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            left, right = masks[left_seed], masks[right_seed]
            modules = sorted(set(left) | set(right))
            module_rows = []
            intersection_total = 0
            union_total = 0
            for module in modules:
                left_indices = left.get(module, torch.empty(0, dtype=torch.int64))
                right_indices = right.get(module, torch.empty(0, dtype=torch.int64))
                intersection = int(torch.isin(left_indices, right_indices).sum().item())
                union = int(left_indices.numel() + right_indices.numel() - intersection)
                intersection_total += intersection
                union_total += union
                module_rows.append(
                    {
                        "module": module,
                        "left_count": int(left_indices.numel()),
                        "right_count": int(right_indices.numel()),
                        "intersection": intersection,
                        "union": union,
                        "jaccard": intersection / union if union else 1.0,
                    }
                )
            pairs.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "left_count": sum(int(value.numel()) for value in left.values()),
                    "right_count": sum(int(value.numel()) for value in right.values()),
                    "intersection": intersection_total,
                    "union": union_total,
                    "jaccard": intersection_total / union_total if union_total else 1.0,
                    "by_module": module_rows,
                }
            )
    payload = {
        "schema_version": 1,
        "masks": {
            str(seed): {"path": str(paths[seed]), "sha256": _sha256(paths[seed])}
            for seed in seeds
        },
        "pairs": pairs,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "pairs": len(pairs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

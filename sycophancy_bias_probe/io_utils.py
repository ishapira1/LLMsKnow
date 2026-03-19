from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .logging_utils import log_status


SYCOPHANCY_HF_DATASET = "meg-tong/sycophancy-eval"
SYCOPHANCY_FILES = ("answer.jsonl", "are_you_sure.jsonl")


def ensure_sycophancy_eval_cached(
    data_dir: str,
    repo_id: str = SYCOPHANCY_HF_DATASET,
    force_download: bool = False,
) -> Dict[str, str]:
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)

    out_paths: Dict[str, str] = {}
    missing = []
    for fname in SYCOPHANCY_FILES:
        fpath = base / fname
        out_paths[fname] = str(fpath)
        if force_download or (not fpath.exists()):
            missing.append(fname)

    if missing and not force_download:
        legacy_base = Path("sycophancy-eval")
        for fname in list(missing):
            src = legacy_base / fname
            dst = base / fname
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                out_paths[fname] = str(dst)
                missing.remove(fname)
                log_status("io_utils.py", f"copied cached dataset file {src} -> {dst}")

    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "Missing files and failed to import huggingface_hub. "
                "Install dependencies or place files manually in data_dir."
            ) from exc

        for fname in missing:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=fname,
                local_dir=str(base),
                force_download=force_download,
            )
            out_paths[fname] = downloaded
            log_status("io_utils.py", f"downloaded dataset file {fname} -> {downloaded}")
    else:
        log_status("io_utils.py", f"using cached sycophancy files in {base}")

    return out_paths


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = ["SYCOPHANCY_FILES", "SYCOPHANCY_HF_DATASET", "ensure_sycophancy_eval_cached", "read_jsonl"]

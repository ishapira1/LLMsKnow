#!/usr/bin/env python3
"""Insert verified random_baseline outputs into a clean paper worktree."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import random_baseline as rb


INSERTION_MARKER = r"\subsection{What Weights Made the Difference?}"
SECTION_LABEL = r"\label{sec:random-mask-baselines}"


def integrate(paper_tex: Path, generated_tex: Path, llama_plot: Path,
              qwen_plot: Path, plot_dir: Path, audit_output: Path) -> dict[str, object]:
    for path in (paper_tex, generated_tex, llama_plot, qwen_plot):
        if not path.is_file():
            raise FileNotFoundError(path)
    original = paper_tex.read_text(encoding="utf-8")
    generated = generated_tex.read_text(encoding="utf-8").strip() + "\n"
    if SECTION_LABEL in original:
        raise ValueError("Random-Mask Baselines subsection already exists")
    if generated.count(SECTION_LABEL) != 1:
        raise ValueError("Generated subsection must contain its section label exactly once")
    if original.count(INSERTION_MARKER) != 1:
        raise ValueError("Paper insertion marker must occur exactly once")
    integrated = original.replace(INSERTION_MARKER,
                                  generated + "\n" + INSERTION_MARKER, 1)
    rb.atomic_text(paper_tex, integrated)
    plot_dir.mkdir(parents=True, exist_ok=True)
    llama_destination = plot_dir / "random_baseline_llama_pareto.pdf"
    qwen_destination = plot_dir / "random_baseline_qwen_pareto.pdf"
    shutil.copy2(llama_plot, llama_destination)
    shutil.copy2(qwen_plot, qwen_destination)
    payload: dict[str, object] = {
        "status": "complete",
        "paper_tex": str(paper_tex.resolve()),
        "paper_tex_before_sha256": rb.sha256_text(original),
        "paper_tex_after_sha256": rb.sha256_file(paper_tex),
        "generated_tex": str(generated_tex.resolve()),
        "generated_tex_sha256": rb.sha256_file(generated_tex),
        "llama_plot_sha256": rb.sha256_file(llama_destination),
        "qwen_plot_sha256": rb.sha256_file(qwen_destination),
        "insertion_marker": INSERTION_MARKER,
        "section_label": SECTION_LABEL,
        "completed_at": rb.utc_now(),
    }
    rb.atomic_json(audit_output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-tex", type=Path, required=True)
    parser.add_argument("--generated-tex", type=Path, required=True)
    parser.add_argument("--llama-plot", type=Path, required=True)
    parser.add_argument("--qwen-plot", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    payload = integrate(args.paper_tex, args.generated_tex, args.llama_plot,
                        args.qwen_plot, args.plot_dir, args.audit_output)
    print(rb.canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

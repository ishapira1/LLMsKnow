#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot raw/chat rendering, token IDs, and exact scored response spans."
    )
    parser.add_argument("--harm-repo", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--hf-cache-dir")
    args = parser.parse_args()

    harm_src = args.harm_repo.expanduser().resolve() / "src"
    sys.path.insert(0, str(harm_src))
    from paper_pruning import encode_completion, load_manifest  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    manifest = args.manifest.expanduser().resolve()
    rows = load_manifest(
        manifest,
        nsamples=args.max_rows,
        expected_model=args.model,
        expected_revision=args.revision,
        expected_tokenizer_revision=args.revision,
    )
    tokenizer_kwargs = {"revision": args.revision, "use_fast": True}
    if args.hf_cache_dir:
        tokenizer_kwargs["cache_dir"] = args.hf_cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

    snapshots = []
    for row in rows:
        for score_format in ("raw", "chat"):
            encoded = encode_completion(row, tokenizer, score_format)
            ids = encoded.input_ids.tolist()
            scored_special_ids = [
                token_id
                for token_id in ids[encoded.response_start :]
                if token_id in set(tokenizer.all_special_ids)
            ]
            if scored_special_ids:
                raise RuntimeError(
                    f"manifest line {row.get('_manifest_line')} format={score_format} "
                    f"scores special token IDs {scored_special_ids}"
                )
            snapshots.append(
                {
                    "example_id": row.get("example_id"),
                    "manifest_line": row.get("_manifest_line"),
                    "score_format": score_format,
                    "rendered_prompt": encoded.rendered_prompt,
                    "target_text": encoded.target_text,
                    "input_ids": ids,
                    "response_start": encoded.response_start,
                    "prompt_token_ids": ids[: encoded.response_start],
                    "response_token_ids": ids[encoded.response_start :],
                    "response_tokens": tokenizer.convert_ids_to_tokens(
                        ids[encoded.response_start :]
                    ),
                    "decoded_response": tokenizer.decode(
                        ids[encoded.response_start :], skip_special_tokens=False
                    ),
                    "special_token_ids_scored": scored_special_ids,
                }
            )
    payload = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer_revision": args.revision,
        "tokenizer_class": tokenizer.__class__.__name__,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "snapshots": snapshots,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "snapshots": len(snapshots)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

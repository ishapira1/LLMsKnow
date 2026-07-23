#!/usr/bin/env python3
"""Validate the frozen Python/package/CUDA contract for intervention jobs."""

from __future__ import print_function

import argparse
import json
import platform
import sys


EXPECTED_VERSIONS = {
    "accelerate": "0.27.0",
    "datasets": "2.21.0",
    "matplotlib": "3.9.2",
    "numpy": "1.26.4",
    "pandas": "1.5.1",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "seaborn": "0.13.2",
    "tokenizers": "0.19.1",
    "torch": "2.2.0",
    "tqdm": "4.66.5",
    "transformers": "4.42.3",
    "wandb": "0.15.12",
}


def _distribution_version(name):
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python 3.8 fallback for a clear failure
        import importlib_metadata as metadata
    return metadata.version(name)


def _base_version(value):
    return str(value).split("+", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    errors = []
    if sys.version_info < (3, 10):
        errors.append(
            "Python >=3.10 is required; found {}".format(platform.python_version())
        )

    actual_versions = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = _distribution_version(distribution)
        except Exception as error:  # noqa: BLE001 - diagnostic should collect all failures
            actual = None
            errors.append("{} is unavailable: {}".format(distribution, error))
        actual_versions[distribution] = actual
        if actual is not None and _base_version(actual) != expected:
            errors.append(
                "{} must be {}, found {}".format(distribution, expected, actual)
            )

    cuda_available = False
    cuda_version = None
    cudnn_version = None
    device_name = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            torch.ones(1, device="cuda").add_(1.0)
    except Exception as error:  # noqa: BLE001
        errors.append("PyTorch runtime check failed: {}".format(error))
    if args.require_cuda and not cuda_available:
        errors.append("CUDA is required for this stage, but torch.cuda.is_available() is false")

    payload = {
        "ok": not errors,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "expected_versions": EXPECTED_VERSIONS,
        "actual_versions": actual_versions,
        "cuda_required": bool(args.require_cuda),
        "cuda_available": cuda_available,
        "torch_cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "device_name": device_name,
        "errors": errors,
    }
    print("[runtime-contract] {}".format(json.dumps(payload, sort_keys=True)))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

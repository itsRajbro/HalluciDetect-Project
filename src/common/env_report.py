"""Print / save the exact software environment (for pinning requirements.txt).

Run on the machine where the Llama benchmark worked, and ideally on all three
teammates' machines; compare the outputs.

    python -m src.common.env_report            # prints + writes results/env_report_<host>.json
"""
from __future__ import annotations

import json
import platform
import socket
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = [
    "torch", "transformers", "bitsandbytes", "accelerate", "huggingface_hub",
    "tokenizers", "safetensors", "numpy", "pandas", "scipy", "scikit-learn",
    "datasets", "psutil", "pytest",
]


def collect() -> dict:
    pkgs = {}
    for name in PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    info = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "cuda": None,
    }
    try:
        import torch
        info["cuda"] = {
            "torch_cuda_build": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
    except Exception as e:  # torch missing or broken
        info["cuda"] = {"error": str(e)}
    return info


def main() -> None:
    info = collect()
    print(json.dumps(info, indent=2))
    out = Path(__file__).resolve().parents[2] / "results" / f"env_report_{info['host']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nsaved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

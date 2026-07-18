from __future__ import annotations

import json
import subprocess
import sys

import fastapi
import torch
import torchvision
import transformers
import uvicorn


def main() -> int:
    report = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "fastapi": fastapi.__version__,
        "uvicorn": uvicorn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["torch"] != "2.11.0+cu128":
        raise RuntimeError("Stage158 requires exact torch 2.11.0+cu128")
    if report["torchvision"] != "0.26.0+cu128":
        raise RuntimeError("Stage158 requires exact torchvision 0.26.0+cu128")
    if report["transformers"] != "5.13.1":
        raise RuntimeError("Stage158 requires exact transformers 5.13.1")
    if report["cuda_available"] is not True:
        raise RuntimeError("Stage158 requires CUDA")
    if report["cuda_version"] != "12.8":
        raise RuntimeError("Stage158 requires CUDA 12.8")
    if report["capability"] != [12, 0]:
        raise RuntimeError("Stage158 requires the confirmed compute capability 12.0")
    return subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""Device choice: CUDA when available, else CPU. Never MPS (local runs are CPU only)."""

from __future__ import annotations


def get_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return "cpu"

"""Print GPU and library versions and check that 4-bit loading works.

Every Kaggle notebook runs this first:  python scripts/gpu_info.py
"""

from __future__ import annotations

import importlib
import platform
import sys

TINY_LM = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def version(mod: str) -> str:
    try:
        return getattr(importlib.import_module(mod), "__version__", "?")
    except Exception as e:  # noqa: BLE001 - report anything
        return f"not available ({type(e).__name__})"


def main() -> None:
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    for mod in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets"):
        print(f"{mod:<13} {version(mod)}")
    try:
        import torch
    except ImportError:
        print("torch not installed")
        return
    print(f"CUDA available: {torch.cuda.is_available()}  (torch CUDA {torch.version.cuda})")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, "
              f"compute capability {p.major}.{p.minor} (bf16 needs >= 8.0)")
    if not torch.cuda.is_available():
        print("4-bit check: skipped (no CUDA)")
        return
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(TINY_LM, quantization_config=q, device_map={"": 0})
        with torch.inference_mode():
            out = m(torch.tensor([[1, 2, 3, 4]], device="cuda")).logits
        print(f"4-bit check: OK (NF4, fp16 compute, logits finite={bool(torch.isfinite(out).all())})")
    except Exception as e:  # noqa: BLE001
        print(f"4-bit check: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()

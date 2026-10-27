import os
from pathlib import Path

import pytest

FIXTURE_MODELS = ("cross-encoder/nli-deberta-v3-xsmall", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


def _models_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    return all(isinstance(try_to_load_from_cache(m, "config.json"), str) for m in FIXTURE_MODELS)


# Tests only use tiny cached models: skip the Hub's network checks (they cost ~20 s and fail
# on a flaky connection). Set LIARS_TEST_ONLINE=1 to allow downloads on a fresh machine.
if not os.environ.get("LIARS_TEST_ONLINE") and _models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.jsonl"


@pytest.fixture
def tiny_path() -> Path:
    return FIXTURE


@pytest.fixture
def tiny():
    from liars.schema import load_jsonl

    return load_jsonl(FIXTURE)

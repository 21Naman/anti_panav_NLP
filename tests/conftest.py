from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.jsonl"


@pytest.fixture
def tiny_path() -> Path:
    return FIXTURE


@pytest.fixture
def tiny():
    from liars.schema import load_jsonl

    return load_jsonl(FIXTURE)

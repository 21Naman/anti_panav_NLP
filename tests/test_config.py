import random

import numpy as np
import pytest

from liars.config import apply_overrides, config_hash, load_config, parse_value, seed_everything


def test_load_and_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\ntrain:\n  lr: 2e-5\n  epochs: 2\nmodel:\n  name: tiny\n")
    cfg = load_config(p, ["train.epochs=3", "model.name=big", "new.deep.key=true"])
    assert cfg["train"]["lr"] == pytest.approx(2e-5)  # plain PyYAML would give a string
    assert cfg["train"]["epochs"] == 3
    assert cfg["model"]["name"] == "big"
    assert cfg["new"]["deep"]["key"] is True


@pytest.mark.parametrize(
    "text, value",
    [("3", 3), ("0.5", 0.5), ("1e-5", 1e-5), ("true", True), ("null", None),
     ("[1, 2]", [1, 2]), ("hello", "hello"), ("a=b", "a=b")],
)
def test_parse_value(text, value):
    assert parse_value(text) == value


def test_bad_overrides():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals"])
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_value_with_equals_sign():
    cfg = apply_overrides({}, ["note=x=y"])
    assert cfg["note"] == "x=y"


def test_config_hash_stable():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_seed_everything_reproducible():
    seed_everything(7)
    a = (random.random(), np.random.rand())
    seed_everything(7)
    b = (random.random(), np.random.rand())
    assert a == b

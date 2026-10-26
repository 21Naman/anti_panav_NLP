"""YAML configs with `--set a.b=value` overrides, config hashing and seeding."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

HASH_LEN = 10


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads `1e-5` as a float (plain PyYAML reads it as a string)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def parse_value(text: str) -> Any:
    """Parse an override value the same way YAML would (ints, floats, bools, lists...).
    Mappings and lists only when written as {...} / [...]: plain YAML would turn
    `Answer:` into {"Answer": None}."""
    value = yaml.load(text, Loader=_Loader)
    if isinstance(value, (dict, list)) and not text.strip().startswith(("{", "[")):
        return text
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply `a.b.c=value` overrides in place. Missing intermediate keys are created."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if not all(parts):
            raise ValueError(f"bad override key {key!r}")
        node = cfg
        for p in parts[:-1]:
            if p not in node or node[p] is None:
                node[p] = {}
            if not isinstance(node[p], dict):
                raise ValueError(f"override {key!r}: {p!r} is not a mapping")
            node = node[p]
        node[parts[-1]] = parse_value(raw)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_Loader) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping")
    return apply_overrides(cfg, overrides)


def config_hash(cfg: dict) -> str:
    """Short stable hash of a config (key order does not matter)."""
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:HASH_LEN]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch and transformers."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if importlib.util.find_spec("transformers") is not None:
        from transformers import set_seed

        set_seed(seed)

"""Branches = candidate methods the loop may run, each with a search space (configs/loop.yaml).

A branch names a training config, the overrides always applied to it, a cost estimate, a
prior (high/med/low, from docs/KNOWLEDGE.md) and a search space: knob -> ordered list of
values. The planner mutates ONE knob at a time to a neighbouring value of the branch's best
run, so every result says what that one change did.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from liars.config import apply_overrides, load_config

PRIOR_RANK = {"high": 0, "med": 1, "low": 2}


@dataclass
class Branch:
    name: str
    family: str
    config: str
    base: list[str] = field(default_factory=list)
    est_gpu_h: float = 1.0
    prior: str = "med"
    space: dict[str, list] = field(default_factory=dict)
    hypothesis: str = ""
    sources: list[str] = field(default_factory=list)
    enabled: bool = True


def load_branches(cfg: dict) -> dict[str, Branch]:
    out = {}
    for name, b in (cfg.get("branches") or {}).items():
        br = Branch(name=name, **b)
        if br.prior not in PRIOR_RANK:
            raise ValueError(f"branch {name}: prior must be high | med | low, got {br.prior!r}")
        out[name] = br
    return out


def signature(branch: str, overrides: list[str]) -> str:
    """Order-insensitive identity of (branch, overrides); later duplicates of a key win."""
    kv: dict[str, str] = {}
    for o in overrides:
        k, _, v = o.partition("=")
        kv[k.strip()] = v.strip()
    blob = json.dumps([branch, sorted(kv.items())])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def as_dict(overrides: list[str]) -> dict[str, str]:
    kv: dict[str, str] = {}
    for o in overrides:
        k, _, v = o.partition("=")
        kv[k.strip()] = v.strip()
    return kv


def merged(overrides: list[str]) -> list[str]:
    """Collapse repeated keys (last wins), stable order."""
    return [f"{k}={v}" for k, v in as_dict(overrides).items()]


def flat_keys(d: dict, prefix: str = "") -> set[str]:
    out = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.add(key)
        if isinstance(v, dict):
            out |= flat_keys(v, key + ".")
    return out


def resolved_config(branch: Branch, overrides: list[str]) -> dict:
    return apply_overrides(load_config(branch.config), merged(branch.base + overrides))


def check_branch(branch: Branch, root: Path = Path(".")) -> list[str]:
    """Problems with a branch definition: missing config, knobs the config does not have."""
    errs = []
    path = root / branch.config
    if not path.exists():
        return [f"{branch.name}: config {branch.config} not found"]
    cfg = apply_overrides(load_config(path), merged(branch.base))
    keys = flat_keys(cfg)
    for k in list(branch.space) + [o.partition("=")[0] for o in branch.base]:
        if k not in keys and k != "seed":
            errs.append(f"{branch.name}: knob {k!r} is not in {branch.config} (typo?)")
    for k, vals in branch.space.items():
        if not isinstance(vals, list) or len(vals) < 2:
            errs.append(f"{branch.name}: space.{k} needs a list of >= 2 values")
            continue
        if k in keys:
            node = cfg
            for part in k.split("."):
                node = node[part]
            if str(node) != str(vals[0]):
                errs.append(f"{branch.name}: space.{k} must start with the config value {node!r} "
                            f"(got {vals[0]!r}); the baseline run is that first value")
    return errs

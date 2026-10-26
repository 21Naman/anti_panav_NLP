"""Model registry. Heavy models are imported lazily."""

from __future__ import annotations

import importlib

from liars.models.base import Model

REGISTRY: dict[str, str] = {
    "majority": "liars.models.baselines:MajorityModel",
    "tfidf_lr": "liars.models.baselines:TfidfLRModel",
    "nli_zeroshot": "liars.models.nli_zeroshot:NLIZeroShotModel",
}


def get_model(name: str) -> Model:
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)()

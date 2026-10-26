"""Stratified folds. Every training run uses the fold column written here."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from liars.schema import Example


def _strata(examples: Sequence[Example]) -> list[str]:
    return [f"{ex.label}|{ex.meta.get('source', '')}" for ex in examples]


def assign_folds(examples: Sequence[Example], k: int = 5, seed: int = 42) -> dict[str, int]:
    """id -> fold, stratified by label (and meta.source when present)."""
    if any(ex.label is None for ex in examples):
        raise ValueError("folds need labelled examples")
    strata = _strata(examples)
    # Rare strata (fewer than k members) fall back to label-only stratification.
    counts = pd.Series(strata).value_counts()
    if (counts < k).any():
        strata = [str(ex.label) for ex in examples]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: dict[str, int] = {}
    for f, (_, idx) in enumerate(skf.split(list(range(len(examples))), strata)):
        for i in idx:
            folds[examples[i].id] = f
    return folds


def folds_path_for(train_path: str | Path, canonical_dir: str | Path) -> Path:
    return Path(canonical_dir) / f"folds_{Path(train_path).stem}.csv"


def save_folds(folds: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(folds), "fold": list(folds.values())}).to_csv(path, index=False)


def load_folds(path: str | Path) -> dict[str, int]:
    df = pd.read_csv(path, dtype={"id": str})
    return dict(zip(df["id"], df["fold"].astype(int)))


def inner_split(examples: Sequence[Example], frac: float, seed: int
                ) -> tuple[list[Example], list[Example]]:
    """Small stratified validation split carved out of the TRAINING folds."""
    labels = [ex.label for ex in examples]
    n_val = max(len(set(labels)), int(round(len(examples) * frac)))
    try:
        tr, va = train_test_split(list(examples), test_size=n_val, random_state=seed,
                                  stratify=labels)
    except ValueError:  # too few examples per class to stratify (debug data)
        tr, va = train_test_split(list(examples), test_size=n_val, random_state=seed)
    return tr, va

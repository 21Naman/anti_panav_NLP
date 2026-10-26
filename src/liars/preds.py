"""Prediction files: parquet with id, p_A..p_D (rows sum to 1). OOF files add fold, label."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from liars.schema import SPEAKERS

PROB_COLS = [f"p_{s}" for s in SPEAKERS]
SUM_TOL = 1e-4


def make_preds(ids: Sequence[str], probs: np.ndarray) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(ids), len(SPEAKERS)):
        raise ValueError(f"probs shape {probs.shape}, expected ({len(ids)}, {len(SPEAKERS)})")
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    df = pd.DataFrame(probs, columns=PROB_COLS)
    df.insert(0, "id", list(ids))
    return df


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def check_preds(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in ["id", *PROB_COLS] if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"]
    if df["id"].duplicated().any():
        errors.append("duplicate ids")
    p = df[PROB_COLS].to_numpy()
    if np.isnan(p).any():
        errors.append("NaN probabilities")
    elif (p < 0).any():
        errors.append("negative probabilities")
    elif not np.allclose(p.sum(axis=1), 1.0, atol=SUM_TOL):
        errors.append("rows do not sum to 1")
    return errors


def argmax_letter(df: pd.DataFrame) -> pd.Series:
    idx = df[PROB_COLS].to_numpy().argmax(axis=1)
    return pd.Series([SPEAKERS[i] for i in idx], index=df.index)


def load_preds(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    errors = check_preds(df)
    if errors:
        raise ValueError(f"{path}: {errors}")
    return df


def save_preds(df: pd.DataFrame, path: str | Path) -> None:
    errors = check_preds(df)
    if errors:
        raise ValueError(f"refusing to save {path}: {errors}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def accuracy(df: pd.DataFrame, labels: pd.Series) -> float:
    """labels: Series indexed like df (letters)."""
    return float((argmax_letter(df).to_numpy() == np.asarray(labels)).mean())

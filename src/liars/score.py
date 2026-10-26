"""`liars score`: accuracy of a prediction file against known labels.

Labels come from a CSV (id,label[,lie_type,difficulty]) such as bench test_labels.csv,
or from a labelled canonical JSONL (lie_type / difficulty read from meta).
Never used on competition test data (no labels exist there).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from liars.preds import PROB_COLS, argmax_letter
from liars.schema import load_jsonl

BREAKDOWNS = ("lie_type", "difficulty", "label")


def load_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [{"id": ex.id, "label": ex.label, "lie_type": ex.meta.get("lie_type"),
                 "difficulty": ex.meta.get("difficulty")} for ex in load_jsonl(path)]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, dtype={"id": str})
    if df["label"].isna().any():
        raise ValueError(f"{path} has rows without labels")
    return df


def score(preds: pd.DataFrame, labels: pd.DataFrame) -> dict:
    preds = preds[["id", *PROB_COLS]]  # OOF files also carry fold/label; labels come from `labels`
    df = preds.assign(pred=argmax_letter(preds)).merge(labels, on="id", how="inner")
    df["correct"] = df["pred"] == df["label"]
    out = {
        "n": len(df),
        "n_preds": len(preds),
        "n_labels": len(labels),
        "accuracy": float(df["correct"].mean()) if len(df) else float("nan"),
        "by": {},
    }
    for col in BREAKDOWNS:
        if col in df.columns and df[col].notna().any():
            g = df.groupby(col)["correct"].agg(n="size", acc="mean")
            out["by"][col] = g
    return out


def format_score(res: dict) -> str:
    lines = [f"accuracy {res['accuracy']:.4f} on {res['n']} rows "
             f"(preds {res['n_preds']}, labels {res['n_labels']})"]
    for col, g in res["by"].items():
        lines.append(f"by {col}:")
        for key, row in g.iterrows():
            lines.append(f"  {str(key):<16} n={int(row['n']):>5}  acc={row['acc']:.4f}")
    return "\n".join(lines)

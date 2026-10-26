"""Dataset summary statistics for `liars stats`."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from liars.schema import SPEAKERS, Example

PERCENTILES = (50, 90, 99)
NULL_LABEL = "null"


def describe(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    out = {"min": arr.min(), "mean": arr.mean()}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(arr, p))
    out["max"] = arr.max()
    return {k: float(v) for k, v in out.items()}


def compute_stats(examples: Sequence[Example]) -> dict:
    labels = Counter(NULL_LABEL if ex.label is None else ex.label for ex in examples)
    lie_types = Counter(ex.meta.get("lie_type") for ex in examples if ex.meta.get("lie_type"))
    return {
        "n": len(examples),
        "labels": dict(labels),
        "lie_types": dict(lie_types),
        "messages_per_chat": describe([len(ex.messages) for ex in examples]),
        "facts_per_chat": describe([len(ex.evidence) for ex in examples]),
        "chat_chars": describe([sum(len(m.text) for m in ex.messages) for ex in examples]),
        "total_chars": describe(
            [
                sum(len(m.text) for m in ex.messages) + sum(len(f) for f in ex.evidence)
                for ex in examples
            ]
        ),
    }


def format_stats(stats: dict) -> str:
    n = stats["n"]
    lines = [f"examples: {n}"]

    order = [s for s in SPEAKERS if s in stats["labels"]]
    order += sorted(k for k in stats["labels"] if k not in SPEAKERS)
    parts = [f"{k} {stats['labels'][k]} ({100 * stats['labels'][k] / n:.1f}%)" for k in order]
    lines.append("labels: " + "  ".join(parts))

    if stats["lie_types"]:
        lt = sorted(stats["lie_types"].items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("lie_types: " + "  ".join(f"{k} {v}" for k, v in lt))

    rows = [
        ("messages/chat", stats["messages_per_chat"]),
        ("facts/chat", stats["facts_per_chat"]),
        ("chat chars", stats["chat_chars"]),
        ("total chars", stats["total_chars"]),
    ]
    cols = ["min", "mean"] + [f"p{p}" for p in PERCENTILES] + ["max"]
    lines.append(f"{'':<14}" + "".join(f"{c:>9}" for c in cols))
    for name, d in rows:
        if d:
            lines.append(f"{name:<14}" + "".join(f"{d[c]:>9.1f}" for c in cols))
    return "\n".join(lines)

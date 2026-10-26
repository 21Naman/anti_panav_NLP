"""Synthetic generator output -> canonical JSONL (+ optional hidden-label CSV)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Sequence

from liars.schema import Example, save_jsonl
from liars.synth.programmatic import GenRecord

# Meta keys that would give the answer away (or are not available on real test data).
HIDDEN_META_KEYS = ("lie_type", "lie_msg_idx", "difficulty", "debug")
LABEL_COLUMNS = ["id", "label", "lie_type", "difficulty", "lie_msg_idx"]


def to_examples(records: Sequence[GenRecord]) -> list[Example]:
    return [r.example for r in records]


def hide_labels(examples: Sequence[Example]) -> tuple[list[Example], list[dict]]:
    """Return copies with label=None and answer-revealing meta removed, plus label rows."""
    hidden, rows = [], []
    for ex in examples:
        rows.append({
            "id": ex.id,
            "label": ex.label,
            "lie_type": ex.meta.get("lie_type", ""),
            "difficulty": ex.meta.get("difficulty", ""),
            "lie_msg_idx": json.dumps(ex.meta.get("lie_msg_idx", [])),
        })
        meta = {k: v for k, v in ex.meta.items() if k not in HIDDEN_META_KEYS}
        hidden.append(Example(ex.id, list(ex.evidence), list(ex.messages), list(ex.speakers),
                              None, meta))
    return hidden, rows


def write_synthetic(examples: Sequence[Example], out: str | Path,
                    labels_csv: str | Path | None = None) -> None:
    if labels_csv is None:
        save_jsonl(examples, out)
        return
    hidden, rows = hide_labels(examples)
    save_jsonl(hidden, out)
    labels_csv = Path(labels_csv)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def format_readable(ex: Example) -> str:
    """Evidence then chat, with the lie lines marked (for humans checking the generator)."""
    lie_idx = set(ex.meta.get("lie_msg_idx", []))
    head = (f"=== {ex.id} | liar: {ex.label} | lie_type: {ex.meta.get('lie_type')} | "
            f"difficulty: {ex.meta.get('difficulty')} ===")
    lines = [head, "EVIDENCE"]
    lines += [f"  {i + 1}. {fact}" for i, fact in enumerate(ex.evidence)]
    lines.append("CHAT")
    names = ex.meta.get("name_map", {})
    for i, m in enumerate(ex.messages):
        who = f"{m.speaker} ({names[m.speaker]})" if m.speaker in names else m.speaker
        mark = "   <-- LIE" if i in lie_idx else ""
        lines.append(f"  [{i:2d}] {who}: {m.text}{mark}")
    return "\n".join(lines)


def write_samples(examples: Sequence[Example], path: str | Path, n: int, seed: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    picked = random.Random(seed).sample(list(examples), min(n, len(examples)))
    path.write_text("\n\n".join(format_readable(ex) for ex in picked) + "\n", encoding="utf-8")
    return path

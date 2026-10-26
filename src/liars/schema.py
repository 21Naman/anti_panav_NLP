"""Canonical example schema: the only data format the pipeline reads and writes.

One JSON object per line (JSONL). Raw competition formats are converted to this
schema in `liars.adapters`; nothing else touches raw formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple

# Adapters map real speaker names to these letters (and keep the mapping in
# meta.name_map). Prediction files use one column per letter: p_A ... p_D.
SPEAKERS: tuple[str, ...] = ("A", "B", "C", "D")

REQUIRED_KEYS = ("id", "evidence", "messages", "speakers")
OPTIONAL_KEYS = ("label", "meta")


@dataclass
class Message:
    speaker: str
    text: str

    @classmethod
    def from_dict(cls, d: Any) -> "Message":
        if not isinstance(d, dict):
            raise ValueError(f"message must be an object, got {type(d).__name__}")
        missing = [k for k in ("speaker", "text") if k not in d]
        if missing:
            raise ValueError(f"message missing keys {missing}")
        extra = sorted(set(d) - {"speaker", "text"})
        if extra:
            raise ValueError(f"message has unknown keys {extra}")
        return cls(speaker=d["speaker"], text=d["text"])

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "text": self.text}


@dataclass
class Example:
    id: str
    evidence: list[str]
    messages: list[Message]
    speakers: list[str] = field(default_factory=lambda: list(SPEAKERS))
    label: str | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "Example":
        """Build an Example from a parsed JSON object. Raises ValueError on bad shape."""
        if not isinstance(d, dict):
            raise ValueError(f"example must be an object, got {type(d).__name__}")
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"missing keys {missing}")
        extra = sorted(set(d) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
        if extra:
            raise ValueError(f"unknown keys {extra}")
        for key in ("evidence", "messages", "speakers"):
            if not isinstance(d[key], list):
                raise ValueError(f"{key} must be a list, got {type(d[key]).__name__}")
        meta = d.get("meta")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be an object, got {type(meta).__name__}")
        return cls(
            id=d["id"],
            evidence=list(d["evidence"]),
            messages=[Message.from_dict(m) for m in d["messages"]],
            speakers=list(d["speakers"]),
            label=d.get("label"),
            meta=dict(meta),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "evidence": list(self.evidence),
            "messages": [m.to_dict() for m in self.messages],
            "speakers": list(self.speakers),
            "label": self.label,
            "meta": dict(self.meta),
        }


def validate(ex: Example) -> list[str]:
    """Return a list of problems with one example (empty list = valid)."""
    errors: list[str] = []

    if not isinstance(ex.id, str) or not ex.id.strip():
        errors.append("id must be a non-empty string")

    # Speakers: exactly the 4 canonical letters, no duplicates.
    if len(ex.speakers) != len(SPEAKERS):
        errors.append(f"expected {len(SPEAKERS)} speakers, got {len(ex.speakers)}")
    if len(set(ex.speakers)) != len(ex.speakers):
        errors.append(f"duplicate speakers {ex.speakers}")
    unknown = [s for s in ex.speakers if s not in SPEAKERS]
    if unknown:
        errors.append(
            f"speakers must be letters {list(SPEAKERS)} (adapters map names), got {unknown}"
        )

    # Label: None (test data) or one of the speakers.
    if ex.label is not None and ex.label not in ex.speakers:
        errors.append(f"label {ex.label!r} not in speakers {ex.speakers}")

    # Evidence.
    if not ex.evidence:
        errors.append("evidence is empty")
    for i, fact in enumerate(ex.evidence):
        if not isinstance(fact, str) or not fact.strip():
            errors.append(f"evidence[{i}] is empty or not a string")

    # Messages.
    if not ex.messages:
        errors.append("messages is empty")
    for i, m in enumerate(ex.messages):
        if m.speaker not in ex.speakers:
            errors.append(f"messages[{i}].speaker {m.speaker!r} not in speakers")
        if not isinstance(m.text, str) or not m.text.strip():
            errors.append(f"messages[{i}].text is empty or not a string")

    # The liar must actually say something.
    if ex.label is not None and ex.label in ex.speakers and ex.messages:
        if not any(m.speaker == ex.label for m in ex.messages):
            errors.append(f"label speaker {ex.label!r} has no messages")

    # Optional synthetic metadata: lie_msg_idx must point at the liar's messages.
    lie_idx = ex.meta.get("lie_msg_idx") if isinstance(ex.meta, dict) else None
    if lie_idx is not None:
        if not isinstance(lie_idx, list) or not all(isinstance(i, int) for i in lie_idx):
            errors.append("meta.lie_msg_idx must be a list of ints")
        else:
            for i in lie_idx:
                if not 0 <= i < len(ex.messages):
                    errors.append(f"meta.lie_msg_idx {i} out of range")
                elif ex.label is not None and ex.messages[i].speaker != ex.label:
                    errors.append(
                        f"meta.lie_msg_idx {i} is spoken by {ex.messages[i].speaker!r}, "
                        f"not the label {ex.label!r}"
                    )

    return errors


def load_jsonl(path: str | Path) -> list[Example]:
    """Load canonical examples. Raises ValueError naming the bad line."""
    examples: list[Example] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                examples.append(Example.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return examples


def save_jsonl(examples: Iterable[Example], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


class FileReport(NamedTuple):
    n_examples: int
    errors: list[str]


def validate_file(path: str | Path) -> FileReport:
    """Validate every line of a JSONL file. Never raises on bad content."""
    errors: list[str] = []
    seen: dict[str, int] = {}
    n = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            n += 1
            try:
                ex = Example.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as e:
                errors.append(f"line {lineno}: {e}")
                continue
            for err in validate(ex):
                errors.append(f"line {lineno} ({ex.id}): {err}")
            if isinstance(ex.id, str):
                if ex.id in seen:
                    errors.append(
                        f"line {lineno}: duplicate id {ex.id!r} (first on line {seen[ex.id]})"
                    )
                else:
                    seen[ex.id] = lineno
    return FileReport(n, errors)

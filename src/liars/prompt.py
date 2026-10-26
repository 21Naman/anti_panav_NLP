"""Chat -> prompt text, shared by the zero-shot LLM and the QLoRA model.

Truncation keeps all evidence. If the prompt is too long, messages are dropped one at a
time, first those furthest from any other message about the same entity (lines that
share no evidence word with any other line, e.g. small talk, go first).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from liars.schema import Example

WORD_RE = re.compile(r"[a-z0-9]+")
MIN_ENTITY_WORD_LEN = 3


def build_user_text(ex: Example, pcfg: dict, keep: Sequence[int] | None = None) -> str:
    idx = range(len(ex.messages)) if keep is None else keep
    facts = "\n".join(pcfg["fact_format"].format(i=i + 1, fact=f) for i, f in enumerate(ex.evidence))
    chat = "\n".join(pcfg["line_format"].format(speaker=ex.messages[i].speaker, text=ex.messages[i].text)
                     for i in idx)
    parts = [pcfg["intro"], f"{pcfg['evidence_header']}\n{facts}", f"{pcfg['chat_header']}\n{chat}",
             pcfg["question"]]
    return "\n\n".join(p for p in parts if p)


def entity_words(ex: Example) -> list[set[str]]:
    """Per message: words it shares with the evidence (names, values)."""
    ev = {w for f in ex.evidence for w in WORD_RE.findall(f.lower()) if len(w) >= MIN_ENTITY_WORD_LEN}
    return [{w for w in WORD_RE.findall(m.text.lower()) if w in ev} for m in ex.messages]


def drop_order(ex: Example) -> list[int]:
    """Message indices in the order they should be dropped."""
    words = entity_words(ex)
    n = len(ex.messages)
    dist = []
    for i in range(n):
        partners = [abs(i - j) for j in range(n) if j != i and words[i] & words[j]]
        dist.append(min(partners) if partners else float("inf"))
    return sorted(range(n), key=lambda i: (-dist[i], i))


def fit_to_budget(ex: Example, pcfg: dict, count_tokens: Callable[[str], int],
                  max_len: int) -> tuple[str, int]:
    """User text that fits max_len tokens (as measured by count_tokens), and how many
    messages were dropped."""
    keep = list(range(len(ex.messages)))
    text = build_user_text(ex, pcfg, keep)
    if count_tokens(text) <= max_len:
        return text, 0
    dropped = 0
    for i in drop_order(ex):
        keep.remove(i)
        dropped += 1
        text = build_user_text(ex, pcfg, keep)
        if count_tokens(text) <= max_len or not keep:
            break
    return text, dropped

"""Ambiguous two-message ("anchor ... it ...") lines.

A reader resolves "it" to the most recently mentioned thing that fits. If another speaker
mentions a different entity between a speaker's anchor ("re the red tent") and their
anaphor ("it was 25 dollars"), the anaphor can be read as being about that entity. That
breaks the label when, under that reading:
  - an HONEST anaphor contradicts the evidence (it looks like a lie), or
  - the LIE anaphor agrees with the evidence (the lie looks honest).
`ambiguous_anaphors` lists those cases; the generator re-draws such chats when
`synth.avoid_ambiguous_anaphors` is true, and verify reports them.
"""

from __future__ import annotations

import re

from liars.synth.programmatic import GenRecord


def _mentions(rec: GenRecord, idx: int) -> set[str]:
    """Entity ids a message refers to: its claims plus entity names in its text."""
    m = rec.messages[idx]
    ents = {c.ent for c in m.claims}
    text = m.text.lower()
    for e in rec.world.entities:
        if re.search(rf"\b{re.escape(e.name.lower())}\b", text):
            ents.add(e.id)
    return ents


def ambiguous_anaphors(rec: GenRecord) -> list[str]:
    world, msgs = rec.world, rec.messages
    ev_facts = {(c.ent, c.attr) for c in rec.evidence}
    target_ent = rec.target[0]
    out = []
    for j, m in enumerate(msgs):
        if m.kind not in ("anaphor", "lie_anaphor") or not m.claims:
            continue
        anchors = [i for i in range(j) if msgs[i].speaker == m.speaker and msgs[i].kind == "anchor"]
        if not anchors:
            continue
        c = m.claims[0]
        others = set().union(*[_mentions(rec, t) for t in range(anchors[-1] + 1, j)]) - {c.ent}
        for ent in sorted(others):
            if (ent, c.attr) not in ev_facts:
                continue  # under that reading the line is unverifiable, not a contradiction
            agrees = (world.value(ent, c.attr) == c.val) == c.pol
            name = world.get(ent).name
            if m.is_lie and agrees:
                out.append(f"lie message {j} could be about {name!r} and then reads as true")
            elif not m.is_lie and not agrees:
                who = "the liar's item" if ent == target_ent else "another evidence item"
                out.append(f"honest message {j} ({m.speaker}) could be about {name!r} "
                           f"({who}) and then contradicts the evidence")
    return out

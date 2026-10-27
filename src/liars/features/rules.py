"""Rule framework: each rule scores every speaker of a chat (higher = more likely the liar).

Rules are FEATURES for the ensemble, never overrides. Add competition-specific rules on the
day: subclass `Rule`, set `name`, implement `score`, decorate with `@register`.
`liars rules` writes long-format scores; `liars rules-report` shows coverage and precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from liars.schema import SPEAKERS, Example

REGISTRY: dict[str, "Rule"] = {}

WORD_RE = re.compile(r"[a-z]+|\d+(?::\d+)?")
NEGATION_RE = re.compile(r"\b(not|no|never|isn't|wasn't|aren't|weren't|doesn't|didn't|don't|n't)\b")
STOP = set("""a an the and or but of to in on at for with by from is was are were be been it its this
that these those i you he she we they me my your our their his her them us as about into over after
before than then so too very just also really still yet not no yes do does did done has have had will
would can could should there here what when where who whom which how why all any some""".split())
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
UNIT_ALIASES = {"bucks": "dollars", "dollar": "dollars", "$": "dollars", "ppl": "people",
                "guests": "people", "guest": "people", "persons": "people"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_FORMS = {d: i for i, d in enumerate(DAYS)}
DAY_FORMS.update({"mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3,
                  "thurs": 3, "fri": 4, "sat": 5, "sun": 6})
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?(?![\d%])", re.I)


class Rule:
    name: str = "rule"

    def score(self, ex: Example) -> dict[str, float]:
        raise NotImplementedError


def register(cls: type[Rule]) -> type[Rule]:
    REGISTRY[cls.name] = cls()
    return cls


# ---------------------------------------------------------------------- helpers
def content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower())
            if len(w) >= 3 and w not in STOP and not w[0].isdigit()
            and w not in DAY_FORMS and w not in UNIT_ALIASES.values()}


def numbers_with_units(text: str) -> list[tuple[int, str]]:
    """(value, unit) pairs: '40 croissants', '$12', 'six people', '12 bucks'."""
    t = text.lower()
    out = []
    for m in re.finditer(r"\$\s?(\d+)", t):
        out.append((int(m.group(1)), "dollars"))
    toks = re.findall(r"\$?\d+|[a-z]+", t)
    for i, tok in enumerate(toks[:-1]):
        if tok.startswith("$"):
            continue
        val = int(tok) if tok.isdigit() else NUMBER_WORDS.get(tok)
        if val is None:
            continue
        unit = UNIT_ALIASES.get(toks[i + 1], toks[i + 1])
        if unit.isalpha() and unit not in STOP:
            out.append((val, unit))
    return out


def times_in(text: str) -> list[int]:
    """Clock times in minutes after midnight ('6 PM', '6pm', '18:00', '6:30 pm')."""
    out = []
    for m in TIME_RE.finditer(text):
        h, mm, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower().replace(".", "")
        if not m.group(2) and not ap:
            continue  # a bare number is not a time
        if h > 23 or mm > 59:
            continue
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        out.append(h * 60 + mm)
    return out


def days_in(text: str) -> list[int]:
    return [DAY_FORMS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in DAY_FORMS]


def _times_equal(a: int, b: int) -> bool:
    return a == b or (a % 720) == (b % 720)  # '5:45' (no am/pm) matches 17:45


@dataclass
class _ValueRule(Rule):
    """Fires when a message shares entity words with an evidence fact, mentions the same
    kind of value, and the values disagree (or agree under a negation)."""

    name: str = "value_rule"
    extract: Callable[[str], list] = None
    equal: Callable = staticmethod(lambda a, b: a == b)
    min_shared_words: int = 1

    def score(self, ex: Example) -> dict[str, float]:
        facts = [(content_words(f), self.extract(f)) for f in ex.evidence]
        facts = [(w, v) for w, v in facts if v]
        scores = {s: 0.0 for s in SPEAKERS}
        for m in ex.messages:
            vals = self.extract(m.text)
            if not vals:
                continue
            words = content_words(m.text)
            negated = bool(NEGATION_RE.search(m.text.lower()))
            for fwords, fvals in facts:
                if len(words & fwords) < self.min_shared_words:
                    continue
                if self._conflict(vals, fvals, negated):
                    scores[m.speaker] = 1.0
        return scores

    def _conflict(self, vals, fvals, negated) -> bool:
        raise NotImplementedError


@register
class NumberMismatch(_ValueRule):
    """'Tickets cost $45' vs a message 'tickets were 25 bucks' about the same entity."""

    name = "number_mismatch"

    def __init__(self):
        super().__init__(name="number_mismatch", extract=numbers_with_units)

    def _conflict(self, vals, fvals, negated):
        for v, u in vals:
            same_unit = [fv for fv, fu in fvals if fu == u]
            if not same_unit:
                continue
            if negated:
                if v in same_unit:
                    return True
            elif v not in same_unit:
                return True
        return False


@register
class DayTimeMismatch(_ValueRule):
    """Day or clock-time disagreement after normalising formats ('6 PM' == '18:00',
    'Fri' == 'Friday')."""

    name = "day_time_mismatch"

    def __init__(self):
        super().__init__(name="day_time_mismatch",
                         extract=lambda t: [("t", x) for x in times_in(t)] + [("d", x) for x in days_in(t)])

    def _conflict(self, vals, fvals, negated):
        for kind, v in vals:
            same = [fv for fk, fv in fvals if fk == kind]
            if not same:
                continue
            eq = any(_times_equal(v, f) if kind == "t" else v == f for f in same)
            if negated == eq:  # stated value differs, or is denied while true
                return True
        return False


# ---------------------------------------------------------------------- outputs
def compute_rules(examples: Sequence[Example], names: Sequence[str] | None = None) -> pd.DataFrame:
    """Long format: id, speaker, rule, score."""
    rules = [REGISTRY[n] for n in (names or REGISTRY)]
    rows = [(ex.id, s, r.name, float(v)) for ex in examples for r in rules
            for s, v in r.score(ex).items()]
    return pd.DataFrame(rows, columns=["id", "speaker", "rule", "score"])


def rules_report(examples: Sequence[Example], scores: pd.DataFrame) -> pd.DataFrame:
    """Per rule and lie_type: coverage (rule fires on a chat) and precision (the single
    top-scoring speaker is the liar)."""
    by_id = {ex.id: ex for ex in examples}
    rows = []
    for (rule, eid), g in scores.groupby(["rule", "id"]):
        ex = by_id[eid]
        top = g["score"].max()
        fired = top > 0
        winners = g.loc[g["score"] == top, "speaker"].tolist()
        rows.append({"rule": rule, "lie_type": ex.meta.get("lie_type", "all"), "fired": fired,
                     "correct": fired and len(winners) == 1 and winners[0] == ex.label})
    df = pd.DataFrame(rows)

    def agg(g):
        f = g["fired"].sum()
        return pd.Series({"n": len(g), "coverage": f / len(g),
                          "precision": g["correct"].sum() / f if f else float("nan")})

    per_type = df.groupby(["rule", "lie_type"])[["fired", "correct"]].apply(agg).reset_index()
    overall = df.groupby("rule")[["fired", "correct"]].apply(agg).reset_index()
    overall.insert(1, "lie_type", "ALL")
    return pd.concat([overall, per_type], ignore_index=True)

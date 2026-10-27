"""Check generated chats against their hidden world. Any error is a generator bug."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from liars.schema import SPEAKERS, validate
from liars.synth.ambiguity import ambiguous_anaphors
from liars.synth.programmatic import EXPLICIT_ATTRS, IMPLICIT_ATTRS, Claim, GenRecord
from liars.synth.world import World


def is_consistent(world: World, claim: Claim) -> bool:
    true = world.value(claim.ent, claim.attr)
    return (true == claim.val) == claim.pol


def verify_record(rec: GenRecord) -> list[str]:
    ex, world = rec.example, rec.world
    errors = [f"schema: {e}" for e in validate(ex)]

    ev_facts = {(c.ent, c.attr) for c in rec.evidence}
    for c in rec.evidence:
        if not is_consistent(world, c):
            errors.append(f"evidence claim {c} disagrees with the world")
    if rec.target not in ev_facts:
        errors.append(f"lie target {rec.target} is not in the evidence")

    for text in list(ex.evidence) + [m.text for m in ex.messages]:
        if "{" in text or "}" in text:
            errors.append(f"unfilled placeholder in {text!r}")

    lie_idx = ex.meta.get("lie_msg_idx", [])
    flagged = [i for i, m in enumerate(rec.messages) if m.is_lie]
    if flagged != lie_idx:
        errors.append(f"lie_msg_idx {lie_idx} != lie messages {flagged}")

    false_claims: list[Claim] = []
    for i, m in enumerate(rec.messages):
        if m.speaker != ex.messages[i].speaker or m.text != ex.messages[i].text:
            errors.append(f"message {i} record does not match the example")
        bad = [c for c in m.claims if not is_consistent(world, c)]
        if bad and not m.is_lie:
            errors.append(f"honest message {i} ({m.speaker}) contradicts the world: {bad}")
        if m.is_lie:
            if m.speaker != ex.label:
                errors.append(f"lie message {i} spoken by {m.speaker}, label is {ex.label}")
            false_claims += bad

    liars = {rec.messages[i].speaker for i in lie_idx}
    if liars != {ex.label}:
        errors.append(f"liars {liars} != label {ex.label}")
    if not false_claims:
        errors.append("lie messages make no false claim")
    for c in false_claims:
        if (c.ent, c.attr) not in ev_facts:
            errors.append(f"false claim {c} is not contradicted by the evidence")

    # Lie type matches what the lie actually does.
    lt = rec.lie_type
    attrs = {c.attr for c in false_claims}
    ent = world.get(rec.target[0])
    if lt in EXPLICIT_ATTRS and not attrs <= EXPLICIT_ATTRS[lt]:
        errors.append(f"{lt} lie changes {attrs}")
    if lt == "negation" and not all(c.attr == "happened" or not c.pol for c in false_claims):
        errors.append("negation lie is neither a 'not' claim nor a happened flip")
    if lt == "implicit" and not attrs <= IMPLICIT_ATTRS:
        errors.append(f"implicit lie on {attrs}")
    if lt == "two_step":
        if len(lie_idx) != 2:
            errors.append(f"two_step lie has {len(lie_idx)} messages")
        else:
            anchor, anaphor = rec.messages[lie_idx[0]], rec.messages[lie_idx[1]]
            if anchor.claims:
                errors.append("two_step anchor makes a claim")
            if ent.name.lower() in anaphor.text.lower():
                errors.append("two_step second message names the entity (not two-step)")
    elif len(lie_idx) != 1:
        errors.append(f"{lt} lie has {len(lie_idx)} messages")
    if ent.cancelled and rec.target[1] != "happened":
        errors.append("lie targets a cancelled event's details")
    return errors


@dataclass
class VerifyReport:
    n: int
    errors: list[str] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    lie_type_counts: dict[str, int] = field(default_factory=dict)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    dup_rate: float = 0.0
    max_label_dev: float = 0.0
    ambiguous: int = 0  # chats with an "it ..." line that could refer to another item

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"verify: {self.n} examples, {len(self.errors)} errors, "
                 f"duplicate rate {self.dup_rate:.4f}, max label deviation {self.max_label_dev:.4f}, "
                 f"ambiguous anaphor chats {self.ambiguous}"]
        lines.append("  labels: " + "  ".join(f"{s} {self.label_counts.get(s, 0)}" for s in SPEAKERS))
        lines.append("  lie types: " + "  ".join(f"{k} {v}" for k, v in sorted(self.lie_type_counts.items())))
        lines.append("  difficulty: " + "  ".join(f"{k} {v}" for k, v in sorted(self.difficulty_counts.items())))
        lines += [f"  ERROR {e}" for e in self.errors[:20]]
        if len(self.errors) > 20:
            lines.append(f"  ... and {len(self.errors) - 20} more")
        return "\n".join(lines)


def verify_records(records: list[GenRecord], expected_probs: dict[str, float],
                   vcfg: dict) -> VerifyReport:
    rep = VerifyReport(n=len(records))
    for rec in records:
        rep.errors += [f"{rec.example.id}: {e}" for e in verify_record(rec)]
        amb = ambiguous_anaphors(rec)
        rep.ambiguous += bool(amb)
        if amb and vcfg.get("ambiguous_is_error"):
            rep.errors += [f"{rec.example.id}: ambiguous: {a}" for a in amb]

    ids = Counter(r.example.id for r in records)
    rep.errors += [f"duplicate id {i}" for i, k in ids.items() if k > 1]

    rep.label_counts = dict(Counter(r.example.label for r in records))
    rep.lie_type_counts = dict(Counter(r.lie_type for r in records))
    rep.difficulty_counts = dict(Counter(r.example.meta["difficulty"] for r in records))
    if records:
        rep.max_label_dev = max(
            abs(rep.label_counts.get(s, 0) / len(records) - expected_probs[s]) for s in SPEAKERS
        )
        if len(records) >= vcfg["balance_min_n"] and rep.max_label_dev > vcfg["balance_tol"]:
            rep.errors.append(f"label distribution off by {rep.max_label_dev:.3f} "
                              f"(tolerance {vcfg['balance_tol']})")
        keys = Counter(
            (tuple(r.example.evidence), tuple(m.text for m in r.example.messages)) for r in records
        )
        dups = sum(k - 1 for k in keys.values())
        rep.dup_rate = dups / len(records)
        if rep.dup_rate > vcfg["max_dup_rate"]:
            rep.errors.append(f"duplicate-chat rate {rep.dup_rate:.4f} > {vcfg['max_dup_rate']}")
    return rep

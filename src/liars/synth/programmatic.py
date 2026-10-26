"""World + template pack -> synthetic Example with a guaranteed label.

Every message records the claims it makes as (entity, attribute, value, polarity),
so `verify.py` can check that only the liar's lie lines disagree with the world and
that the lie disagrees with a fact that is in the evidence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from liars.schema import SPEAKERS, Example, Message
from liars.synth.pack import Pack, bool_key
from liars.synth.world import Entity, World, alternative_value, build_world

LIE_TYPES = (
    "number_change", "time_shift", "day_change", "location_swap",
    "owner_swap", "negation", "implicit", "two_step",
)
EXPLICIT_ATTRS = {
    "number_change": {"price", "count"},
    "time_shift": {"time"},
    "day_change": {"day"},
    "location_swap": {"location"},
    "owner_swap": {"owner"},
}
IMPLICIT_ATTRS = {"weather", "happened"}
# Kinds that state a value and may be hedged ("I think ..."). Lies and honest lines share
# the same treatment, so hedging never marks a line as honest or as a lie.
CLAIM_KINDS = {"restate_ev", "restate_world", "neg_true", "same_entity", "anaphor", "cover",
               "lie_claim", "lie_anaphor"}
DIFFICULTIES = ("easy", "med", "hard")


@dataclass(frozen=True)
class Claim:
    ent: str
    attr: str
    val: Any
    pol: bool = True  # False = "X is NOT val"


@dataclass
class MsgRec:
    speaker: str
    text: str
    kind: str
    claims: list[Claim] = field(default_factory=list)
    is_lie: bool = False


@dataclass
class GenRecord:
    example: Example
    world: World
    messages: list[MsgRec]
    evidence: list[Claim]
    target: tuple[str, str]
    lie_type: str


def fact_key(ent: Entity, attr: str) -> str:
    return f"{ent.type}.{attr}"


# ---------------------------------------------------------------------------
# Lie targets
# ---------------------------------------------------------------------------
def eligible_targets(world: World, pack: Pack, lie_type: str) -> list[tuple[str, str]]:
    out = []
    for ent in world.entities:
        for attr in ent.attrs:
            key = fact_key(ent, attr)
            if ent.cancelled and attr != "happened":
                continue  # facts about a cancelled event are never lie targets
            if lie_type in EXPLICIT_ATTRS:
                ok = attr in EXPLICIT_ATTRS[lie_type]
            elif lie_type == "negation":
                ok = attr == "happened" or bool(pack["claims"][key].get("neg"))
            elif lie_type == "implicit":
                ok = attr in IMPLICIT_ATTRS and key in pack["implicit"]
            elif lie_type == "two_step":
                ok = ent.type in pack["anchors"] and key in pack["anaphors"]
            else:
                raise ValueError(f"unknown lie type {lie_type!r}")
            if ok:
                out.append((ent.id, attr))
    return out


# ---------------------------------------------------------------------------
# Clause builders (return text clause + claims)
# ---------------------------------------------------------------------------
class _Ctx:
    def __init__(self, rng, world, pack, cfg, preset, target, label, evidence_keys):
        self.rng: random.Random = rng
        self.world: World = world
        self.pack: Pack = pack
        self.wcfg: dict = cfg["world"]
        self.lcfg: dict = cfg["lies"]
        self.chat: dict = cfg["chat"]
        self.preset: dict = preset
        self.target = target
        self.label = label
        self.evidence_keys = evidence_keys

    def alt(self, eid: str, attr: str) -> Any:
        return alternative_value(self.rng, self.world, self.wcfg, self.lcfg, eid, attr)


def _value_clause(ctx: _Ctx, templates: list[str], ent: Entity, attr: str, val: Any) -> str:
    tmpl = ctx.rng.choice(templates)
    return tmpl.format(e=ent.name, v=ctx.pack.render_value(ctx.rng, attr, val))


def claim_clause(ctx: _Ctx, ent: Entity, attr: str, val: Any, pol: bool) -> tuple[str, Claim]:
    key = fact_key(ent, attr)
    claims = ctx.pack["claims"][key]
    if attr == "happened":  # value-keyed templates; always stored as a positive claim
        stated = val if pol else not val
        tmpl = ctx.rng.choice(claims[bool_key(stated)])
        return tmpl.format(e=ent.name), Claim(ent.id, attr, stated, True)
    part = "pos" if pol else "neg"
    return _value_clause(ctx, claims[part], ent, attr, val), Claim(ent.id, attr, val, pol)


def implicit_clause(ctx: _Ctx, ent: Entity, attr: str, val: Any) -> tuple[str, Claim]:
    lines = ctx.pack["implicit"][fact_key(ent, attr)][bool_key(val) if attr == "happened" else val]
    return ctx.rng.choice(lines).format(e=ent.name), Claim(ent.id, attr, val, True)


def anchor_clause(ctx: _Ctx, ent: Entity) -> str:
    return ctx.rng.choice(ctx.pack["anchors"][ent.type]).format(e=ent.name)


def anaphor_clause(ctx: _Ctx, ent: Entity, attr: str, val: Any) -> tuple[str, Claim]:
    tmpl = ctx.rng.choice(ctx.pack["anaphors"][fact_key(ent, attr)])
    return tmpl.format(v=ctx.pack.render_value(ctx.rng, attr, val)), Claim(ent.id, attr, val, True)


def has_implicit(pack: Pack, ent: Entity, attr: str, val: Any) -> bool:
    if attr not in IMPLICIT_ATTRS:
        return False
    by_value = pack["implicit"].get(fact_key(ent, attr), {})
    return bool(by_value.get(bool_key(val) if attr == "happened" else val))


# ---------------------------------------------------------------------------
# Lie
# ---------------------------------------------------------------------------
def make_lie(ctx: _Ctx, lie_type: str) -> list[tuple[str, str, list[Claim]]]:
    """Return 1 (or 2 for two_step) (kind, clause, claims) items for the liar."""
    eid, attr = ctx.target
    ent = ctx.world.get(eid)
    true = ent.attrs[attr]
    if lie_type in EXPLICIT_ATTRS:
        clause, c = claim_clause(ctx, ent, attr, ctx.alt(eid, attr), True)
        return [("lie_claim", clause, [c])]
    if lie_type == "negation":
        if attr == "happened":
            clause, c = claim_clause(ctx, ent, attr, not true, True)
        else:  # "X is not <true value>"
            clause, c = claim_clause(ctx, ent, attr, true, False)
        return [("lie_claim", clause, [c])]
    if lie_type == "implicit":
        if attr == "happened":
            wrong = not true
        else:
            # Wrong weather drawn like real weather, so lie values look like honest ones.
            wcfg = ctx.wcfg
            options = [(w, wt) for w, wt in zip(wcfg["weather"], wcfg["weather_weights"])
                       if w != true and has_implicit(ctx.pack, ent, attr, w)]
            wrong = ctx.rng.choices([o[0] for o in options], weights=[o[1] for o in options])[0]
        clause, c = implicit_clause(ctx, ent, attr, wrong)
        return [("lie_implicit", clause, [c])]
    if lie_type == "two_step":
        anaphor, c = anaphor_clause(ctx, ent, attr, ctx.alt(eid, attr))
        return [("anchor", anchor_clause(ctx, ent), []), ("lie_anaphor", anaphor, [c])]
    raise ValueError(lie_type)


# ---------------------------------------------------------------------------
# Honest messages
# ---------------------------------------------------------------------------
def _allowed(ctx: _Ctx, speaker: str, fact: tuple[str, str]) -> bool:
    # The liar never also states the lied-about fact truthfully (would contradict itself).
    return not (speaker == ctx.label and fact == ctx.target)


def _neg_ok(ctx: _Ctx, fact: tuple[str, str]) -> bool:
    ent = ctx.world.get(fact[0])
    return fact[1] == "happened" or bool(ctx.pack["claims"][fact_key(ent, fact[1])].get("neg"))


def honest_message(ctx: _Ctx, speaker: str, kind: str) -> tuple[str, str, list[Claim]]:
    rng, world, pack = ctx.rng, ctx.world, ctx.pack
    all_facts = [f for f in world.facts() if _allowed(ctx, speaker, f)]
    ev = [f for f in ctx.evidence_keys if _allowed(ctx, speaker, f)]
    non_ev = [f for f in all_facts if f not in ctx.evidence_keys]

    if kind == "same_entity" and speaker == ctx.label:
        kind = "restate_ev"
    if kind == "restate_world" and not non_ev:
        kind = "restate_ev"
    if kind == "implicit_true":
        imp = [f for f in all_facts
               if has_implicit(pack, world.get(f[0]), f[1], world.value(*f))]
        if not imp:
            kind = "restate_ev"

    if kind in ("restate_ev", "restate_world", "same_entity", "neg_true"):
        if kind == "restate_ev":
            fact = rng.choice(ev or all_facts)
        elif kind == "restate_world":
            fact = rng.choice(non_ev)
        elif kind == "same_entity":
            if rng.random() < ctx.preset["same_attr_rate"]:
                fact = ctx.target
            else:
                same = [f for f in all_facts if f[0] == ctx.target[0]]
                fact = rng.choice(same)
        else:
            fact = rng.choice([f for f in all_facts if _neg_ok(ctx, f)])
        ent = world.get(fact[0])
        if kind == "neg_true" and fact[1] != "happened":
            clause, c = claim_clause(ctx, ent, fact[1], ctx.alt(*fact), False)
        else:
            clause, c = claim_clause(ctx, ent, fact[1], world.value(*fact), True)
        return kind, clause, [c]
    if kind == "implicit_true":
        fact = rng.choice(imp)
        clause, c = implicit_clause(ctx, world.get(fact[0]), fact[1], world.value(*fact))
        return kind, clause, [c]
    if kind == "question":
        ent = rng.choice(world.entities)
        return kind, rng.choice(pack["questions"][ent.type]).format(e=ent.name), []
    if kind == "small_talk":
        return kind, rng.choice(pack["small_talk"]), []
    if kind == "personal":
        events = [e for e in world.entities if e.type == "event" and not e.cancelled]
        if events and rng.random() < ctx.chat["personal_entity_rate"]:
            ent = rng.choice(events)
            return kind, rng.choice(pack["personal_entity"]).format(e=ent.name), []
        return kind, rng.choice(pack["personal"]), []
    raise ValueError(f"unknown message kind {kind!r}")


def honest_pair(ctx: _Ctx, speaker: str) -> list[tuple[str, str, list[Claim]]] | None:
    facts = [
        f for f in ctx.world.facts()
        if _allowed(ctx, speaker, f) and f[1] != "happened"
        and not ctx.world.get(f[0]).cancelled
        and ctx.world.get(f[0]).type in ctx.pack["anchors"]
        and fact_key(ctx.world.get(f[0]), f[1]) in ctx.pack["anaphors"]
    ]
    if not facts:
        return None
    eid, attr = ctx.rng.choice(facts)
    ent = ctx.world.get(eid)
    anaphor, c = anaphor_clause(ctx, ent, attr, ent.attrs[attr])
    return [("anchor", anchor_clause(ctx, ent), []), ("anaphor", anaphor, [c])]


def cover_line(ctx: _Ctx, speaker: str, lie_type: str) -> tuple[str, str, list[Claim]] | None:
    """An honest line in the same FORM as the lie (same attribute, same polarity or
    implicit style), so line forms are equally common for liars and honest speakers.
    Not used for two_step (its cover is an honest anchor + 'it' pair)."""
    rng, world, preset = ctx.rng, ctx.world, ctx.preset
    t_eid, attr = ctx.target
    same_attr = [f for f in world.facts() if f[1] == attr and _allowed(ctx, speaker, f)]
    if lie_type == "implicit":
        same_attr = [f for f in same_attr
                     if has_implicit(ctx.pack, world.get(f[0]), attr, world.value(*f))]
    elif lie_type == "negation" and attr != "happened":
        same_attr = [f for f in same_attr if _neg_ok(ctx, f)]
    if not same_attr:
        return None
    if ctx.target in same_attr and rng.random() < preset["cover_same_fact_rate"]:
        fact = ctx.target
    else:
        ev = [f for f in same_attr if f in ctx.evidence_keys and f != ctx.target]
        rest = [f for f in same_attr if f != ctx.target] or same_attr
        fact = rng.choice(ev if ev and rng.random() < preset["cover_evidence_rate"] else rest)
    ent = world.get(fact[0])
    true = world.value(*fact)
    if lie_type == "implicit":
        clause, c = implicit_clause(ctx, ent, attr, true)
        return "implicit_true", clause, [c]
    if lie_type == "negation" and attr != "happened":
        clause, c = claim_clause(ctx, ent, attr, ctx.alt(*fact), False)
        return "cover", clause, [c]
    clause, c = claim_clause(ctx, ent, attr, true, True)
    return "cover", clause, [c]


def pick_target(ctx_rng: random.Random, targets: list[tuple[str, str]], world: World,
                p_cancelled: float) -> tuple[str, str] | None:
    """Choose the lie target. A lie about `happened` states "it went ahead" about as often
    as honest lines do (most events go ahead), so the stated value carries no signal."""
    stated_happened = ctx_rng.random() >= p_cancelled
    ok = [t for t in targets if t[1] != "happened" or world.value(*t) != stated_happened]
    return ctx_rng.choice(ok) if ok else None


# ---------------------------------------------------------------------------
# Chat assembly
# ---------------------------------------------------------------------------
def speaker_sequence(rng: random.Random, n: int, min_each: int, activity_floor: float) -> list[str]:
    """Interleaved speaker order. Counts are independent of who the liar is."""
    counts = {s: min_each for s in SPEAKERS}
    weights = [rng.random() + activity_floor for _ in SPEAKERS]
    for _ in range(n - min_each * len(SPEAKERS)):
        counts[rng.choices(SPEAKERS, weights=weights)[0]] += 1
    seq: list[str] = []
    prev = None
    for _ in range(n):
        options = [s for s in SPEAKERS if counts[s] > 0 and s != prev]
        if not options:
            options = [s for s in SPEAKERS if counts[s] > 0]
        s = rng.choices(options, weights=[counts[o] for o in options])[0]
        seq.append(s)
        counts[s] -= 1
        prev = s
    return seq


def _pick_close_pair(rng: random.Random, pairs: list[tuple[int, int]]) -> tuple[int, int]:
    """Two consecutive turns of one speaker; closer turns are more likely, so the
    'it ...' line usually follows its anchor soon."""
    return rng.choices(pairs, weights=[1 / (b - a) for a, b in pairs])[0]


def _finish(ctx: _Ctx, kind: str, clause: str) -> str:
    rng, preset = ctx.rng, ctx.preset
    if kind in CLAIM_KINDS:
        if rng.random() < preset["hedge_rate"]:
            clause = rng.choice(ctx.pack["hedges"]).format(c=clause)
    if rng.random() < preset["filler_rate"]:
        clause = rng.choice(ctx.pack["fillers"]) + clause
    return ctx.pack.finish(rng, clause)


def render_evidence(rng: random.Random, pack: Pack, world: World, fact: tuple[str, str]) -> str:
    ent = world.get(fact[0])
    val = ent.attrs[fact[1]]
    templates = pack["evidence"][fact_key(ent, fact[1])]
    if fact[1] == "happened":
        text = rng.choice(templates[bool_key(val)]).format(e=ent.name)
    else:
        text = rng.choice(templates).format(e=ent.name, v=pack.render_value(rng, fact[1], val))
    return pack.finish(rng, text, casual=False)


def generate_example(
    rng: random.Random, world: World, pack: Pack, cfg: dict, preset: dict,
    ex_id: str, label: str, lie_type: str, difficulty: str,
) -> GenRecord | None:
    targets = eligible_targets(world, pack, lie_type)
    target = pick_target(rng, targets, world, cfg["world"]["p_cancelled"]) if targets else None
    if target is None:
        return None

    # Evidence: the lie target plus random other facts, shuffled.
    facts = world.facts()
    n_facts = min(rng.randint(*preset["facts"]), len(facts))
    others = [f for f in facts if f != target]
    evidence_keys = [target] + rng.sample(others, n_facts - 1)
    rng.shuffle(evidence_keys)
    ctx = _Ctx(rng, world, pack, cfg, preset, target, label, evidence_keys)

    chat = cfg["chat"]
    n_msgs = rng.randint(*preset["messages"])
    seq = speaker_sequence(rng, n_msgs, chat["min_msgs_per_speaker"], chat["activity_floor"])
    slots: dict[str, list[int]] = {s: [i for i, x in enumerate(seq) if x == s] for s in SPEAKERS}
    plan: dict[int, tuple[str, str, list[Claim]]] = {}

    # Every turn first draws an honest kind. The lie then replaces one of the liar's
    # CLAIM turns, and each honest speaker gets a cover line of the same form in one
    # of their claim turns, so the mix of line types does not depend on who lies.
    kinds = preset["kinds"]
    drawn = {i: rng.choices(list(kinds), weights=list(kinds.values()))[0] for i in range(n_msgs)}

    def free(s: str) -> list[int]:
        return [i for i in slots[s] if i not in plan]

    def claim_turn(s: str) -> int:
        f = free(s)
        claims = [i for i in f if drawn[i] in CLAIM_KINDS]
        return rng.choice(claims or f)

    lie_items = make_lie(ctx, lie_type)
    if lie_type == "two_step":
        for s in SPEAKERS:
            pairs = list(zip(free(s), free(s)[1:]))
            if s == label:
                a, b = _pick_close_pair(rng, pairs)
                plan[a], plan[b] = lie_items
                lie_idx = [a, b]
            elif pairs and rng.random() < preset["cover_rate"]:
                pair = honest_pair(ctx, s)
                if pair:
                    a, b = _pick_close_pair(rng, pairs)
                    plan[a], plan[b] = pair
    else:
        i = claim_turn(label)
        plan[i] = lie_items[0]
        lie_idx = [i]
        for s in SPEAKERS:
            if s != label and rng.random() < preset["cover_rate"]:
                item = cover_line(ctx, s, lie_type)
                if item:
                    plan[claim_turn(s)] = item
        # Optional honest anchor+anaphor pair for any speaker (liar included).
        if rng.random() < preset["honest_pair_rate"]:
            candidates = [(s, a, b) for s in SPEAKERS for a, b in zip(free(s), free(s)[1:])]
            if candidates:
                weights = [1 / (b - a) for _, a, b in candidates]
                s, a, b = rng.choices(candidates, weights=weights)[0]
                pair = honest_pair(ctx, s)
                if pair:
                    plan[a], plan[b] = pair

    # Remaining honest turns. Retry a few times to avoid identical lines and a speaker
    # repeating a fact they already stated.
    used_clauses = {item[1] for item in plan.values()}
    said: dict[str, set] = {sp: set() for sp in SPEAKERS}
    for i, item in plan.items():
        said[seq[i]].update((c.ent, c.attr) for c in item[2])
    for i, s in enumerate(seq):
        if i in plan:
            continue
        for attempt in range(chat["max_retries"]):
            kind = drawn[i] if attempt == 0 else rng.choices(list(kinds), weights=list(kinds.values()))[0]
            item = honest_message(ctx, s, kind)
            facts = {(c.ent, c.attr) for c in item[2]}
            if item[1] not in used_clauses and not (facts & said[s]):
                break
        plan[i] = item
        used_clauses.add(item[1])
        said[s].update(facts)

    msgs: list[MsgRec] = []
    for i, s in enumerate(seq):
        kind, clause, claims = plan[i]
        msgs.append(MsgRec(s, _finish(ctx, kind, clause), kind, claims, is_lie=i in lie_idx))

    evidence_claims = [Claim(e, a, world.value(e, a), True) for e, a in evidence_keys]
    evidence_text = [render_evidence(rng, pack, world, f) for f in evidence_keys]

    meta: dict[str, Any] = {
        "source": "synthetic",
        "pack": pack.name,
        "difficulty": difficulty,
        "lie_type": lie_type,
        "lie_msg_idx": lie_idx,
    }
    if cfg["speaker_names"] == "first_names":
        names = rng.sample(pack["vocab"]["speaker_names"], len(SPEAKERS))
        meta["name_map"] = dict(zip(SPEAKERS, names))
    if cfg.get("debug_meta"):
        meta["debug"] = {
            "target": [world.get(target[0]).name, target[1]],
            "kinds": [m.kind for m in msgs],
        }
    example = Example(
        id=ex_id,
        evidence=evidence_text,
        messages=[Message(m.speaker, m.text) for m in msgs],
        speakers=list(SPEAKERS),
        label=label,
        meta=meta,
    )
    return GenRecord(example, world, msgs, evidence_claims, target, lie_type)


# ---------------------------------------------------------------------------
# Many examples
# ---------------------------------------------------------------------------
def liar_probs(setting: Any) -> dict[str, float]:
    if setting in (None, "uniform"):
        return {s: 1 / len(SPEAKERS) for s in SPEAKERS}
    if not isinstance(setting, dict) or set(setting) - set(SPEAKERS):
        raise ValueError(f"liar_position must be 'uniform' or a dict over {SPEAKERS}")
    total = sum(float(setting.get(s, 0)) for s in SPEAKERS)
    return {s: float(setting.get(s, 0)) / total for s in SPEAKERS}


def allocate_labels(rng: random.Random, n: int, probs: dict[str, float]) -> list[str]:
    """Exact label counts (largest remainder), shuffled."""
    raw = {s: n * p for s, p in probs.items()}
    counts = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - counts[s], reverse=True)[: n - sum(counts.values())]:
        counts[s] += 1
    labels = [s for s in SPEAKERS for _ in range(counts[s])]
    rng.shuffle(labels)
    return labels


def difficulty_weights(difficulty: str, cfg: dict) -> dict[str, float]:
    if difficulty == "mixed":
        return dict(cfg["mixed_weights"])
    names = [d.strip() for d in difficulty.split(",") if d.strip()]
    bad = [d for d in names if d not in cfg["difficulty"]]
    if bad or not names:
        raise ValueError(f"unknown difficulty {difficulty!r}; use easy, med, hard, mixed or e.g. med,hard")
    return {d: 1.0 for d in names}


def generate_many(n: int, seed: int, difficulty: str, pack: Pack, cfg: dict,
                  id_prefix: str | None = None) -> list[GenRecord]:
    """`cfg` is the `synth` section of configs/synth.yaml."""
    prefix = id_prefix if id_prefix is not None else f"{pack.name}_s{seed}_"
    master = random.Random(seed)
    labels = allocate_labels(master, n, liar_probs(cfg["liar_position"]))
    dweights = difficulty_weights(difficulty, cfg)
    records: list[GenRecord] = []
    for i in range(n):
        rng = random.Random(seed * 1_000_003 + i)
        diff = rng.choices(list(dweights), weights=list(dweights.values()))[0]
        preset = cfg["difficulty"][diff]
        lt = preset["lie_type_weights"]
        lie_type = rng.choices(list(lt), weights=list(lt.values()))[0]
        for _ in range(cfg["max_world_tries"]):
            world = build_world(rng, pack, cfg["world"])
            rec = generate_example(rng, world, pack, cfg, preset, f"{prefix}{i:06d}",
                                   labels[i], lie_type, diff)
            if rec is not None:
                records.append(rec)
                break
        else:
            raise RuntimeError(f"could not build a world for lie type {lie_type!r}")
    return records

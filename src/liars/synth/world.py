"""The hidden world behind each synthetic chat.

A World is a handful of entities (events, objects, places, people) with attribute
values. Evidence and honest lines are read off the world; a lie states a value the
world does not have. Because the world is known, every label is guaranteed.

Value encodings: day = 0..6 (Monday first), time = minutes after midnight,
price / count = int, weather = "sunny" | "rainy" | "snowy", happened = bool,
location / owner / companion = a name string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from liars.synth.pack import Pack

N_DAYS = 7


@dataclass
class Entity:
    id: str
    type: str  # event | object | place | person
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def cancelled(self) -> bool:
        return self.type == "event" and self.attrs.get("happened") is False


@dataclass
class World:
    entities: list[Entity]
    people: list[str]  # owners / organisers / companions (never entities themselves)
    places: list[str]  # location values (never place entities themselves)
    weather_by_day: dict[int, str]

    def get(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise KeyError(eid)

    def facts(self) -> list[tuple[str, str]]:
        return [(e.id, a) for e in self.entities for a in e.attrs]

    def value(self, eid: str, attr: str) -> Any:
        return self.get(eid).attrs[attr]


def _pick_time(rng: random.Random, hours: list[int], minutes: list[int]) -> int:
    return rng.randint(hours[0], hours[1]) * 60 + rng.choice(minutes)


def build_world(rng: random.Random, pack: Pack, wcfg: dict) -> World:
    vocab = pack["vocab"]
    n = rng.randint(*wcfg["n_entities"])

    types = [t for t, k in wcfg["min_types"].items() for _ in range(k)]
    weights = wcfg["type_weights"]
    while len(types) < n:
        types.append(rng.choices(list(weights), weights=list(weights.values()))[0])
    rng.shuffle(types)

    # Name pools. People and places used as *values* are kept apart from people and
    # places that are entities, so no cross-entity facts can clash.
    all_people = list(vocab["people"])
    rng.shuffle(all_people)
    n_person_ents = types.count("person")
    person_names = all_people[:n_person_ents]
    pool_size = rng.randint(*wcfg["n_people"])
    people = all_people[n_person_ents : n_person_ents + pool_size]

    all_places = list(vocab["places"])
    rng.shuffle(all_places)
    n_place_ents = types.count("place")
    place_names = all_places[:n_place_ents]
    places = all_places[n_place_ents : n_place_ents + wcfg["n_location_values"]]

    event_names = list(vocab["events"])
    rng.shuffle(event_names)
    personal = [t.format(p=p) for t in vocab["events_personal"] for p in people]
    rng.shuffle(personal)

    weather_by_day = {
        d: rng.choices(wcfg["weather"], weights=wcfg["weather_weights"])[0]
        for d in range(N_DAYS)
    }

    used_objects: set[str] = set()
    entities: list[Entity] = []
    counters = {t: 0 for t in ("event", "object", "place", "person")}
    for t in types:
        counters[t] += 1
        eid = f"{t}{counters[t]}"
        if t == "event":
            if personal and rng.random() < wcfg["personal_event_rate"]:
                name = personal.pop()
            else:
                name = event_names.pop()
            # A cancelled event only has "happened" and "owner": phrasings like "they
            # held X on Sunday" would otherwise imply it took place.
            attrs: dict[str, Any] = {
                "owner": rng.choice(people),
                "happened": rng.random() >= wcfg["p_cancelled"],
            }
            if attrs["happened"]:
                attrs["day"] = rng.randrange(N_DAYS)
                attrs["location"] = rng.choice(places)
                attrs["time"] = _pick_time(rng, wcfg["event_hours"], wcfg["minutes"])
                attrs["price"] = rng.choice(wcfg["prices"])
                attrs["count"] = rng.randint(*wcfg["counts"])
                attrs["weather"] = weather_by_day[attrs["day"]]
        elif t == "object":
            while True:
                name = vocab["object_phrase"].format(
                    colour=rng.choice(vocab["colours"]), object=rng.choice(vocab["objects"])
                )
                if name not in used_objects:
                    used_objects.add(name)
                    break
            attrs = {
                "owner": rng.choice(people),
                "location": rng.choice(places),
                "price": rng.choice(wcfg["prices"]),
            }
        elif t == "place":
            name = place_names.pop()
            attrs = {
                "time": _pick_time(rng, wcfg["open_hours"], wcfg["minutes"]),
                "day": rng.randrange(N_DAYS),
                "price": rng.choice(wcfg["prices"]),
            }
        else:  # person
            name = person_names.pop()
            attrs = {"location": rng.choice(places), "companion": rng.choice(people)}
        entities.append(Entity(eid, t, name, attrs))

    return World(entities, people, places, weather_by_day)


def alternative_value(rng: random.Random, world: World, wcfg: dict, lcfg: dict,
                      eid: str, attr: str) -> Any:
    """A value for (eid, attr) that differs from the world's value (used for lies and
    for honest 'X is not V' statements)."""
    true = world.value(eid, attr)
    if attr == "day":
        return rng.choice([d for d in range(N_DAYS) if d != true])
    if attr == "time":
        lo, hi = lcfg["time_shift_hours"]
        for _ in range(50):
            shift = rng.randint(lo, hi) * rng.choice([-1, 1]) * 60
            if rng.random() < lcfg["time_change_minutes_rate"]:
                shift += rng.choice(wcfg["minutes"]) - true % 60
            new = true + shift
            lo_h, hi_h = lcfg["time_bounds_hours"]
            if lo_h * 60 <= new <= hi_h * 60 + 59 and new != true:
                return new
        # Fallback: nearest whole-hour shift that stays in bounds (never 12 h, which
        # would be ambiguous in formats without am/pm).
        lo_h, hi_h = lcfg["time_bounds_hours"]
        for shift in (60, -60, 120, -120):
            if lo_h * 60 <= true + shift <= hi_h * 60 + 59:
                return true + shift
        raise ValueError(f"cannot shift time {true}")
    if attr == "price":
        return rng.choice([p for p in wcfg["prices"] if p != true])
    if attr == "count":
        lo, hi = wcfg["counts"]
        gap = max(lcfg["min_count_change"], round(true * lcfg["min_count_change_frac"]))
        options = [c for c in range(lo, hi + 1) if abs(c - true) >= gap]
        return rng.choice(options)
    if attr == "weather":
        return rng.choice([w for w in wcfg["weather"] if w != true])
    if attr == "happened":
        return not true
    if attr == "location":
        return rng.choice([p for p in world.places if p != true])
    if attr in ("owner", "companion"):
        return rng.choice([p for p in world.people if p != true])
    raise ValueError(f"no alternative for attribute {attr!r}")

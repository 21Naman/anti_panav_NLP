"""Template packs: load a YAML pack and turn structured values into text."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TEMPLATE_DIR = Path(__file__).parent / "templates"

FACT_KEYS = (
    "event.day", "event.time", "event.location", "event.price", "event.count",
    "event.weather", "event.happened", "event.owner",
    "object.owner", "object.location", "object.price",
    "place.time", "place.day", "place.price",
    "person.location", "person.companion",
)
ENTITY_TYPES = ("event", "object", "place", "person")
BOOL_KEYS = ("true", "false")


def bool_key(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class Pack:
    name: str
    data: dict

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    # ---- value rendering -------------------------------------------------
    def day(self, rng: random.Random, day_idx: int) -> str:
        return rng.choice(self.data["values"]["days"][day_idx])

    def time(self, rng: random.Random, minutes: int) -> str:
        h24, mm = divmod(minutes, 60)
        h12 = h24 % 12 or 12
        fields = {
            "h12": h12,
            "h24": h24,
            "hh24": f"{h24:02d}",
            "mm": f"{mm:02d}",
            "ampm": "am" if h24 < 12 else "pm",
            "AMPM": "AM" if h24 < 12 else "PM",
        }
        key = "time_on_hour" if mm == 0 else "time_with_minutes"
        return rng.choice(self.data["values"][key]).format(**fields)

    def price(self, rng: random.Random, n: int) -> str:
        return rng.choice(self.data["values"]["price"]).format(n=n)

    def count(self, rng: random.Random, n: int) -> str:
        words = self.data["values"]["number_words"]
        if n < len(words) and rng.random() < self.data["values"]["count_word_rate"]:
            return words[n]
        return str(n)

    def weather(self, rng: random.Random, value: str) -> str:
        return rng.choice(self.data["values"]["weather"][value])

    def render_value(self, rng: random.Random, attr: str, value: Any) -> str:
        if attr == "day":
            return self.day(rng, value)
        if attr == "time":
            return self.time(rng, value)
        if attr == "price":
            return self.price(rng, value)
        if attr == "count":
            return self.count(rng, value)
        if attr == "weather":
            return self.weather(rng, value)
        return str(value)  # names: location, owner, companion

    # ---- surface style -----------------------------------------------------
    def finish(self, rng: random.Random, clause: str, casual: bool = True) -> str:
        """Clause -> chat line: capitalise, punctuate, then apply the pack style."""
        text = clause.strip()
        text = text[0].upper() + text[1:]
        style = self.data["style"]
        if text[-1] not in "?!.":
            text += "!" if rng.random() < style["exclaim_rate"] else "."
        if not casual:
            return text
        if style["slang"] and rng.random() < style["slang_rate"]:
            for word, repl in style["slang"].items():
                text = re.sub(rf"\b{re.escape(word)}\b", repl, text)
        if rng.random() < style["lowercase_rate"]:
            text = text.lower()
        if text.endswith(".") and rng.random() < style["drop_period_rate"]:
            text = text[:-1]
        if style["emojis"] and rng.random() < style["emoji_rate"]:
            text = f"{text} {rng.choice(style['emojis'])}"
        return text


def load_pack(name_or_path: str | Path) -> Pack:
    path = Path(name_or_path)
    if not path.suffix:
        path = TEMPLATE_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pack = Pack(name=data.get("name", path.stem), data=data)
    errors = check_pack(pack)
    if errors:
        raise ValueError(f"template pack {path} is invalid:\n  " + "\n  ".join(errors))
    return pack


def _try_format(errors: list[str], where: str, template: str, **fields: str) -> None:
    try:
        out = template.format(**fields)
    except (KeyError, IndexError, ValueError) as e:
        errors.append(f"{where}: bad placeholder in {template!r} ({e})")
        return
    if "{" in out or "}" in out:
        errors.append(f"{where}: leftover brace in {template!r}")


def check_pack(pack: Pack) -> list[str]:
    """Render every template with dummy values; report missing sections and bad placeholders."""
    d = pack.data
    errors: list[str] = []
    required = ["style", "values", "vocab", "evidence", "claims", "implicit", "anchors",
                "anaphors", "questions", "small_talk", "personal", "personal_entity",
                "hedges", "fillers"]
    missing = [k for k in required if k not in d]
    if missing:
        return [f"missing sections {missing}"]

    ev = dict(e="the x", v="y")
    for key in FACT_KEYS:
        if key not in d["evidence"]:
            errors.append(f"evidence.{key} missing")
        if key not in d["claims"]:
            errors.append(f"claims.{key} missing")
    for key, val in d["evidence"].items():
        lists = val.values() if isinstance(val, dict) else [val]
        for lst in lists:
            if not lst:
                errors.append(f"evidence.{key} is empty")
            for t in lst:
                _try_format(errors, f"evidence.{key}", t, **ev)
    for key, val in d["claims"].items():
        parts = BOOL_KEYS if key == "event.happened" else ("pos",)
        for part in parts:
            if not val.get(part):
                errors.append(f"claims.{key}.{part} is empty")
        for part, lst in val.items():
            for t in lst or []:
                _try_format(errors, f"claims.{key}.{part}", t, **ev)
    for key, by_value in d["implicit"].items():
        for value, lst in by_value.items():
            for t in lst:
                _try_format(errors, f"implicit.{key}.{value}", t, **ev)
    for etype in ("event", "object", "place"):
        if not d["anchors"].get(etype):
            errors.append(f"anchors.{etype} missing")
    for key, lst in list(d["anchors"].items()) + list(d["questions"].items()):
        for t in lst:
            _try_format(errors, key, t, e="the x")
    for key, lst in d["anaphors"].items():
        for t in lst:
            _try_format(errors, f"anaphors.{key}", t, v="y")
    for t in d["personal_entity"]:
        _try_format(errors, "personal_entity", t, e="the x")
    for t in d["hedges"]:
        _try_format(errors, "hedges", t, c="claim")
        if "{c}" not in t:
            errors.append(f"hedges: {t!r} has no {{c}}")
    for t in d["vocab"]["events_personal"]:
        _try_format(errors, "vocab.events_personal", t, p="Sam")
    _try_format(errors, "vocab.object_phrase", d["vocab"].get("object_phrase", "{missing}"),
                colour="red", object="bike")
    for etype in ENTITY_TYPES:
        if not d["questions"].get(etype):
            errors.append(f"questions.{etype} missing")

    v = d["values"]
    if len(v["days"]) != 7:
        errors.append("values.days must have 7 entries (Monday first)")
    for key in ("time_on_hour", "time_with_minutes"):
        for t in v[key]:
            _try_format(errors, f"values.{key}", t, h12="7", h24="19", hh24="19", mm="30",
                        ampm="pm", AMPM="PM")
    for t in v["price"]:
        _try_format(errors, "values.price", t, n="5")

    voc = d["vocab"]
    overlap = set(voc["speaker_names"]) & set(voc["people"])
    if overlap:
        errors.append(f"vocab.speaker_names and vocab.people overlap: {sorted(overlap)}")
    return errors

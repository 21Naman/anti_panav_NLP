import copy
import csv
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from liars.adapters.synthetic import format_readable, hide_labels, to_examples, write_synthetic
from liars.cli import app
from liars.config import load_config
from liars.schema import SPEAKERS, load_jsonl, validate, validate_file
from liars.synth.pack import check_pack, load_pack
from liars.synth.programmatic import LIE_TYPES, Claim, generate_many, liar_probs
from liars.synth.verify import verify_record, verify_records
from liars.synth.world import alternative_value, build_world

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "synth.yaml")
SCFG = CFG["synth"]
PACKS = ["default", "alt"]


@pytest.fixture(scope="module")
def packs():
    return {name: load_pack(name) for name in PACKS}


@pytest.mark.parametrize("name", PACKS)
def test_pack_templates_render(name):
    assert check_pack(load_pack(name)) == []


def test_bad_pack_rejected(packs):
    data = copy.deepcopy(packs["default"].data)
    data["claims"]["event.day"]["pos"].append("{e} is on {when}")
    del data["anaphors"]
    from liars.synth.pack import Pack

    assert check_pack(Pack("bad", data))  # missing section
    data["anaphors"] = packs["default"].data["anaphors"]
    errs = check_pack(Pack("bad", data))
    assert any("bad placeholder" in e for e in errs)


@pytest.mark.parametrize("name", PACKS)
@pytest.mark.parametrize("difficulty", ["easy", "med", "hard", "mixed", "med,hard"])
def test_generate_and_verify(packs, name, difficulty):
    recs = generate_many(80, 3, difficulty, packs[name], SCFG)
    rep = verify_records(recs, liar_probs("uniform"), CFG["verify"])
    assert rep.errors == []
    for r in recs:
        assert validate(r.example) == []
        counts = Counter(m.speaker for m in r.example.messages)
        assert min(counts[s] for s in SPEAKERS) >= SCFG["chat"]["min_msgs_per_speaker"]


def test_deterministic(packs):
    a = [r.example.to_dict() for r in generate_many(30, 9, "mixed", packs["default"], SCFG)]
    b = [r.example.to_dict() for r in generate_many(30, 9, "mixed", packs["default"], SCFG)]
    c = [r.example.to_dict() for r in generate_many(30, 10, "mixed", packs["default"], SCFG)]
    assert a == b
    assert a != c


def test_label_balance_and_skew(packs):
    recs = generate_many(100, 1, "easy", packs["default"], SCFG)
    assert Counter(r.example.label for r in recs) == {s: 25 for s in SPEAKERS}
    skew = dict(SCFG, liar_position={"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2})
    recs = generate_many(100, 1, "easy", packs["default"], skew)
    assert Counter(r.example.label for r in recs) == {"A": 40, "B": 20, "C": 20, "D": 20}


def test_all_lie_types_appear(packs):
    recs = generate_many(300, 4, "hard", packs["alt"], SCFG)
    assert set(r.lie_type for r in recs) == set(LIE_TYPES)
    for r in recs:
        n_idx = len(r.example.meta["lie_msg_idx"])
        assert n_idx == (2 if r.lie_type == "two_step" else 1)


def test_first_names_fill_name_map(packs):
    cfg = dict(SCFG, speaker_names="first_names")
    rec = generate_many(5, 2, "med", packs["alt"], cfg)[0]
    nm = rec.example.meta["name_map"]
    assert set(nm) == set(SPEAKERS) and len(set(nm.values())) == 4


def test_verify_catches_planted_errors(packs):
    rec = generate_many(10, 6, "med", packs["default"], SCFG)[0]
    assert verify_record(rec) == []

    # An honest message that disagrees with the world must be reported.
    bad = copy.deepcopy(rec)
    honest = next(m for m in bad.messages if not m.is_lie and m.claims)
    c = honest.claims[0]
    wrong = alternative_value(random.Random(0), bad.world, SCFG["world"], SCFG["lies"], c.ent, c.attr) \
        if c.attr != "happened" else not c.val
    honest.claims[0] = replace(c, val=wrong, pol=True)
    assert any("honest message" in e for e in verify_record(bad))

    # A lie about a fact that is not in the evidence must be reported.
    bad2 = copy.deepcopy(rec)
    bad2.evidence = [cl for cl in bad2.evidence if (cl.ent, cl.attr) != bad2.target]
    errs = verify_record(bad2)
    assert any("not in the evidence" in e or "not contradicted" in e for e in errs)

    # Label pointing at someone else must be reported.
    bad3 = copy.deepcopy(rec)
    bad3.example.label = next(s for s in SPEAKERS if s != rec.example.label)
    assert verify_record(bad3)


def test_alternative_values_differ(packs):
    rng = random.Random(0)
    for _ in range(50):
        world = build_world(rng, packs["default"], SCFG["world"])
        for eid, attr in world.facts():
            alt = alternative_value(rng, world, SCFG["world"], SCFG["lies"], eid, attr)
            true = world.value(eid, attr)
            assert alt != true
            if attr == "time":
                assert abs(alt - true) != 12 * 60


def test_hide_labels_strips_answers(tmp_path, packs):
    examples = to_examples(generate_many(20, 8, "med", packs["alt"], SCFG))
    hidden, rows = hide_labels(examples)
    assert all(h.label is None for h in hidden)
    assert all("lie_msg_idx" not in h.meta and "lie_type" not in h.meta for h in hidden)
    assert [r["label"] for r in rows] == [e.label for e in examples]
    out, lab = tmp_path / "t.jsonl", tmp_path / "labels.csv"
    write_synthetic(examples, out, lab)
    assert validate_file(out).errors == []
    with open(lab) as f:
        assert [r["id"] for r in csv.DictReader(f)] == [e.id for e in examples]
    assert all(e.label is None for e in load_jsonl(out))


def test_readable_marks_lie(packs):
    ex = generate_many(1, 1, "med", packs["default"], SCFG)[0].example
    assert "<-- LIE" in format_readable(ex)


def test_cli_synth(tmp_path):
    out = tmp_path / "s.jsonl"
    result = CliRunner().invoke(app, [
        "synth", "--n", "40", "--seed", "1", "--difficulty", "mixed", "--pack", "alt",
        "--out", str(out), "--config", str(ROOT / "configs" / "synth.yaml"),
        "--set", f"synth.samples_dir={tmp_path}",
    ])
    assert result.exit_code == 0, result.output
    assert "0 errors" in result.output
    assert len(load_jsonl(out)) == 40
    assert (tmp_path / "synth_alt_mixed.txt").exists()

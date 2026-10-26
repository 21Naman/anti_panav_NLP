import copy
import json

import pytest

from liars.schema import (
    SPEAKERS,
    Example,
    load_jsonl,
    save_jsonl,
    validate,
    validate_file,
)


def _good() -> dict:
    return {
        "id": "x1",
        "evidence": ["The shop opens at 9 AM."],
        "messages": [
            {"speaker": "A", "text": "It opens at 9."},
            {"speaker": "B", "text": "It opens at noon."},
        ],
        "speakers": ["A", "B", "C", "D"],
        "label": "B",
        "meta": {"lie_msg_idx": [1]},
    }


def _errors(d: dict) -> list[str]:
    return validate(Example.from_dict(d))


def test_good_example_is_valid():
    assert _errors(_good()) == []


def test_null_label_is_valid():
    d = _good()
    d["label"] = None
    d["meta"] = {}
    assert _errors(d) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda d: d.update(speakers=["A", "B", "C"]), "expected 4 speakers"),
        (lambda d: d.update(speakers=["A", "B", "C", "C"]), "duplicate speakers"),
        (lambda d: d.update(speakers=["A", "B", "C", "Raj"]), "must be letters"),
        (lambda d: d.update(label="E"), "not in speakers"),
        (lambda d: d.update(evidence=[]), "evidence is empty"),
        (lambda d: d.update(evidence=["  "]), "evidence[0]"),
        (lambda d: d.update(messages=[]), "messages is empty"),
        (lambda d: d["messages"].append({"speaker": "E", "text": "hi"}), "not in speakers"),
        (lambda d: d["messages"].append({"speaker": "C", "text": ""}), "text is empty"),
        (lambda d: d.update(label="D"), "has no messages"),
        (lambda d: d["meta"].update(lie_msg_idx=[0]), "not the label"),
        (lambda d: d["meta"].update(lie_msg_idx=[7]), "out of range"),
        (lambda d: d.update(id=""), "id must be"),
    ],
)
def test_bad_examples_rejected(mutate, expect):
    d = _good()
    mutate(d)
    errs = _errors(d)
    assert any(expect in e for e in errs), errs


@pytest.mark.parametrize(
    "d, expect",
    [
        ({"id": "x"}, "missing keys"),
        ({**_good(), "extra": 1}, "unknown keys"),
        ({**_good(), "evidence": "one fact"}, "evidence must be a list"),
        ({**_good(), "messages": [{"speaker": "A"}]}, "message missing keys"),
        ({**_good(), "meta": []}, "meta must be an object"),
        ([1, 2], "must be an object"),
    ],
)
def test_malformed_shapes_raise(d, expect):
    with pytest.raises(ValueError, match=expect):
        Example.from_dict(d)


def test_roundtrip(tmp_path, tiny):
    out = tmp_path / "rt.jsonl"
    save_jsonl(tiny, out)
    again = load_jsonl(out)
    assert [e.to_dict() for e in again] == [e.to_dict() for e in tiny]


def test_validate_file_reports_bad_lines(tmp_path):
    good = _good()
    dup = copy.deepcopy(good)
    bad = copy.deepcopy(good)
    bad["id"] = "x2"
    bad["label"] = "Z"
    p = tmp_path / "bad.jsonl"
    p.write_text(
        "\n".join([json.dumps(good), json.dumps(dup), "{not json", json.dumps(bad)]) + "\n"
    )
    report = validate_file(p)
    assert report.n_examples == 4
    text = "\n".join(report.errors)
    assert "duplicate id 'x1'" in text
    assert "line 3" in text
    assert "label 'Z'" in text


def test_load_jsonl_names_bad_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(_good()) + "\n{oops\n")
    with pytest.raises(ValueError, match=":2:"):
        load_jsonl(p)


def test_fixture_is_valid(tiny_path):
    report = validate_file(tiny_path)
    assert report.errors == []
    assert report.n_examples == 32


def test_fixture_design(tiny):
    from collections import Counter

    assert Counter(e.label for e in tiny) == {s: 8 for s in SPEAKERS}
    lie_types = Counter(e.meta["lie_type"] for e in tiny)
    assert len(lie_types) == 8 and set(lie_types.values()) == {4}
    for e in tiny:
        assert 6 <= len(e.messages) <= 12, e.id
        assert {m.speaker for m in e.messages} == set(SPEAKERS), e.id
        idx = e.meta["lie_msg_idx"]
        assert all(e.messages[i].speaker == e.label for i in idx)
        assert len(idx) == (2 if e.meta["lie_type"] == "two_step" else 1)
    # honest hedges exist
    assert sum("I think" in m.text for e in tiny for m in e.messages if m.speaker != e.label) >= 16

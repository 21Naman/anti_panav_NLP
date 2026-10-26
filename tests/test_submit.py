import numpy as np
import pandas as pd
import pytest

from liars.preds import make_preds
from liars.schema import SPEAKERS
from liars.submit import decode_label, make_submission, validate_submission

FORMATS = [
    {"label_format": "letter", "int_order": "as_listed"},
    {"label_format": "int", "int_order": "as_listed"},
    {"label_format": "int", "int_order": "first_appearance"},
    {"label_format": "name", "int_order": "as_listed"},
]


def _with_names(tiny):
    names = ["Priya", "Tom", "Lena", "Raj"]
    for i, ex in enumerate(tiny):
        rot = names[i % 4:] + names[: i % 4]
        ex.meta["name_map"] = dict(zip(SPEAKERS, rot))
    return tiny


def _perfect_preds(examples):
    probs = np.array([[float(s == ex.label) for s in SPEAKERS] for ex in examples])
    return make_preds([ex.id for ex in examples], probs)


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_reproduces_labels(tiny, fmt):
    tiny = _with_names(tiny)
    scfg = {"id_column": "chat_id", "label_column": "liar", **fmt}
    sub = make_submission(_perfect_preds(tiny), tiny, scfg)
    assert validate_submission(sub, tiny, scfg) == []
    by_id = {ex.id: ex for ex in tiny}
    decoded = [decode_label(v, by_id[i], scfg) for i, v in zip(sub["chat_id"], sub["liar"])]
    assert decoded == [ex.label for ex in tiny]


def test_matches_sample_order_and_columns(tiny):
    scfg = {"id_column": "id", "label_column": "label", "label_format": "letter",
            "int_order": "as_listed"}
    sample = pd.DataFrame({"id": [ex.id for ex in reversed(tiny)], "label": "A"})
    sub = make_submission(_perfect_preds(tiny), tiny, scfg, sample)
    assert list(sub["id"]) == list(sample["id"])
    assert validate_submission(sub, tiny, scfg, sample) == []


def test_validation_catches_problems(tiny):
    scfg = {"id_column": "id", "label_column": "label", "label_format": "int",
            "int_order": "as_listed"}
    sub = make_submission(_perfect_preds(tiny), tiny, scfg)
    assert any("missing" in e for e in validate_submission(sub.iloc[1:], tiny, scfg))
    bad = sub.copy()
    bad.loc[0, "label"] = 7
    assert any("invalid label" in e for e in validate_submission(bad, tiny, scfg))
    assert validate_submission(sub[["label", "id"]], tiny, scfg)
    with pytest.raises(ValueError, match="no prediction"):
        make_submission(_perfect_preds(tiny[1:]), tiny, scfg)

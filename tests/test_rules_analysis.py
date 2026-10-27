import pandas as pd

from liars.analysis import build_report
from liars.features.rules import REGISTRY, compute_rules, days_in, numbers_with_units, rules_report, times_in
from liars.preds import make_preds
from liars.schema import SPEAKERS, Example, Message

import numpy as np


def test_normalisers():
    assert times_in("at 6 PM") == times_in("at 18:00") == times_in("at 6pm") == [1080]
    assert times_in("7 people") == []
    assert days_in("Fri or friday") == [4, 4]
    assert (45, "dollars") in numbers_with_units("tickets cost $45")
    assert (6, "people") in numbers_with_units("six guests came")


def test_rules_fire_on_the_liar():
    ex = Example("r1", ["The jazz concert starts at 6 PM on Friday.", "Tickets for the jazz concert cost 45 dollars."], [
        Message("A", "The jazz concert starts at 18:00, right?"),
        Message("B", "Tickets for the jazz concert are 25 bucks."),
        Message("C", "The jazz concert isn't on Saturday."),
        Message("D", "The jazz concert is on Sat."),
    ], label="B")
    s = {r: REGISTRY[r].score(ex) for r in REGISTRY}
    assert s["number_mismatch"] == {"A": 0, "B": 1, "C": 0, "D": 0}
    assert s["day_time_mismatch"]["D"] == 1 and s["day_time_mismatch"]["A"] == 0
    assert s["day_time_mismatch"]["C"] == 0  # true negation


def test_rules_report_on_fixture(tiny):
    rep = rules_report(tiny, compute_rules(tiny))
    assert set(rep["rule"]) == set(REGISTRY) and (rep["lie_type"] == "ALL").sum() == len(REGISTRY)
    assert rep["coverage"].between(0, 1).all()


def test_html_report(tiny):
    rng = np.random.default_rng(0)
    oof = make_preds([e.id for e in tiny], rng.dirichlet(np.ones(4), size=32))
    oof.insert(1, "fold", [i % 5 for i in range(32)])
    other = make_preds([e.id for e in tiny], rng.dirichlet(np.ones(4), size=32))
    acfg = {"length_bins": [0, 8, 1000], "confidence_bins": [0, 0.5, 1.0]}
    html = build_report("r", oof, tiny, acfg, 5, ("r2", other))
    assert html.startswith("<!doctype html>") and "Confusion matrix" in html
    assert "Disagreements with r2" in html and "class=\"true\"" in html

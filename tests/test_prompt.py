from liars.config import load_config
from liars.prompt import build_user_text, drop_order, fit_to_budget
from liars.schema import Example, Message

from pathlib import Path

PCFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "llm_zeroshot.yaml")["prompt"]


def test_user_text_layout(tiny):
    t = build_user_text(tiny[0], PCFG)
    assert "1. " + tiny[0].evidence[0] in t
    assert f"{tiny[0].messages[0].speaker}: {tiny[0].messages[0].text}" in t
    assert t.strip().endswith("Answer with one letter.")


def _ex():
    return Example("x", ["The jazz concert is on Friday.", "Tom owns the red bike."], [
        Message("A", "haha nice"),
        Message("B", "The jazz concert is on Friday."),
        Message("C", "lol"),
        Message("D", "No, the jazz concert is on Saturday."),
        Message("A", "Tom owns the red bike."),
    ], label="D")


def test_drop_order_small_talk_first():
    order = drop_order(_ex())
    assert set(order[:2]) == {0, 2}  # lines sharing no evidence word with another line


def test_fit_to_budget_keeps_evidence():
    ex = _ex()
    words = lambda t: len(t.split())
    full, dropped = fit_to_budget(ex, PCFG, words, 10_000)
    assert dropped == 0
    short, dropped = fit_to_budget(ex, PCFG, words, words(full) - 3)
    assert dropped >= 1 and "Tom owns the red bike." in short and "1. The jazz concert" in short
    assert "haha nice" not in short

from liars.stats import compute_stats, describe, format_stats


def test_describe():
    d = describe([1, 2, 3, 4])
    assert d["min"] == 1 and d["max"] == 4 and d["mean"] == 2.5 and d["p50"] == 2.5
    assert describe([]) == {}


def test_stats_on_fixture(tiny):
    s = compute_stats(tiny)
    assert s["n"] == 32
    assert s["labels"] == {"A": 8, "B": 8, "C": 8, "D": 8}
    assert 6 <= s["messages_per_chat"]["min"] <= s["messages_per_chat"]["max"] <= 12
    assert "messages/chat" in format_stats(s)

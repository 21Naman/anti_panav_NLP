import pandas as pd
import pytest

from liars.preds import make_preds, merge_shards, split_shards


def test_split_shards_cover_everything_in_order():
    items = list(range(11))
    shards = split_shards(items, 2)
    assert [len(s) for s in shards] == [6, 5]
    assert sum(shards, []) == items
    assert split_shards(items[:1], 2) == [[0], []]


def test_merge_shards_restores_input_order_and_checks_ids():
    ids = [f"x{i}" for i in range(6)]
    p = make_preds(ids, [[0.25] * 4] * 6)
    a, b = p.iloc[:3], p.iloc[3:]
    merged = merge_shards([b, a], ids)  # shard order does not matter
    assert list(merged["id"]) == ids
    with pytest.raises(ValueError, match="missing"):
        merge_shards([a], ids)
    with pytest.raises(ValueError, match="duplicate"):
        merge_shards([a, a, b], ids)


def test_predict_options_exist():
    from typer.testing import CliRunner

    from liars.cli import app
    out = CliRunner().invoke(app, ["predict", "--help"]).output
    assert "--gpus" in out and "--test-preds" in out

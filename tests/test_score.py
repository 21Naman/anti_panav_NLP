import numpy as np
import pandas as pd

from liars.preds import make_preds
from liars.schema import SPEAKERS
from liars.score import format_score, load_labels, score


def test_score_from_jsonl_and_csv(tiny_path, tiny, tmp_path):
    probs = np.array([[1.0 if s == ex.label else 0.0 for s in SPEAKERS] for ex in tiny])
    probs[0] = [0.0, 0.0, 0.0, 1.0] if tiny[0].label != "D" else [1.0, 0, 0, 0]
    preds = make_preds([ex.id for ex in tiny], probs)
    res = score(preds, load_labels(tiny_path))
    assert res["n"] == 32 and abs(res["accuracy"] - 31 / 32) < 1e-9
    assert "lie_type" in res["by"] and "accuracy" in format_score(res)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"id": [e.id for e in tiny], "label": [e.label for e in tiny]}).to_csv(csv_path, index=False)
    assert score(preds, load_labels(csv_path))["accuracy"] == res["accuracy"]

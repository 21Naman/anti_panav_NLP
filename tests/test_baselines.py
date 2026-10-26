import copy

import numpy as np

from liars.config import load_config
from liars.models import get_model
from liars.preds import PROB_COLS, accuracy
from liars.schema import Message

from pathlib import Path

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "baselines.yaml",
                  ["model.tfidf.min_df=1"])


def _planted(tiny):
    """Copy of the fixture where every liar line contains a giveaway word."""
    out = []
    for ex in copy.deepcopy(tiny):
        ex.messages = [Message(m.speaker, m.text + (" zebra" if m.speaker == ex.label else ""))
                       for m in ex.messages]
        out.append(ex)
    return out


def test_tfidf_learns_planted_signal(tiny):
    data = _planted(tiny)
    m = get_model("tfidf_lr")
    m.fit(data[:24], None, CFG)
    p = m.predict_proba(data[24:])
    assert np.allclose(p[PROB_COLS].sum(axis=1), 1)
    import pandas as pd
    assert accuracy(p, pd.Series([e.label for e in data[24:]])) == 1.0
    assert "zebra" in m.top_features(3)["feature"].tolist()


def test_majority(tiny):
    m = get_model("majority")
    m.fit(tiny, None, CFG)
    p = m.predict_proba(tiny)
    assert np.allclose(p[PROB_COLS].to_numpy(), 0.25)

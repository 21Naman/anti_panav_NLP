from pathlib import Path

import numpy as np
import pytest

import liars.models.nli_zeroshot as nz
from liars.config import load_config
from liars.preds import PROB_COLS
from liars.schema import SPEAKERS

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "nli_zeroshot.yaml")
NCFG = CFG["nli"]


class FakeScorer:
    """Contradiction = 1 when the hypothesis contains a lie line, else 0.1."""

    def __init__(self, lie_texts):
        self.lie_texts = lie_texts
        self.calls = 0

    def score(self, pairs, progress=False):
        self.calls += len(pairs)
        out = np.zeros((len(pairs), 3))
        for i, (_, h) in enumerate(pairs):
            out[i, 0] = 1.0 if any(t in h for t in self.lie_texts) else 0.1
        return out


def _lie_texts(examples):
    return {ex.messages[i].text for ex in examples for i in ex.meta["lie_msg_idx"]}


@pytest.mark.parametrize("mode", nz.MODES)
def test_speaker_scores_find_liar(tiny, mode):
    # two_step lie lines are short "it ..." lines; keep every line for this check
    ncfg = dict(NCFG, filter={"skip_questions": False, "min_words": 1})
    s = nz.speaker_scores(tiny, mode, FakeScorer(_lie_texts(tiny)), ncfg)
    assert s.shape == (32, 4)
    assert all(SPEAKERS[s[i].argmax()] == ex.label for i, ex in enumerate(tiny))


def test_filter_skips_questions_and_short_lines():
    f = {"skip_questions": True, "min_words": 3}
    assert not nz.keep_message("When is the concert?", f)
    assert not nz.keep_message("Haha nice", f)
    assert nz.keep_message("The concert is on Friday.", f)


def test_fit_picks_mode_and_temperature(tiny, monkeypatch, tmp_path):
    fake = FakeScorer(_lie_texts(tiny))
    monkeypatch.setattr(nz, "get_scorer", lambda ncfg, cache_dir: fake)
    cfg = load_config(ROOT / "configs" / "nli_zeroshot.yaml",
                      [f"paths.data_dir={tmp_path}", "nli.filter.min_words=1",
                       "nli.filter.skip_questions=false", "nli.progress=false"])
    m = nz.NLIZeroShotModel()
    m.fit(tiny, None, cfg)
    assert m.mode in nz.MODES and m.temperature in NCFG["temperatures"]
    p = m.predict_proba(tiny)
    assert np.allclose(p[PROB_COLS].sum(axis=1), 1)
    assert (p[PROB_COLS].to_numpy().argmax(1) == [SPEAKERS.index(e.label) for e in tiny]).all()
    assert "lie_type" in m.report() and "chosen mode" in m.report()


def test_label_indices_read_from_checkpoint():
    assert nz.label_indices({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}) == [2, 0, 1]
    assert nz.label_indices({"0": "contradiction", "1": "entailment", "2": "neutral"}) == [0, 1, 2]
    with pytest.raises(ValueError):
        nz.label_indices({0: "entailment", 1: "not_entailment"})


def test_pair_cache_roundtrip(tmp_path):
    c = nz.PairCache(tmp_path / "c.parquet")
    c.data["k"] = np.array([0.7, 0.2, 0.1])
    c.dirty = True
    c.save()
    again = nz.PairCache(tmp_path / "c.parquet")
    assert np.allclose(again.data["k"], [0.7, 0.2, 0.1])

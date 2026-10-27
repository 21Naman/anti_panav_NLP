from pathlib import Path

import numpy as np
import pytest

from liars.config import load_config
from liars.preds import make_preds
from liars.pseudo import PseudoDisabled, select_pseudo
from liars.schema import SPEAKERS, save_jsonl
from liars.trainer import run_training

ROOT = Path(__file__).resolve().parents[1]


def _preds(tiny):
    p = np.full((32, 4), 0.01)
    for i, ex in enumerate(tiny):
        p[i, SPEAKERS.index(ex.label)] = 0.97 if i % 2 == 0 else 0.5
    return make_preds([e.id for e in tiny], p)


def test_refuses_when_disabled(tiny):
    with pytest.raises(PseudoDisabled):
        select_pseudo(_preds(tiny), tiny, 0.95, enabled=False)


def test_selects_confident_only(tiny):
    out = select_pseudo(_preds(tiny), tiny, 0.95, enabled=True)
    assert len(out) == 16 and all(e.meta["source"] == "pseudo" for e in out)
    assert all(e.label == t.label for e, t in zip(out, tiny[::2]))


def test_extra_train_is_training_side_only(tiny, tmp_path):
    extra = [e for e in select_pseudo(_preds(tiny), tiny, 0.95, enabled=True)]
    for e in extra:
        e.id = "pseudo_" + e.id
    path = tmp_path / "pseudo.jsonl"
    save_jsonl(extra, path)
    fixture = ROOT / "tests" / "fixtures" / "tiny.jsonl"
    cfg = load_config(ROOT / "configs" / "baselines.yaml", [
        f"paths.data_dir={tmp_path}", f"debug.data.train={fixture}", f"debug.data.test={fixture}",
        f"train.extra_train=[{path}]"])
    out = run_training(cfg, ROOT / "configs" / "baselines.yaml", debug=True, log=lambda *_: None)
    import pandas as pd
    oof = pd.read_parquet(out["oof_path"])
    assert not oof["id"].str.startswith("pseudo_").any() and len(oof) == 32

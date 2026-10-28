import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from liars.config import load_config
from liars.preds import PROB_COLS
from liars.trainer import run_training

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baselines.yaml"


def _cfg(tmp_path, *extra):
    fixture = ROOT / "tests" / "fixtures" / "tiny.jsonl"
    return load_config(CONFIG, [f"paths.data_dir={tmp_path}", f"debug.data.train={fixture}",
                                f"debug.data.test={fixture}", *extra])


@pytest.mark.parametrize("mode", ["kfold", "holdout"])
def test_debug_run(tmp_path, mode):
    out = run_training(_cfg(tmp_path, f"train.mode={mode}", "train.refit_full=true"),
                       CONFIG, debug=True, log=lambda *_: None)
    oof = pd.read_parquet(out["oof_path"])
    preds = pd.read_parquet(out["preds_path"])
    assert len(preds) == 32 and (preds[PROB_COLS].sum(axis=1).round(6) == 1).all()
    if mode == "kfold":
        assert len(oof) == 32 and oof["fold"].nunique() == 5
    else:
        assert set(oof["fold"]) == {0} and len(oof) < 32
    assert {"id", "fold", "label", *PROB_COLS} <= set(oof.columns)
    runs = tmp_path / "debug" / "runs.csv"
    with open(runs) as f:
        row = list(csv.DictReader(f))[-1]
    assert row["mode"] == mode
    assert len(json.loads(row["fold_acc"])) == (5 if mode == "kfold" else 1)
    assert (tmp_path / "debug" / "runs" / f"{out['run']}.yaml").exists()


def test_missing_folds_file_is_explained(tmp_path):
    # data.test=null: the test must not depend on generated data under data/ (fresh checkout)
    cfg = _cfg(tmp_path, f"data.train={ROOT / 'tests' / 'fixtures' / 'tiny.jsonl'}", "data.test=null")
    with pytest.raises(FileNotFoundError, match="liars folds"):
        run_training(cfg, CONFIG, debug=False, log=lambda *_: None)


def test_tta_check_compares_first_variant_with_mean(tiny):
    import numpy as np

    from liars.models.llm_zeroshot import LLMZeroShotModel
    from liars.schema import SPEAKERS
    from liars.trainer import tta_check

    ex = tiny[:4]
    y = [SPEAKERS.index(e.label) for e in ex]
    logits = np.zeros((4, 2, 4))
    for i, k in enumerate(y):
        logits[i, 1, k] = 5.0            # variant 2 always right
        logits[i, 0, (k + 1) % 4] = 1.0  # variant 1 always wrong, but weaker
    m = LLMZeroShotModel()
    m.temperature, m._last_tta_logits = 1.0, logits
    msg = tta_check(m, ex)
    assert "first variant only 0.0000" in msg and "mean of 2 variants 1.0000" in msg
    m._last_tta_logits = logits[:, :1]
    assert tta_check(m, ex) is None  # tta_n 1: nothing to compare


def test_subsample_is_stratified_and_deterministic(tiny):
    from collections import Counter

    from liars.trainer import subsample
    a = subsample(tiny, 16, seed=1)
    assert len(a) == 16 and Counter(e.label for e in a) == Counter({s: 4 for s in "ABCD"})
    assert [e.id for e in a] == [e.id for e in subsample(tiny, 16, seed=1)]
    assert subsample(tiny, None, 1) == list(tiny) and subsample(tiny, 999, 1) == list(tiny)


def test_latest_run_resolves_newest_config(tmp_path):
    import os
    import time

    from liars.paths import Paths
    from liars.trainer import latest_run
    paths = Paths(tmp_path)
    paths.run_configs.mkdir(parents=True)
    for i, name in enumerate(["llm_qlora_a", "llm_qlora_b", "deberta_mc_c"]):
        f = paths.run_configs / f"{name}.yaml"
        f.write_text("x: 1\n")
        os.utime(f, (time.time() + i, time.time() + i))
    assert latest_run(paths, "latest:llm_qlora") == "llm_qlora_b"
    assert latest_run(paths, "plain_name") == "plain_name"

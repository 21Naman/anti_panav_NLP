import numpy as np
import pandas as pd

from liars.ensemble import fit_stack, fit_weights, predict_stack, run_ensemble, stack_features
from liars.paths import Paths
from liars.preds import make_preds, save_preds
from liars.schema import SPEAKERS

CFG = {"seed": 0, "inner_k": 3, "stack_l2": 0.01, "stack_margin": 0.003}


def _write_run(paths, name, ids, labels, quality, rng, test_ids):
    def probs(n, lab):
        p = rng.dirichlet(np.ones(4), size=n)
        if lab is not None:
            for i, l in enumerate(lab):
                if rng.random() < quality:
                    p[i, SPEAKERS.index(l)] += 2.0
        return p / p.sum(1, keepdims=True)
    oof = make_preds(ids, probs(len(ids), labels))
    oof.insert(1, "fold", [i % 5 for i in range(len(ids))])
    oof.insert(2, "label", labels)
    save_preds(oof, paths.oof / f"{name}.parquet")
    save_preds(make_preds(test_ids, probs(len(test_ids), None)), paths.preds / f"{name}.parquet")


def test_ensemble_beats_or_matches_best(tmp_path):
    rng = np.random.default_rng(0)
    paths = Paths(tmp_path)
    ids = [f"x{i}" for i in range(300)]
    labels = [SPEAKERS[i % 4] for i in range(300)]
    test_ids = [f"t{i}" for i in range(50)]
    _write_run(paths, "good", ids, labels, 0.8, rng, test_ids)
    _write_run(paths, "ok", ids, labels, 0.6, rng, test_ids)
    _write_run(paths, "noise", ids, labels, 0.0, rng, test_ids)
    out = run_ensemble(paths, ["good", "ok", "noise"], "auto", "t", CFG, log=lambda *_: None)
    t = out["table"].set_index("model")["accuracy"]
    assert out["n_aligned"] == 300
    assert t[f"weighted (inner 3-fold)"] >= t["good"] - 0.02
    assert out["weights"]["noise"] < 0.1
    ens = pd.read_parquet(paths.preds / "ens_t.parquet")
    assert len(ens) == 50 and np.allclose(ens.iloc[:, 1:].sum(axis=1), 1)


def test_weights_and_stack_shapes():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 200)
    good = np.full((200, 4), 0.1)
    good[np.arange(200), y] = 0.7
    noise = rng.dirichlet(np.ones(4), size=200)
    P = np.stack([good, noise])
    w = fit_weights(P, y)
    assert w[0] > 0.8 and abs(w.sum() - 1) < 1e-6
    X = stack_features(P, None)
    assert X.shape == (200, 4, 6)
    p = predict_stack(X, fit_stack(X, y, 0.01))
    assert (p.argmax(1) == y).mean() > 0.95

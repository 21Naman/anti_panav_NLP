"""`liars ensemble`: combine runs on honest held-out predictions.

OOF files are aligned on the ids every run has (with any holdout-mode run that is the
holdout fold). Methods:
  mean     - plain average of probabilities (nothing fitted).
  weighted - non-negative weights (sum 1) minimising OOF log-loss.
  stack    - conditional logit: per-speaker features (p, log p, rank of each run, rule
             scores) with weights shared across speakers, softmax across the 4.
Weighted and stack are scored with an inner K-fold over the aligned rows (fit on k-1
parts, predict the rest), so their accuracy is honest. The final weights are fitted on
all aligned rows and applied to the test predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from liars.paths import Paths
from liars.preds import PROB_COLS, load_preds, make_preds, save_preds
from liars.schema import SPEAKERS

EPS = 1e-6


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, None))))


def _acc(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


# ------------------------------------------------------------------ loading
def load_runs(paths: Paths, runs: Sequence[str]):
    """Aligned OOF stack (runs, n, 4), labels, folds, ids; test stack (runs, m, 4), test ids."""
    oofs = [pd.read_parquet(paths.oof / f"{r}.parquet") for r in runs]
    common = set(oofs[0]["id"])
    for o in oofs[1:]:
        common &= set(o["id"])
    if not common:
        raise ValueError("the runs share no OOF ids")
    ids = sorted(common)
    ref = oofs[0].set_index("id").loc[ids]
    y = np.array([SPEAKERS.index(l) for l in ref["label"]])
    folds = ref["fold"].to_numpy()
    P = np.stack([o.set_index("id").loc[ids, PROB_COLS].to_numpy() for o in oofs])
    test_P, test_ids = None, None
    tests = [paths.preds / f"{r}.parquet" for r in runs]
    if all(t.exists() for t in tests):
        tdfs = [load_preds(t) for t in tests]
        test_ids = tdfs[0]["id"].tolist()
        for r, t in zip(runs, tdfs):
            if sorted(t["id"]) != sorted(test_ids):
                raise ValueError(f"test ids of {r} differ from {runs[0]}")
        test_P = np.stack([t.set_index("id").loc[test_ids, PROB_COLS].to_numpy() for t in tdfs])
    return P, y, folds, ids, test_P, test_ids


def load_rule_features(paths: Paths, ids: Sequence[str]) -> np.ndarray | None:
    """(n, 4, n_rules) rule scores from data/features/rules*.parquet, or None."""
    files = sorted(paths.features.glob("rules*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates(["id", "speaker", "rule"])
    wide = df.pivot_table(index=["id", "speaker"], columns="rule", values="score", fill_value=0.0)
    rules = list(wide.columns)
    out = np.zeros((len(ids), len(SPEAKERS), len(rules)))
    for i, eid in enumerate(ids):
        for j, s in enumerate(SPEAKERS):
            if (eid, s) in wide.index:
                out[i, j] = wide.loc[(eid, s)].to_numpy()
    return out


# ------------------------------------------------------------------ methods
def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = P.shape[0]
    if r == 1:
        return np.ones(1)

    def loss(w):
        return _nll(np.tensordot(w, P, axes=1), y)

    res = minimize(loss, np.full(r, 1 / r), method="SLSQP", bounds=[(0, 1)] * r,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def stack_features(P: np.ndarray, rules: np.ndarray | None) -> np.ndarray:
    """(n, 4, f): for each run p, log p, rank (0 = top); then rule scores."""
    feats = []
    for p in P:
        feats += [p, np.log(np.clip(p, EPS, None)), (-p).argsort(1).argsort(1).astype(float)]
    X = np.stack(feats, axis=2)
    if rules is not None:
        X = np.concatenate([X, rules], axis=2)
    return X


def fit_stack(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conditional logit with shared weights; features standardised. Returns (w, mu, sd)."""
    mu = X.reshape(-1, X.shape[2]).mean(0)
    sd = X.reshape(-1, X.shape[2]).std(0) + EPS
    Z = (X - mu) / sd
    n = len(y)

    def f(w):
        s = Z @ w
        s = s - s.max(1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(1, keepdims=True))
        p = np.exp(logp)
        loss = -logp[np.arange(n), y].mean() + l2 * (w @ w)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), y] = 1
        grad = np.einsum("ns,nsf->f", p - onehot, Z) / n + 2 * l2 * w
        return loss, grad

    res = minimize(f, np.zeros(Z.shape[2]), jac=True, method="L-BFGS-B")
    return res.x, mu, sd


def predict_stack(X: np.ndarray, model) -> np.ndarray:
    w, mu, sd = model
    s = ((X - mu) / sd) @ w
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def honest_oof(method: str, P, y, rules, k: int, seed: int, l2: float) -> np.ndarray:
    """Out-of-fold ensemble probabilities over the aligned rows (inner K-fold)."""
    out = np.zeros(P.shape[1:])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        if method == "weighted":
            out[te] = np.tensordot(fit_weights(P[:, tr], y[tr]), P[:, te], axes=1)
        else:
            X = stack_features(P, rules)
            out[te] = predict_stack(X[te], fit_stack(X[tr], y[tr], l2))
    return out


def run_ensemble(paths: Paths, runs: Sequence[str], method: str, name: str, ecfg: dict,
                 log=print) -> dict:
    P, y, folds, ids, test_P, test_ids = load_runs(paths, runs)
    log(f"aligned OOF rows (ids every run has): {len(ids)}; test rows: "
        f"{len(test_ids) if test_ids else 0}")
    rules = load_rule_features(paths, ids)
    test_rules = load_rule_features(paths, test_ids) if (rules is not None and test_ids) else None
    if rules is not None:
        log(f"rule features: {rules.shape[2]} rules")
        if test_rules is not None and not test_rules.any() and rules.any():
            log("WARNING: no rule scores for the test ids (run `liars rules --in <test file>`)")
    k, seed, l2 = ecfg["inner_k"], ecfg["seed"], ecfg["stack_l2"]

    rows = [(r, _acc(p, y), _nll(p, y)) for r, p in zip(runs, P)]
    mean_p = P.mean(0)
    rows.append(("mean", _acc(mean_p, y), _nll(mean_p, y)))
    oof = {"mean": mean_p}
    for m in ("weighted", "stack"):
        oof[m] = honest_oof(m, P, y, rules, k, seed, l2)
        rows.append((f"{m} (inner {k}-fold)", _acc(oof[m], y), _nll(oof[m], y)))
    table = pd.DataFrame(rows, columns=["model", "accuracy", "log_loss"])
    log(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    weights = fit_weights(P, y)
    log("weights: " + ", ".join(f"{r} {w:.3f}" for r, w in zip(runs, weights)))
    if method == "auto":  # best of mean / weighted; stack only if clearly better
        base = max(("weighted", "mean"), key=lambda m: _acc(oof[m], y))
        method = "stack" if _acc(oof["stack"], y) > _acc(oof[base], y) + ecfg["stack_margin"] else base
        log(f"auto -> {method} (stack must beat {base} by > {ecfg['stack_margin']:.3f})")
    out = {"method": method, "table": table, "weights": dict(zip(runs, weights)), "n_aligned": len(ids)}

    oof_df = make_preds(ids, oof[method])
    oof_df.insert(1, "fold", folds)
    oof_df.insert(2, "label", [SPEAKERS[i] for i in y])
    save_preds(oof_df, paths.oof / f"ens_{name}.parquet")
    if test_P is not None:
        if method == "mean":
            tp = test_P.mean(0)
        elif method == "weighted":
            tp = np.tensordot(weights, test_P, axes=1)
        else:
            model = fit_stack(stack_features(P, rules), y, l2)
            tp = predict_stack(stack_features(test_P, test_rules if rules is not None else None), model)
        save_preds(make_preds(test_ids, tp), paths.preds / f"ens_{name}.parquet")
        log(f"test preds -> {paths.preds / f'ens_{name}.parquet'}")
    log(f"oof -> {paths.oof / f'ens_{name}.parquet'} ({method}, honest acc {_acc(oof[method], y):.4f})")
    return out

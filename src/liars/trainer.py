"""`liars train`: k-fold or holdout training with honest out-of-fold predictions.

kfold   : for each fold f, fit on the other folds, predict fold f and the test set;
          test predictions are averaged over folds. OOF covers every train row.
holdout : fit once on all folds except cv.holdout_fold, predict that fold + test.
          Models with uses_val get an inner split of the TRAINING folds for early
          stopping, never the holdout fold. Optional train.refit_full retrains on 100%
          of train and uses that model's test predictions (OOF stays from the holdout run).
"""

from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from liars.config import seed_everything
from liars.cv import assign_folds, folds_path_for, inner_split, load_folds
from liars.models import get_model
from liars.models.base import Model
from liars.paths import get_paths
from liars.preds import PROB_COLS, accuracy, make_preds, save_preds
from liars.runlog import log_run
from liars.schema import Example, load_jsonl

DEBUG_SUBDIR = "debug"


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def apply_debug(cfg: dict) -> dict:
    """Merge the config's `debug:` block and send outputs to <data_dir>/debug."""
    cfg = copy.deepcopy(cfg)
    deep_update(cfg, cfg.get("debug") or {})
    paths = cfg.setdefault("paths", {})
    paths["data_dir"] = str(get_paths(cfg).data / DEBUG_SUBDIR)
    return cfg


def _fit(cfg: dict, train: Sequence[Example]) -> Model:
    seed_everything(cfg["seed"])
    model = get_model(cfg["model"]["name"])
    if model.uses_val:
        tr, va = inner_split(train, cfg["train"]["inner_val_frac"], cfg["seed"])
    else:
        tr, va = list(train), None
    model.fit(tr, va, cfg)
    return model


def _labels(examples: Sequence[Example]) -> pd.Series:
    return pd.Series([ex.label for ex in examples])


def _resolve_folds(cfg: dict, train: Sequence[Example], train_path: str, debug: bool) -> dict:
    k, seed = cfg["cv"]["k"], cfg["cv"]["seed"]
    if debug:
        return assign_folds(train, k, seed)
    path = cfg["data"].get("folds") or folds_path_for(train_path, get_paths(cfg).canonical)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"folds file {path} not found. Run: liars folds --in {train_path} --k {k}"
        )
    folds = load_folds(path)
    missing = [ex.id for ex in train if ex.id not in folds]
    if missing:
        raise ValueError(f"{len(missing)} train ids have no fold in {path} (e.g. {missing[:3]})")
    return folds


def run_training(cfg: dict, config_path: str | Path, test_path: str | Path | None = None,
                 debug: bool = False, log=print) -> dict:
    if debug:
        cfg = apply_debug(cfg)
    paths = get_paths(cfg)
    t0 = time.time()

    train_path = cfg["data"]["train"]
    train = load_jsonl(train_path)
    test_path = test_path or cfg["data"].get("test")
    test = load_jsonl(test_path) if test_path else []
    folds = _resolve_folds(cfg, train, train_path, debug)
    mode = cfg["train"]["mode"]
    run = f"{Path(config_path).stem}_{datetime.now().strftime('%m%d-%H%M%S')}"
    if debug:
        run += "_debug"
    log(f"run {run}: model={cfg['model']['name']} mode={mode} train={len(train)} test={len(test)}")

    fold_ids = sorted(set(folds.values()))
    if mode == "kfold":
        todo = fold_ids
    elif mode == "holdout":
        todo = [cfg["cv"]["holdout_fold"]]
    else:
        raise ValueError(f"train.mode must be kfold or holdout, got {mode!r}")

    oof_parts, test_parts, fold_acc = [], [], {}
    for f in todo:
        tr = [ex for ex in train if folds[ex.id] != f]
        va = [ex for ex in train if folds[ex.id] == f]
        tf = time.time()
        model = _fit(cfg, tr)
        p = model.predict_proba(va)
        fold_acc[f] = accuracy(p, _labels(va))
        p.insert(1, "fold", f)
        p.insert(2, "label", [ex.label for ex in va])
        oof_parts.append(p)
        if test:
            test_parts.append(model.predict_proba(test))
        if hasattr(model, "report"):  # optional model-specific report (fit + predict stats)
            paths.reports.mkdir(parents=True, exist_ok=True)
            (paths.reports / f"{run}_fold{f}.md").write_text(model.report(), encoding="utf-8")
        log(f"  fold {f}: train {len(tr)} held-out {len(va)} acc {fold_acc[f]:.4f} "
            f"({time.time() - tf:.1f}s)")

    if mode == "holdout" and cfg["train"].get("refit_full") and test:
        refit_cfg = copy.deepcopy(cfg)
        refit_cfg["train"]["_refit"] = model.refit_info()
        log("  refit on 100% of train for test predictions")
        test_parts = [_fit(refit_cfg, train).predict_proba(test)]

    oof = pd.concat(oof_parts, ignore_index=True)
    save_preds(oof, paths.oof / f"{run}.parquet")
    out = {"run": run, "fold_acc": fold_acc, "mean_acc": float(np.mean(list(fold_acc.values()))),
           "oof_path": str(paths.oof / f"{run}.parquet"), "preds_path": None}
    if test_parts:
        ids = test_parts[0]["id"].tolist()
        for tp in test_parts[1:]:
            if tp["id"].tolist() != ids:
                raise ValueError("test prediction ids differ between folds")
        avg = np.mean([tp[PROB_COLS].to_numpy() for tp in test_parts], axis=0)
        save_preds(make_preds(ids, avg), paths.preds / f"{run}.parquet")
        out["preds_path"] = str(paths.preds / f"{run}.parquet")

    paths.run_configs.mkdir(parents=True, exist_ok=True)
    with open(paths.run_configs / f"{run}.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    wall = time.time() - t0
    log_run(run, config_path, cfg, mode, fold_acc, wall, runs_csv=paths.runs_csv)
    out["wall_time_s"] = wall
    log(f"done: mean held-out acc {out['mean_acc']:.4f} in {wall:.1f}s")
    log(f"  oof   -> {out['oof_path']}")
    if out["preds_path"]:
        log(f"  preds -> {out['preds_path']}")
    return out

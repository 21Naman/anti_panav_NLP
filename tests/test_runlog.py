import csv
import json

import pytest

from liars.runlog import COLUMNS, log_run


def test_log_run_appends(tmp_path):
    path = tmp_path / "sub" / "runs.csv"
    cfg = {"seed": 1}
    log_run("r1", "configs/x.yaml", cfg, "kfold", {0: 0.5, 1: 0.7}, 12.34, gpu="cpu", runs_csv=path)
    log_run("r2", "configs/x.yaml", cfg, "holdout", {0: 0.9}, 1.0, gpu="T4", runs_csv=path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == COLUMNS
    assert [r["run"] for r in rows] == ["r1", "r2"]
    assert float(rows[0]["mean_acc"]) == pytest.approx(0.6)
    assert json.loads(rows[0]["fold_acc"]) == {"0": 0.5, "1": 0.7}
    assert rows[1]["mode"] == "holdout" and rows[1]["gpu"] == "T4"
    assert rows[0]["config_hash"] == rows[1]["config_hash"]


def test_log_run_rejects_foreign_header(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        log_run("r", "c.yaml", {}, "kfold", {0: 1.0}, 1.0, gpu="cpu", runs_csv=path)

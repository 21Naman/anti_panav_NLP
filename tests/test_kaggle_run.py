import csv
import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "scripts" / "kaggle_run.py")
kr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kr)


def test_notebook_is_valid_and_thin():
    nb = kr.build_notebook(["liars train --config configs/x.yaml", "liars score --preds a --labels b"],
                           ref="abc123")
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert nb["nbformat"] == 4
    assert "REF = 'abc123'" in src and "liars score --preds a --labels b" in src
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<cell>", "exec")  # valid python
    assert "ghp_" not in src and "github_pat_" not in src  # no token literals


def test_template_in_sync():
    committed = json.loads((ROOT / "notebooks" / "kaggle" / "run_cli.ipynb").read_text())
    expected = kr.build_notebook(
        ["liars train --config configs/nli_zeroshot.yaml --test data/bench/test.jsonl"])
    assert committed == json.loads(json.dumps(expected)), "run: python scripts/kaggle_run.py template"


def _write_runs(path, runs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "mean_acc"])
        w.writeheader()
        w.writerows({"run": r, "mean_acc": 0.5} for r in runs)


def test_merge_outputs(tmp_path):
    data = tmp_path / "data"
    (data / "oof").mkdir(parents=True)
    (data / "oof" / "old.parquet").write_text("local")
    _write_runs(data / "runs.csv", ["old"])
    stage = tmp_path / "stage" / "data"
    (stage / "oof").mkdir(parents=True)
    (stage / "oof" / "old.parquet").write_text("remote")
    (stage / "oof" / "new.parquet").write_text("remote")
    _write_runs(stage / "runs.csv", ["old", "new"])
    z = tmp_path / "outputs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in stage.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(stage.parent))
    added = kr.merge_outputs(z, data)
    assert (data / "oof" / "old.parquet").read_text() == "local"  # never overwritten
    assert (data / "oof" / "new.parquet").exists()
    with open(data / "runs.csv") as f:
        assert [r["run"] for r in csv.DictReader(f)] == ["old", "new"]
    assert any("runs.csv += new" in a for a in added)


def test_refresh_code_uploads_only_when_commit_changed():
    state = {"commit": "old", "uploads": 0}

    def upload():
        state["uploads"] += 1
        state["pending"] = 2  # Kaggle serves the new version after 2 polls

    def current(slug):
        if state.get("pending"):
            state["pending"] -= 1
            if not state["pending"]:
                state["commit"] = "new"
        return state["commit"]

    assert kr.refresh_code("new", "liars-code", current, upload, sleep=lambda s: None) is True
    assert state["uploads"] == 1
    assert kr.refresh_code("new", "liars-code", current, upload, sleep=lambda s: None) is False
    assert state["uploads"] == 1


def test_push_refuses_dirty_tree(monkeypatch):
    import pytest
    monkeypatch.setattr(kr, "git", lambda *a: " M src/x.py" if a[0] == "status" else "abc")
    args = kr.argparse.Namespace(allow_dirty=False)
    with pytest.raises(SystemExit, match="uncommitted"):
        kr.cmd_push(args)

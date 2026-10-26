"""Append one row per training run to data/runs.csv."""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from liars.config import config_hash

DEFAULT_RUNS_CSV = Path("data") / "runs.csv"

COLUMNS = [
    "run",
    "config_path",
    "config_hash",
    "mode",
    "fold_acc",  # JSON object {fold: accuracy}
    "mean_acc",
    "wall_time_s",
    "gpu",
    "timestamp",  # UTC, ISO 8601
    "notes",
]


def detect_gpu() -> str:
    """CUDA GPU name (with count if > 1), or 'cpu'. Never reports MPS: local runs are CPU."""
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    return f"{name} x{n}" if n > 1 else name


def log_run(
    run: str,
    config_path: str | Path,
    cfg: dict,
    mode: str,
    fold_acc: Mapping[int | str, float],
    wall_time_s: float,
    gpu: str | None = None,
    runs_csv: str | Path = DEFAULT_RUNS_CSV,
    notes: str = "",
) -> dict:
    """Append a run to runs.csv (header written on first use). Returns the row."""
    accs = [float(v) for v in fold_acc.values()]
    row = {
        "run": run,
        "config_path": str(config_path),
        "config_hash": config_hash(cfg),
        "mode": mode,
        "fold_acc": json.dumps({str(k): round(float(v), 6) for k, v in fold_acc.items()}),
        "mean_acc": round(sum(accs) / len(accs), 6) if accs else "",
        "wall_time_s": round(float(wall_time_s), 1),
        "gpu": gpu if gpu is not None else detect_gpu(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }

    path = Path(runs_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {COLUMNS}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row

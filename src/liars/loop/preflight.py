"""`liars loop doctor`: is this machine + config ready for the loop? Nothing is run.

Errors block `liars loop run` (it escalates instead of starting); warnings are shown.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

NEEDED_LOCAL = ("torch", "transformers", "peft", "accelerate", "pyarrow", "sklearn", "scipy", "yaml")


def environment(executor: str, data_dir: str, min_free_gb: float = 5.0) -> tuple[list[str], list[str], list[str]]:
    errors, warnings, info = [], [], []
    if sys.version_info < (3, 10):
        errors.append(f"Python {sys.version.split()[0]} < 3.10")
    info.append(f"python {sys.version.split()[0]}")
    missing = [m for m in NEEDED_LOCAL if importlib.util.find_spec(m) is None]
    if "pyarrow" in missing or "scipy" in missing or "sklearn" in missing:
        errors.append(f"missing packages needed by the loop itself: {[m for m in missing if m in ('pyarrow', 'scipy', 'sklearn', 'yaml')]}")
    if executor == "local" and missing:
        errors.append(f"local executor needs {missing} (pip install -e '.[nlp]')")
    elif missing:
        warnings.append(f"not installed here (fine if training runs on Kaggle): {missing}")
    if importlib.util.find_spec("torch") is not None:
        import torch

        info.append("CUDA: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"))
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_dir).free / 2**30
    (errors if free < min_free_gb else info).append(f"free disk under {data_dir}: {free:.1f} GB (need {min_free_gb})")
    if executor == "kaggle":
        exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
        if not Path(exe).exists():
            errors.append("kaggle CLI not found (pip install kaggle in the venv)")
        import os

        home = Path.home() / ".kaggle"
        legacy = (home / "kaggle.json").exists()
        token = (home / "access_token").exists() or bool(os.environ.get("KAGGLE_API_TOKEN"))
        oauth = any(p.name != "kaggle.json" and p.name != "access_token" for p in home.glob("*")) if home.exists() else False
        if not (legacy or token or oauth):
            errors.append("no Kaggle credentials: `kaggle auth login`, or save an API token to "
                          "~/.kaggle/access_token (kaggle.com -> Settings -> API -> Generate New Token)")
        if not legacy and not os.environ.get("KAGGLE_USERNAME"):
            errors.append("set KAGGLE_USERNAME (the newer Kaggle logins store no username): "
                          "echo 'export KAGGLE_USERNAME=<username>' >> ~/.bashrc")
        try:
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, check=True).stdout.strip()
            if dirty:
                errors.append("uncommitted changes: Kaggle runs use the last commit; commit first")
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.append("not a git checkout: cannot check for uncommitted code")
    return errors, warnings, info


def summarise_state(ledger_dir: str) -> str:
    p = Path(ledger_dir) / "state.json"
    if not p.exists():
        return "no loop state yet (fresh start)"
    st = json.loads(p.read_text())
    return (f"loop state: {st.get('status')} at iteration {st.get('iteration')}, champion "
            f"{st.get('champion')}, {st.get('gpu_h_used', 0):.1f} GPU h used")

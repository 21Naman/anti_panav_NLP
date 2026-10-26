"""Run `liars` commands on a Kaggle GPU from the terminal (developer tool, not a dependency).

  python scripts/kaggle_run.py data   [--slug liars-bench] FILE...     # private data dataset
  python scripts/kaggle_run.py code   [--slug liars-code]              # private code snapshot (fallback)
  python scripts/kaggle_run.py push   --name NAME --cmd "liars ..." [--cmd ...] [--wait]
  python scripts/kaggle_run.py status --name NAME
  python scripts/kaggle_run.py wait   --name NAME
  python scripts/kaggle_run.py fetch  --name NAME                       # download + merge into data/
  python scripts/kaggle_run.py merge  --zip outputs.zip                 # merge a manually downloaded zip
  python scripts/kaggle_run.py template                                 # regenerate notebooks/kaggle/run_cli.ipynb

The notebook (built by `build_notebook`) is thin: install the repo, copy data, print GPU
info, run the commands, zip the outputs. Code comes from a git clone using the GH_TOKEN
Kaggle Secret; if that secret is not attached it falls back to the private code dataset.
Tokens are never printed or written to files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Panavdoesstuff/liars-chatroom"
BUILD = ROOT / "build" / "kaggle"
TEMPLATE_PATH = ROOT / "notebooks" / "kaggle" / "run_cli.ipynb"
DEFAULT_DATASETS = ["liars-bench"]
CODE_DATASET = "liars-code"
ACCELERATOR = "NvidiaTeslaT4"  # T4 x2
OUTPUT_ITEMS = ["data/oof", "data/preds", "data/runs.csv", "data/reports", "data/runs", "data/logs"]
POLL_SECONDS = 60
MANIFEST = "manifest.json"

# --------------------------------------------------------------------------------------
# Notebook
# --------------------------------------------------------------------------------------
INSTALL_CELL = r'''# Install: git clone with the GH_TOKEN Kaggle Secret, else the private code dataset.
import glob, json, os, shutil, subprocess, sys, tarfile, time
WORK = "/tmp/liars"   # outside /kaggle/working, so nothing here is saved as output

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception as e:
        print(f"Kaggle secret {name}: not available ({type(e).__name__})")
        return None

def find_input(slug):
    direct = f"/kaggle/input/{slug}"
    if os.path.isdir(direct):
        return direct
    hits = [p for p in glob.glob("/kaggle/input/**/" + slug, recursive=True) if os.path.isdir(p)]
    if not hits:
        raise FileNotFoundError(f"dataset {slug!r} is not attached to this notebook")
    return hits[0]

hf = secret("HF_TOKEN")
if hf:
    os.environ["HF_TOKEN"] = hf
gh = secret("GH_TOKEN")
shutil.rmtree(WORK, ignore_errors=True)
code_source = None
if gh:
    r = subprocess.run(["git", "clone", "-q", f"https://x-access-token:{gh}@github.com/{REPO}.git", WORK],
                       capture_output=True, text=True)
    if r.returncode == 0:
        subprocess.run(["git", "-C", WORK, "remote", "set-url", "origin", f"https://github.com/{REPO}.git"], check=True)
        subprocess.run(["git", "-C", WORK, "checkout", "-q", REF], check=True)
        code_source = "git " + subprocess.run(["git", "-C", WORK, "rev-parse", "--short", "HEAD"],
                                              capture_output=True, text=True).stdout.strip()
    else:
        print("git clone failed (token or repo access); trying the code dataset")  # never echo the URL
        shutil.rmtree(WORK, ignore_errors=True)
if code_source is None:
    src = find_input(CODE_DATASET)
    tars = glob.glob(src + "/**/liars-src.tar.gz", recursive=True)
    if tars:
        with tarfile.open(tars[0]) as t:
            t.extractall(WORK)
    else:
        root = os.path.dirname(glob.glob(src + "/**/pyproject.toml", recursive=True)[0])
        shutil.copytree(root, WORK)
    code_source = "dataset " + CODE_DATASET
print("code source:", code_source)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", WORK + "[nlp]"], check=True)
'''

PATHS_CELL = r'''# Paths: copy attached data datasets into <repo>/data/ following each manifest.json.
os.chdir(WORK)
for slug in DATA_DATASETS:
    src = find_input(slug)
    manifests = glob.glob(src + "/**/manifest.json", recursive=True)
    if not manifests:
        raise FileNotFoundError(f"{slug}: no manifest.json")
    base = os.path.dirname(manifests[0])
    for name, rel in json.load(open(manifests[0])).items():
        dst = os.path.join(WORK, "data", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(base, name), dst)
        print(f"data: {slug}/{name} -> data/{rel}")
os.makedirs("data/logs", exist_ok=True)
'''

GPU_CELL = r'''subprocess.run([sys.executable, "scripts/gpu_info.py"], check=True)
'''

RUN_CELL = r'''# Run the commands (one per line). Wall time per command goes to data/logs/kaggle_timing.json.
timing = []
for line in [l.strip() for l in COMMANDS.strip().splitlines()]:
    if not line or line.startswith("#"):
        continue
    print(f"\n$ {line}", flush=True)
    t0 = time.time()
    subprocess.run(line, shell=True, check=True)
    timing.append({"cmd": line, "seconds": round(time.time() - t0, 1)})
    json.dump({"code_source": code_source, "commands": timing}, open("data/logs/kaggle_timing.json", "w"), indent=1)
print(json.dumps(timing, indent=1))
'''

ZIP_CELL = r'''# Zip outputs to /kaggle/working/outputs.zip (download it, then: python scripts/kaggle_run.py merge --zip outputs.zip)
import zipfile
with zipfile.ZipFile("/kaggle/working/outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for item in OUTPUT_ITEMS:
        paths = [item] if os.path.isfile(item) else glob.glob(item + "/**/*", recursive=True)
        for p in paths:
            if os.path.isfile(p):
                z.write(p)
print("outputs.zip:", round(os.path.getsize("/kaggle/working/outputs.zip") / 2**20, 2), "MB")
'''


def _code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build_notebook(commands: list[str], ref: str = "main", repo: str = REPO,
                   code_dataset: str = CODE_DATASET,
                   data_datasets: list[str] | None = None) -> dict:
    data_datasets = DEFAULT_DATASETS if data_datasets is None else data_datasets
    cfg = (
        "# Settings (edit COMMANDS for a manual run)\n"
        f"REPO = {repo!r}\n"
        f"REF = {ref!r}  # branch, tag or commit\n"
        f"CODE_DATASET = {code_dataset!r}  # fallback when the GH_TOKEN secret is not attached\n"
        f"DATA_DATASETS = {data_datasets!r}\n"
        f"OUTPUT_ITEMS = {OUTPUT_ITEMS!r}\n"
        'COMMANDS = r"""\n' + "\n".join(commands) + '\n"""\n'
    )
    md = ("# liars: run CLI commands on a Kaggle GPU\n"
          "Thin notebook: install the repo, copy data, GPU info, run `liars` commands, zip outputs.\n"
          "Settings: GPU T4 x2, Internet on. Run with Save Version -> Save & Run All.\n")
    return {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)},
                  _code_cell(cfg), _code_cell(INSTALL_CELL), _code_cell(PATHS_CELL),
                  _code_cell(GPU_CELL), _code_cell(RUN_CELL), _code_cell(ZIP_CELL)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------------------
# Kaggle CLI helpers
# --------------------------------------------------------------------------------------
def kaggle_user() -> str:
    import os

    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    with open(Path.home() / ".kaggle" / "kaggle.json") as f:
        return json.load(f)["username"]  # only the username is read


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def kernel_slug(name: str) -> str:
    return f"liars-{name}"


def upload_dataset(slug: str, files: dict[Path, str], message: str) -> None:
    """Create or version a PRIVATE dataset. files: local path -> name in the dataset."""
    user = kaggle_user()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for src, name in files.items():
            shutil.copy(src, d / name)
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
        exists = kaggle("datasets", "status", f"{user}/{slug}", check=False).returncode == 0
        if exists:
            r = kaggle("datasets", "version", "-p", str(d), "-m", message)
        else:
            r = kaggle("datasets", "create", "-p", str(d))  # private unless --public
        print(r.stdout.strip())


def cmd_data(a) -> None:
    files, manifest = {}, {}
    for f in a.files:
        p = (ROOT / f).resolve()
        rel = str(p.relative_to(ROOT / "data"))
        name = rel.replace("/", "__")
        files[p] = name
        manifest[name] = rel
    with tempfile.TemporaryDirectory() as tmp:
        mpath = Path(tmp) / MANIFEST
        mpath.write_text(json.dumps(manifest, indent=1))
        files[mpath] = MANIFEST
        upload_dataset(a.slug, files, a.message)
    print(f"dataset {a.slug}: {manifest}")


def cmd_code(a) -> None:
    subprocess.run(["bash", str(ROOT / "scripts" / "build_wheel.sh")], check=True)
    tar = ROOT / "build" / "kaggle_code" / "liars-src.tar.gz"
    wheels = list((ROOT / "dist").glob("liars-*.whl"))
    upload_dataset(a.slug, {tar: tar.name, **{w: w.name for w in wheels}},
                   f"code {git('rev-parse', '--short', 'HEAD')}")


def cmd_push(a) -> None:
    if git("status", "--porcelain", "--untracked-files=no"):
        print("WARNING: uncommitted changes are NOT in the run (it uses the pushed commit).")
    ref = a.ref or git("rev-parse", "HEAD")
    if not a.ref and not git("branch", "-r", "--contains", ref):
        raise SystemExit("HEAD is not pushed to origin; run `git push` first (Kaggle clones from GitHub).")
    user, slug = kaggle_user(), kernel_slug(a.name)
    d = BUILD / a.name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    datasets = a.datasets or DEFAULT_DATASETS
    nb = build_notebook(a.cmd, ref=ref, data_datasets=datasets)
    (d / "run.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": "run.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/{s}" for s in [*datasets, CODE_DATASET]],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    r = kaggle("kernels", "push", "-p", str(d), "--accelerator", a.accelerator)
    print(r.stdout.strip())
    print(f"pushed {user}/{slug} @ {ref[:8]}: {a.cmd}")
    if a.wait:
        cmd_wait(a)


def _status(name: str) -> str:
    r = kaggle("kernels", "status", f"{kaggle_user()}/{kernel_slug(name)}", check=False)
    return (r.stdout + r.stderr).strip()


def cmd_status(a) -> None:
    print(_status(a.name))


def cmd_wait(a) -> None:
    t0 = time.time()
    while True:
        s = _status(a.name)
        low = s.lower()
        print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)
        if any(k in low for k in ("complete", "error", "cancel")):
            return
        time.sleep(POLL_SECONDS)


def merge_outputs(zip_path: Path, data_dir: Path = ROOT / "data") -> list[str]:
    """Unzip a Kaggle outputs.zip into data/: new files are copied, runs.csv rows appended
    (deduplicated by run name). Existing files are never overwritten."""
    added = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        src_data = Path(tmp) / "data"
        for p in sorted(src_data.rglob("*")):
            if not p.is_file() or p.name == "runs.csv":
                continue
            dst = data_dir / p.relative_to(src_data)
            if p.parent.name == "logs" or not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
                added.append(str(dst.relative_to(data_dir.parent)))
        runs = src_data / "runs.csv"
        if runs.exists():
            local = data_dir / "runs.csv"
            have = set()
            if local.exists():
                with open(local, newline="") as f:
                    have = {row["run"] for row in csv.DictReader(f)}
            with open(runs, newline="") as f:
                reader = csv.DictReader(f)
                new_rows = [row for row in reader if row["run"] not in have]
                fields = reader.fieldnames
            if new_rows:
                write_header = not local.exists()
                with open(local, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    if write_header:
                        w.writeheader()
                    w.writerows(new_rows)
                added += [f"runs.csv += {row['run']}" for row in new_rows]
    return added


def cmd_fetch(a) -> None:
    out = ROOT / "data" / "kaggle" / a.name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    kaggle("kernels", "output", f"{kaggle_user()}/{kernel_slug(a.name)}", "-p", str(out))
    print("downloaded:", ", ".join(p.name for p in out.iterdir()))
    z = out / "outputs.zip"
    if not z.exists():
        raise SystemExit(f"no outputs.zip in {out} (did the run fail? see the .log file there)")
    for line in merge_outputs(z):
        print("  +", line)


def cmd_merge(a) -> None:
    for line in merge_outputs(Path(a.zip)):
        print("  +", line)


def cmd_template(a) -> None:
    nb = build_notebook(["liars train --config configs/nli_zeroshot.yaml --test data/bench/test.jsonl"])
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {TEMPLATE_PATH}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    p = sub.add_parser("data")
    p.add_argument("--slug", default=DEFAULT_DATASETS[0])
    p.add_argument("--message", default="update")
    p.add_argument("files", nargs="+", help="files under data/, e.g. data/bench/train.jsonl")
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("code")
    p.add_argument("--slug", default=CODE_DATASET)
    p.set_defaults(fn=cmd_code)
    p = sub.add_parser("push")
    p.add_argument("--name", required=True)
    p.add_argument("--cmd", action="append", required=True, help="a liars command (repeatable)")
    p.add_argument("--ref", default=None, help="git ref (default: HEAD, must be pushed)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--accelerator", default=ACCELERATOR)
    p.add_argument("--wait", action="store_true")
    p.set_defaults(fn=cmd_push)
    for name, fn in (("status", cmd_status), ("wait", cmd_wait), ("fetch", cmd_fetch)):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.set_defaults(fn=fn)
    p = sub.add_parser("merge")
    p.add_argument("--zip", required=True)
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("template")
    p.set_defaults(fn=cmd_template)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

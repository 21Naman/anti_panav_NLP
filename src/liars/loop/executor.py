"""ACT: how an experiment's training command actually runs.

  LocalExecutor   `liars train ...` as a subprocess on this machine (CPU debug, or a GPU box)
  KaggleExecutor  scripts/kaggle_run.py push -> wait -> fetch, up to max_concurrent jobs at once
  DryExecutor     no training: writes simulated OOF/test predictions from the real train file
                  and folds, with a quality that depends on the branch and overrides, and can
                  inject failures. Exercises the whole loop (and its recovery) in seconds.
Every executor returns one ExecResult per experiment; it never decides what to do next.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from liars.loop.ledger import Experiment

RUN_RE = re.compile(r"^run (\S+): model=", re.MULTILINE)
FETCH_RE = re.compile(r"runs\.csv \+= (\S+)")


@dataclass
class ExecResult:
    ok: bool
    run: str | None
    log_tail: str
    gpu_h: float
    wall_s: float


def train_args(exp: Experiment, data: dict, data_dir: str, predict_test: bool) -> list[str]:
    """Arguments after `liars` for one training experiment."""
    args = ["train", "--config", exp.config,
            "--set", f"data.train={data['train']}", "--set", f"paths.data_dir={data_dir}"]
    if data.get("folds"):
        args += ["--set", f"data.folds={data['folds']}"]
    if predict_test and data.get("test"):
        args += ["--test", data["test"]]
    else:
        args += ["--set", "data.test=null"]
    for o in exp.overrides:
        args += ["--set", o]
    return args


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:]


class LocalExecutor:
    def __init__(self, log_dir: Path, timeout_h: float, uses_gpu: bool = False):
        self.log_dir, self.timeout_s, self.uses_gpu = Path(log_dir), timeout_h * 3600, uses_gpu
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_batch(self, exps: Sequence[Experiment], argv: Callable[[Experiment], list[str]]) -> list[ExecResult]:
        return [self._run(e, argv(e)) for e in exps]

    def _run(self, exp: Experiment, args: list[str]) -> ExecResult:
        log = self.log_dir / f"{exp.id}_try{exp.attempts}.log"
        cmd = [sys.executable, "-m", "liars.cli", *args]
        t0 = time.time()
        try:
            with open(log, "w") as fh:
                fh.write("$ " + " ".join(cmd) + "\n")
                fh.flush()
                r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=self.timeout_s)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -9
            with open(log, "a") as fh:
                fh.write(f"\nexceeded the time limit of {self.timeout_s / 3600:.1f} h\n")
        wall = time.time() - t0
        text = log.read_text(errors="replace")
        m = RUN_RE.findall(text)
        ok = code == 0 and bool(m)
        return ExecResult(ok, m[-1] if m else None, _tail(text), wall / 3600 if self.uses_gpu else 0.0, wall)


class KaggleExecutor:
    """One Kaggle notebook per experiment (name loop-<id>-<attempt>). Pushes up to
    max_concurrent, waits for all, fetches each. GPU hours = wall time from push to done
    (what the weekly quota counts, roughly)."""

    def __init__(self, root: Path, datasets: list[str], max_concurrent: int, poll_s: int = 60,
                 timeout_h: float = 12.0, runner: Callable[..., subprocess.CompletedProcess] | None = None,
                 sleep=time.sleep, max_unreadable: int = 10, fetch_tries: int = 3):
        self.max_unreadable, self.fetch_tries = max_unreadable, fetch_tries
        self.root, self.datasets, self.max_concurrent = Path(root), datasets, max_concurrent
        self.poll_s, self.timeout_s, self.sleep = poll_s, timeout_h * 3600, sleep
        self.runner = runner or (lambda *a: subprocess.run(
            [sys.executable, str(self.root / "scripts" / "kaggle_run.py"), *a],
            capture_output=True, text=True, cwd=str(self.root)))

    @staticmethod
    def job_name(exp: Experiment) -> str:
        return f"loop-{exp.id}-{exp.attempts}".lower()

    def run_batch(self, exps: Sequence[Experiment], argv: Callable[[Experiment], list[str]]) -> list[ExecResult]:
        out: list[ExecResult] = []
        for i in range(0, len(exps), self.max_concurrent):
            out += self._wave(exps[i : i + self.max_concurrent], argv)
        return out

    ENUM_RE = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
    WORD_RE = re.compile(r'has status "?([A-Za-z_]+)"?')
    DONE = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCELED"}

    def _state(self, text: str) -> str | None:
        """Kernel state from `kaggle_run.py status` output ('... has status
        "KernelWorkerStatus.COMPLETE"' or '... has status "complete"'), or None if it could not
        be read (CLI/network hiccup: the kernel is NOT assumed finished)."""
        m = self.ENUM_RE.findall(text)
        if m:
            return m[-1].upper()
        m = [w for w in self.WORD_RE.findall(text) if w != "KernelWorkerStatus"]
        return m[-1].upper() if m else None

    def _wave(self, exps, argv) -> list[ExecResult]:
        started, results, unreadable = {}, {}, {}
        for e in exps:
            cmd = "liars " + " ".join(_quote(a) for a in argv(e))
            r = self.runner("push", "--name", self.job_name(e), "--datasets", *self.datasets, "--cmd", cmd)
            if r.returncode != 0:
                results[e.id] = ExecResult(False, None, _tail(r.stdout + r.stderr), 0.0, 0.0)
            else:
                started[e.id] = time.time()
        pending = dict(started)
        while pending:
            for e in exps:
                if e.id not in pending:
                    continue
                s = self.runner("status", "--name", self.job_name(e))
                state = self._state(s.stdout + s.stderr)
                elapsed = time.time() - pending[e.id]
                if state is None:
                    unreadable[e.id] = unreadable.get(e.id, 0) + 1
                    if unreadable[e.id] < self.max_unreadable:
                        continue
                if state in self.DONE:
                    pending.pop(e.id)
                    results[e.id] = self._fetch(e, elapsed, state == "COMPLETE")
                elif elapsed > self.timeout_s or state is None:
                    pending.pop(e.id)
                    why = ("exceeded the time limit" if state is not None else
                           f"status unreadable {unreadable[e.id]} times")
                    results[e.id] = ExecResult(
                        False, None, f"KAGGLE_TIMEOUT_STILL_RUNNING: {self.job_name(e)} {why} after "
                        f"{elapsed / 3600:.1f} h; it may still be running and using quota. Cancel it on "
                        "kaggle.com, then fetch or resume.", elapsed / 3600, elapsed)
            if pending:
                self.sleep(self.poll_s)
        return [results[e.id] for e in exps]

    def _fetch(self, e: Experiment, wall: float, completed: bool) -> ExecResult:
        text, runs, code = "", [], 1
        for attempt in range(self.fetch_tries):  # re-fetch, never re-train, on a download hiccup
            f = self.runner("fetch", "--name", self.job_name(e))
            text, code = f.stdout + f.stderr, f.returncode
            runs = FETCH_RE.findall(text)
            if code == 0 or "no outputs.zip" in text:
                break
            self.sleep(self.poll_s)
        logs = sorted((self.root / "data" / "kaggle" / self.job_name(e)).glob("*.log"))
        if logs:
            text += "\n--- kernel log ---\n" + _tail(logs[-1].read_text(errors="replace"), 6000)
        if completed and code != 0 and "no outputs.zip" not in text:
            text = (f"KAGGLE_OUTPUTS_NOT_FETCHED: {self.job_name(e)} completed but its outputs could not "
                    f"be downloaded; run `python scripts/kaggle_run.py fetch --name {self.job_name(e)}`.\n" + text)
        ok = completed and code == 0 and bool(runs)
        return ExecResult(ok, runs[-1] if runs else None, _tail(text), wall / 3600, wall)


def _quote(a: str) -> str:
    return a if re.fullmatch(r"[\w./=:,@+-]+", a) else "'" + a.replace("'", "'\\''") + "'"


class DryExecutor:
    """Simulated runs. quality(branch, overrides) = base[branch] + a deterministic +-effect per
    override, so the loop has something real to optimise. `failures` maps an experiment id
    (or branch name) to a list of log texts returned on its 1st, 2nd, ... attempt."""

    def __init__(self, data_dir: Path, train_path: str, test_path: str | None, folds,
                 cfgs: dict[str, dict], base_quality: dict[str, float], est_gpu_h: dict[str, float],
                 failures: dict[str, list[str]] | None = None, effect: float = 0.06):
        self.data_dir = Path(data_dir)
        self._paths, self._data = (train_path, test_path), None
        self._folds, self.cfgs, self.base, self.est = folds, cfgs, base_quality, est_gpu_h
        self.failures, self.effect = failures or {}, effect
        self.calls: list[str] = []

    def _load(self):
        if self._data is None:
            from liars.schema import load_jsonl

            tr, te = self._paths
            self._data = (load_jsonl(tr), load_jsonl(te) if te else [])
        return self._data

    @property
    def train(self):
        return self._load()[0]

    @property
    def test(self):
        return self._load()[1]

    @property
    def folds(self) -> dict[str, int]:
        return self._folds() if callable(self._folds) else self._folds

    def run_batch(self, exps, argv) -> list[ExecResult]:
        return [self._run(e) for e in exps]

    def _quality(self, e: Experiment) -> float:
        q = self.base.get(e.branch, 0.6)
        for o in sorted(e.overrides):
            h = int(hashlib.sha1(f"{e.branch}|{o}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            q += (h - 0.5) * 2 * self.effect
        return float(np.clip(q, 0.05, 0.98))

    def _run(self, e: Experiment) -> ExecResult:
        from liars.preds import make_preds, save_preds
        from liars.schema import SPEAKERS

        self.calls.append(e.id)
        plan = self.failures.get(e.id) or self.failures.get(e.branch)
        if plan and e.attempts <= len(plan) and plan[e.attempts - 1]:
            return ExecResult(False, None, plan[e.attempts - 1], 0.1, 1.0)
        cfg = self.cfgs[e.id]
        mode, hf = cfg["train"]["mode"], cfg["cv"]["holdout_fold"]
        held = [ex for ex in self.train if mode == "kfold" or self.folds[ex.id] == hf]
        run = f"dry_{e.branch}_{e.id}_{e.attempts}"
        seed = int(hashlib.sha1(run.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        q = self._quality(e)

        def probs(exs, labelled: bool):
            p = rng.dirichlet(np.ones(4), size=len(exs))
            for i, ex in enumerate(exs):
                if labelled and rng.random() < q:
                    p[i, SPEAKERS.index(ex.label)] += 2.0
                elif not labelled:
                    p[i, rng.integers(4)] += 2.0 * q
            return p / p.sum(1, keepdims=True)

        oof = make_preds([ex.id for ex in held], probs(held, True))
        oof.insert(1, "fold", [self.folds[ex.id] for ex in held])
        oof.insert(2, "label", [ex.label for ex in held])
        save_preds(oof, self.data_dir / "oof" / f"{run}.parquet")
        if self.test:
            save_preds(make_preds([ex.id for ex in self.test], probs(self.test, False)),
                       self.data_dir / "preds" / f"{run}.parquet")
        return ExecResult(True, run, f"run {run}: model=dry\ndone", self.est.get(e.branch, 0.5), 0.1)

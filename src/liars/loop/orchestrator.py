"""The cycle: PLAN -> (guardrails) -> ACT (+ recovery) -> TEST -> EVALUATE -> IMPROVE -> repeat.

One iteration:
  1 plan        planner.plan(): proposals, else branch baselines, else one-knob mutations
  2 guard       guardrails.pre_run() on each; failures are logged as skipped, never run
  3 smoke       first run of a branch: `liars train --debug` on CPU first (catches config and
                code errors before GPU time is spent), if loop.smoke_test
  4 act         executor.run_batch(); failures go through recovery.decide(): retry with
                backoff, mutate-and-retry (OOM / too slow / NaN), fail, or escalate
  5 test        evaluate.sanity(): files valid, held-out coverage, above chance, no letter
                collapse, test ids match; failing runs are rejected
  6 evaluate    metrics + paired comparison with the champion (McNemar + bootstrap,
                Bonferroni over the round's candidates)
  7 improve     promote or reject; then recombine the best run of each branch (vote,
                Caruana, weighted, ...) and evaluate those the same way
  8 stop?       budget, max_iters, patience, target, holdout-query budget, nothing left
Everything is written to the ledger; report.md is regenerated after every iteration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd

from liars.config import apply_overrides, load_config
from liars.cv import folds_path_for, load_folds
from liars.loop import evaluate as ev
from liars.loop.catalog import Branch, check_branch, merged
from liars.loop.executor import LocalExecutor, train_args
from liars.loop.guardrails import check_budget, check_overrides, check_test_unlabelled, pre_run
from liars.loop.ledger import (FAILED, PLANNED, PROMOTED, REJECTED, RUNNING, SKIPPED, SUCCEEDED,
                               Experiment, Ledger, State)
from liars.loop.planner import plan, plan_recombination
from liars.loop.recovery import classify, decide
from liars.paths import Paths
from liars.schema import SPEAKERS, load_jsonl


class Loop:
    def __init__(self, lcfg: dict, branches: dict[str, Branch], executor, log: Callable[[str], None] = print,
                 smoke_executor=None, sleep=time.sleep, root: Path = Path(".")):
        self.cfg, self.branches, self.ex, self.log, self.sleep = lcfg, branches, executor, log, sleep
        self.root = Path(root)
        self.ledger = Ledger(lcfg["ledger_dir"])
        self.paths = Paths(Path(lcfg["data_dir"]))
        self.data = dict(lcfg["data"])
        # Folds always live next to the canonical data (data/canonical/folds_<stem>.csv), NOT
        # under loop.data_dir (that is only where run outputs go). Resolved once and passed to
        # every run explicitly, so the trainer and the loop use the same file.
        if not self.data.get("folds"):
            self.data["folds"] = str(folds_path_for(self.data["train"], Paths(Path("data")).canonical))
        self.smoke = smoke_executor
        self._train = None
        self._folds = None

    # ------------------------------------------------------------------ helpers
    @property
    def train(self):
        if self._train is None:
            self._train = load_jsonl(self.data["train"])
        return self._train

    @property
    def folds(self) -> dict[str, int]:
        if self._folds is None:
            self._folds = load_folds(self.data["folds"])
        return self._folds

    def exp_cfg(self, e: Experiment) -> dict:
        b = self.branches[e.branch]
        return apply_overrides(load_config(self.root / b.config), merged(b.base + e.overrides))

    def expected_oof_n(self, e: Experiment) -> int:
        c = self.exp_cfg(e)
        if c["train"]["mode"] == "kfold":
            return len(self.train)
        hf = c["cv"]["holdout_fold"]
        return sum(1 for ex in self.train if self.folds.get(ex.id) == hf)

    def family_keys(self, e: Experiment) -> dict:
        """Knob names of the branch's model family + their CURRENT values from the resolved
        config (so recovery halves the real micro batch, not a guessed default)."""
        fam = dict((self.cfg.get("families") or {}).get(self.branches[e.branch].family) or {})
        cfg = self.exp_cfg(e)
        for role in ("micro_batch", "grad_accum", "epochs", "lr"):
            key = fam.get(role)
            if not key:
                continue
            node = cfg
            try:
                for part in key.split("."):
                    node = node[part]
                fam[f"{role}_default"] = node
            except (KeyError, TypeError):
                fam.pop(role)
        return fam

    def argv(self, e: Experiment) -> list[str]:
        full = Experiment(**{**e.__dict__, "overrides": merged(self.branches[e.branch].base + e.overrides)})
        return train_args(full, self.data, self.cfg["data_dir"], self.cfg["predict_test"])

    def _read(self, path: Path) -> pd.DataFrame | None:
        return pd.read_parquet(path) if path.exists() else None

    # ------------------------------------------------------------------ preflight
    def preflight(self) -> list[str]:
        errs = []
        for k in ("train",):
            if not Path(self.data[k]).exists():
                errs.append(f"data.{k} {self.data[k]} not found")
        if self.data.get("test") and not Path(self.data["test"]).exists():
            errs.append(f"data.test {self.data['test']} not found")
        if errs:
            return errs
        try:
            missing = [ex.id for ex in self.train if ex.id not in self.folds]
            if missing:
                errs.append(f"{len(missing)} train ids have no fold (run `liars folds --in {self.data['train']}`)")
        except FileNotFoundError:
            errs.append(f"no folds file {self.data['folds']} for {self.data['train']} "
                        f"(run `liars folds --in {self.data['train']} --k 5`)")
        if self.data.get("test"):
            c = check_test_unlabelled(load_jsonl(self.data["test"]), bool(self.cfg.get("allow_labelled_test")))
            if not c.ok:
                errs.append(c.why)
        for b in self.branches.values():
            if b.enabled:
                errs += check_branch(b, self.root)
                c = check_overrides(b.base, self.cfg["guardrails"], bool(self.cfg.get("allow_pseudo")))
                if not c.ok:
                    errs.append(f"{b.name}: base overrides: {c.why}")
        return errs

    # ------------------------------------------------------------------ main
    def run(self, max_iters: int | None = None) -> State:
        st = self.ledger.load_state()
        if st.status != "active":
            self.log(f"loop is {st.status} ({st.stop_reason}); `liars loop resume` to continue")
            return st
        self.recover_interrupted()
        errs = self.preflight()
        if errs:
            self.ledger.escalate(st, "preflight failed", {"problems": "; ".join(errs)})
            self.log("ESCALATED: preflight failed:\n  " + "\n  ".join(errs))
            self.write_report(st)
            return st
        iters = 0
        limit = max_iters if max_iters is not None else self.cfg["stop"]["max_iters"]
        while st.status == "active" and iters < limit:
            iters += 1
            st.iteration += 1
            self.log(f"=== iteration {st.iteration} ===")
            improved = self.iteration(st)
            if st.status != "active":
                break
            st.no_improve_iters = 0 if improved else st.no_improve_iters + 1
            self.check_stop(st)
            self.ledger.save_state(st)
            self.write_report(st)
        self.ledger.save_state(st)
        self.write_report(st)
        self.log(f"loop {st.status}" + (f": {st.stop_reason}" if st.stop_reason else "")
                 + (f"; champion {st.champion}" if st.champion else ""))
        return st

    def recover_interrupted(self) -> None:
        """Jobs left planned/running by a crash, Ctrl-C or an escalation mid-batch are marked
        failed ('interrupted') so the planner may try them again; they never sit forever."""
        for e in self.ledger.all().values():
            if e.status in (PLANNED, RUNNING):
                e.status, e.decision = FAILED, "interrupted (loop stopped before it finished); may be re-planned"
                e.errors.append({"class": "interrupted", "attempt": e.attempts, "tail": ""})
                self.ledger.put(e)
                self.log(f"  {e.id} was left unfinished: marked interrupted")

    def scaled_estimate(self, e: Experiment) -> float:
        """Branch estimates assume the branch config's train.mode; kfold costs k fits."""
        b = self.branches[e.branch]
        base = apply_overrides(load_config(self.root / b.config), merged(b.base))
        cur = self.exp_cfg(e)
        est = e.est_gpu_h
        if base["train"]["mode"] != "kfold" and cur["train"]["mode"] == "kfold":
            est *= cur["cv"]["k"]
        elif base["train"]["mode"] == "kfold" and cur["train"]["mode"] != "kfold":
            est /= base["cv"]["k"]
        return est

    def iteration(self, st: State) -> bool:
        champ_before = st.champion
        done = self.ledger.all()
        batch = plan(self.ledger, st, self.branches, self.cfg)
        runnable = []
        for e in batch:
            if e.kind == "train" and e.branch in self.branches:
                try:
                    e.est_gpu_h = self.scaled_estimate(e)
                except (KeyError, ValueError):
                    pass  # a bad override; pre_run / the run itself reports it
            c = pre_run(e, st, done, self.cfg)
            if not c.ok:
                e.status, e.decision = SKIPPED, c.why
                self.log(f"  skip {e.id} ({e.branch}): {c.why}")
            else:
                runnable.append(e)
                self.log(f"  plan {e.id} {e.branch} {e.overrides or '(baseline)'}: {e.hypothesis}")
            self.ledger.put(e)
            done[e.id] = e
        if runnable:
            runnable = self.smoke_test(st, runnable)
        if runnable and st.status == "active":
            self.act(st, runnable)
        if st.status != "active":
            return False
        evaluated = [e for e in runnable if e.status == SUCCEEDED]
        self.evaluate_round(st, evaluated)
        if st.status == "active" and self.cfg["recombine"]["enabled"]:
            ens = []
            for e in plan_recombination(self.ledger, self.cfg["recombine"]):
                c = pre_run(e, st, self.ledger.all(), self.cfg)
                if c.ok:
                    ens.append(self.run_ensemble(e))
            self.evaluate_round(st, [e for e in ens if e.status == SUCCEEDED])
        if not batch and not runnable:
            left = self.cfg["budget"]["gpu_hours"] - st.gpu_h_used
            st.stop_reason = st.stop_reason or (
                f"search exhausted: nothing left to plan that fits the remaining {left:.1f} GPU h "
                "(untried ideas may need more budget; see report)")
        return st.champion != champ_before

    # ------------------------------------------------------------------ smoke
    def smoke_test(self, st: State, exps: list[Experiment]) -> list[Experiment]:
        """CPU `--debug` run of each branch's first experiment; a branch that cannot even
        run on tiny data never reaches the GPU."""
        if not (self.cfg.get("smoke_test") and self.smoke):
            return exps
        seen = {e.branch for e in self.ledger.all().values() if e.kind == "train" and e.status in
                (SUCCEEDED, PROMOTED, REJECTED)}
        keep = []
        for e in exps:
            if e.branch in seen:
                keep.append(e)
                continue
            args = ["train", "--config", e.config, "--debug",
                    *sum((["--set", o] for o in merged(self.branches[e.branch].base + e.overrides)), [])]
            res = self.smoke.run_batch([e], lambda _e: args)[0]
            if res.ok:
                self.log(f"  smoke {e.id} ok")
                keep.append(e)
                continue
            cls = classify(res.log_tail, self.cfg["recovery"].get("patterns"))
            e.status, e.decision = FAILED, f"smoke test ({cls}) failed before any GPU time"
            e.errors.append({"class": cls, "stage": "smoke", "tail": res.log_tail[-800:]})
            self.ledger.put(e)
            self.log(f"  smoke {e.id} FAILED ({cls})")
            if cls == "human":
                for other in exps:
                    if other.status not in (FAILED,) and other is not e:
                        other.status, other.decision = FAILED, "interrupted: loop escalated during smoke tests"
                        other.errors.append({"class": "interrupted", "attempt": 0, "tail": ""})
                        self.ledger.put(other)
                self.ledger.escalate(st, f"smoke test of {e.id} needs a human ({cls})",
                                     {"log tail": res.log_tail[-600:]})
                return []
        return keep

    # ------------------------------------------------------------------ act + recovery
    def act(self, st: State, exps: list[Experiment]) -> None:
        rcfg = self.cfg["recovery"]
        pending = list(exps)
        for e in pending:
            e.status = RUNNING
        while pending and st.status == "active":
            for e in pending:
                e.attempts += 1
                self.ledger.put(e)
            results = self.ex.run_batch(pending, self.argv)
            nxt = []
            escalation = None
            for e, res in zip(pending, results):
                st.gpu_h_used += res.gpu_h
                e.gpu_h += res.gpu_h
                if res.ok:
                    e.run, e.status = res.run, SUCCEEDED
                    self.ledger.put(e)
                    self.log(f"  done {e.id} -> run {res.run} ({res.gpu_h:.2f} GPU h)")
                    continue
                cls = classify(res.log_tail, rcfg.get("patterns"))
                e.errors.append({"class": cls, "attempt": e.attempts, "tail": res.log_tail[-800:]})
                hit = st.error_counts.setdefault(cls, [])
                if e.id not in hit:
                    hit.append(e.id)
                fam = self.family_keys(e)
                action = decide(cls, e.attempts, merged(self.branches[e.branch].base + e.overrides), rcfg, fam)
                systemic = cls not in ("transient",) and len(hit) >= rcfg["systemic_after"]
                self.log(f"  FAIL {e.id} attempt {e.attempts} [{cls}] -> {action.kind}: {action.why}")
                if action.kind == "escalate" or systemic:
                    e.status, e.decision = FAILED, action.why
                    self.ledger.put(e)
                    why = (f"error class '{cls}' hit {len(hit)} experiments ({', '.join(hit)}): "
                           "systemic problem" if systemic and action.kind != "escalate" else action.why)
                    escalation = escalation or (why, {"experiment": e.id, "branch": e.branch,
                                                      "error class": cls, "log tail": res.log_tail[-600:]})
                    continue  # the other results of this batch are still recorded
                if action.kind == "fail":
                    e.status, e.decision = FAILED, action.why
                    self.ledger.put(e)
                    continue
                if action.kind == "mutate":
                    if not e.planned_overrides:
                        e.planned_overrides = list(e.overrides)   # the planner keeps it as tried
                    e.overrides = merged(e.overrides + action.overrides)
                    c = check_overrides(e.overrides, self.cfg["guardrails"], bool(self.cfg.get("allow_pseudo")))
                    if not c.ok:
                        e.status, e.decision = FAILED, f"recovery change blocked by guardrails: {c.why}"
                        self.ledger.put(e)
                        continue
                    e.decision = f"recovery: {action.why}"
                b = check_budget(e, st, self.cfg["budget"])
                if not b.ok:
                    e.status, e.decision = FAILED, f"no budget left to retry: {b.why}"
                    self.ledger.put(e)
                    continue
                if action.wait_s:
                    self.sleep(action.wait_s)
                self.ledger.put(e)
                nxt.append(e)
            if escalation:
                for e in nxt:   # retries that will not happen now
                    e.status, e.decision = FAILED, "interrupted: loop escalated before this retry"
                    self.ledger.put(e)
                self.ledger.escalate(st, *escalation)
                return
            pending = nxt
            self.ledger.save_state(st)

    # ------------------------------------------------------------------ test + evaluate
    def evaluate_round(self, st: State, exps: list[Experiment]) -> None:
        if not exps:
            return
        pcfg = self.cfg["promotion"]
        test_ids = [ex.id for ex in load_jsonl(self.data["test"])] if (
            self.data.get("test") and self.cfg["predict_test"]) else None
        k = max(1, len(exps)) if pcfg.get("bonferroni") else 1
        lie_types = {ex.id: ex.meta.get("lie_type") for ex in self.train} if pcfg.get("by_group") else {}
        # sort so the strongest candidate is compared first
        scored = []
        for e in exps:
            oof = self._read(self.paths.oof / f"{e.run}.parquet")
            test = self._read(self.paths.preds / f"{e.run}.parquet") if test_ids is not None else None
            s = ev.sanity(oof, test, test_ids, self.cfg["sanity"],
                          self.expected_oof_n(e) if e.kind == "train" else None)
            if not s.ok:
                e.status, e.decision = REJECTED, "sanity: " + "; ".join(s.failures)
                e.metrics["warnings"] = s.warnings
                self.ledger.put(e)
                self.log(f"  reject {e.id}: {e.decision}")
                continue
            ids, P, y = ev.oof_arrays(oof)
            e.metrics = ev.metrics(P, y, self.cfg["sanity"]["ece_bins"])
            e.metrics["warnings"] = s.warnings
            if lie_types and any(lie_types.values()):
                g = pd.DataFrame({"t": [lie_types.get(i) for i in ids], "c": P.argmax(1) == y})
                e.metrics["by_lie_type"] = {str(k_): round(float(v), 4) for k_, v in g.groupby("t")["c"].mean().items()}
            scored.append((e, oof, s))
        scored.sort(key=lambda t: (-t[0].metrics["acc"], t[0].metrics["nll"]))
        for i, (e, oof, s) in enumerate(scored):
            if s.warnings:
                self.log(f"  warn {e.id}: {'; '.join(s.warnings)}")
                if self.cfg["guardrails"].get("escalate_on_suspicious") and any("too good" in w for w in s.warnings):
                    e.decision = "suspiciously good: held for a human"
                    self.ledger.put(e)
                    for later, _, _ in scored[i + 1:]:   # not compared yet: leave them re-plannable
                        later.status, later.decision = FAILED, "interrupted: loop escalated before its evaluation"
                        later.errors.append({"class": "interrupted", "attempt": later.attempts, "tail": ""})
                        self.ledger.put(later)
                    self.ledger.escalate(st, f"{e.id} looks too good to be true", {"warnings": s.warnings,
                                                                                     "metrics": e.metrics})
                    return
            self.decide_promotion(st, e, oof, k)
            if st.status != "active":
                return

    def decide_promotion(self, st: State, e: Experiment, oof: pd.DataFrame, k: int) -> None:
        pcfg = dict(self.cfg["promotion"])
        pcfg["alpha"] = pcfg["alpha"] / k
        champ = self.ledger.get(st.champion) if st.champion else None
        if champ is None:
            verdict = ev.Verdict(True, "first_valid_run", {})
        else:
            st.holdout_queries += 1
            champ_oof = self._read(self.paths.oof / f"{champ.run}.parquet")
            verdict = ev.compare(oof, champ_oof, pcfg, seed=st.iteration)
        e.metrics["vs_champion"] = verdict.detail
        if verdict.promote:
            st.champion = e.id
            st.n_promotions += 1
            st.champion_history.append({"id": e.id, "run": e.run, "acc": e.metrics["acc"],
                                        "rule": verdict.rule, "iteration": st.iteration})
            e.status, e.decision = PROMOTED, f"promoted ({verdict.rule})"
            self.log(f"  PROMOTE {e.id} acc {e.metrics['acc']:.4f} ({verdict.rule})")
        else:
            e.status = SUCCEEDED
            e.decision = f"kept, not promoted ({verdict.rule})"
            d = verdict.detail
            if d:
                self.log(f"  keep {e.id} acc {e.metrics['acc']:.4f}: gain {d.get('acc_gain', 0):+.4f}, "
                         f"McNemar p {d.get('mcnemar_p', 1):.3f} (alpha {pcfg['alpha']:.3f})")
        self.ledger.put(e)

    # ------------------------------------------------------------------ recombine
    def run_ensemble(self, e: Experiment) -> Experiment:
        from liars.ensemble import run_ensemble

        ecfg = load_config(self.root / self.cfg["recombine"]["ensemble_config"])["ensemble"]
        name = f"loop_{e.id}"
        try:
            run_ensemble(self.paths, e.members, e.method, name, ecfg, log=lambda *a: None)
            e.run, e.status = f"ens_{name}", SUCCEEDED
            self.log(f"  ensemble {e.id} {e.method} over {len(e.members)} runs")
        except Exception as err:  # an ensemble failing is never fatal for the loop
            e.status, e.decision = FAILED, f"ensemble failed: {type(err).__name__}: {err}"
            self.log(f"  ensemble {e.id} FAILED: {err}")
        self.ledger.put(e)
        return e

    # ------------------------------------------------------------------ stop
    def check_stop(self, st: State) -> None:
        s, b = self.cfg["stop"], self.cfg["budget"]
        champ = self.ledger.get(st.champion) if st.champion else None
        left = b["gpu_hours"] - st.gpu_h_used
        cheapest = min((x.est_gpu_h for x in self.branches.values() if x.enabled), default=0)
        reason = ""
        if st.stop_reason.startswith("search exhausted"):
            reason = st.stop_reason
        elif champ and s.get("target_acc") and champ.metrics.get("acc", 0) >= s["target_acc"]:
            reason = f"target accuracy {s['target_acc']} reached by {champ.id}"
        elif st.no_improve_iters >= s["patience"]:
            reason = f"no promotion for {st.no_improve_iters} iterations (patience {s['patience']})"
        elif left < cheapest * b["margin"]:
            reason = f"GPU budget nearly spent ({left:.1f} h left < cheapest run)"
        elif st.iteration >= s["max_iters"]:
            reason = f"max_iters {s['max_iters']} reached"
        q = st.holdout_queries
        if not reason and q >= s["holdout_query_budget"]:
            self.ledger.escalate(st, f"holdout looked at {q} times (budget {s['holdout_query_budget']}): "
                                     "further promotions risk fitting the holdout; refresh it or raise the budget",
                                 {"champion": st.champion})
            return
        if reason:
            st.status, st.stop_reason = "stopped", reason

    # ------------------------------------------------------------------ report
    def write_report(self, st: State) -> Path:
        exps = self.ledger.all()
        champ = exps.get(st.champion) if st.champion else None
        rows = []
        for e in exps.values():
            m = e.metrics or {}
            rows.append({"id": e.id, "branch": e.branch, "status": e.status,
                         "acc": round(m["acc"], 4) if "acc" in m else None,
                         "nll": round(m["nll"], 4) if "nll" in m else None,
                         "gpu_h": round(e.gpu_h, 2), "change": " ".join(e.overrides) or e.method or "baseline",
                         "decision": (e.decision or "")[:90]})
        df = pd.DataFrame(rows)
        lines = ["# Improvement loop report", "",
                 f"- status: **{st.status}**" + (f" ({st.stop_reason})" if st.stop_reason else ""),
                 f"- iteration: {st.iteration}; experiments: {len(exps)}; promotions: {st.n_promotions}",
                 f"- GPU hours used: {st.gpu_h_used:.2f} of {self.cfg['budget']['gpu_hours']}",
                 f"- holdout comparisons so far: {st.holdout_queries} "
                 f"(budget {self.cfg['stop']['holdout_query_budget']})"]
        if champ:
            m = champ.metrics
            lines += [f"- champion: **{champ.id}** ({champ.branch}, run `{champ.run}`): acc {m.get('acc', 0):.4f}, "
                      f"log-loss {m.get('nll', 0):.4f}, ECE {m.get('ece', 0):.3f} on {m.get('n')} held-out chats"]
            n = m.get("n") or 1
            lines += [f"- smallest gain McNemar can reliably see here (10% disagreement): "
                      f"~{100 * ev.min_detectable_gain(n, 0.10):.1f} points"]
            if self.data.get("test") and self.cfg["predict_test"]:
                lines += ["", "Submit the champion (a human decides; the loop never uploads):", "",
                          f"    liars submit --preds {self.paths.preds}/{champ.run}.parquet "
                          f"--out {self.paths.subs}/{champ.id}.csv --test {self.data['test']}"]
        if st.status == "escalated":
            lines += ["", f"**ESCALATED** — see {self.ledger.root / 'ESCALATE.md'}"]
        lines += ["", "## Experiments", ""]
        if len(df):
            df = df.sort_values(["acc", "id"], ascending=[False, True], na_position="last")
            lines.append(df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False))
        lines += ["", "## Champion history", ""]
        lines += [f"- iter {h['iteration']}: {h['id']} acc {h['acc']:.4f} ({h['rule']})" for h in st.champion_history] or ["- none yet"]
        path = self.ledger.root / "report.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def build_loop(lcfg: dict, executor_name: str, log=print, sleep=time.sleep, dry_failures=None) -> Loop:
    """Construct the loop with the chosen executor (dry | local | kaggle)."""
    from liars.loop.catalog import load_branches
    from liars.loop.executor import DryExecutor, KaggleExecutor

    branches = load_branches(lcfg)
    loop_cfg = lcfg["loop"]
    loop_cfg.setdefault("families", lcfg.get("families") or {})
    smoke = LocalExecutor(Path(loop_cfg["ledger_dir"]) / "logs" / "smoke", loop_cfg["smoke_timeout_h"]) \
        if loop_cfg.get("smoke_test") and executor_name != "dry" else None
    if executor_name == "local":
        ex = LocalExecutor(Path(loop_cfg["ledger_dir"]) / "logs", loop_cfg["budget"]["per_run_max_h"] * 1.5,
                           uses_gpu=loop_cfg.get("local_counts_gpu", False))
    elif executor_name == "kaggle":
        if Path(loop_cfg["data_dir"]).resolve() != (Path(".") / "data").resolve():
            raise ValueError("the kaggle executor merges outputs into ./data: set loop.data_dir: data")
        ex = KaggleExecutor(Path("."), loop_cfg["kaggle"]["datasets"], loop_cfg["budget"]["max_concurrent"],
                            loop_cfg["kaggle"]["poll_s"], loop_cfg["budget"]["per_run_max_h"] * 1.5, sleep=sleep)
    elif executor_name == "dry":
        # simulated runs make independent errors, so a vote of them can look "too good":
        # in dry mode that check only warns (it still escalates for real executors)
        loop_cfg["guardrails"]["escalate_on_suspicious"] = False
        loop = Loop(loop_cfg, branches, None, log=log, sleep=sleep)
        cfgs = _CfgView(loop)
        ex = DryExecutor(Path(loop_cfg["data_dir"]), loop_cfg["data"]["train"],
                         loop_cfg["data"].get("test") if loop_cfg["predict_test"] else None,
                         lambda: loop.folds, cfgs, {k: v for k, v in (loop_cfg.get("dry") or {}).get("base_quality", {}).items()},
                         {b.name: b.est_gpu_h for b in branches.values()}, failures=dry_failures)
        loop.ex = ex
        return loop
    else:
        raise ValueError(f"executor must be dry | local | kaggle, got {executor_name!r}")
    return Loop(loop_cfg, branches, ex, log=log, smoke_executor=smoke, sleep=sleep)


class _CfgView(dict):
    """Lazy experiment-id -> resolved config, for the dry executor."""

    def __init__(self, loop: Loop):
        super().__init__()
        self.loop = loop

    def __getitem__(self, exp_id):
        return self.loop.exp_cfg(self.loop.ledger.get(exp_id))

"""PLAN: decide the next batch of experiments.

Order of business each iteration (first one that yields work wins, up to max_concurrent):
  1. proposals  validated entries from data/loop/proposals.yaml (the LLM planner agent's ideas)
  2. explore    a baseline run for every enabled branch that has none yet (prior high -> low,
                then cheapest first)
  3. exploit    one-knob mutations of the best run of the top `exploit_top_k` branches: each knob
                in the branch's space moves to a neighbouring value (greedy coordinate search,
                like AIDE's improve step); untried mutations of the best branch come first
  4. nothing    -> the orchestrator recombines and/or stops
Recombination (ensembles of the best run per branch) is planned separately after every
iteration, because it costs no GPU.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from liars.loop.catalog import PRIOR_RANK, Branch, as_dict, signature
from liars.loop.ledger import FAILED, PROMOTED, SKIPPED, SUCCEEDED, Experiment, Ledger, State

GOOD = {SUCCEEDED, PROMOTED}


def best_per_branch(exps: dict[str, Experiment]) -> dict[str, Experiment]:
    out: dict[str, Experiment] = {}
    for e in exps.values():
        if e.kind != "train" or e.status not in GOOD or "acc" not in e.metrics:
            continue
        cur = out.get(e.branch)
        if cur is None or (e.metrics["acc"], -e.metrics["nll"]) > (cur.metrics["acc"], -cur.metrics["nll"]):
            out[e.branch] = e
    return out


def _tried(exps: dict[str, Experiment]) -> set[str]:
    """Signatures never to plan again: everything except failed runs (which may be retried
    as new experiments). Skipped ones count (a guardrail said no, it will say no again), and a
    run changed by recovery also blocks its ORIGINAL overrides (the config that failed)."""
    out = set()
    for e in exps.values():
        if e.kind != "train" or e.status == FAILED:
            continue
        out.add(signature(e.branch, e.overrides))
        if e.planned_overrides:
            out.add(signature(e.branch, e.planned_overrides))
    return out


def _failed_baselines(exps: dict[str, Experiment]) -> dict[str, int]:
    """Failed baseline attempts per branch, not counting interruptions or failures that went
    to a human (those are not the branch's fault)."""
    out: dict[str, int] = {}
    for e in exps.values():
        if e.kind == "train" and e.source == "planner:baseline" and e.status == FAILED:
            classes = {err.get("class") for err in e.errors}
            if classes & {"interrupted", "human"}:
                continue
            out[e.branch] = out.get(e.branch, 0) + 1
    return out


def _neighbours(values: list, current) -> list:
    """Values next to `current` in the ordered list (both sides); all values if not in it."""
    strs = [str(v) for v in values]
    if str(current) not in strs:
        return values
    i = strs.index(str(current))
    return [values[j] for j in (i + 1, i - 1) if 0 <= j < len(values)]


def mutations(branch: Branch, best: Experiment, tried: set[str]) -> list[tuple[list[str], str]]:
    cur = as_dict(best.overrides)
    out = []
    for knob, values in branch.space.items():
        # a knob the best run did not override is at the config default = the first listed
        # value (catalog.check_branch enforces this)
        current = cur.get(knob, values[0])
        for v in _neighbours(values, current):
            ov = dict(cur)
            ov[knob] = str(v)
            o = [f"{k}={val}" for k, val in ov.items()]
            if signature(branch.name, o) not in tried:
                out.append((o, f"{knob}: {cur.get(knob, values[0])} -> {v} (from {best.id})"))
    return out


def load_proposals(path: Path, seen: set[str]) -> list[dict]:
    """New entries of proposals.yaml (a list of {id, branch, overrides, hypothesis, by}).
    Entries whose id was already ingested are ignored, so the file can just grow."""
    if not path.exists():
        return []
    items = yaml.safe_load(path.read_text()) or []
    if not isinstance(items, list):
        raise ValueError(f"{path}: must be a YAML list of proposals")
    out = []
    for it in items:
        if not isinstance(it, dict) or "id" not in it or "branch" not in it:
            raise ValueError(f"{path}: every proposal needs at least id and branch: {it}")
        if f"proposal:{it['id']}" not in seen:
            out.append(it)
    return out


def plan(ledger: Ledger, st: State, branches: dict[str, Branch], lcfg: dict) -> list[Experiment]:
    exps = ledger.all()
    tried = _tried(exps)
    n_max = lcfg["budget"]["max_concurrent"]
    left = lcfg["budget"]["gpu_hours"] - st.gpu_h_used
    out: list[Experiment] = []
    counter = [ledger.max_num()]

    def new(branch: Branch, overrides: list[str], hypothesis: str, source: str, parent=None) -> Experiment:
        counter[0] += 1
        return Experiment(id=f"e{counter[0]:03d}", branch=branch.name, kind="train", config=branch.config,
                          overrides=overrides, hypothesis=hypothesis, source=source, parent=parent,
                          est_gpu_h=branch.est_gpu_h)

    def room(b: Branch) -> bool:
        used = sum(e.est_gpu_h for e in out)
        return len(out) < n_max and used + b.est_gpu_h * lcfg["budget"]["margin"] <= left + 1e-9

    # 1. proposals
    seen = {e.source for e in exps.values()}
    for p in load_proposals(Path(lcfg["ledger_dir"]) / "proposals.yaml", seen):
        b = branches.get(p["branch"])
        if b is None:
            e = Experiment(id=f"e{counter[0] + 1:03d}", branch=str(p["branch"]), kind="train", config="",
                           source=f"proposal:{p['id']}", status=SKIPPED,
                           decision=f"unknown branch {p['branch']!r}")
            counter[0] += 1
            ledger.put(e)
            continue
        if not room(b):
            break
        e = new(b, [str(o) for o in p.get("overrides") or []], str(p.get("hypothesis", "")),
                f"proposal:{p['id']}", p.get("parent"))
        if p.get("est_gpu_h") is not None:   # a proposal may raise the estimate, never lower it
            e.est_gpu_h = max(b.est_gpu_h, float(p["est_gpu_h"]))
        out.append(e)
    if out:
        return out

    # 2. explore: baselines
    has_base = {e.branch for e in exps.values() if e.kind == "train" and e.source == "planner:baseline"
                and e.status not in (FAILED, SKIPPED)}
    failed = _failed_baselines(exps)
    max_fail = lcfg["planner"].get("max_baseline_failures", 2)
    todo = [b for b in branches.values() if b.enabled and b.name not in has_base
            and failed.get(b.name, 0) < max_fail]
    for b in sorted(todo, key=lambda b: (PRIOR_RANK[b.prior], b.est_gpu_h, b.name)):
        if room(b) and signature(b.name, []) not in tried:
            out.append(new(b, [], b.hypothesis or f"baseline of {b.name}", "planner:baseline"))
    if out:
        return out

    # 3. exploit: one-knob mutations of the best branches
    best = best_per_branch(exps)
    ranked = sorted(best.values(), key=lambda e: (-e.metrics["acc"], e.metrics["nll"]))
    for e in ranked[: lcfg["planner"]["exploit_top_k"]]:
        b = branches.get(e.branch)
        if b is None or not b.enabled:
            continue
        for ov, why in mutations(b, e, tried):
            if not room(b):
                break
            out.append(new(b, ov, why, "planner:mutate", e.id))
            tried.add(signature(b.name, ov))
    return out


def plan_recombination(ledger: Ledger, rcfg: dict) -> list[Experiment]:
    """Ensembles of the best run per branch (top max_members by accuracy), one per method,
    once at least min_members branches have a good run. Re-planned whenever the member set
    changes; the duplicate guardrail skips sets already evaluated."""
    exps = ledger.all()
    best = sorted(best_per_branch(exps).values(), key=lambda e: (-e.metrics["acc"], e.metrics["nll"]))
    members = [e.run for e in best[: rcfg["max_members"]] if e.run]
    if len(members) < rcfg["min_members"]:
        return []
    out = []
    n = ledger.max_num()
    for m in rcfg["methods"]:
        n += 1
        out.append(Experiment(id=f"e{n:03d}", branch=f"ensemble:{m}", kind="ensemble", config="",
                              members=members, method=m, source="planner:recombine",
                              hypothesis=f"{m} over the best run of {len(members)} branches"))
    return out

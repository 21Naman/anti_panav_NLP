"""Checks every experiment must pass BEFORE it runs. A failed check never runs the job;
the experiment is marked skipped (or the loop escalates for state-level problems).

  budget      GPU hours left >= estimate x margin; estimate <= per-run cap
  overrides   keys must start with an allowed prefix; forbidden key=value pairs blocked
              (pseudo-labelling stays off unless loop.allow_pseudo; data.* is set by the loop)
  duplicate   the same branch + overrides never runs twice (unless the earlier one failed)
  volume      at most max_experiments in total
  fair play   the test file must be unlabelled unless loop.allow_labelled_test (bench only)
"""

from __future__ import annotations

from dataclasses import dataclass

from liars.loop.catalog import as_dict, signature
from liars.loop.ledger import FAILED, SKIPPED, Experiment, State


@dataclass
class Check:
    ok: bool
    why: str = ""


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def check_overrides(overrides: list[str], gcfg: dict, allow_pseudo: bool) -> Check:
    for k, v in as_dict(overrides).items():
        if k.startswith("data.") or k.startswith("paths."):
            return Check(False, f"override {k} not allowed: the loop sets data and paths itself")
        if not any(k == p or k.startswith(p) for p in gcfg["allowed_override_prefixes"]):
            return Check(False, f"override {k} is outside the allowed prefixes {gcfg['allowed_override_prefixes']}")
        if k in ("pseudo.enabled",) and _truthy(v) and not allow_pseudo:
            return Check(False, "pseudo-labelling is off until the organizers allow it (loop.allow_pseudo)")
        if k == "train.extra_train" and v not in ("null", "None", "[]", "") and not allow_pseudo:
            return Check(False, "train.extra_train (extra/pseudo data) needs loop.allow_pseudo")
        for bad_k, bad_v in (gcfg.get("forbidden") or {}).items():
            if k == bad_k and str(v) == str(bad_v):
                return Check(False, f"{k}={v} is forbidden by loop.guardrails.forbidden")
    return Check(True)


def check_budget(exp: Experiment, st: State, bcfg: dict) -> Check:
    if exp.est_gpu_h > bcfg["per_run_max_h"]:
        return Check(False, f"estimate {exp.est_gpu_h:.1f} GPU h > per-run cap {bcfg['per_run_max_h']} h")
    left = bcfg["gpu_hours"] - st.gpu_h_used
    need = exp.est_gpu_h * bcfg["margin"]
    if need > left + 1e-9:
        return Check(False, f"needs ~{need:.1f} GPU h (estimate x margin), {left:.1f} h left")
    return Check(True)


def check_duplicate(exp: Experiment, done: dict[str, Experiment]) -> Check:
    if exp.kind != "train":
        key = (exp.method, tuple(sorted(exp.members)))
        for e in done.values():
            if e.id != exp.id and e.kind == "ensemble" and (e.method, tuple(sorted(e.members))) == key:
                return Check(False, f"same ensemble already evaluated as {e.id}")
        return Check(True)
    sig = signature(exp.branch, exp.overrides)
    for e in done.values():
        if e.id != exp.id and e.kind == "train" and e.status not in (FAILED, SKIPPED) and \
                sig in {signature(e.branch, e.overrides), signature(e.branch, e.planned_overrides or e.overrides)}:
            return Check(False, f"same branch + overrides already planned/run as {e.id}")
    return Check(True)


def pre_run(exp: Experiment, st: State, done: dict[str, Experiment], lcfg: dict) -> Check:
    if st.status != "active":
        return Check(False, f"loop is {st.status}")
    n_real = sum(1 for e in done.values() if e.status != SKIPPED and e.id != exp.id)
    if n_real >= lcfg["guardrails"]["max_experiments"]:
        return Check(False, f"max_experiments {lcfg['guardrails']['max_experiments']} reached")
    for c in (check_overrides(exp.overrides, lcfg["guardrails"], bool(lcfg.get("allow_pseudo"))),
              check_duplicate(exp, done),
              check_budget(exp, st, lcfg["budget"]) if exp.kind == "train" else Check(True)):
        if not c.ok:
            return c
    return Check(True)


def check_test_unlabelled(test_examples, allow_labelled: bool) -> Check:
    n = sum(ex.label is not None for ex in test_examples)
    if n and not allow_labelled:
        return Check(False, f"test file has {n} labelled rows: the loop never uses test labels; "
                            "set loop.allow_labelled_test only for synthetic bench sets")
    return Check(True)

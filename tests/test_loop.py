"""The improvement loop: recovery, guardrails, statistics, recombination, catalogue, and
full dry runs (simulated training) including failures, escalation, resume and proposals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from liars.config import load_config
from liars.cv import assign_folds, save_folds
from liars.ensemble import caruana_weights, vote
from liars.loop import evaluate as ev
from liars.loop.catalog import check_branch, load_branches, signature
from liars.loop.guardrails import check_budget, check_overrides, pre_run
from liars.loop.ledger import Experiment, Ledger, State
from liars.loop.planner import _neighbours
from liars.loop.recovery import classify, decide
from liars.preds import make_preds
from liars.schema import SPEAKERS, save_jsonl

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "configs" / "loop.yaml"


def _lcfg(overrides=()):
    return load_config(LOOP, list(overrides))


# ------------------------------------------------------------------ recovery
RCFG = _lcfg()["loop"]["recovery"]
FAM = {"micro_batch": "qlora.micro_batch_size", "grad_accum": "qlora.grad_accum", "epochs": "qlora.epochs",
       "lr": "qlora.lr", "micro_batch_default": 2, "grad_accum_default": 8, "epochs_default": 2, "lr_default": 1e-4}


@pytest.mark.parametrize("text,cls", [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", "oom"),
    ("RuntimeError: estimated 6.1 GPU h > qlora.max_gpu_hours 4.0: stopping", "time"),
    ("uncommitted changes: the run would NOT include them", "human"),
    ("requests.exceptions.ConnectionError: Temporary failure in name resolution", "transient"),
    ("KeyError: 'qlora'", "config"),
    ("RuntimeError: training collapsed 2 time(s)", "numeric"),
    ("Segmentation fault", "unknown"),
    ("step 401/1520 val acc 0.91\nRuntimeError: CUDA error: an illegal memory access", "unknown"),
    ("step 500/1520\nSegmentation fault", "unknown"),
    ("requests.exceptions.HTTPError: 503 Server Error: Service Unavailable", "transient"),
    ("401 Client Error: Unauthorized for url", "human"),
    ("ModuleNotFoundError: No module named 'torch'", "human"),
])
def test_classify(text, cls):
    assert classify(text) == cls


def test_recovery_actions():
    a = decide("oom", 1, [], RCFG, FAM)
    assert a.kind == "mutate" and a.overrides == ["qlora.micro_batch_size=1", "qlora.grad_accum=16"]
    assert decide("oom", 1, ["qlora.micro_batch_size=1"], RCFG, FAM).kind == "fail"
    assert decide("time", 1, [], RCFG, FAM).overrides == ["qlora.epochs=1"]
    assert decide("time", 2, ["qlora.epochs=1"], RCFG, FAM).overrides[0].startswith("train.subsample=")
    n = decide("numeric", 1, [], RCFG, FAM)
    assert n.kind == "mutate" and "qlora.lr=5e-05" in n.overrides and "seed=43" in n.overrides
    assert decide("human", 1, [], RCFG, FAM).kind == "escalate"
    t1, t2 = decide("transient", 1, [], RCFG, FAM), decide("transient", 2, [], RCFG, FAM)
    assert t1.kind == "retry" and t2.wait_s == 2 * t1.wait_s           # exponential backoff
    assert decide("transient", RCFG["max_retries"]["transient"] + 1, [], RCFG, FAM).kind == "fail"
    assert decide("config", 1, [], RCFG, FAM).kind == "fail"           # no blind retries on config errors


# ------------------------------------------------------------------ guardrails
def test_guardrails():
    g = _lcfg()["loop"]["guardrails"]
    assert check_overrides(["qlora.lr=5e-5", "seed=7"], g, False).ok
    assert not check_overrides(["data.train=x.jsonl"], g, False).ok
    assert not check_overrides(["paths.data_dir=/tmp"], g, False).ok
    assert not check_overrides(["pseudo.enabled=true"], g, False).ok   # not an allowed prefix either
    assert not check_overrides(["train.extra_train=[p.jsonl]"], {**g, "allowed_override_prefixes": ["train."]}, False).ok
    assert not check_overrides(["evil.key=1"], g, False).ok
    b = {"gpu_hours": 10, "per_run_max_h": 4, "margin": 1.2}
    assert check_budget(Experiment("e1", "x", "train", "c", est_gpu_h=3), State(gpu_h_used=6), b).ok
    assert not check_budget(Experiment("e1", "x", "train", "c", est_gpu_h=3), State(gpu_h_used=7), b).ok
    assert not check_budget(Experiment("e1", "x", "train", "c", est_gpu_h=5), State(), b).ok


def test_duplicates_are_blocked():
    lcfg = _lcfg()["loop"]
    a = Experiment("e1", "qlora_gemma", "train", "c", overrides=["qlora.lr=5e-5", "qlora.r=32"])
    b = Experiment("e2", "qlora_gemma", "train", "c", overrides=["qlora.r=32", "qlora.lr=5e-5"])
    assert signature(a.branch, a.overrides) == signature(b.branch, b.overrides)
    assert not pre_run(b, State(), {"e1": a}, lcfg).ok
    a.status = "failed"
    assert pre_run(b, State(), {"e1": a}, lcfg).ok                    # a failed run may be retried as new


# ------------------------------------------------------------------ evaluate
def _oof(n, acc, seed, collapse=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    p = rng.dirichlet(np.ones(4), n)
    hit = rng.random(n) < acc
    p[np.arange(n), np.where(hit, y, (y + 1) % 4)] += 3
    if collapse:
        p[:, 0] += 10
    p /= p.sum(1, keepdims=True)
    df = make_preds([f"x{i}" for i in range(n)], p)
    df.insert(1, "fold", 0)
    df.insert(2, "label", [SPEAKERS[i] for i in y])
    return df


def test_sanity():
    s = _lcfg()["loop"]["sanity"]
    assert ev.sanity(_oof(400, 0.9, 0), None, None, s, 400).ok
    assert not ev.sanity(_oof(400, 0.9, 0), None, None, s, 300).ok          # wrong held-out coverage
    assert not ev.sanity(_oof(400, 0.9, 0, collapse=True), None, None, s).ok  # letter collapse
    assert not ev.sanity(_oof(400, 0.0, 0), None, None, s).ok               # at/below chance
    assert not ev.sanity(_oof(400, 0.9, 0), None, ["t1"], s).ok             # test preds missing
    assert ev.sanity(_oof(400, 1.0, 0), None, None, s).warnings            # too good -> warning


def test_compare_needs_significance():
    pcfg = _lcfg()["loop"]["promotion"]
    base = _oof(1000, 0.80, 1)
    better = base.copy()
    worse_rows = better.index[(better[["p_A", "p_B", "p_C", "p_D"]].to_numpy().argmax(1)
                               != better["label"].map(SPEAKERS.index).to_numpy())]
    fix = worse_rows[:80]                       # +8 points on the same chats
    for i in fix:
        lab = better.loc[i, "label"]
        better.loc[i, ["p_A", "p_B", "p_C", "p_D"]] = [0.97 if s == lab else 0.01 for s in SPEAKERS]
    v = ev.compare(better, base, pcfg)
    assert v.promote and v.rule == "significant_accuracy_win" and v.detail["mcnemar_p"] < 1e-6
    tiny = base.copy()
    for i in worse_rows[:3]:                    # +0.3 points: noise, not progress
        lab = tiny.loc[i, "label"]
        tiny.loc[i, ["p_A", "p_B", "p_C", "p_D"]] = [0.4 if s == lab else 0.2 for s in SPEAKERS]
    assert not ev.compare(tiny, base, {**pcfg, "min_nll_gain": 1.0}).promote
    assert not ev.compare(better.iloc[:50], base, pcfg).promote             # too few aligned
    assert 0.02 < ev.min_detectable_gain(1000, 0.10) < 0.03


# ------------------------------------------------------------------ recombine
def test_vote_and_caruana():
    y = np.array([0, 1, 2, 3, 0])
    eye = np.eye(4)
    good, ok = eye[y] * 0.7 + 0.075, eye[[0, 1, 2, 0, 0]] * 0.7 + 0.075
    bad = eye[[1, 2, 3, 0, 1]] * 0.7 + 0.075
    P = np.stack([good, ok, bad])
    v = vote(P)
    assert np.allclose(v.sum(1), 1) and (v.argmax(1) == np.array([0, 1, 2, 0, 0])).all()
    w = caruana_weights(P, y, 10, 1)
    assert w[0] == w.max() and w[2] == 0 and np.isclose(w.sum(), 1)


# ------------------------------------------------------------------ catalogue
def test_real_branches_are_consistent():
    cfg = _lcfg()
    branches = load_branches(cfg)
    assert {"qlora_gemma", "deberta", "line_mil"} <= set(branches)
    errs = [e for b in branches.values() for e in check_branch(b, ROOT)]
    assert errs == []
    g = cfg["loop"]["guardrails"]
    for b in branches.values():                 # every knob the planner may mutate is allowed
        for k in b.space:
            assert check_overrides([f"{k}={b.space[k][1]}"], g, False).ok, k
    for fam in cfg["families"].values():
        assert set(fam) <= {"micro_batch", "grad_accum", "epochs", "lr"}


def test_neighbours():
    assert _neighbours([1, 2, 3], 2) == [3, 1]
    assert _neighbours([1, 2, 3], 1) == [2]
    assert _neighbours([1, 2, 3], 9) == [1, 2, 3]


# ------------------------------------------------------------------ full dry runs
@pytest.fixture
def dry_data(tmp_path, tiny):
    """tiny.jsonl split into train (labelled) + test (unlabelled), with folds."""
    import copy
    train = [copy.deepcopy(ex) for ex in tiny]
    test = []
    for ex in tiny[:10]:
        t = copy.deepcopy(ex)
        t.id, t.label = "t_" + t.id, None
        test.append(t)
    save_jsonl(train, tmp_path / "train.jsonl")
    save_jsonl(test, tmp_path / "test.jsonl")
    save_folds(assign_folds(train, 2, 0), tmp_path / "folds.csv")
    return tmp_path


def _dry(tmp, extra=(), failures=None):
    from liars.loop.orchestrator import build_loop
    cfg = load_config(LOOP, [f"loop.data.train={tmp}/train.jsonl", f"loop.data.test={tmp}/test.jsonl",
                             f"loop.data.folds={tmp}/folds.csv", f"loop.data_dir={tmp}/out",
                             f"loop.ledger_dir={tmp}/out/loop", "loop.promotion.min_aligned=5",
                             "loop.sanity.min_margin_over_chance=-1", "loop.sanity.max_pred_share_excess=1",
                             "loop.guardrails.escalate_on_suspicious=false", *extra])
    logs = []
    loop = build_loop(cfg, "dry", log=logs.append, sleep=lambda s: None, dry_failures=failures)
    return loop, logs


def test_dry_loop_runs_and_stops(dry_data):
    loop, logs = _dry(dry_data)
    st = loop.run(6)
    exps = loop.ledger.all()
    assert st.champion is not None and st.status in ("active", "stopped")
    assert st.gpu_h_used <= loop.cfg["budget"]["gpu_hours"] + 1e-9
    ens = [e for e in exps.values() if e.kind == "ensemble"]
    assert ens and all(e.status in ("promoted", "succeeded", "rejected") for e in ens), \
        [(e.id, e.status, e.decision) for e in ens]                     # recombination really ran
    assert {e.branch for e in exps.values() if e.source == "planner:baseline"} >= {"qlora_gemma", "deberta"}
    assert (loop.ledger.root / "report.md").exists()
    assert json.loads((loop.ledger.root / "state.json").read_text())["champion"] == st.champion
    # resumable: a second call continues from the saved state, never repeats an experiment
    n = len(exps)
    loop2, _ = _dry(dry_data)
    loop2.run(1)
    sigs = [signature(e.branch, e.overrides) for e in loop2.ledger.all().values()
            if e.kind == "train" and e.status not in ("failed", "skipped")]
    assert len(sigs) == len(set(sigs)) and len(loop2.ledger.all()) >= n


def test_dry_loop_recovers_from_oom(dry_data):
    loop, logs = _dry(dry_data, failures={"qlora_gemma": ["torch.OutOfMemoryError: CUDA out of memory", ""]})
    loop.run(1)
    e = next(x for x in loop.ledger.all().values() if x.branch == "qlora_gemma")
    assert e.status in ("promoted", "succeeded") and e.attempts == 2
    assert "qlora.micro_batch_size=1" in e.overrides and "qlora.grad_accum=16" in e.overrides


def test_dry_loop_escalates_and_resumes(dry_data):
    loop, logs = _dry(dry_data, failures={"qlora_gemma": ["uncommitted changes: commit first"]})
    st = loop.run(3)
    assert st.status == "escalated" and (loop.ledger.root / "ESCALATE.md").exists()
    assert loop.run(1).status == "escalated"          # stays stopped until a human resumes
    st = loop.ledger.load_state()
    st.status, st.stop_reason = "active", ""
    loop.ledger.save_state(st)
    loop2, _ = _dry(dry_data)
    assert loop2.run(1).status in ("active", "stopped")


def test_systemic_failure_escalates(dry_data):
    bad = ["Segmentation fault"] * 5
    loop, logs = _dry(dry_data, ["loop.recovery.systemic_after=2"],
                      failures={"qlora_gemma": bad, "qlora_qwen": bad})
    st = loop.run(2)
    assert st.status == "escalated" and "systemic" in st.stop_reason


def test_proposals_are_ingested_first_and_guarded(dry_data):
    loop, _ = _dry(dry_data)
    (loop.ledger.root).mkdir(parents=True, exist_ok=True)
    (loop.ledger.root / "proposals.yaml").write_text(yaml.safe_dump([
        {"id": "p1", "branch": "deberta", "overrides": ["deberta.lr=1e-5"], "hypothesis": "lower lr", "by": "test"},
        {"id": "p2", "branch": "nope", "overrides": [], "hypothesis": "unknown branch"},
        {"id": "p3", "branch": "deberta", "overrides": ["data.train=other.jsonl"], "hypothesis": "sneaky"},
    ]))
    loop.run(1)
    by_src = {e.source: e for e in loop.ledger.all().values()}
    assert by_src["proposal:p1"].status in ("promoted", "succeeded")
    assert by_src["proposal:p2"].status == "skipped"
    assert by_src["proposal:p3"].status == "skipped" and "data" in by_src["proposal:p3"].decision


def test_kaggle_executor_push_wait_fetch(tmp_path):
    """Fake kaggle_run.py: push ok, status 'running' once then 'complete', fetch prints the
    merged run name. Two jobs per wave; job names are slug-safe."""
    import subprocess
    from liars.loop.executor import KaggleExecutor

    calls, status_n = [], {}

    def runner(*args):
        calls.append(args)
        name = args[args.index("--name") + 1]
        if args[0] == "push":
            assert "--cmd" in args and args[args.index("--cmd") + 1].startswith("liars train --config ")
            return subprocess.CompletedProcess(args, 0, "pushed", "")
        if args[0] == "status":
            status_n[name] = status_n.get(name, 0) + 1
            return subprocess.CompletedProcess(args, 0, "running" if status_n[name] == 1 else
                                               f'{name} has status "KernelWorkerStatus.COMPLETE"', "")
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 0, f"  + runs.csv += llm_qlora_{name}", "")
        raise AssertionError(args)

    ex = KaggleExecutor(tmp_path, ["liars-comp"], 2, poll_s=0, runner=runner, sleep=lambda s: None)
    exps = [Experiment(f"e00{i}", "qlora_gemma", "train", "configs/llm_qlora.yaml", attempts=1) for i in (1, 2, 3)]
    res = ex.run_batch(exps, lambda e: ["train", "--config", e.config, "--set", "qlora.lr=5e-05"])
    assert [r.ok for r in res] == [True, True, True]
    assert res[0].run == "llm_qlora_loop-e001-1"
    pushes = [c for c in calls if c[0] == "push"]
    assert len(pushes) == 3
    # first wave = 2 jobs pushed before any fetch
    order = [c[0] for c in calls]
    assert order.index("fetch") > 1 and order[:2] == ["push", "push"]


def test_kaggle_executor_push_failure_is_reported(tmp_path):
    import subprocess
    from liars.loop.executor import KaggleExecutor
    from liars.loop.recovery import classify

    runner = lambda *a: subprocess.CompletedProcess(a, 1, "", "uncommitted changes: the run would NOT include them")
    ex = KaggleExecutor(tmp_path, ["liars-comp"], 2, poll_s=0, runner=runner, sleep=lambda s: None)
    r = ex.run_batch([Experiment("e001", "x", "train", "c.yaml", attempts=1)], lambda e: ["train"])[0]
    assert not r.ok and classify(r.log_tail) == "human"



def test_escalation_mid_batch_keeps_the_other_result(dry_data):
    # qlora_gemma and qlora_qwen run in the same batch; gemma escalates, qwen succeeds
    loop, _ = _dry(dry_data, failures={"qlora_gemma": ["uncommitted changes"]})
    st = loop.run(1)
    exps = {e.branch: e for e in loop.ledger.all().values() if e.kind == "train"}
    assert st.status == "escalated"
    assert exps["qlora_qwen"].status == "succeeded" and exps["qlora_qwen"].gpu_h > 0
    assert abs(st.gpu_h_used - sum(e.gpu_h for e in loop.ledger.all().values())) < 1e-9
    assert not [e for e in loop.ledger.all().values() if e.status in ("planned", "running")]


def test_failed_baseline_is_retried_after_resume(dry_data):
    loop, _ = _dry(dry_data, failures={"qlora_gemma": ["uncommitted changes"]})
    loop.run(1)
    st = loop.ledger.load_state()
    st.status, st.stop_reason, st.error_counts = "active", "", {}
    loop.ledger.save_state(st)
    loop2, _ = _dry(dry_data)          # the human fixed it: no more failures
    loop2.run(2)
    gem = [e for e in loop2.ledger.all().values() if e.branch == "qlora_gemma"]
    assert any(e.status in ("promoted", "succeeded") for e in gem)


def test_skipped_experiments_are_not_replanned(dry_data):
    # per-run cap below every GPU branch: they are skipped once, never re-planned
    loop, _ = _dry(dry_data, ["loop.budget.per_run_max_h=0.3"])
    loop.run(3)
    skipped = [e for e in loop.ledger.all().values() if e.status == "skipped"]
    sigs = [signature(e.branch, e.overrides) for e in skipped]
    assert skipped and len(sigs) == len(set(sigs))


def test_ids_never_collide(dry_data):
    loop, _ = _dry(dry_data)
    loop.run(4)
    lines = [json.loads(l) for l in (loop.ledger.root / "ledger.jsonl").read_text().splitlines()]
    first = {}
    for d in lines:                    # an id always refers to the same experiment
        key = (d["branch"], d["kind"], d["source"])
        assert first.setdefault(d["id"], key) == key, d["id"]


def test_kaggle_status_parsing(tmp_path):
    from liars.loop.executor import KaggleExecutor
    ex = KaggleExecutor(tmp_path, [], 2)
    assert ex._state('panavrg/liars-x has status "KernelWorkerStatus.COMPLETE"') == "COMPLETE"
    assert ex._state('panavrg/liars-x has status "KernelWorkerStatus.RUNNING"') == "RUNNING"
    assert ex._state('x has status "complete"') == "COMPLETE"
    assert ex._state("requests.exceptions.ConnectionError: NewConnectionError ... error") is None


def test_max_experiments_and_proposal_estimates():
    lcfg = _lcfg(["loop.guardrails.max_experiments=2"])["loop"]
    done = {f"e{i}": Experiment(f"e{i}", "deberta", "train", "c", overrides=[f"seed={i}"]) for i in (1, 2)}
    assert not pre_run(Experiment("e3", "deberta", "train", "c", overrides=["seed=3"]), State(), done, lcfg).ok
    done["e2"].status = "skipped"      # skipped ones do not count
    assert pre_run(Experiment("e3", "deberta", "train", "c", overrides=["seed=3"]), State(), done, lcfg).ok



def test_default_folds_path_ignores_loop_data_dir(dry_data, monkeypatch):
    """loop.data_dir only says where outputs go; folds default to data/canonical/folds_<stem>.csv."""
    import shutil
    from liars.loop.orchestrator import build_loop
    monkeypatch.chdir(dry_data)
    (dry_data / "data" / "canonical").mkdir(parents=True)
    shutil.copy(dry_data / "folds.csv", dry_data / "data" / "canonical" / "folds_train.csv")
    cfg = load_config(LOOP, [f"loop.data.train={dry_data}/train.jsonl", f"loop.data.test={dry_data}/test.jsonl",
                             "loop.data.folds=null", "loop.data_dir=elsewhere", "loop.ledger_dir=elsewhere/loop"])
    loop = build_loop(cfg, "dry", log=lambda *a: None)
    assert loop.data["folds"] == str(Path("data") / "canonical" / "folds_train.csv")
    assert not [e for e in loop.preflight() if "folds" in e]



def test_suspicious_result_escalates_on_real_executors_only(tmp_path):
    from liars.loop.orchestrator import Loop, build_loop
    from liars.loop.catalog import load_branches
    cfg = _lcfg()
    assert cfg["loop"]["guardrails"]["escalate_on_suspicious"] is True        # real runs: escalate
    loop = build_loop(_lcfg([f"loop.ledger_dir={tmp_path}/l"]), "dry", log=lambda *a: None)
    assert loop.cfg["guardrails"]["escalate_on_suspicious"] is False          # dry: warn only

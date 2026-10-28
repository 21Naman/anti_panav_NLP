"""Error taxonomy and the recovery policy: retry / mutate-and-retry / give up / escalate.

Classes (matched on the run's log tail, first match wins; patterns in configs/loop.yaml):
  transient  network, Kaggle API hiccups, DNS, 429/5xx  -> retry the SAME run with backoff
  oom        CUDA out of memory                         -> retry with a smaller micro batch
  time       the model's time probe / max_gpu_hours     -> retry with fewer epochs / subsample
  numeric    NaN / inf loss, fp16 overflow, collapse    -> retry with lower lr + new seed
  config     bad override / missing key / file missing  -> no retry (fix needed): fail
  human      uncommitted changes, quota, auth, missing package -> escalate at once
  unknown    anything else                              -> one plain retry, then fail
A class that fails `systemic_after` different experiments escalates: something is wrong
with the pipeline, not with one config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PATTERNS: dict[str, list[str]] = {
    "human": [r"uncommitted changes", r"quota (exceeded|exhausted|limit)|out of (GPU )?quota", r"HTTP.?401|401 Client Error|403 Forbidden|403 Client Error|Unauthorized",
              r"is PUBLIC",
              r"not pushed to origin", r"gated repo|accept the licen[cs]e",
              r"ModuleNotFoundError|No module named",    # broken environment, not a bad config
              r"KAGGLE_TIMEOUT_STILL_RUNNING",           # kernel may still burn quota: a human cancels it
              r"KAGGLE_OUTPUTS_NOT_FETCHED"],            # outputs exist on Kaggle; fetch by hand, do not rerun
    "oom": [r"CUDA out of memory", r"OutOfMemoryError", r"CUBLAS_STATUS_ALLOC_FAILED"],
    "time": [r"max_gpu_hours", r"exceeded the time limit", r"Kernel.*timed? ?out"],
    "numeric": [r"\bnan\b.*loss|loss.*\bnan\b", r"non-finite", r"overflow", r"collapsed"],
    "config": [r"KeyError", r"unknown model", r"FileNotFoundError", r"No such file",
               r"must be .* got", r"override must look like"],
    "transient": [r"ConnectionError|ConnectionReset|RemoteDisconnected", r"Temporary failure in name resolution",
                  r"NameResolutionError", r"HTTP.?(429|5\d\d)|(429|5\d\d) (Client|Server) Error|Too Many Requests",
                  r"Read timed out|ReadTimeout", r"SSLError|TLS handshake"],
}
ORDER = ["human", "oom", "time", "numeric", "config", "transient"]


def classify(log_tail: str, patterns: dict[str, list[str]] | None = None) -> str:
    pats = patterns or DEFAULT_PATTERNS
    for cls in ORDER:
        for p in pats.get(cls, []):
            if re.search(p, log_tail, flags=re.IGNORECASE):
                return cls
    return "unknown"


@dataclass
class Action:
    kind: str                  # retry | mutate | fail | escalate
    overrides: list[str]       # extra overrides for mutate
    wait_s: float = 0.0
    why: str = ""


def _get(overrides: list[str], key: str, default):
    for o in reversed(overrides):
        k, _, v = o.partition("=")
        if k == key:
            try:
                return type(default)(v) if default is not None else v
            except ValueError:
                return default
    return default


def decide(err_class: str, attempt: int, overrides: list[str], rcfg: dict, family_keys: dict) -> Action:
    """attempt = how many times this experiment has already failed (1 after the first failure).
    family_keys: names of the knobs for this branch's model family, e.g.
    {"micro_batch": "qlora.micro_batch_size", "grad_accum": "qlora.grad_accum", "epochs": ..., "lr": ...}."""
    max_retries = rcfg["max_retries"].get(err_class, 0)
    if err_class == "human":
        return Action("escalate", [], why="needs a human (credentials, quota, uncommitted code, licence)")
    if attempt > max_retries:
        return Action("fail", [], why=f"{err_class}: {attempt} failure(s) > max_retries {max_retries}")
    backoff = rcfg["backoff_s"] * (2 ** (attempt - 1))
    if err_class == "transient":
        return Action("retry", [], wait_s=backoff, why="transient error: same run again after backoff")
    if err_class == "unknown":
        return Action("retry", [], wait_s=backoff, why="unknown error: one plain retry")
    if err_class == "config":
        return Action("fail", [], why="configuration error: needs a code/config fix, not a retry")
    fk = family_keys or {}
    if err_class == "oom" and fk.get("micro_batch"):
        mb = _get(overrides, fk["micro_batch"], None)
        mb = int(mb) if mb is not None else int(fk.get("micro_batch_default", 2))
        if mb <= 1:
            return Action("fail", [], why="OOM at micro batch 1: needs a smaller model or max_len")
        ga = int(_get(overrides, fk["grad_accum"], None) or fk.get("grad_accum_default", 8))
        return Action("mutate", [f"{fk['micro_batch']}={mb // 2}", f"{fk['grad_accum']}={ga * 2}"],
                      why=f"OOM: micro batch {mb} -> {mb // 2}, grad accum {ga} -> {ga * 2} (same effective batch)")
    if err_class == "time" and fk.get("epochs"):
        ep = float(_get(overrides, fk["epochs"], None) or fk.get("epochs_default", 2))
        if ep > 1:
            return Action("mutate", [f"{fk['epochs']}=1"], why=f"too slow: epochs {ep:g} -> 1")
        return Action("mutate", [f"train.subsample={rcfg['time_subsample']}"],
                      why=f"too slow at 1 epoch: subsample {rcfg['time_subsample']} training chats")
    if err_class == "numeric" and fk.get("lr"):
        lr = float(_get(overrides, fk["lr"], None) or fk.get("lr_default", 1e-4))
        seed = int(_get(overrides, "seed", None) or 42)
        return Action("mutate", [f"{fk['lr']}={lr * rcfg['numeric_lr_factor']:.3g}", f"seed={seed + 1}"],
                      why=f"numeric instability: lr {lr:g} -> {lr * rcfg['numeric_lr_factor']:.3g}, seed {seed + 1}")
    return Action("fail", [], why=f"{err_class}: no automatic fix for this branch")

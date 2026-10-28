"""Loop state (state.json, rewritten atomically) and the experiment ledger (ledger.jsonl,
append-only: every status change is a new line, the last line per id wins). Resumable:
a crashed or interrupted loop picks up from here."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Experiment statuses
PLANNED, RUNNING, SUCCEEDED, FAILED, REJECTED, PROMOTED, ESCALATED, SKIPPED = (
    "planned", "running", "succeeded", "failed", "rejected", "promoted", "escalated", "skipped")
DONE = {SUCCEEDED, FAILED, REJECTED, PROMOTED, ESCALATED, SKIPPED}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Experiment:
    id: str
    branch: str
    kind: str                      # train | ensemble
    config: str                    # config path (train) or "" (ensemble)
    overrides: list[str] = field(default_factory=list)
    planned_overrides: list[str] = field(default_factory=list)  # before recovery changed them
    hypothesis: str = ""
    source: str = "planner"        # planner | proposal:<who> | recovery
    parent: str | None = None
    members: list[str] = field(default_factory=list)   # ensemble: run names
    method: str = ""               # ensemble: vote | caruana | weighted | mean | stack | auto
    est_gpu_h: float = 0.0
    status: str = PLANNED
    attempts: int = 0
    run: str | None = None         # trainer run name (data/oof/<run>.parquet)
    metrics: dict = field(default_factory=dict)
    decision: str = ""             # why it was promoted / rejected / escalated
    errors: list[dict] = field(default_factory=list)
    gpu_h: float = 0.0
    created: str = field(default_factory=now)
    updated: str = field(default_factory=now)

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class State:
    status: str = "active"         # active | stopped | escalated
    iteration: int = 0
    champion: str | None = None    # experiment id
    champion_history: list[dict] = field(default_factory=list)
    gpu_h_used: float = 0.0
    no_improve_iters: int = 0
    stop_reason: str = ""
    error_counts: dict = field(default_factory=dict)   # error class -> count (systemic failures)
    n_promotions: int = 0
    holdout_queries: int = 0       # champion comparisons made on the holdout (reuse budget)
    updated: str = field(default_factory=now)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Ledger:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.log_path = self.root / "ledger.jsonl"

    # ------------------------------------------------------------------ state
    def load_state(self) -> State:
        if not self.state_path.exists():
            return State()
        return State.from_dict(json.loads(self.state_path.read_text()))

    def save_state(self, st: State) -> None:
        st.updated = now()
        _atomic_write(self.state_path, json.dumps(asdict(st), indent=1))

    # ------------------------------------------------------------------ experiments
    def put(self, exp: Experiment) -> None:
        exp.updated = now()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(exp)) + "\n")

    def all(self) -> dict[str, Experiment]:
        out: dict[str, Experiment] = {}
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = Experiment.from_dict(json.loads(line))
                    out[e.id] = e
        return out

    def get(self, exp_id: str) -> Experiment | None:
        return self.all().get(exp_id)

    def max_num(self) -> int:
        """Largest numeric experiment id so far (ids are never reused, even if some planned
        experiments were never written)."""
        nums = [int(i[1:]) for i in self.all() if i[1:].isdigit()]
        return max(nums, default=0)

    def next_id(self, prefix: str = "e") -> str:
        return f"{prefix}{self.max_num() + 1:03d}"

    def escalate(self, st: State, why: str, context: dict[str, Any]) -> Path:
        """Stop the loop and leave a human-readable note."""
        st.status = "escalated"
        st.stop_reason = why
        self.save_state(st)
        path = self.root / "ESCALATE.md"
        lines = [f"# Loop escalated ({now()})", "", f"**Why:** {why}", "",
                 "The loop has stopped and will not restart until a human acts.", "",
                 "## Context", ""]
        lines += [f"- **{k}**: {v}" for k, v in context.items()]
        lines += ["", "## To resume", "",
                  "1. Fix the cause (or decide to skip it).",
                  "2. `liars loop resume` (sets the status back to active), then `liars loop run`."]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

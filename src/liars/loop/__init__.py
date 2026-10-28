"""Self-improving experiment loop: plan -> act -> test -> evaluate -> improve -> repeat.

Modules
  ledger     persistent state (data/loop/state.json) + append-only experiment log (ledger.jsonl)
  catalog    branches (candidate methods) and their search spaces, from configs/loop.yaml
  planner    what to run next: explore branches -> exploit (one-knob mutations) -> recombine -> stop
  guardrails checks before a run (budget, allowed overrides, fair-play switches, duplicates)
  executor   how a run happens: local subprocess, Kaggle GPU job, or a simulated dry run
  recovery   error taxonomy -> retry / mutate-and-retry / give up / escalate to a human
  evaluate   sanity tests on outputs, metrics, statistical champion promotion
  recombine  majority vote, Caruana ensemble selection, plus the existing weighted/stack ensemble
  preflight  runtime-environment checks (`liars loop doctor`)
  orchestrator  the cycle itself (`liars loop run`)

The deterministic code makes every decision reproducible and auditable. An LLM coding agent
(Claude Code: .claude/agents/loop-planner.md) adds judgement by writing proposals to
data/loop/proposals.yaml, which go through the same guardrails as everything else. No model
API is ever called from this package.
"""
